"""Spatial Binary Markov Chain with Duration (BMCD) — model, likelihood and I/O.

Implements the spatial extension of the single-site BMCD described in
``Spatialisation-Rain-Occurrence-Generator-article/main.tex``. Re-uses the
single-site fits produced by
``article_code/notebooks_article_pipeline/01_prepare_data_and_fit_distributions.ipynb``.

This module implements the **direct Linear Model of Coregionalization (LMC)**
construction (the ``\\new{}`` blue model in ``main.tex``): a bivariate centred
Gaussian field ``(Z^(0), Z^(1))``, one component per spell type, built from three
independent latent fields ``W^c, W^(0), W^(1)`` and a mixing weight ``lambda``,

    Z^(r)(s) = (-1)^r * sqrt(lambda) * W^c(s) + sqrt(1-lambda) * W^(r)(s),

selected by the current spell type, ``Z(s) = Z^(R(s))(s)`` (no sign flip). The
model parameter is ``theta = (lambda, sigma_wc, sigma_w0, sigma_w1)`` and the
state-dependent covariance blocks are

    C^(0,0)(h) = lambda * rho_wc(h) + (1-lambda) * rho_w0(h),
    C^(1,1)(h) = lambda * rho_wc(h) + (1-lambda) * rho_w1(h),
    C^(0,1)(h) = -lambda * rho_wc(h),

with exponential latent correlations ``rho_k(h) = exp(-h / sigma_k)``. It reduces
to the original single-latent-field model when ``lambda = 1`` (then ``sigma_wc``
plays the role of the old scalar range ``sigma``); that frozen single-field
implementation is kept under ``archive_old_single_latent_field/``.

The module also supports the **state-dependent-lambda variant** (the blue
"alternative suggestion", Eqs. ``lmc_fields_direct_state_dep`` /
``lmc_blocks_inline_state_dep`` in ``main.tex``): one mixing weight per spell
type,

    Z^(r)(s) = (-1)^r * sqrt(lambda_r) * W^c(s) + sqrt(1-lambda_r) * W^(r)(s),

with ``theta = (lambda_0, lambda_1, sigma_wc, sigma_w0, sigma_w1)`` and blocks

    C^(0,0)(h) = lambda_0 * rho_wc(h) + (1-lambda_0) * rho_w0(h),
    C^(1,1)(h) = lambda_1 * rho_wc(h) + (1-lambda_1) * rho_w1(h),
    C^(0,1)(h) = -sqrt(lambda_0 * lambda_1) * rho_wc(h).

Every function taking ``theta`` accepts both parametrisations (dispatch by
length/keys, see :func:`normalize_theta`); the shared-lambda model is the
special case ``lambda_0 = lambda_1``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.stats
from scipy.optimize import minimize
from scipy.special import erf
from scipy.stats import multivariate_normal, norm
from tqdm import tqdm

from article_code.util_files import config
from article_code.util_files.data_load import (
    from_date_to_season,
    list_station_files,
)
from article_code.util_files.spell_models import (
    make_cdf_fitted_hdeGPD_from_params,
)
from article_code.util_files.statistics import get_proba_leaving_state_n_kozu


# ---------------------------------------------------------------------------
# Single-site model integration
# ---------------------------------------------------------------------------


def _make_cdf_fitted_mix_geom_from_params(pi: float, p1: float, p2: float) -> Callable:
    """CDF of a 2-component mixture of geometric distributions on {1, 2, ...}.

    Closed form (mirrors the inline definition in the original notebook):
        F(z) = pi * (1 - (1-p1)**z) + (1-pi) * (1 - (1-p2)**z),  z >= 1
        F(z) = 0,                                                z < 1
    """
    pi, p1, p2 = float(pi), float(p1), float(p2)

    def cdf(z):
        if z <= 0:
            return 0.0
        return pi * (1.0 - (1.0 - p1) ** z) + (1.0 - pi) * (1.0 - (1.0 - p2) ** z)

    return cdf


def _normalise_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``x_norm`` / ``y_norm`` columns in [0, 1] from ``lon`` / ``lat``."""
    out = df.copy()
    min_lat, max_lat = out["lat"].min(), out["lat"].max()
    min_lon, max_lon = out["lon"].min(), out["lon"].max()
    out["x_norm"] = (out["lon"] - min_lon) / (max_lon - min_lon)
    out["y_norm"] = (out["lat"] - min_lat) / (max_lat - min_lat)
    return out


def build_dict_model_params(
    df_fit_dry: pd.DataFrame,
    df_fit_wet: pd.DataFrame,
    spells: dict,
    season: str = "spring",
    stations: Optional[Iterable[str]] = None,
    show_progress: bool = True,
) -> Dict[str, dict]:
    """Build the per-station spatial-model parameter dictionary.

    For every station in ``stations`` (or every common station if ``None``)
    the returned dict holds:

        x_norm, y_norm                 -- normalised coordinates (set externally)
        params_dry = (f_1, xi, sigma, kappa)
        params_wet = (pi, p1, p2)
        q_d_dry_function : callable    -- d -> q^(0)(d), see Eq. (1) of the article
        q_d_wet_function : callable    -- d -> q^(1)(d)

    ``df_fit_dry`` and ``df_fit_wet`` must already carry the columns produced
    by notebook 01 plus ``city``, ``season``, ``x_norm``, ``y_norm`` (see
    :func:`prepare_station_fit_table`).
    """
    if stations is None:
        stations = sorted(
            set(df_fit_dry["city"]).intersection(df_fit_wet["city"]).intersection(spells.keys())
        )
    iterator = tqdm(stations) if show_progress else stations

    dict_model_params: Dict[str, dict] = {}
    for city in iterator:
        row_dry = df_fit_dry[
            (df_fit_dry["city"] == city) & (df_fit_dry["season"] == season)
        ]
        row_wet = df_fit_wet[
            (df_fit_wet["city"] == city) & (df_fit_wet["season"] == season)
        ]
        if row_dry.empty or row_wet.empty:
            continue

        xi, sigma_gpd, kappa = (
            row_dry["xi"].item(),
            row_dry["sigma"].item(),
            row_dry["kappa"].item(),
        )
        pi, p1, p2 = (
            row_wet["pi"].item(),
            row_wet["p1"].item(),
            row_wet["p2"].item(),
        )

        # f_1 = P(dry spell duration == 1) estimated on this season's durations
        all_durations = spells[city]["dry_spell"]["duration_spell"]
        all_dates = spells[city]["dry_spell"]["start_date_spell"]
        season_durations = [
            dur
            for dur, date in zip(all_durations, all_dates)
            if from_date_to_season(date) == season
        ]
        f_1 = sum(s == 1 for s in season_durations) / len(season_durations)

        cdf_dry = make_cdf_fitted_hdeGPD_from_params(f_1, xi, sigma_gpd, kappa)
        cdf_wet = _make_cdf_fitted_mix_geom_from_params(pi, p1, p2)

        dict_model_params[city] = {
            "x_norm": float(row_dry["x_norm"].item()),
            "y_norm": float(row_dry["y_norm"].item()),
            "lon": float(row_dry["lon"].item()),
            "lat": float(row_dry["lat"].item()),
            "country": row_dry["country"].item(),
            "params_dry": (f_1, xi, sigma_gpd, kappa),
            "params_wet": (pi, p1, p2),
            "q_d_dry_function": lambda d, cdf=cdf_dry: get_proba_leaving_state_n_kozu(cdf, d),
            "q_d_wet_function": lambda d, cdf=cdf_wet: get_proba_leaving_state_n_kozu(cdf, d),
        }
    return dict_model_params


def prepare_station_fit_table(
    df_fit_dry: pd.DataFrame,
    df_fit_wet: pd.DataFrame,
    stations_metadata: pd.DataFrame,
    countries: Optional[Sequence[str]] = None,
    season: str = "spring",
):
    """Merge fit parameters with station metadata and add normalised coords.

    Returns ``(df_fit_dry_subset, df_fit_wet_subset)`` filtered to ``season``
    and ``countries`` (if provided). Both frames carry ``city``, ``country``,
    ``lat``, ``lon``, ``x_norm``, ``y_norm`` columns ready for
    :func:`build_dict_model_params`.
    """
    df_dry = df_fit_dry.copy()
    df_wet = df_fit_wet.copy()
    if "city" not in df_dry.columns:
        df_dry["city"] = df_dry["data_source"].map(lambda s: s.split()[0])
        df_dry["season"] = df_dry["data_source"].map(lambda s: s.split()[-1])
    if "season" not in df_wet.columns:
        df_wet["season"] = df_wet["data_source"].map(lambda s: s.split()[-1])

    df_dry = df_dry.merge(stations_metadata, on="city", how="inner")
    df_wet = df_wet.merge(stations_metadata, on="city", how="inner")

    mask_dry = df_dry["season"] == season
    mask_wet = df_wet["season"] == season
    if countries is not None:
        mask_dry &= df_dry["country"].isin(list(countries))
        mask_wet &= df_wet["country"].isin(list(countries))
    df_dry = _normalise_coords(df_dry[mask_dry])
    df_wet_subset = df_wet[mask_wet].merge(
        df_dry[["city", "x_norm", "y_norm"]], on="city", how="inner"
    )
    return df_dry.reset_index(drop=True), df_wet_subset.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Spatial Markov step and simulation
# ---------------------------------------------------------------------------


def exponential_cov_func(x_norm1, x_norm2, y_norm1, y_norm2, sigma: float = 0.3):
    """Exponential covariance ``exp(-d / sigma)`` between normalised coordinates."""
    d = np.sqrt((x_norm2 - x_norm1) ** 2 + (y_norm2 - y_norm1) ** 2)
    return np.exp(-d / sigma)


def build_C_from_sigma(
    station_names: Sequence[str],
    params_by_station: Dict[str, dict],
    sigma: float,
) -> np.ndarray:
    """Build the spatial covariance matrix C from a scalar range parameter ``sigma``."""
    sigma = float(sigma)
    return np.array(
        [
            [
                exponential_cov_func(
                    params_by_station[c1]["x_norm"],
                    params_by_station[c2]["x_norm"],
                    params_by_station[c1]["y_norm"],
                    params_by_station[c2]["y_norm"],
                    sigma=sigma,
                )
                for c1 in station_names
            ]
            for c2 in station_names
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# LMC parameter theta = (lambda, sigma_wc, sigma_w0, sigma_w1) and covariance
# ---------------------------------------------------------------------------


def normalize_theta(theta) -> Dict[str, float]:
    """Coerce ``theta`` to a dict; dispatches shared vs state-dependent lambda.

    Accepted inputs:

    - **shared-lambda model** (Eq. ``lmc_fields_direct``): a length-4 sequence
      ``(lam, sigma_wc, sigma_w0, sigma_w1)`` or a mapping with key ``lam``
      (also spelled ``lambda``). The returned dict has keys
      ``lam, lam0, lam1, sigma_wc, sigma_w0, sigma_w1`` with ``lam0 = lam1 = lam``.
    - **state-dependent-lambda model** (Eq. ``lmc_fields_direct_state_dep``):
      a length-5 sequence ``(lam0, lam1, sigma_wc, sigma_w0, sigma_w1)`` or a
      mapping with keys ``lam0`` / ``lam1`` (and no ``lam``). The returned dict
      has keys ``lam0, lam1, sigma_wc, sigma_w0, sigma_w1`` — **no** ``lam`` key,
      even if ``lam0 == lam1``.

    Downstream code reads ``lam0`` / ``lam1`` (always present); the presence of
    the ``lam`` key marks the shared parametrisation.
    """
    if isinstance(theta, dict):
        if "lam" in theta or "lambda" in theta:
            lam = float(theta.get("lam", theta.get("lambda")))
            lam0 = lam1 = lam
            shared = True
        else:
            lam0, lam1 = float(theta["lam0"]), float(theta["lam1"])
            shared = False
        s_wc, s_w0, s_w1 = theta["sigma_wc"], theta["sigma_w0"], theta["sigma_w1"]
    else:
        vals = tuple(float(v) for v in theta)
        if len(vals) == 4:
            lam, s_wc, s_w0, s_w1 = vals
            lam0 = lam1 = lam
            shared = True
        elif len(vals) == 5:
            lam0, lam1, s_wc, s_w0, s_w1 = vals
            shared = False
        else:
            raise ValueError(
                f"theta must have 4 (shared lambda) or 5 (state-dependent lambda) "
                f"components, got {len(vals)}"
            )
    out = {
        "lam0": float(lam0),
        "lam1": float(lam1),
        "sigma_wc": float(s_wc),
        "sigma_w0": float(s_w0),
        "sigma_w1": float(s_w1),
    }
    if shared:
        out = {"lam": float(lam0), **out}
    return out


def build_field_cov_matrices(
    station_names: Sequence[str],
    params_by_station: Dict[str, dict],
    theta,
) -> Dict[str, np.ndarray]:
    """The three latent exponential correlation matrices ``rho_wc, rho_w0, rho_w1``.

    Each is the ``J x J`` matrix ``exp(-h / sigma_k)`` of the corresponding latent
    field, reusing :func:`build_C_from_sigma`.
    """
    th = normalize_theta(theta)
    return {
        "wc": build_C_from_sigma(station_names, params_by_station, th["sigma_wc"]),
        "w0": build_C_from_sigma(station_names, params_by_station, th["sigma_w0"]),
        "w1": build_C_from_sigma(station_names, params_by_station, th["sigma_w1"]),
    }


def build_lmc_blocks(
    station_names: Sequence[str],
    params_by_station: Dict[str, dict],
    theta,
) -> Dict[str, np.ndarray]:
    """The LMC covariance blocks ``C^(0,0), C^(1,1), C^(0,1)`` — Eq. (lmc_blocks_direct).

    Returns ``{"C00", "C11", "C01"}``, each a ``J x J`` matrix:

        C00 = lambda_0 * rho_wc + (1-lambda_0) * rho_w0,
        C11 = lambda_1 * rho_wc + (1-lambda_1) * rho_w1,
        C01 = -sqrt(lambda_0 * lambda_1) * rho_wc.

    Shared-lambda model: ``lambda_0 = lambda_1 = lambda`` (then ``C01`` reduces
    to ``-lambda * rho_wc``, Eq. lmc_blocks_inline); state-dependent variant:
    Eq. lmc_blocks_inline_state_dep. Diagonals of ``C00`` and ``C11`` are 1
    (unit-variance sites). ``C01`` is the negative dry-wet cross-block carrying
    the sign.
    """
    th = normalize_theta(theta)
    lam0, lam1 = th["lam0"], th["lam1"]
    rho = build_field_cov_matrices(station_names, params_by_station, th)
    return {
        "C00": lam0 * rho["wc"] + (1.0 - lam0) * rho["w0"],
        "C11": lam1 * rho["wc"] + (1.0 - lam1) * rho["w1"],
        "C01": -np.sqrt(lam0 * lam1) * rho["wc"],
    }


def build_state_selected_cov(blocks: Dict[str, np.ndarray], R: np.ndarray) -> np.ndarray:
    """Assemble the state-selected covariance ``Sigma^(R)`` — Eq. (state_selected_cov).

    Entry ``(j, j')`` is the LMC block selected by the spell types ``(R[j], R[j'])``:
    ``C00`` for dry-dry, ``C11`` for wet-wet, ``C01`` otherwise. ``R`` is the 0/1
    spell-type configuration; ``blocks`` comes from :func:`build_lmc_blocks`
    (already restricted to the same station ordering as ``R``).
    """
    R = np.asarray(R).astype(int)
    rj = R[:, None]
    rk = R[None, :]
    C00, C11, C01 = blocks["C00"], blocks["C11"], blocks["C01"]
    return np.where(
        (rj == 0) & (rk == 0), C00,
        np.where((rj == 1) & (rk == 1), C11, C01),
    )


def _assemble_selected_field(R, lam0, lam1, Wc, W0, W1):
    """Assemble the state-selected latent variable ``Z = Z^(R)`` — Eq. (lmc_fields_direct).

    ``Z^(r) = (-1)^r sqrt(lambda_r) Wc + sqrt(1-lambda_r) W^(r)``, then select the
    component matching the current spell type ``R``. No further sign flip.
    Shared-lambda model: ``lam0 == lam1``; state-dependent variant:
    Eq. (lmc_fields_direct_state_dep).
    """
    Z0 = np.sqrt(lam0) * Wc + np.sqrt(1.0 - lam0) * W0    # r = 0, (-1)^0 = +1
    Z1 = -np.sqrt(lam1) * Wc + np.sqrt(1.0 - lam1) * W1   # r = 1, (-1)^1 = -1
    return np.where(R == 0, Z0, Z1)


def _exit_probs(R, D, params_by_station, station_names, clip_q):
    """Per-station exit probabilities ``q`` and Gaussian thresholds ``z``."""
    m = len(station_names)
    q = np.empty(m, dtype=float)
    for j, name in enumerate(station_names):
        f = params_by_station[name][
            "q_d_dry_function" if R[j] == 0 else "q_d_wet_function"
        ]
        q[j] = float(f(D[j]))
    q = np.clip(q, clip_q, 1.0 - clip_q)
    return q, norm.ppf(q)


def step_spatial_markov(
    R: np.ndarray,
    D: np.ndarray,
    field_cov: Dict[str, np.ndarray],
    lam: float,
    params_by_station: Dict[str, dict],
    rng: Optional[np.random.Generator] = None,
    station_names: Optional[Sequence[str]] = None,
    clip_q: float = 1e-12,
):
    """One step of the spatial LMC Markov chain — see Eq. (spatialized_markov) of main.tex.

    Draws the three latent fields ``W^c, W^(0), W^(1)`` (each a centred Gaussian
    with the corresponding exponential correlation in ``field_cov``), assembles the
    state-selected variable ``Z`` and applies the switch/persist rule. The Gaussian
    vectors are drawn with ``rng.multivariate_normal`` (SVD factorisation) — the
    per-step ("svd") simulator. ``lam`` is either a scalar (shared-lambda model)
    or a pair ``(lam0, lam1)`` (state-dependent variant).
    Returns ``(R_next, D_next, switch, q, z)``.
    """
    lam0, lam1 = (float(lam), float(lam)) if np.isscalar(lam) else (
        float(lam[0]), float(lam[1])
    )
    if station_names is None:
        station_names = sorted(list(params_by_station.keys()))
    m = len(station_names)
    mean = np.zeros(m)

    def _draw(C):
        if rng is not None:
            return rng.multivariate_normal(mean=mean, cov=C)
        return multivariate_normal(mean=mean, cov=C).rvs()

    Wc, W0, W1 = _draw(field_cov["wc"]), _draw(field_cov["w0"]), _draw(field_cov["w1"])
    Z = _assemble_selected_field(R, lam0, lam1, Wc, W0, W1)

    q, z = _exit_probs(R, D, params_by_station, station_names, clip_q)
    switch = Z <= z

    R_next = R.copy()
    D_next = D.copy()
    R_next[switch] = 1 - R_next[switch]
    D_next[switch] = 1
    D_next[~switch] = D_next[~switch] + 1
    return R_next, D_next, switch, q, z


def simulate_cholesky(
    theta,
    params_by_station: Dict[str, dict],
    n_steps: int,
    n_burn: int = 500,
    station_names: Optional[Sequence[str]] = None,
    R0: Optional[np.ndarray] = None,
    D0: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    clip_q: float = 1e-12,
    jitter: float = 0.0,
) -> dict:
    """Simulate a spatial LMC-BMCD trajectory by Cholesky factorisation — Section 5 of main.tex.

    ``theta = (lambda, sigma_wc, sigma_w0, sigma_w1)`` for the shared-lambda
    model, or ``(lambda_0, lambda_1, sigma_wc, sigma_w0, sigma_w1)`` for the
    state-dependent variant (tuple or dict, see :func:`normalize_theta`).
    Three steps, mirroring the article:

    1. **Spatial Cholesky factorisation** — build the three latent covariance
       matrices ``Sigma_wc, Sigma_w0, Sigma_w1`` (exponential with ranges
       ``sigma_wc, sigma_w0, sigma_w1``) on the rescaled stations and factor each
       ``Sigma_k = L_k L_k^T`` once. ``jitter`` defaults to ``0``; set a small
       positive value (e.g. ``1e-6``) to stabilise near-singular factorisations.
    2. **Initialisation + burn-in** — if ``R0`` / ``D0`` are not provided, start
       deterministically all-dry (``R0[j] = 0``, ``D0[j] = 1``), then run
       ``n_burn`` steps and discard them.
    3. **Sequential update** — for each kept day draw independent
       ``V^c, V^(0), V^(1) ~ N(0, I_J)``, form ``W^k = L_k V^k``, assemble
       ``Z^(r) = (-1)^r sqrt(lambda) W^c + sqrt(1-lambda) W^(r)`` and select the
       component ``Z = Z^(R)`` (Eq. z_selection_direct), then apply the threshold.

    Returns the same ``history`` dict layout as :func:`simulate_history` (keys
    ``station_names``, ``R``, ``D``, ``S``, ``q``, ``z``, ``theta``) so the
    simulated trajectory can be fed back into :func:`mle_theta_pairwise`.
    """
    th = normalize_theta(theta)
    lam0, lam1 = th["lam0"], th["lam1"]
    if station_names is None:
        station_names = sorted(list(params_by_station.keys()))
    m = len(station_names)

    # Step 1: factor the three latent covariance matrices once.
    field_cov = build_field_cov_matrices(station_names, params_by_station, th)

    def _chol(C):
        return np.linalg.cholesky(C + jitter * np.eye(m)) if jitter > 0 else np.linalg.cholesky(C)

    Lc, L0, L1 = _chol(field_cov["wc"]), _chol(field_cov["w0"]), _chol(field_cov["w1"])

    rng = np.random.default_rng(seed)

    # Step 2: initialisation (deterministic all-dry start by default).
    if R0 is None:
        R = np.zeros(m, dtype=int)
    else:
        R = np.asarray(R0, dtype=int).copy()
    if D0 is None:
        D = np.ones(m, dtype=int)
    else:
        D = np.asarray(D0, dtype=int).copy()

    def step(R, D):
        Wc = Lc @ rng.standard_normal(m)
        W0 = L0 @ rng.standard_normal(m)
        W1 = L1 @ rng.standard_normal(m)
        Z = _assemble_selected_field(R, lam0, lam1, Wc, W0, W1)
        q, z = _exit_probs(R, D, params_by_station, station_names, clip_q)
        switch = Z <= z
        R_next = R.copy()
        D_next = D.copy()
        R_next[switch] = 1 - R_next[switch]
        D_next[switch] = 1
        D_next[~switch] = D_next[~switch] + 1
        return R_next, D_next, switch.astype(float), q, z

    # Burn-in.
    for _ in range(n_burn):
        R, D, _, _, _ = step(R, D)

    # Step 3: sequential update, recording the trajectory.
    R_hist = np.empty((n_steps + 1, m), dtype=float)
    D_hist = np.empty((n_steps + 1, m), dtype=float)
    S_hist = np.empty((n_steps, m), dtype=float)
    q_hist = np.empty((n_steps, m), dtype=float)
    z_hist = np.empty((n_steps, m), dtype=float)
    R_hist[0], D_hist[0] = R, D
    for n in range(n_steps):
        R, D, S, q, z = step(R, D)
        R_hist[n + 1], D_hist[n + 1] = R, D
        S_hist[n], q_hist[n], z_hist[n] = S, q, z

    return {
        "station_names": list(station_names),
        "R": R_hist,
        "D": D_hist,
        "S": S_hist,
        "q": q_hist,
        "z": z_hist,
        "theta": th,
    }


def simulate_cholesky_seasonal(
    dates,
    theta_by_season: Dict[str, object],
    params_by_season: Dict[str, Dict[str, dict]],
    station_names: Optional[Sequence[str]] = None,
    n_burn: int = 1000,
    seed: Optional[int] = None,
    clip_q: float = 1e-12,
    jitter: float = 0.0,
) -> dict:
    """Simulate one continuous LMC-BMCD trajectory across successive seasons.

    Same Cholesky scheme as :func:`simulate_cholesky` (Section 5 of main.tex),
    but with one fitted parameter set per meteorological season and the
    following season-transition rule:

    - **spatial field**: the latent fields ``W^c, W^(0), W^(1)`` driving the
      transition from day ``n`` to ``n+1`` use ``theta_by_season[s(n)]`` where
      ``s(n)`` is the season of day ``n`` — the field parameters switch on the
      first day of each new season;
    - **exit probabilities**: the ongoing spell at station ``j`` keeps the
      ``q``-closures of the season in which the spell *started* until its next
      state switch (a dry spell starting on the last day of spring keeps the
      spring exit probabilities until it ends, even in summer); the spell that
      starts on day ``n+1`` after a switch is stamped with season ``s(n+1)``.

    Otherwise each transition is exactly Eq. (spatialized_markov_model_option2)
    of main.tex.

    ``dates`` is the daily calendar to simulate (consecutive days). The burn-in
    extends the calendar *backwards* by ``n_burn`` days, so the burn-in follows
    the real season sequence preceding ``dates[0]`` and is then discarded.

    Per-season covariances are built from that season's own ``params_by_season``
    entries (normalised coordinates), consistent with how each seasonal ``theta``
    was fitted. Returns the :func:`simulate_cholesky` history layout (``R``/``D``
    have one row per day of ``dates``, ``S``/``q``/``z`` one row per transition)
    plus ``dates`` and ``theta_by_season``.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    if not (np.diff(dates.to_numpy()) == np.timedelta64(1, "D")).all():
        raise ValueError("`dates` must be consecutive daily dates")
    if station_names is None:
        station_names = sorted(
            set.intersection(*(set(p) for p in params_by_season.values()))
        )
    station_names = list(station_names)
    m = len(station_names)
    seasons = sorted(theta_by_season)
    if sorted(params_by_season) != seasons:
        raise ValueError("theta_by_season and params_by_season must cover the same seasons")

    # One (lambda, Cholesky factors) set per season, from that season's theta
    # and normalised coordinates.
    def _chol(C):
        return np.linalg.cholesky(C + jitter * np.eye(m)) if jitter > 0 else np.linalg.cholesky(C)

    lam_by_season, factors_by_season = {}, {}
    for s in seasons:
        th = normalize_theta(theta_by_season[s])
        cov = build_field_cov_matrices(station_names, params_by_season[s], th)
        lam_by_season[s] = (th["lam0"], th["lam1"])
        factors_by_season[s] = tuple(_chol(cov[k]) for k in ("wc", "w0", "w1"))

    # Burn-in calendar: n_burn real days preceding dates[0].
    burn_dates = pd.date_range(end=dates[0] - pd.Timedelta(days=1), periods=n_burn, freq="D")
    full_dates = burn_dates.append(dates)
    season_seq = season_of_dates(full_dates)
    missing = set(season_seq) - set(seasons)
    if missing:
        raise ValueError(f"no fitted parameters for season(s) {sorted(missing)}")

    rng = np.random.default_rng(seed)

    # All-dry start; the initial spells are stamped with the first day's season.
    R = np.zeros(m, dtype=int)
    D = np.ones(m, dtype=int)
    spell_season = np.full(m, season_seq[0], dtype=object)

    def _q_thresholds(R, D, spell_season):
        q = np.empty(m, dtype=float)
        for j, name in enumerate(station_names):
            f = params_by_season[spell_season[j]][name][
                "q_d_dry_function" if R[j] == 0 else "q_d_wet_function"
            ]
            q[j] = float(f(D[j]))
        q = np.clip(q, clip_q, 1.0 - clip_q)
        return q, norm.ppf(q)

    def step(R, D, spell_season, n):
        Lc, L0, L1 = factors_by_season[season_seq[n]]
        Wc = Lc @ rng.standard_normal(m)
        W0 = L0 @ rng.standard_normal(m)
        W1 = L1 @ rng.standard_normal(m)
        Z = _assemble_selected_field(R, *lam_by_season[season_seq[n]], Wc, W0, W1)
        q, z = _q_thresholds(R, D, spell_season)
        switch = Z <= z
        R_next = R.copy()
        D_next = D.copy()
        R_next[switch] = 1 - R_next[switch]
        D_next[switch] = 1
        D_next[~switch] = D_next[~switch] + 1
        spell_season_next = spell_season.copy()
        spell_season_next[switch] = season_seq[n + 1]
        return R_next, D_next, spell_season_next, switch.astype(float), q, z

    # Burn-in transitions over the prepended real calendar.
    for n in range(n_burn):
        R, D, spell_season, _, _, _ = step(R, D, spell_season, n)

    # Kept trajectory: one state row per day of `dates`, one S/q/z row per transition.
    n_steps = len(dates) - 1
    R_hist = np.empty((n_steps + 1, m), dtype=float)
    D_hist = np.empty((n_steps + 1, m), dtype=float)
    S_hist = np.empty((n_steps, m), dtype=float)
    q_hist = np.empty((n_steps, m), dtype=float)
    z_hist = np.empty((n_steps, m), dtype=float)
    R_hist[0], D_hist[0] = R, D
    for k in range(n_steps):
        R, D, spell_season, S, q, z = step(R, D, spell_season, n_burn + k)
        R_hist[k + 1], D_hist[k + 1] = R, D
        S_hist[k], q_hist[k], z_hist[k] = S, q, z

    return {
        "station_names": station_names,
        "dates": dates,
        "R": R_hist,
        "D": D_hist,
        "S": S_hist,
        "q": q_hist,
        "z": z_hist,
        "theta_by_season": {s: normalize_theta(theta_by_season[s]) for s in seasons},
    }


def simulate_history(
    n_steps: int,
    R0: np.ndarray,
    D0: np.ndarray,
    theta,
    params_by_station: Dict[str, dict],
    station_names: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
) -> dict:
    """Simulate ``n_steps`` of the spatial LMC-BMCD; returns a ``history`` dict.

    Per-step ("svd") simulator: each day the three latent fields are drawn with
    ``rng.multivariate_normal`` (no burn-in). ``theta`` is the LMC parameter
    ``(lambda, sigma_wc, sigma_w0, sigma_w1)`` or its state-dependent 5-component
    form (see :func:`normalize_theta`).
    """
    th = normalize_theta(theta)
    lam = (th["lam0"], th["lam1"])
    if station_names is None:
        station_names = sorted(list(params_by_station.keys()))
    m = len(station_names)
    field_cov = build_field_cov_matrices(station_names, params_by_station, th)

    rng = np.random.default_rng(seed)
    R_hist = np.empty((n_steps + 1, m), dtype=float)
    D_hist = np.empty((n_steps + 1, m), dtype=float)
    S_hist = np.empty((n_steps, m), dtype=float)
    q_hist = np.empty((n_steps, m), dtype=float)
    z_hist = np.empty((n_steps, m), dtype=float)

    R = np.array(R0, dtype=float).copy()
    D = np.array(D0, dtype=float).copy()
    R_hist[0], D_hist[0] = R, D

    for n in range(n_steps):
        R, D, S, q, z = step_spatial_markov(
            R, D, field_cov, lam, params_by_station, rng=rng, station_names=station_names
        )
        R_hist[n + 1], D_hist[n + 1] = R, D
        S_hist[n], q_hist[n], z_hist[n] = S, q, z

    return {
        "station_names": list(station_names),
        "R": R_hist,
        "D": D_hist,
        "S": S_hist,
        "q": q_hist,
        "z": z_hist,
        "theta": th,
    }


# ---------------------------------------------------------------------------
# Pairwise composite likelihood
# ---------------------------------------------------------------------------


def phi2(z1: float, z2: float, rho: float) -> float:
    """Bivariate standard-normal CDF ``Phi_2(z1, z2; rho)``.

    Exact scipy baseline: builds a fresh ``multivariate_normal`` object on every
    call. Correct but slow when called on millions of pairs inside the MLE; see
    :func:`phi2_vec` for the fast vectorised approximation.
    """
    rho = float(np.clip(rho, -0.999999, 0.999999))
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
    return float(multivariate_normal(mean=[0.0, 0.0], cov=cov).cdf([z1, z2]))


# Tsay & Ke (2021), Theorem 1 fitted constants (from approxcdf's src/other.cpp).
_TSAY_C1 = -1.0950081470333
_TSAY_C2 = -0.75651138383854
_SQRT2 = np.sqrt(2.0)


def phi2_vec(z1, z2, rho, indep_tol: float = 1e-14):
    """Vectorised approximation of the bivariate standard-normal CDF ``Phi_2``.

    Closed-form, branch-free port of the Tsay & Ke (2021) approximation
    (``approxcdf``'s ``norm_cdf_2d_vfast`` = Theorem 1's ``F_app``). Accurate to
    ~1e-3 vs :func:`phi2`, needs no extra dependency, and being closed-form in
    ``rho`` it vectorises over the per-pair correlations. ``z1``, ``z2``, ``rho``
    broadcast against each other. Derived and validated in
    ``approx_cdf_adaptation/approx_cdf_explained.ipynb``.
    """
    rho = np.clip(rho, -0.999999, 0.999999)
    p, q, rho = np.broadcast_arrays(
        np.asarray(z1, float), np.asarray(z2, float), np.asarray(rho, float)
    )
    c1, c2 = _TSAY_C1, _TSAY_C2
    with np.errstate(all="ignore"):
        denom = np.sqrt(1.0 - rho * rho)
        a = -rho / denom
        b = p / denom
        aqb = a * q + b  # inner argument a*nu + b at the upper limit nu = q
        aa = a * a
        sqrt2b = _SQRT2 * b
        sqrt2q = _SQRT2 * q
        a_sq_c1 = aa * c1
        a_sq_c2 = aa * c2
        r = 1.0 - a_sq_c2
        s = np.sqrt(r)
        temp = 1.0 / (4.0 * s)
        a_c1 = a * c1
        twicea_s = 2.0 * a * s
        # branch A: Theorem 1 case 1 (a > 0, aqb >= 0)
        t1A = a_sq_c1 * c1 + 2.0 * b * b * c2
        t2A = 2.0 * sqrt2b * c1
        t3A = 4.0 * r
        A = (
            0.5 * (erf(q / _SQRT2) + erf(b / (_SQRT2 * a)))
            + temp * np.exp((t1A - t2A) / t3A) * (1.0 - erf((sqrt2b - a_sq_c1) / twicea_s))
            - temp * np.exp((t1A + t2A) / t3A)
            * (
                erf((sqrt2q - sqrt2q * a_sq_c2 - sqrt2b * a * c2 - a * c1) / (2.0 * s))
                + erf((a_sq_c1 + sqrt2b) / twicea_s)
            )
        )
        # branch B: Theorem 1 case 3 (a > 0, aqb < 0)
        B = (
            temp
            * np.exp((a_c1 * a_c1 - 2.0 * sqrt2b * c1 + 2.0 * b * b * c2) / (4.0 * r))
            * (1.0 + erf((sqrt2q - sqrt2q * a_sq_c2 - sqrt2b * a * c2 + a_c1) / (2.0 * s)))
        )
        # branch C: Theorem 1 case 4 (a <= 0, aqb >= 0)
        Cc = (
            0.5
            + 0.5 * erf(q / _SQRT2)
            - temp
            * np.exp((a_c1 * a_c1 + 2.0 * sqrt2b * c1 + 2.0 * b * b * c2) / (4.0 * r))
            * (1.0 + erf((sqrt2q - sqrt2q * a_sq_c2 - sqrt2b * a * c2 - a_c1) / (2.0 * s)))
        )
        # branch D: Theorem 1 case 5 (a <= 0, aqb < 0)
        t1D = a_c1 * a_c1 + 2.0 * b * b * c2
        t2D = 2.0 * sqrt2b * c1
        t3D = 4.0 * r
        D = (
            0.5
            - 0.5 * erf(b / (_SQRT2 * a))
            - temp * np.exp((t1D + t2D) / t3D) * (1.0 - erf((sqrt2b + a * a_c1) / (2.0 * a * s)))
            + temp * np.exp((t1D - t2D) / t3D)
            * (
                erf((sqrt2q - sqrt2q * a_sq_c2 - sqrt2b * a * c2 + a_c1) / (2.0 * s))
                + erf((sqrt2b - a * a_c1) / (2.0 * a * s))
            )
        )
        out = np.where(a > 0, np.where(aqb >= 0, A, B), np.where(aqb >= 0, Cc, D))
    indep = np.abs(rho) <= indep_tol  # Theorem 1 case 2 (rho = 0): independence
    if np.any(indep):
        out = np.where(indep, norm.cdf(p) * norm.cdf(q), out)
    return out


def _select_pair_rho(blocks_v, Rv, j, k):
    """LMC off-diagonal ``rho = C^(r_j, r_k)`` for a single pair — Eq. (rho_direct)."""
    if Rv[j] == 0 and Rv[k] == 0:
        return blocks_v["C00"][j, k]
    if Rv[j] == 1 and Rv[k] == 1:
        return blocks_v["C11"][j, k]
    return blocks_v["C01"][j, k]


def pairwise_loglik_one_step_nanaware(
    Rn, Dn, Sn, blocks, params_by_station,
    station_names=None, eps: float = 1e-15, clip_q: float = 1e-12,
):
    """One-step pairwise composite log-likelihood, robust to NaN stations.

    ``blocks`` are the LMC covariance blocks from :func:`build_lmc_blocks`
    (full station set); each pair's correlation is the state-selected block
    ``rho = C^(r_j, r_k)(s_j, s_k)`` (Eq. rho_direct), no sign flip.
    """
    if station_names is None:
        station_names = sorted(list(params_by_station.keys()))
    Rn, Dn, Sn = np.asarray(Rn), np.asarray(Dn), np.asarray(Sn)

    valid = np.isfinite(Rn) & np.isfinite(Dn)
    if Sn.dtype != bool:
        valid = valid & np.isfinite(Sn)
    idx = np.flatnonzero(valid)
    if idx.size < 2:
        return 0.0, idx.size

    station_names_v = [station_names[i] for i in idx]
    blocks_v = {k: blocks[k][np.ix_(idx, idx)] for k in ("C00", "C11", "C01")}
    Rv = Rn[idx].astype(int)
    Dv = Dn[idx].astype(int)
    Sv = Sn[idx].astype(bool)
    m = idx.size

    q = np.empty(m, dtype=float)
    for j, name in enumerate(station_names_v):
        f = params_by_station[name][
            "q_d_dry_function" if Rv[j] == 0 else "q_d_wet_function"
        ]
        q[j] = float(f(Dv[j]))
    q = np.clip(q, clip_q, 1.0 - clip_q)
    z = norm.ppf(q)

    ll = 0.0
    for j in range(m - 1):
        for k in range(j + 1, m):
            rho_star = _select_pair_rho(blocks_v, Rv, j, k)
            q_joint = phi2(z[j], z[k], rho_star)
            if Sv[j] and Sv[k]:
                p = q_joint
            elif Sv[j] and (not Sv[k]):
                p = q[j] - q_joint
            elif (not Sv[j]) and Sv[k]:
                p = q[k] - q_joint
            else:
                p = 1.0 - q[j] - q[k] + q_joint
            if p < eps:
                p = eps
            ll += np.log(p)
    return ll, m


def pairwise_loglik_one_step_nanaware_vec(
    Rn, Dn, Sn, blocks, params_by_station,
    station_names=None, eps: float = 1e-15, clip_q: float = 1e-12,
):
    """Vectorised one-step pairwise composite log-likelihood, robust to NaN stations.

    Same NaN-valid masking and ``(ll, m)`` return contract as the scalar
    :func:`pairwise_loglik_one_step_nanaware`, but the double ``for j,k`` pair
    loop is replaced by an ``np.triu_indices`` rewrite that calls the closed-form
    :func:`phi2_vec` once over all pairs. Each pair's correlation is the
    state-selected LMC block ``rho = C^(r_j, r_k)`` (Eq. rho_direct). Returns the
    same number to ~1e-3 (the approximation error of ``phi2_vec``).
    """
    if station_names is None:
        station_names = sorted(list(params_by_station.keys()))
    Rn, Dn, Sn = np.asarray(Rn), np.asarray(Dn), np.asarray(Sn)

    valid = np.isfinite(Rn) & np.isfinite(Dn)
    if Sn.dtype != bool:
        valid = valid & np.isfinite(Sn)
    idx = np.flatnonzero(valid)
    if idx.size < 2:
        return 0.0, idx.size

    station_names_v = [station_names[i] for i in idx]
    blocks_v = {k: blocks[k][np.ix_(idx, idx)] for k in ("C00", "C11", "C01")}
    Rv = Rn[idx].astype(int)
    Dv = Dn[idx].astype(int)
    Sv = Sn[idx].astype(bool)
    m = idx.size

    q = np.empty(m, dtype=float)
    for j, name in enumerate(station_names_v):
        f = params_by_station[name][
            "q_d_dry_function" if Rv[j] == 0 else "q_d_wet_function"
        ]
        q[j] = float(f(Dv[j]))
    q = np.clip(q, clip_q, 1.0 - clip_q)
    z = norm.ppf(q)

    j, k = np.triu_indices(m, 1)
    rj, rk = Rv[j], Rv[k]
    rho_star = np.where(
        (rj == 0) & (rk == 0), blocks_v["C00"][j, k],
        np.where((rj == 1) & (rk == 1), blocks_v["C11"][j, k], blocks_v["C01"][j, k]),
    )
    q_joint = np.asarray(phi2_vec(z[j], z[k], rho_star))
    qj, qk = q[j], q[k]
    sj, sk = Sv[j], Sv[k]
    p = np.where(
        sj & sk, q_joint,
        np.where(
            sj & ~sk, qj - q_joint,
            np.where(~sj & sk, qk - q_joint, 1.0 - qj - qk + q_joint),
        ),
    )
    ll = float(np.sum(np.log(np.maximum(p, eps))))
    return ll, m


def pairwise_loglik_from_history(
    history, blocks, params_by_station, eps=1e-15, clip_q=1e-12, vectorized=False,
):
    """Pairwise composite log-likelihood summed over all time steps of one history.

    ``blocks`` are the LMC covariance blocks from :func:`build_lmc_blocks`.
    ``vectorized=True`` uses the fast closed-form :func:`phi2_vec` kernel via
    :func:`pairwise_loglik_one_step_nanaware_vec`; the default uses the exact
    scipy :func:`phi2`.
    """
    R_hist, D_hist, S_hist = history["R"], history["D"], history["S"]
    station_names = history["station_names"]
    one_step = (
        pairwise_loglik_one_step_nanaware_vec if vectorized
        else pairwise_loglik_one_step_nanaware
    )
    T = S_hist.shape[0]
    total, nb_obs = 0.0, 0.0
    for n in range(T):
        # skip rows where all stations are NaN
        if not np.any(np.isfinite(R_hist[n])):
            continue
        ll, m = one_step(
            R_hist[n], D_hist[n], S_hist[n], blocks, params_by_station,
            station_names=station_names, eps=eps, clip_q=clip_q,
        )
        total += ll
        nb_obs += m
    return total, nb_obs


def pairwise_loglik_given_theta(history, params_by_station, theta, eps=1e-15, vectorized=False):
    station_names = history["station_names"]
    blocks = build_lmc_blocks(station_names, params_by_station, theta)
    return pairwise_loglik_from_history(
        history, blocks, params_by_station, eps=eps, vectorized=vectorized
    )[0]


# Default optimiser box (shared-lambda model): lambda in [0, 1], each range in [1e-3, 5].
THETA_BOUNDS = ((0.0, 1.0), (1e-3, 5.0), (1e-3, 5.0), (1e-3, 5.0))
THETA_X0 = (0.5, 0.3, 0.3, 0.3)

# State-dependent-lambda variant (Eq. lmc_fields_direct_state_dep):
# theta = (lambda_0, lambda_1, sigma_wc, sigma_w0, sigma_w1).
THETA_BOUNDS_STATE_DEP = ((0.0, 1.0), (0.0, 1.0), (1e-3, 5.0), (1e-3, 5.0), (1e-3, 5.0))
THETA_X0_STATE_DEP = (0.5, 0.5, 0.3, 0.3, 0.3)


def mle_theta_pairwise(
    history, params_by_station, bounds=THETA_BOUNDS, x0=THETA_X0,
    eps=1e-15, vectorized=False,
):
    """Maximiser of the pairwise composite likelihood in ``theta``.

    Optimises ``theta = (lambda, sigma_wc, sigma_w0, sigma_w1)`` with L-BFGS-B over
    the box ``bounds``. Pass ``bounds=THETA_BOUNDS_STATE_DEP, x0=THETA_X0_STATE_DEP``
    to fit the 5-parameter state-dependent-lambda variant instead.
    ``vectorized=True`` evaluates the likelihood with the fast
    :func:`phi2_vec` kernel; the default uses the exact scipy :func:`phi2`.
    """
    def neg_ll(x):
        return -pairwise_loglik_given_theta(
            history, params_by_station, x, eps=eps, vectorized=vectorized
        )

    res = minimize(neg_ll, x0=np.asarray(x0, float), method="L-BFGS-B", bounds=bounds)
    theta_hat = normalize_theta(res.x)
    ll_hat = float(-res.fun)
    station_names = history["station_names"]
    blocks_hat = build_lmc_blocks(station_names, params_by_station, theta_hat)
    nb_obs = pairwise_loglik_from_history(
        history, blocks_hat, params_by_station, eps=eps, vectorized=vectorized
    )[1]
    return {
        "theta_hat": theta_hat,
        "ll_hat": ll_hat,
        "opt_result": res,
        "total_nb_observed_stations": nb_obs,
    }


def pairwise_loglik_from_histories(histories, blocks, params_by_station, eps=1e-15, vectorized=False):
    total, nb_obs = 0.0, 0.0
    for h in histories:
        ll, m = pairwise_loglik_from_history(
            h, blocks, params_by_station, eps=eps, vectorized=vectorized
        )
        total += ll
        nb_obs += m
    return total, nb_obs


def mle_theta_pairwise_histories(
    histories, params_by_station, bounds=THETA_BOUNDS, x0=THETA_X0,
    eps=1e-15, vectorized=False,
):
    """Pairwise-MLE of ``theta`` summing the likelihood across several histories.

    Pass ``bounds=THETA_BOUNDS_STATE_DEP, x0=THETA_X0_STATE_DEP`` to fit the
    5-parameter state-dependent-lambda variant. ``vectorized=True`` uses the
    fast :func:`phi2_vec` kernel.
    """
    station_names = histories[0]["station_names"]

    def neg_ll(x):
        blocks = build_lmc_blocks(station_names, params_by_station, x)
        return -pairwise_loglik_from_histories(
            histories, blocks, params_by_station, eps=eps, vectorized=vectorized
        )[0]

    res = minimize(neg_ll, x0=np.asarray(x0, float), method="L-BFGS-B", bounds=bounds)
    return {"theta_hat": normalize_theta(res.x), "ll_hat": float(-res.fun), "opt_result": res}


# ---------------------------------------------------------------------------
# NaN robustness
# ---------------------------------------------------------------------------


def inject_nans_in_history(history, frac_nan: float = 0.10, seed: int = 0, mask_S: bool = True):
    """Replace a random fraction of stations per time step by NaN, for stress-testing."""
    rng = np.random.default_rng(seed)
    R = history["R"].astype(float, copy=True)
    D = history["D"].astype(float, copy=True)
    S = history["S"].astype(float, copy=True) if mask_S else history["S"].copy()
    station_names = history["station_names"]
    T, m = S.shape[0], R.shape[1]
    k = max(1, int(round(frac_nan * m)))

    for n in range(T):
        miss = rng.choice(m, size=k, replace=False)
        R[n, miss] = np.nan
        D[n, miss] = np.nan
        if mask_S:
            S[n, miss] = np.nan
    miss = rng.choice(m, size=k, replace=False)
    R[T, miss] = np.nan
    D[T, miss] = np.nan

    out = dict(history)
    out["R"], out["D"], out["S"], out["station_names"] = R, D, S, station_names
    return out


# ---------------------------------------------------------------------------
# DRY helper: simulate -> estimate experiment
# ---------------------------------------------------------------------------


def run_simulation_mle_experiment(
    dict_model_params: Dict[str, dict],
    theta_true,
    nb_stations: int,
    nb_steps: int,
    nb_estimations: int,
    inject_nan_frac: float = 0.0,
    nan_seed: int = 123,
    seed_base: int = 0,
    bounds=None,
    x0=None,
    vectorized: bool = False,
    simulator: str = "cholesky",
    n_burn: int = 200,
) -> pd.DataFrame:
    """Simulate ``nb_estimations`` spatial histories and fit ``theta`` on each.

    ``theta_true = (lambda, sigma_wc, sigma_w0, sigma_w1)`` is the LMC parameter;
    the 5-component form ``(lambda_0, lambda_1, sigma_wc, sigma_w0, sigma_w1)``
    selects the state-dependent-lambda variant (see :func:`normalize_theta`).
    Unless overridden, ``bounds`` / ``x0`` follow the parametrisation of
    ``theta_true``, so the fitted model matches the simulated one.
    Each returned row carries the recovered mixing weight(s) (``lam_hat`` for the
    shared model, ``lam0_hat`` / ``lam1_hat`` for the state-dependent one), the
    ``sigma_*_hat`` ranges and ``ll_hat``.

    ``simulator`` selects how the latent Gaussian fields are drawn:

    - ``"cholesky"`` (default) — :func:`simulate_cholesky` (factor each
      ``Sigma_k = L_k L_k^T`` once), with a burn-in of ``n_burn`` days. Used by
      the clean pipeline notebook and the parameter-estimation tests.
    - ``"svd"`` — :func:`simulate_history` / :func:`step_spatial_markov`, i.e.
      numpy's ``multivariate_normal`` (SVD factorisation) per step, no burn-in.
      Kept for the explicit Cholesky-vs-SVD comparison in ``tests_simulation_methods``.
    """
    th_true = normalize_theta(theta_true)
    shared_lam = "lam" in th_true
    if bounds is None:
        bounds = THETA_BOUNDS if shared_lam else THETA_BOUNDS_STATE_DEP
    if x0 is None:
        x0 = THETA_X0 if shared_lam else THETA_X0_STATE_DEP
    extract_stations = sorted(list(dict_model_params.keys()))[:nb_stations]
    params_by_station = {c: dict_model_params[c] for c in extract_stations}
    station_names = extract_stations  # already sorted

    if simulator not in ("cholesky", "svd"):
        raise ValueError(f"simulator must be 'cholesky' or 'svd', got {simulator!r}")

    if simulator == "svd":
        R0 = np.ones(len(extract_stations), dtype=int)
        D0 = np.ones(len(extract_stations), dtype=int)

    rows = []
    for i in range(nb_estimations):
        if simulator == "cholesky":
            history = simulate_cholesky(
                theta=th_true, params_by_station=params_by_station,
                n_steps=nb_steps, n_burn=n_burn,
                station_names=station_names, seed=seed_base + i,
            )
        else:
            history = simulate_history(
                n_steps=nb_steps, R0=R0, D0=D0, theta=th_true,
                params_by_station=params_by_station,
                station_names=station_names, seed=seed_base + i,
            )
        if inject_nan_frac > 0:
            history = inject_nans_in_history(
                history, frac_nan=inject_nan_frac, seed=nan_seed, mask_S=True
            )
        mle = mle_theta_pairwise(
            history, params_by_station, bounds=bounds, x0=x0, vectorized=vectorized
        )
        th_hat = mle["theta_hat"]
        row = {"i": i}
        if "lam" in th_hat:
            row["lam_hat"] = th_hat["lam"]
        else:
            row["lam0_hat"] = th_hat["lam0"]
            row["lam1_hat"] = th_hat["lam1"]
        row.update({
            "sigma_wc_hat": th_hat["sigma_wc"],
            "sigma_w0_hat": th_hat["sigma_w0"],
            "sigma_w1_hat": th_hat["sigma_w1"],
            "ll_hat": mle["ll_hat"],
            "mean_nb_obs_stations": mle["total_nb_observed_stations"] / nb_steps,
        })
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["theta_true"] = th_true
    df.attrs["nb_stations"] = nb_stations
    df.attrs["nb_steps"] = nb_steps
    df.attrs["inject_nan_frac"] = inject_nan_frac
    return df


# ---------------------------------------------------------------------------
# Real-data ECAD I/O
# ---------------------------------------------------------------------------


def read_ecad_rr_file(file_path) -> pd.DataFrame:
    """Read a single ECAD ``RR_SOUID*.txt`` file -> DataFrame[DATE, RR, Q_RR]."""
    file_path = Path(file_path)
    header_idx = None
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            s = line.strip().replace(" ", "")
            if (
                s.startswith("STAID,") and "SOUID" in s and "DATE" in s
                and "RR" in s and "Q_RR" in s
            ):
                header_idx = i
                break
    if header_idx is None:
        raise ValueError(f"Could not find header line in {file_path.name}")

    df = pd.read_csv(file_path, skiprows=header_idx, sep=",", header=0, engine="python")
    df.columns = [c.strip() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"].astype(str).str.strip(), format="%Y%m%d", errors="coerce")
    df["RR"] = pd.to_numeric(df["RR"], errors="coerce")
    df["Q_RR"] = pd.to_numeric(df["Q_RR"], errors="coerce").astype("Int64")
    df.loc[df["RR"] == -9999, "RR"] = np.nan
    df.loc[df["RR"] < 0, "RR"] = np.nan
    df = df.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)
    return df[["DATE", "RR", "Q_RR"]]


def load_all_station_rr(
    data_dir,
    stations_to_get: Optional[Iterable[str]],
    verbose: bool = True,
    candidates_csv: str = "df_candidates_kept.csv",
) -> Dict[str, pd.DataFrame]:
    """Load every station RR file in ``data_dir`` whose city is in ``stations_to_get``."""
    data_dir = Path(data_dir)
    df_info = pd.read_csv(data_dir / candidates_csv)
    souid_re = re.compile(r"SOUID(\d+)")
    station_files = list_station_files(data_dir)
    stations_to_get = set(stations_to_get) if stations_to_get is not None else None

    dfs_by_city: Dict[str, pd.DataFrame] = {}
    nb_rejected = 0
    iterator = (
        tqdm(station_files, desc="Loading station RR files") if verbose else station_files
    )
    for file_path in iterator:
        m = souid_re.search(file_path.name)
        if not m:
            continue
        souid = int(m.group(1))
        city_row = df_info.loc[df_info["souid"] == souid]
        city_name = city_row["city"].iloc[0] if not city_row.empty else f"SOUID_{souid}"
        if verbose and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(value=f"{city_name} -- {souid}")
        if stations_to_get is not None and city_name not in stations_to_get:
            continue
        try:
            dfs_by_city[city_name] = read_ecad_rr_file(file_path)
        except Exception as e:
            nb_rejected += 1
            if verbose:
                print(f"Rejected {file_path.name} ({city_name}, SOUID={souid}) -> {e}")
    if verbose:
        print(f"Finished RR load -- Rejected {nb_rejected} station files")
    return dfs_by_city


_MONTH_TO_SEASON = np.array(
    ["winter", "winter",                       # Jan, Feb
     "spring", "spring", "spring",             # Mar, Apr, May
     "summer", "summer", "summer",             # Jun, Jul, Aug
     "autumn", "autumn", "autumn",             # Sep, Oct, Nov
     "winter"],                                # Dec
    dtype=object,
)


def season_of_dates(dates) -> np.ndarray:
    """Meteorological season label of each date (DJF/MAM/JJA/SON)."""
    dates = pd.to_datetime(dates)
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(dates)
    return _MONTH_TO_SEASON[dates.month.to_numpy() - 1]


def _season_mask(dates: pd.DatetimeIndex, season: str) -> np.ndarray:
    season = season.strip().lower()
    labels = season_of_dates(dates)
    if season == "all":
        return np.ones(labels.shape[0], dtype=bool)
    if season not in ("winter", "spring", "summer", "autumn"):
        raise ValueError(f"Unknown season: {season!r}")
    return labels == season


def build_joint_df_occurrence_from_raw_data(
    dfs_by_station: Dict[str, pd.DataFrame],
    season: str,
    station_names: Sequence[str],
    rr_col: str = "RR",
    date_col: str = "DATE",
    wet_day_threshold: float = float(config.WET_DAY_THRESHOLD),
) -> pd.DataFrame:
    """Build the wet/dry occurrence matrix (days x stations) for a given season.

    ``season="all"`` keeps the full calendar (no season filtering). NaNs in the
    raw RR data propagate as NaNs in the occurrence matrix.
    """
    all_dates = []
    for name in station_names:
        df = dfs_by_station[name].copy()
        df[date_col] = pd.to_datetime(df[date_col])
        all_dates.append(df[date_col].min())
        all_dates.append(df[date_col].max())
    full_index = pd.date_range(min(all_dates), max(all_dates), freq="D")
    RR = pd.DataFrame(index=full_index, columns=list(station_names), dtype=float)
    for name in station_names:
        df = dfs_by_station[name].copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        tmp = df[[date_col, rr_col]].groupby(date_col)[rr_col].mean()
        RR[name] = tmp.reindex(full_index)
    RR = RR.loc[_season_mask(RR.index, season)]
    Rbin = (RR >= wet_day_threshold).astype(float)
    Rbin[RR.isna()] = np.nan
    return Rbin


def history_from_Rbin_drop_ambiguous_spell_after_nan(Rbin: pd.DataFrame) -> dict:
    """Build a history dict from a binary occurrence DataFrame.

    Spells whose start is ambiguous (interior of an unknown run, beginning of a
    new season but possibly continuing the previous one) are masked with NaN
    so they contribute neither to ``D`` nor to ``S``.
    """
    station_names = list(Rbin.columns)
    Rbin = Rbin.copy()
    Rbin.index = pd.to_datetime(Rbin.index)
    Rbin = Rbin.sort_index()
    R_in = Rbin.to_numpy(dtype=float)
    T, m = R_in.shape

    R_out = R_in.copy()
    D_out = np.full((T, m), np.nan, dtype=float)

    for j in range(m):
        mode = "gap"
        last_r = np.nan
        d_prev = np.nan
        for t in range(T):
            r = R_in[t, j]
            if not np.isfinite(r):
                R_out[t, j], D_out[t, j] = np.nan, np.nan
                mode = "gap"
                last_r, d_prev = np.nan, np.nan
                continue
            r_int = int(r)
            if mode == "gap":
                R_out[t, j], D_out[t, j] = np.nan, np.nan
                mode = "pending"
                last_r, d_prev = r_int, np.nan
                continue
            if mode == "pending":
                if r_int == last_r:
                    R_out[t, j], D_out[t, j] = np.nan, np.nan
                    last_r = r_int
                    continue
                R_out[t, j], D_out[t, j] = float(r_int), 1.0
                mode = "known"
                last_r, d_prev = r_int, 1.0
                continue

            # mode == "known"
            R_out[t, j] = float(r_int)
            if np.isfinite(last_r):
                D_out[t, j] = (d_prev + 1.0) if r_int == last_r and np.isfinite(d_prev) else 1.0
            else:
                D_out[t, j] = np.nan
            last_r = r_int
            d_prev = D_out[t, j]

            # Drop the first spell of a new season (it may be the tail of last season)
            spell_from_last_season = False
            if t == 0:
                spell_from_last_season = True
            else:
                last_day = Rbin.index[t - 1]
                current_day = Rbin.index[t]
                if current_day.day == 1 and current_day.month in (3, 6, 9, 12):
                    if current_day - last_day != pd.Timedelta("1d"):
                        spell_from_last_season = True
                    if R_in[t, j] == R_in[t - 1, j]:
                        spell_from_last_season = True
            if spell_from_last_season:
                R_out[t, j], D_out[t, j] = np.nan, np.nan
                mode = "pending"
                last_r, d_prev = r_int, np.nan
                continue

    if T >= 2:
        S_out = np.full((T - 1, m), np.nan, dtype=float)
        for t in range(T - 1):
            r0 = R_out[t, :]
            r1 = R_out[t + 1, :]
            valid = np.isfinite(r0) & np.isfinite(r1)
            S_out[t, valid] = (r1[valid].astype(int) != r0[valid].astype(int)).astype(float)
    else:
        S_out = np.empty((0, m), dtype=float)

    return {
        "station_names": list(station_names),
        "dates": Rbin.index,
        "R": R_out,
        "D": D_out,
        "S": S_out,
    }


def split_history_by_year(history: dict) -> List[dict]:
    """Split a single history dict into one history per calendar year.

    Useful to feed :func:`mle_theta_pairwise_histories`: each season-year
    becomes an (approximately) independent segment.
    """
    dates = pd.DatetimeIndex(history["dates"])
    years = dates.year.unique()
    out = []
    for y in years:
        mask = (dates.year == y)
        if mask.sum() < 2:
            continue
        mask_S = mask[:-1] & mask[1:]
        out.append({
            "station_names": history["station_names"],
            "dates": dates[mask],
            "R": history["R"][mask],
            "D": history["D"][mask],
            "S": history["S"][mask_S],
        })
    return out
