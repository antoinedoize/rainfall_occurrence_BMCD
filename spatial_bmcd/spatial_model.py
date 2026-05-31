"""Spatial Binary Markov Chain with Duration (BMCD) — model, likelihood and I/O.

Implements the spatial extension of the single-site BMCD described in
``spatial_analysis_article.tex``. Re-uses the single-site fits produced by
``article_code/notebooks_article_pipeline/01_prepare_data_and_fit_distributions.ipynb``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.stats
from scipy.optimize import minimize_scalar
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


def step_spatial_markov(
    R: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    params_by_station: Dict[str, dict],
    rng: Optional[np.random.Generator] = None,
    station_names: Optional[Sequence[str]] = None,
    clip_q: float = 1e-12,
):
    """One step of the spatial Markov chain — see Eq. (3) of the article.

    Returns ``(R_next, D_next, switch, q, z)``.
    """
    if station_names is None:
        station_names = sorted(list(params_by_station.keys()))
    m = len(station_names)

    if rng is not None:
        Y = rng.multivariate_normal(mean=np.zeros(m), cov=C)
    else:
        Y = multivariate_normal(mean=np.zeros(m), cov=C).rvs()
    Z = np.where(R == 0, Y, -Y)

    q = np.empty(m, dtype=float)
    for j, name in enumerate(station_names):
        f = params_by_station[name][
            "q_d_dry_function" if R[j] == 0 else "q_d_wet_function"
        ]
        q[j] = float(f(D[j]))
    q = np.clip(q, clip_q, 1.0 - clip_q)
    z = norm.ppf(q)
    switch = Z <= z

    R_next = R.copy()
    D_next = D.copy()
    R_next[switch] = 1 - R_next[switch]
    D_next[switch] = 1
    D_next[~switch] = D_next[~switch] + 1
    return R_next, D_next, switch, q, z


def simulate_fitted_model(
    sigma: float,
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
    """Simulate from a fitted spatial BMCD — algorithm of Section 5 of the article.

    Three steps, mirroring the article:

    1. **Spatial Cholesky factorisation** — build the covariance ``Sigma(sigma)``
       on the rescaled stations and factor ``Sigma = L L^T`` once. ``jitter``
       defaults to ``0``; set it to a small positive value (e.g. ``1e-6``) to
       stabilise the factorisation if ``Sigma`` is near-singular.
    2. **Initialisation + burn-in** — if ``R0`` / ``D0`` are not provided, set
       ``R0[j] = 0`` and ``D0[j] = 1`` for every station (arbitrary deterministic
       all-dry start), then run ``n_burn`` steps and discard them so the chain
       reaches its stationary regime.
    3. **Sequential update** — for each kept day, draw ``V ~ N(0, I_J)``, set
       ``Y = L V`` (so ``Y ~ N(0, Sigma)``), and apply the threshold rule of
       Eq. (3) of the article via the sign-adjusted field
       ``Z[j] = (-1)^R[j] * Y[j]``.

    Returns the same ``history`` dict layout as :func:`simulate_history` (keys
    ``station_names``, ``R``, ``D``, ``S``, ``q``, ``z``) so the simulated
    trajectory can be fed back into :func:`mle_sigma_pairwise` for diagnostics.
    For an ensemble of ``M`` independent trajectories, call this function ``M``
    times with different ``seed`` values.
    """
    if station_names is None:
        station_names = sorted(list(params_by_station.keys()))
    m = len(station_names)

    # Step 1: spatial Cholesky factorisation (done once).
    C = build_C_from_sigma(station_names, params_by_station, sigma)
    L = np.linalg.cholesky(C + jitter * np.eye(m)) if jitter > 0 else np.linalg.cholesky(C)

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
        V = rng.standard_normal(m)
        Y = L @ V
        Z = np.where(R == 0, Y, -Y)
        q = np.empty(m, dtype=float)
        for j, name in enumerate(station_names):
            f = params_by_station[name][
                "q_d_dry_function" if R[j] == 0 else "q_d_wet_function"
            ]
            q[j] = float(f(D[j]))
        q = np.clip(q, clip_q, 1.0 - clip_q)
        z = norm.ppf(q)
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
        "sigma": float(sigma),
    }


def simulate_history(
    n_steps: int,
    R0: np.ndarray,
    D0: np.ndarray,
    C: np.ndarray,
    params_by_station: Dict[str, dict],
    station_names: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
) -> dict:
    """Simulate ``n_steps`` of the spatial BMCD; returns a ``history`` dict."""
    if station_names is None:
        station_names = sorted(list(params_by_station.keys()))
    m = len(station_names)

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
            R, D, C, params_by_station, rng=rng, station_names=station_names
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
    }


# ---------------------------------------------------------------------------
# Pairwise composite likelihood
# ---------------------------------------------------------------------------


def phi2(z1: float, z2: float, rho: float) -> float:
    """Bivariate standard-normal CDF ``Phi_2(z1, z2; rho)``."""
    rho = float(np.clip(rho, -0.999999, 0.999999))
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
    return float(multivariate_normal(mean=[0.0, 0.0], cov=cov).cdf([z1, z2]))


def pairwise_loglik_one_step_nanaware(
    Rn, Dn, Sn, C, params_by_station,
    station_names=None, eps: float = 1e-15, clip_q: float = 1e-12,
):
    """One-step pairwise composite log-likelihood, robust to NaN stations."""
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
    C_v = C[np.ix_(idx, idx)]
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
            rho_star = ((-1) ** (Rv[j] + Rv[k])) * C_v[j, k]
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


def pairwise_loglik_from_history(history, C, params_by_station, eps=1e-15, clip_q=1e-12):
    """Pairwise composite log-likelihood summed over all time steps of one history."""
    R_hist, D_hist, S_hist = history["R"], history["D"], history["S"]
    station_names = history["station_names"]
    T = S_hist.shape[0]
    total, nb_obs = 0.0, 0.0
    for n in range(T):
        # skip rows where all stations are NaN
        if not np.any(np.isfinite(R_hist[n])):
            continue
        ll, m = pairwise_loglik_one_step_nanaware(
            R_hist[n], D_hist[n], S_hist[n], C, params_by_station,
            station_names=station_names, eps=eps, clip_q=clip_q,
        )
        total += ll
        nb_obs += m
    return total, nb_obs


def pairwise_loglik_given_sigma(history, params_by_station, sigma, eps=1e-15):
    station_names = history["station_names"]
    C_sigma = build_C_from_sigma(station_names, params_by_station, sigma)
    return pairwise_loglik_from_history(history, C_sigma, params_by_station, eps=eps)[0]


def mle_sigma_pairwise(history, params_by_station, bounds=(1e-3, 5.0), eps=1e-15):
    """Bounded 1-D maximiser of the pairwise composite likelihood in ``sigma``."""
    def neg_ll(sigma):
        return -pairwise_loglik_given_sigma(history, params_by_station, sigma, eps=eps)

    res = minimize_scalar(neg_ll, bounds=bounds, method="bounded")
    sigma_hat = float(res.x)
    ll_hat = float(-res.fun)
    station_names = history["station_names"]
    C_hat = build_C_from_sigma(station_names, params_by_station, sigma_hat)
    nb_obs = pairwise_loglik_from_history(history, C_hat, params_by_station, eps=eps)[1]
    return {
        "sigma_hat": sigma_hat,
        "ll_hat": ll_hat,
        "opt_result": res,
        "total_nb_observed_stations": nb_obs,
    }


def pairwise_loglik_from_histories(histories, C, params_by_station, eps=1e-15):
    total, nb_obs = 0.0, 0.0
    for h in histories:
        ll, m = pairwise_loglik_from_history(h, C, params_by_station, eps=eps)
        total += ll
        nb_obs += m
    return total, nb_obs


def mle_sigma_pairwise_histories(
    histories, params_by_station, bounds=(1e-3, 5.0), eps=1e-15
):
    """Pairwise-MLE of ``sigma`` summing the likelihood across several histories."""
    station_names = histories[0]["station_names"]

    def neg_ll(sigma):
        C_sigma = build_C_from_sigma(station_names, params_by_station, sigma)
        return -pairwise_loglik_from_histories(histories, C_sigma, params_by_station, eps=eps)[0]

    res = minimize_scalar(neg_ll, bounds=bounds, method="bounded")
    return {"sigma_hat": float(res.x), "ll_hat": float(-res.fun), "opt_result": res}


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
    sigma_true: float,
    nb_stations: int,
    nb_steps: int,
    nb_estimations: int,
    inject_nan_frac: float = 0.0,
    nan_seed: int = 123,
    seed_base: int = 0,
    bounds=(1e-3, 2.0),
) -> pd.DataFrame:
    """Simulate ``nb_estimations`` spatial histories and fit ``sigma`` on each.

    Mirrors the repeated ``Max likelihood estimation`` / ``Check that the
    estimation works with NaN values`` cells in the original notebook.
    """
    extract_stations = sorted(list(dict_model_params.keys()))[:nb_stations]
    params_by_station = {c: dict_model_params[c] for c in extract_stations}
    station_names = extract_stations  # already sorted

    R0 = np.ones(len(extract_stations), dtype=int)
    D0 = np.ones(len(extract_stations), dtype=int)
    C = build_C_from_sigma(station_names, params_by_station, sigma=sigma_true)

    rows = []
    for i in range(nb_estimations):
        history = simulate_history(
            n_steps=nb_steps, R0=R0, D0=D0, C=C,
            params_by_station=params_by_station,
            station_names=station_names, seed=seed_base + i,
        )
        if inject_nan_frac > 0:
            history = inject_nans_in_history(
                history, frac_nan=inject_nan_frac, seed=nan_seed, mask_S=True
            )
        mle = mle_sigma_pairwise(history, params_by_station, bounds=bounds)
        rows.append({
            "i": i,
            "sigma_hat": mle["sigma_hat"],
            "ll_hat": mle["ll_hat"],
            "mean_nb_obs_stations": mle["total_nb_observed_stations"] / nb_steps,
        })
    df = pd.DataFrame(rows)
    df.attrs["sigma_true"] = sigma_true
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


def _season_mask(dates: pd.DatetimeIndex, season: str) -> np.ndarray:
    dates = pd.to_datetime(dates)
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(dates)
    m = dates.month
    season = season.strip().lower()
    if season == "winter":
        return m.isin([12, 1, 2])
    if season == "spring":
        return m.isin([3, 4, 5])
    if season == "summer":
        return m.isin([6, 7, 8])
    if season == "autumn":
        return m.isin([9, 10, 11])
    raise ValueError(f"Unknown season: {season!r}")


def build_joint_df_occurrence_from_raw_data(
    dfs_by_station: Dict[str, pd.DataFrame],
    season: str,
    station_names: Sequence[str],
    rr_col: str = "RR",
    date_col: str = "DATE",
    wet_day_threshold: float = float(config.WET_DAY_THRESHOLD),
) -> pd.DataFrame:
    """Build the wet/dry occurrence matrix (days x stations) for a given season.

    NaNs in the raw RR data propagate as NaNs in the occurrence matrix.
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

    Useful to feed :func:`mle_sigma_pairwise_histories`: each season-year
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
