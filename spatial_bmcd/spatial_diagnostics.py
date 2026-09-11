"""Diagnostics comparing simulations from the fitted spatial BMCD with observations.

Statistics and figures used by ``diagnostics_fitted_model.ipynb`` to assess how
close the fitted model (``sigma_hat`` from the pairwise composite MLE) is to the
observed ECAD occurrence field.

Methodology shared by every diagnostic
--------------------------------------
- The observed record is a list of per-year season *blocks* (e.g. 41 blocks of
  92 spring days) built from the raw 0/1/NaN occurrence matrix ``Rbin`` with
  :func:`blocks_from_Rbin`.
- The simulated side is an ensemble of replicates with the *same block layout*,
  produced by :func:`simulate_block_ensemble` from the stationary fitted model
  ("perpetual spring": season-specific exit probabilities and covariance are
  time-invariant).
- Every statistic is computed by the *same* function on observed and simulated
  blocks, with identical censoring at block boundaries (running dry-spell ages
  restart at each block start; spells touching a block edge are dropped), so
  that observed values can be compared to the ensemble envelope without bias.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import squareform
from scipy.stats import norm
from tqdm import tqdm

from spatial_bmcd.spatial_model import (
    normalize_theta,
    phi2_vec,
    simulate_cholesky,
    station_distance_matrix,
)
from spatial_bmcd.spatial_plotting import _ensure_figures_dir


# ---------------------------------------------------------------------------
# Shared helpers: distances, station ordering, block construction
# ---------------------------------------------------------------------------

# ``station_distance_matrix`` (haversine, kilometres) is defined in
# ``spatial_model`` so that the diagnostics' distance axes and the model's own
# latent correlations exp(-h / sigma_k) read the exact same metric -- the model
# curves of :func:`lmc_block_curves` are overlaid on those axes. It is re-exported
# here for the callers that have always imported it from this module.


def spatial_station_order(
    params_by_station: Dict[str, dict],
    station_names: Optional[Sequence[str]] = None,
) -> List[str]:
    """Order stations so that adjacent entries are spatially close.

    Hierarchical clustering on the inter-station distance matrix with optimal
    leaf ordering (scipy): the returned permutation is the standard seriation
    of the distance matrix, used as the y-axis order of the occurrence rasters
    so that spatially coherent wet/dry episodes appear as contiguous blocks.
    """
    if station_names is None:
        station_names = sorted(params_by_station.keys())
    D = station_distance_matrix(params_by_station, station_names)
    Z = linkage(squareform(D, checks=False), method="average", optimal_ordering=True)
    return [station_names[i] for i in leaves_list(Z)]


def blocks_from_Rbin(Rbin: pd.DataFrame) -> List[dict]:
    """Split the raw occurrence matrix into one block per calendar year.

    Mirrors :func:`spatial_model.split_history_by_year` but keeps the *raw*
    0/1/NaN values of ``Rbin`` (no ambiguous-spell masking): diagnostics need
    actual occupancy, and any censoring is re-applied symmetrically to the
    simulated blocks by the statistics functions below.
    """
    Rbin = Rbin.copy()
    Rbin.index = pd.to_datetime(Rbin.index)
    Rbin = Rbin.sort_index()
    out: List[dict] = []
    for year in sorted(Rbin.index.year.unique()):
        sub = Rbin.loc[Rbin.index.year == year]
        if len(sub) < 2:
            continue
        out.append({
            "station_names": list(Rbin.columns),
            "dates": sub.index,
            "year": int(year),
            "R": sub.to_numpy(dtype=float),
        })
    return out


def simulate_block_ensemble(
    theta,
    params_by_station: Dict[str, dict],
    n_blocks: int,
    block_len: int,
    n_sims: int,
    n_burn: int = 2000,
    gap: int = 500,
    seed: int = 0,
    station_names: Optional[Sequence[str]] = None,
    show_progress: bool = True,
) -> List[List[dict]]:
    """Simulate ``n_sims`` replicates of the observed season-block layout.

    .. note::
       **Superseded for the seasonal fits.** This builds a *stationary* ensemble
       from a single ``theta``; the multi-season notebooks
       (``spatial_lin_mod_coregion_fit_spain_portugal.ipynb``) instead simulate
       replicates with :func:`spatial_model.simulate_cholesky_seasonal`, one
       ``theta_hat_s`` per season on the real calendar and masked with the
       observed NaN pattern, then feed the plotting helpers below with
       ``[blocks_from_Rbin(rep) for rep in reps]``. Prefer that route; see
       ``NOTES_spatial_diagnostics.md``. This function remains valid for the
       single-season / stationary studies it was written for.

    The fitted model is stationary ("perpetual spring": the season-specific
    exit probabilities and the spatial covariance are time-invariant), while
    the observed record consists of disjoint season windows. Each replicate
    therefore reproduces that layout from the stationary process:

    1. one long :func:`spatial_model.simulate_cholesky` run per replicate, with
       a single burn-in of ``n_burn`` days discarded (the chain reaches its
       stationary regime, removing the all-dry initialisation) — cheaper than
       one burn-in per block, with the same stationary law;
    2. ``n_blocks`` windows of ``block_len`` days are cut out, separated by
       ``gap`` discarded days so consecutive blocks are effectively independent
       (like distinct years) even for heavy-tailed dry spells.

    Block boundaries are then treated by the statistics functions exactly like
    the observed season boundaries (ages restart, edge spells dropped), so the
    censoring is symmetric even though the simulated blocks come from a
    continuous trajectory.

    Returns a list of replicates; each replicate is a list of ``n_blocks``
    dicts with the same ``{station_names, R}`` layout as :func:`blocks_from_Rbin`.
    """
    period = block_len + gap
    n_steps = n_blocks * period - gap
    iterator = tqdm(range(n_sims), desc="Simulating ensemble") if show_progress else range(n_sims)
    sims: List[List[dict]] = []
    for i in iterator:
        hist = simulate_cholesky(
            theta=theta,
            params_by_station=params_by_station,
            n_steps=n_steps,
            n_burn=n_burn,
            station_names=station_names,
            seed=None if seed is None else seed + i,
        )
        R = hist["R"][1:]  # drop the initial-state row -> one row per simulated day
        blocks = [
            {
                "station_names": list(hist["station_names"]),
                "block": k,
                "R": R[k * period: k * period + block_len].copy(),
            }
            for k in range(n_blocks)
        ]
        sims.append(blocks)
    return sims


def mask_like_obs(sim_blocks: List[dict], obs_blocks: List[dict]) -> List[dict]:
    """Copy the observed NaN pattern onto simulated blocks (sensitivity check).

    With the mask applied, every statistic on the simulated blocks uses exactly
    the same station-days as the observed one, so the ensemble envelope has the
    same sampling noise as the observed estimate.
    """
    if len(sim_blocks) != len(obs_blocks):
        raise ValueError(f"{len(sim_blocks)} simulated vs {len(obs_blocks)} observed blocks")
    out = []
    for sb, ob in zip(sim_blocks, obs_blocks):
        R_sim = np.asarray(sb["R"], dtype=float).copy()
        R_obs = np.asarray(ob["R"], dtype=float)
        if R_sim.shape != R_obs.shape:
            raise ValueError(f"block shape mismatch: sim {R_sim.shape} vs obs {R_obs.shape}")
        R_sim[~np.isfinite(R_obs)] = np.nan
        out.append({**sb, "R": R_sim})
    return out


def running_dry_age(R: np.ndarray) -> np.ndarray:
    """Per-station running dry-spell age, censored at block start and NaNs.

    ``age[t, j]`` is the number of consecutive dry days at station ``j`` up to
    and including day ``t`` (0 on wet days, NaN on missing days). The counter
    restarts after a NaN and at the block start, so ages near those edges are
    lower bounds — applying the same function to observed and simulated blocks
    keeps the censoring symmetric.
    """
    R = np.asarray(R, dtype=float)
    T, m = R.shape
    age = np.full((T, m), np.nan, dtype=float)
    prev = np.zeros(m, dtype=float)
    for t in range(T):
        r = R[t]
        is_dry = r == 0
        is_nan = ~np.isfinite(r)
        prev = np.where(is_dry, prev + 1.0, 0.0)
        row = np.where(is_dry, prev, 0.0)
        row[is_nan] = np.nan
        prev[is_nan] = 0.0
        age[t] = row
    return age


# ---------------------------------------------------------------------------
# Statistics on blocks
# ---------------------------------------------------------------------------


def _pairwise_binary_stats(
    X: np.ndarray,
    station_names: Sequence[str],
    dist_matrix: np.ndarray,
    min_common_days: int = 200,
) -> pd.DataFrame:
    """Per-pair joint statistics of a binary (0/1/NaN) day x station matrix.

    For every station pair with at least ``min_common_days`` jointly observed
    days: ``p11`` = P(both 1), ``p00`` = P(both 0), ``phi`` = Pearson
    correlation of the two binary series (NaN if one series is constant), and
    ``phi_dry`` the same correlation built from the *complementary* (0-valued)
    indicators,

        phi_dry = (p00 - (1 - p_j)(1 - p_k)) / sqrt(p_j(1-p_j) p_k(1-p_k)).

    The phi coefficient is invariant under the 0<->1 relabelling
    (Corr(1-X, 1-Y) = Corr(X, Y)), so ``phi_dry`` equals ``phi`` exactly; it is
    kept as the explicit dry-indicator form (and a consistency check).
    """
    X = np.asarray(X, dtype=float)
    m = X.shape[1]
    records = []
    with np.errstate(invalid="ignore"):
        for j in range(m - 1):
            for k in range(j + 1, m):
                xj, xk = X[:, j], X[:, k]
                valid = np.isfinite(xj) & np.isfinite(xk)
                n = int(valid.sum())
                if n < min_common_days:
                    continue
                a, b = xj[valid], xk[valid]
                phi = (
                    float(np.corrcoef(a, b)[0, 1])
                    if a.std() > 0 and b.std() > 0
                    else np.nan
                )
                pj, pk = float(a.mean()), float(b.mean())
                p00 = float(np.mean((a == 0) & (b == 0)))
                denom = np.sqrt(pj * (1.0 - pj) * pk * (1.0 - pk))
                phi_dry = (
                    float((p00 - (1.0 - pj) * (1.0 - pk)) / denom)
                    if denom > 0
                    else np.nan
                )
                records.append({
                    "station_j": station_names[j],
                    "station_k": station_names[k],
                    "dist": float(dist_matrix[j, k]),
                    "n_days": n,
                    "p11": float(np.mean((a == 1) & (b == 1))),
                    "p00": p00,
                    "phi": phi,
                    "phi_dry": phi_dry,
                })
    return pd.DataFrame(records)


def pairwise_occurrence_stats(
    blocks: List[dict],
    params_by_station: Dict[str, dict],
    min_common_days: int = 200,
) -> pd.DataFrame:
    """Pairwise wet/dry co-occurrence statistics pooled over all blocks.

    Columns: ``station_j, station_k, dist, n_days, p11, p00, phi, phi_dry``
    where 1 = wet and 0 = dry (so ``p11`` = P(both wet), ``p00`` = P(both dry),
    ``phi`` = correlation of the wet indicators, ``phi_dry`` = correlation of the
    dry indicators, equal to ``phi`` by the 0<->1 invariance of the phi
    coefficient).
    """
    station_names = list(blocks[0]["station_names"])
    X = np.vstack([np.asarray(b["R"], dtype=float) for b in blocks])
    D = station_distance_matrix(params_by_station, station_names)
    return _pairwise_binary_stats(X, station_names, D, min_common_days=min_common_days)


def heavy_dry_stats(
    blocks: List[dict],
    params_by_station: Dict[str, dict],
    heavy_dry_min_days: int,
    min_common_days: int = 200,
) -> pd.DataFrame:
    """Pairwise synchrony of *heavy* dry spells (running age >= threshold).

    Same columns as :func:`pairwise_occurrence_stats`, but computed on the
    daily indicator "station currently in a dry spell of age >=
    ``heavy_dry_min_days``" (from :func:`running_dry_age`, hence block-start
    censored identically for observed and simulated blocks). ``p11`` is then
    P(both stations simultaneously in a heavy dry spell).
    """
    station_names = list(blocks[0]["station_names"])
    mats = []
    for b in blocks:
        age = running_dry_age(b["R"])
        with np.errstate(invalid="ignore"):
            H = np.where(np.isfinite(age), (age >= heavy_dry_min_days).astype(float), np.nan)
        mats.append(H)
    X = np.vstack(mats)
    D = station_distance_matrix(params_by_station, station_names)
    return _pairwise_binary_stats(X, station_names, D, min_common_days=min_common_days)


def daily_wet_fraction(blocks: List[dict], min_stations: int = 10) -> np.ndarray:
    """Daily fraction of wet stations among the observed ones, pooled over blocks.

    Days with fewer than ``min_stations`` observed stations are dropped (the
    fraction would be too noisy to compare).
    """
    out = []
    for b in blocks:
        R = np.asarray(b["R"], dtype=float)
        n_obs = np.isfinite(R).sum(axis=1)
        n_wet = (R == 1).sum(axis=1)
        keep = n_obs >= min_stations
        out.append(n_wet[keep] / n_obs[keep])
    return np.concatenate(out) if out else np.array([])


def heavy_dry_daily_fraction(
    blocks: List[dict],
    heavy_dry_min_days: int,
    min_stations: int = 10,
) -> np.ndarray:
    """Daily fraction of stations currently in a heavy dry spell (drought extent)."""
    out = []
    for b in blocks:
        age = running_dry_age(b["R"])
        n_obs = np.isfinite(age).sum(axis=1)
        with np.errstate(invalid="ignore"):
            n_heavy = (age >= heavy_dry_min_days).sum(axis=1)
        keep = n_obs >= min_stations
        out.append(n_heavy[keep] / n_obs[keep])
    return np.concatenate(out) if out else np.array([])


def extract_complete_spells(blocks: List[dict], kind: str = "dry") -> List[int]:
    """Durations of *complete* spells pooled over blocks and stations.

    Within each block and station, the series is split into NaN-free segments;
    the first and last run of every segment are dropped (their start or end is
    censored by the block boundary or by a gap), and the remaining interior
    runs of the requested type are returned. The identical rule on observed
    and simulated blocks makes the censoring symmetric.
    """
    if kind not in ("dry", "wet"):
        raise ValueError(f"kind must be 'dry' or 'wet', got {kind!r}")
    target = 0.0 if kind == "dry" else 1.0
    durations: List[int] = []
    for b in blocks:
        R = np.asarray(b["R"], dtype=float)
        T, m = R.shape
        for j in range(m):
            col = R[:, j]
            finite = np.isfinite(col)
            t = 0
            while t < T:
                if not finite[t]:
                    t += 1
                    continue
                s = t
                while t < T and finite[t]:
                    t += 1
                runs = _runs(col[s:t])
                for value, length in runs[1:-1]:
                    if value == target:
                        durations.append(int(length))
    return durations


def _runs(values: np.ndarray) -> List[tuple]:
    """Run-length encode a 1-D array without NaNs -> list of (value, length)."""
    out = []
    start = 0
    n = len(values)
    for i in range(1, n + 1):
        if i == n or values[i] != values[start]:
            out.append((values[start], i - start))
            start = i
    return out


# ---------------------------------------------------------------------------
# Diagnostic 1 — spatially ordered occurrence raster (obs vs sim)
# ---------------------------------------------------------------------------


_RASTER_COLORS = ["#4c78a8", "#f5deb3", "#8c1d18", "#d9d9d9"]
_RASTER_LABELS = ["wet", "dry", "heavy dry", "missing"]


def _category_matrix(R: np.ndarray, heavy_dry_min_days: int) -> np.ndarray:
    """0 = wet, 1 = dry (age < h), 2 = heavy dry (age >= h), 3 = missing."""
    R = np.asarray(R, dtype=float)
    age = running_dry_age(R)
    cat = np.full(R.shape, 3.0)
    with np.errstate(invalid="ignore"):
        cat[R == 1] = 0.0
        cat[(R == 0) & (age < heavy_dry_min_days)] = 1.0
        cat[(R == 0) & (age >= heavy_dry_min_days)] = 2.0
    return cat


def _block_label(block: dict, i: int) -> str:
    if "year" in block:
        return str(block["year"])
    if "block" in block:
        return f"sim {block['block'] + 1}"
    return f"block {i + 1}"


def _draw_raster(ax, blocks, order_idx, heavy_dry_min_days, station_labels):
    cats = [_category_matrix(b["R"], heavy_dry_min_days) for b in blocks]
    mat = np.vstack(cats)[:, order_idx].T  # stations x days
    cmap = ListedColormap(_RASTER_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    # season separators + x labels at block centres
    lengths = [b["R"].shape[0] for b in blocks]
    edges = np.cumsum(lengths)
    for e in edges[:-1]:
        ax.axvline(e - 0.5, color="white", lw=2)
    centers = edges - np.asarray(lengths) / 2.0
    ax.set_xticks(centers)
    ax.set_xticklabels([_block_label(b, i) for i, b in enumerate(blocks)], fontsize=8)
    ax.set_yticks(range(len(station_labels)))
    ax.set_yticklabels(station_labels, fontsize=5)
    return ax


def plot_station_order_map(
    params_by_station: Dict[str, dict],
    station_names: Optional[Sequence[str]] = None,
    order: Optional[List[str]] = None,
    save: bool = True,
    filename: str = "diag_station_order_map.pdf",
):
    """Map of the seriation order: where each raster row sits geographically.

    The :math:`J` stations are placed at their normalised coordinates and
    coloured by their rank in the spatial order (:func:`spatial_station_order`),
    so the colour gradient traces the path that the occurrence-raster rows
    (top → bottom) follow across the map. Used as the common station axis of the
    rasters in :func:`plot_occurrence_raster_obs_vs_sim`.
    """
    if station_names is None:
        station_names = sorted(params_by_station.keys())
    station_names = list(station_names)
    if order is None:
        order = spatial_station_order(params_by_station, station_names)

    fig, ax_map = plt.subplots(figsize=(5.5, 5.0))
    xs = [params_by_station[c]["x_norm"] for c in order]
    ys = [params_by_station[c]["y_norm"] for c in order]
    sc = ax_map.scatter(xs, ys, c=range(len(order)), cmap="viridis", s=55, zorder=3)
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax_map.annotate(str(i + 1), (x, y), fontsize=5.5, xytext=(3, 3),
                        textcoords="offset points")
    plt.colorbar(sc, ax=ax_map, label="raster row (top → bottom)", shrink=0.8)
    ax_map.set_xlabel("x_norm (lon)")
    ax_map.set_ylabel("y_norm (lat)")
    ax_map.set_title("Station order on the map", fontsize=10)

    plt.tight_layout()
    if save:
        fig.savefig(_ensure_figures_dir() / filename, bbox_inches="tight")
    return fig


def plot_occurrence_raster_obs_vs_sim(
    obs_blocks: List[dict],
    sim_blocks: List[dict],
    params_by_station: Dict[str, dict],
    order: Optional[List[str]] = None,
    heavy_dry_min_days: int = 10,
    n_seasons: int = 2,
    first_season: int = 0,
    save: bool = True,
    filename: str = "diag_raster_obs_vs_sim.pdf",
    figsize = (10, 7.5),
):
    """Observed vs simulated occurrence rasters with spatially ordered stations.

    Two stacked day x station rasters (observed on top, one simulated replicate
    below) share the same y-axis: stations ordered by spatial proximity
    (:func:`spatial_station_order`), so synchronous wet/dry episodes at nearby
    stations appear as vertically contiguous patches — and heavy dry spells
    (running age >= ``heavy_dry_min_days``) as dark-red blocks. The geographic
    location of each raster row is shown separately by
    :func:`plot_station_order_map`.
    """
    station_names = list(obs_blocks[0]["station_names"])
    if list(sim_blocks[0]["station_names"]) != station_names:
        raise ValueError("observed and simulated blocks have different station lists")
    if order is None:
        order = spatial_station_order(params_by_station, station_names)
    order_idx = [station_names.index(c) for c in order]

    obs_sel = obs_blocks[first_season: first_season + n_seasons]
    sim_sel = sim_blocks[first_season: first_season + n_seasons]

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, hspace=0.25)
    ax_obs = fig.add_subplot(gs[0, 0])
    ax_sim = fig.add_subplot(gs[1, 0])

    _draw_raster(ax_obs, obs_sel, order_idx, heavy_dry_min_days, order)
    ax_obs.set_title("Observed", fontsize=10, loc="left")
    _draw_raster(ax_sim, sim_sel, order_idx, heavy_dry_min_days, order)
    ax_sim.set_title("Simulated (one replicate)", fontsize=10, loc="left")
    ax_sim.set_xlabel("season (years / simulated blocks)")

    handles = [Patch(facecolor=c, label=l) for c, l in zip(_RASTER_COLORS, _RASTER_LABELS)]
    handles[2] = Patch(facecolor=_RASTER_COLORS[2],
                       label=f"heavy dry (≥ {heavy_dry_min_days} d)")
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Occurrence raster, stations ordered by spatial proximity",
                 y=1.04, fontsize=12)

    if save:
        fig.savefig(_ensure_figures_dir() / filename, bbox_inches="tight")
    return fig


_SEASON_COLORS = {
    "winter": "#7570b3",
    "spring": "#66a61e",
    "summer": "#e6ab02",
    "autumn": "#a6761d",
}


def history_to_Rbin(history: dict) -> pd.DataFrame:
    """Seasonal simulated history -> days x stations 0/1 occurrence DataFrame.

    ``history`` comes from :func:`spatial_model.simulate_cholesky_seasonal`,
    whose ``R`` rows are aligned one-to-one with ``history["dates"]``.
    """
    return pd.DataFrame(
        history["R"],
        index=pd.DatetimeIndex(history["dates"]),
        columns=history["station_names"],
    )


def _draw_time_rows_panel(ax_season, ax, Rbin, order, heavy_dry_min_days):
    """One season-strip + raster panel of the time-rows occurrence raster."""
    from spatial_bmcd.spatial_model import season_of_dates

    station_names = list(Rbin.columns)
    order_idx = [station_names.index(c) for c in order]
    dates = pd.DatetimeIndex(pd.to_datetime(Rbin.index))
    cat = _category_matrix(Rbin.to_numpy(dtype=float), heavy_dry_min_days)[:, order_idx]

    # Main raster: time on rows, seriated stations on columns.
    cmap = ListedColormap(_RASTER_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    ax.imshow(cat, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=5)
    ax.tick_params(top=False, bottom=False, labeltop=True, labelbottom=True,
                   left=False, labelleft=False)

    # Season sidebar strip (one colour per meteorological season).
    seasons = season_of_dates(dates)
    season_keys = list(_SEASON_COLORS)
    season_int = np.array([season_keys.index(s) for s in seasons])[:, None]
    ax_season.imshow(
        season_int, aspect="auto", interpolation="nearest",
        cmap=ListedColormap(list(_SEASON_COLORS.values())),
        norm=BoundaryNorm(np.arange(len(season_keys) + 1) - 0.5, len(season_keys)),
    )
    ax_season.set_xticks([])

    # Year separators + year labels at each 1 January.
    year_starts = np.flatnonzero(np.r_[True, dates.year[1:] != dates.year[:-1]])
    for y0 in year_starts[1:]:
        ax.axhline(y0 - 0.5, color="white", lw=0.8)
        ax_season.axhline(y0 - 0.5, color="white", lw=0.8)
    ax_season.set_yticks(year_starts)
    ax_season.set_yticklabels(dates.year[year_starts], fontsize=7)


def _raster_legend_handles(heavy_dry_min_days):
    """Legend patches for the raster categories + the four season colours."""
    handles = [Patch(facecolor=c, label=l) for c, l in zip(_RASTER_COLORS, _RASTER_LABELS)]
    handles[2] = Patch(facecolor=_RASTER_COLORS[2],
                       label=f"heavy dry (≥ {heavy_dry_min_days} d)")
    handles += [Patch(facecolor=c, label=s) for s, c in _SEASON_COLORS.items()]
    return handles


def plot_occurrence_raster_time_rows(
    Rbin: pd.DataFrame,
    params_by_station: Dict[str, dict],
    order: Optional[List[str]] = None,
    heavy_dry_min_days: int = 10,
    time_window: Optional[Tuple[str, str]] = None,
    station_width: Optional[float] = None,
    title: str = "Occurrence raster",
    save: bool = True,
    filename: str = "diag_raster_time_rows.pdf",
    figsize=(12, 40),
):
    """Calendar occurrence raster: one row per day, stations on the columns.

    Transposed variant of :func:`plot_occurrence_raster_obs_vs_sim` for a
    *continuous multi-season* record (observed ``Rbin`` or simulated one via
    :func:`history_to_Rbin`): time runs top to bottom over the full calendar,
    stations are ordered by spatial proximity (:func:`spatial_station_order`),
    and a narrow left strip colours each day by its meteorological season.
    Cells encode wet / dry / heavy dry (running dry age >= ``heavy_dry_min_days``,
    censored only at NaNs and at the record start) / missing. Year separators
    (thin white lines) and year labels mark each 1 January.

    ``time_window=(start, end)`` restricts the plot to that date range
    (inclusive, anything ``pd.Timestamp`` accepts, e.g. ``("1990", "1995")``);
    heavy-dry ages are censored at the window start like at a record start.
    ``station_width`` (inches per station column) overrides the width in
    ``figsize`` so column size is set directly.
    """
    if time_window is not None:
        Rbin = Rbin.loc[time_window[0]:time_window[1]]
    if order is None:
        order = spatial_station_order(params_by_station, list(Rbin.columns))
    if station_width is not None:
        figsize = (len(order) * station_width + 1.0, figsize[1])

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2, width_ratios=[1, 50], wspace=0.02,
                  left=0.07, right=0.99, top=0.96, bottom=0.03)
    ax_season = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1], sharey=ax_season)
    _draw_time_rows_panel(ax_season, ax, Rbin, order, heavy_dry_min_days)

    fig.legend(handles=_raster_legend_handles(heavy_dry_min_days),
               loc="upper center", ncol=8, frameon=False,
               bbox_to_anchor=(0.5, 0.99), fontsize=8)
    fig.suptitle(title, y=0.995, fontsize=12)

    if save:
        fig.savefig(_ensure_figures_dir() / filename, bbox_inches="tight")
    return fig


def plot_occurrence_raster_time_rows_pair(
    Rbin_left: pd.DataFrame,
    Rbin_right: pd.DataFrame,
    params_by_station: Dict[str, dict],
    order: Optional[List[str]] = None,
    heavy_dry_min_days: int = 10,
    time_window: Optional[Tuple[str, str]] = None,
    station_width: Optional[float] = None,
    titles: Sequence[str] = ("Observed", "Simulated"),
    suptitle: Optional[str] = None,
    save: bool = True,
    filename: str = "diag_raster_time_rows_pair.pdf",
    figsize=(24, 44),
):
    """Two calendar occurrence rasters side by side (e.g. observed vs simulated).

    Each half is one :func:`plot_occurrence_raster_time_rows` panel (season
    strip + raster) with its own ``titles`` entry; the legend (and optional
    ``suptitle``) is shared. Both records use the same station ``order`` so the
    columns are directly comparable. ``time_window`` and ``station_width``
    behave as in :func:`plot_occurrence_raster_time_rows` (the window is
    applied to both records; the width accounts for the two panels).
    """
    if time_window is not None:
        Rbin_left = Rbin_left.loc[time_window[0]:time_window[1]]
        Rbin_right = Rbin_right.loc[time_window[0]:time_window[1]]
    if order is None:
        order = spatial_station_order(params_by_station, list(Rbin_left.columns))
    if station_width is not None:
        figsize = (2 * len(order) * station_width + 2.0, figsize[1])

    fig = plt.figure(figsize=figsize)
    outer = GridSpec(1, 2, wspace=0.10,
                     left=0.045, right=0.99, top=0.95, bottom=0.03)
    for Rbin, spec, panel_title in zip((Rbin_left, Rbin_right), outer, titles):
        inner = spec.subgridspec(1, 2, width_ratios=[1, 50], wspace=0.02)
        ax_season = fig.add_subplot(inner[0, 0])
        ax = fig.add_subplot(inner[0, 1], sharey=ax_season)
        _draw_time_rows_panel(ax_season, ax, Rbin, order, heavy_dry_min_days)
        # y unset -> matplotlib auto-places the title above the top tick labels.
        ax.set_title(panel_title, fontsize=11)

    fig.legend(handles=_raster_legend_handles(heavy_dry_min_days),
               loc="upper center", ncol=8, frameon=False,
               bbox_to_anchor=(0.5, 0.998), fontsize=8)
    if suptitle:
        fig.suptitle(suptitle, y=0.9995, fontsize=13)

    if save:
        fig.savefig(_ensure_figures_dir() / filename, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Diagnostics 2 & 3 — pairwise statistics vs distance (obs dots vs sim band)
# ---------------------------------------------------------------------------


_STAT_LABELS = {
    "phi": "correlation of wet indicators",
    "phi_dry": "correlation of dry indicators",
    "p11": "P(both wet)",
    "p00": "P(both dry)",
}


def _ensemble_band(ax, sim_dfs, stat, n_bins, color, label):
    """Shaded 2.5-97.5% band + median of pooled (pair x replicate) values, by distance bin."""
    pooled = pd.concat(sim_dfs, ignore_index=True)[["dist", stat]].dropna()
    edges = np.unique(np.quantile(pooled["dist"], np.linspace(0, 1, n_bins + 1)))
    centers, lo, med, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        vals = pooled.loc[(pooled["dist"] >= a) & (pooled["dist"] <= b), stat]
        if len(vals) == 0:
            continue
        centers.append(0.5 * (a + b))
        lo.append(np.percentile(vals, 2.5))
        med.append(np.percentile(vals, 50))
        hi.append(np.percentile(vals, 97.5))
    ax.fill_between(centers, lo, hi, color=color, alpha=0.25, label=f"{label} (95% band)")
    ax.plot(centers, med, color=color, lw=1.6, label=f"{label} (median)")


def plot_pairwise_stat_vs_distance(
    obs_df: pd.DataFrame,
    sim_dfs: List[pd.DataFrame],
    stats: Sequence[str] = ("phi", "phi_dry", "p11", "p00"),
    sim_dfs_alt: Optional[List[pd.DataFrame]] = None,
    labels: Sequence[str] = ("simulated", "simulated, obs-NaN mask"),
    stat_labels: Optional[Dict[str, str]] = None,
    n_bins: int = 10,
    sigma_marker: Optional[float] = None,
    save: bool = True,
    filename: str = "diag_pairwise_vs_distance.pdf",
):
    """Observed pairwise statistics vs distance against the simulated envelope.

    One panel per statistic. Observed pairs are black dots; the simulated
    ensemble is summarised by per-pair means (light dots) and a distance-binned
    2.5-97.5% band pooling pairs and replicates. ``sim_dfs_alt`` overlays a
    second band (e.g. NaN-masked simulations) for sensitivity checks;
    ``sigma_marker`` draws a vertical line at the fitted range parameter.
    """
    stat_labels = {**_STAT_LABELS, **(stat_labels or {})}
    fig, axes = plt.subplots(1, len(stats), figsize=(4.8 * len(stats), 4.0), squeeze=False)
    for ax, stat in zip(axes[0], stats):
        _ensemble_band(ax, sim_dfs, stat, n_bins, "steelblue", labels[0])
        per_pair = (
            pd.concat(sim_dfs, ignore_index=True)
            .groupby(["station_j", "station_k"], as_index=False)
            .agg(dist=("dist", "first"), val=(stat, "mean"))
        )
        ax.scatter(per_pair["dist"], per_pair["val"], s=9, color="steelblue",
                   alpha=0.35, linewidths=0, label=f"{labels[0]} (pair means)")
        if sim_dfs_alt is not None:
            _ensemble_band(ax, sim_dfs_alt, stat, n_bins, "darkorange", labels[1])
        ax.scatter(obs_df["dist"], obs_df[stat], s=12, color="black", alpha=0.7,
                   linewidths=0, zorder=3, label="observed")
        if sigma_marker is not None:
            ax.axvline(sigma_marker, color="grey", ls="--", lw=1)
            ax.text(sigma_marker, ax.get_ylim()[1], r" $\hat\sigma$",
                    va="top", ha="left", color="grey")
        ax.set_xlabel("inter-station distance (km)")
        ax.set_title(stat_labels.get(stat, stat), fontsize=10)
    axes[0][0].legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    if save:
        fig.savefig(_ensure_figures_dir() / filename, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Diagnostics 3 & 4 — distribution of a daily fraction (obs vs sim envelope)
# ---------------------------------------------------------------------------


def plot_fraction_distribution(
    obs_values: np.ndarray,
    sim_values_list: List[np.ndarray],
    bins: Optional[np.ndarray] = None,
    sim_values_list_alt: Optional[List[np.ndarray]] = None,
    labels: Sequence[str] = ("simulated", "simulated, obs-NaN mask"),
    xlabel: str = "daily fraction of stations",
    title: str = "",
    log_y: bool = False,
    ax: Optional[plt.Axes] = None,
    save: bool = True,
    filename: str = "diag_fraction_distribution.pdf",
):
    """Histogram of a daily fraction: observed steps vs simulated per-bin envelope.

    The envelope is the 2.5-97.5% range, across replicates, of each bin's
    density; ``sim_values_list_alt`` overlays a second envelope.
    """
    if bins is None:
        bins = np.linspace(0.0, 1.0, 21)
    centers = 0.5 * (bins[:-1] + bins[1:])
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 4.2))

    def _band(values_list, color, label):
        dens = np.array([np.histogram(v, bins=bins, density=True)[0] for v in values_list])
        lo, med, hi = np.percentile(dens, [2.5, 50, 97.5], axis=0)
        ax.fill_between(centers, lo, hi, step="mid", color=color, alpha=0.25,
                        label=f"{label} (95% band)")
        ax.step(centers, med, where="mid", color=color, lw=1.6, label=f"{label} (median)")

    _band(sim_values_list, "steelblue", labels[0])
    if sim_values_list_alt is not None:
        _band(sim_values_list_alt, "darkorange", labels[1])
    obs_dens, _ = np.histogram(obs_values, bins=bins, density=True)
    ax.step(centers, obs_dens, where="mid", color="black", lw=1.8, label="observed")

    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    plt.tight_layout()
    if save:
        ax.figure.savefig(_ensure_figures_dir() / filename, bbox_inches="tight")
    return ax.figure


# ---------------------------------------------------------------------------
# Diagnostic 5 — spell-duration survival curves (obs vs sim envelope)
# ---------------------------------------------------------------------------


def plot_spell_survival_obs_vs_sim(
    obs_blocks: List[dict],
    sim_blocks_list: List[List[dict]],
    kinds: Sequence[str] = ("dry", "wet"),
    d_max: Optional[int] = None,
    save: bool = True,
    filename: str = "diag_spell_survival.pdf",
):
    """Pooled spell-length survival P(tau >= d), observed curve vs simulated band.

    Spells are the *complete* ones from :func:`extract_complete_spells`
    (identical block-edge censoring on both sides), pooled over stations and
    blocks. Log-scale y; one panel per spell type.
    """
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.4 * len(kinds), 4.2), squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        obs_sp = np.asarray(extract_complete_spells(obs_blocks, kind=kind))
        sim_sps = [np.asarray(extract_complete_spells(bl, kind=kind)) for bl in sim_blocks_list]
        dm = d_max
        if dm is None:
            longest = max([obs_sp.max() if obs_sp.size else 1]
                          + [sp.max() if sp.size else 1 for sp in sim_sps])
            dm = int(min(80, longest))
        ds = np.arange(1, dm + 1)
        surv_obs = np.array([(obs_sp >= d).mean() if obs_sp.size else np.nan for d in ds])
        surv_sim = np.array([
            [(sp >= d).mean() if sp.size else np.nan for d in ds] for sp in sim_sps
        ])
        lo, med, hi = np.nanpercentile(surv_sim, [2.5, 50, 97.5], axis=0)
        ax.fill_between(ds, lo, hi, color="steelblue", alpha=0.25, label="simulated (95% band)")
        ax.plot(ds, med, color="steelblue", lw=1.6, label="simulated (median)")
        ax.plot(ds, surv_obs, color="black", lw=1.8, marker=".", ms=4,
                label=f"observed (n={obs_sp.size})")
        ax.set_yscale("log")
        ax.set_xlabel("spell duration d (days)")
        ax.set_ylabel(r"P($\tau \geq$ d)")
        ax.set_title(f"{kind} spells (complete, pooled over stations)", fontsize=10)
        ax.legend(fontsize=8)
    plt.tight_layout()
    if save:
        fig.savefig(_ensure_figures_dir() / filename, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Switch-dependence diagnostics (main.tex, empirical motivation of the spatial
# model): per-pair statistics of the transition indicators S, stratified by
# the current state pair of the two stations
# ---------------------------------------------------------------------------

# The model's spatial dependence acts on the switch indicators
# ``S[n, j] = 1{R[n+1, j] != R[n, j]}`` conditionally on the current states, so
# every statistic below is stratified by the state pair of the two stations:
_SWITCH_CLASSES = ("dry-dry", "wet-wet", "dry-wet")


def _switch_class_labels(rj: np.ndarray, rk: np.ndarray) -> np.ndarray:
    """State-pair class of each day: dry-dry (0,0), wet-wet (1,1), else dry-wet."""
    return np.where(
        (rj == 0) & (rk == 0), "dry-dry",
        np.where((rj == 1) & (rk == 1), "wet-wet", "dry-wet"),
    )


def exit_prob_matrix(
    history: dict,
    params_by_station: Dict[str, dict],
    clip_q: float = 1e-12,
) -> np.ndarray:
    """Fitted exit probabilities ``q[n, j] = q^(R_nj)_{s_j}(D_nj)`` aligned with ``S``.

    Same lookup as ``spatial_model._exit_probs`` (single-site fits treated as
    known inputs), vectorised over days by caching the per-duration values.
    Rows where ``R``, ``D`` or ``S`` is NaN at station ``j`` stay NaN.
    """
    R, D, S = history["R"], history["D"], history["S"]
    names = history["station_names"]
    T = S.shape[0]
    Q = np.full((T, len(names)), np.nan, dtype=float)
    for j, name in enumerate(names):
        funcs = {
            0: params_by_station[name]["q_d_dry_function"],
            1: params_by_station[name]["q_d_wet_function"],
        }
        Rj, Dj = R[:T, j], D[:T, j]
        valid = np.isfinite(Rj) & np.isfinite(Dj) & np.isfinite(S[:, j])
        for r, f in funcs.items():
            idx = np.flatnonzero(valid & (Rj == r))
            if idx.size == 0:
                continue
            ds = Dj[idx].astype(int)
            lut = {d: float(f(d)) for d in np.unique(ds)}
            Q[idx, j] = [lut[d] for d in ds]
    return np.clip(Q, clip_q, 1.0 - clip_q)


def switch_marginal_calibration(
    histories: List[dict],
    params_by_station: Dict[str, dict],
) -> pd.DataFrame:
    """Per-station ratio (observed switches) / (sum of fitted exit probabilities).

    Sanity check of the single-site fits used to standardise the pairwise
    statistics: under a well-calibrated marginal model the ratio is ~1 for
    every station (separately per current state), so any pairwise ratio far
    from 1 must come from *dependence between* stations, not from the marginals.
    """
    names = histories[0]["station_names"]
    n_sw = np.zeros((2, len(names)))
    sum_q = np.zeros((2, len(names)))
    n_days = np.zeros((2, len(names)), dtype=int)
    for h in histories:
        Q = exit_prob_matrix(h, params_by_station)
        T = h["S"].shape[0]
        R, S = h["R"][:T], h["S"]
        for r in (0, 1):
            m = np.isfinite(Q) & (R == r)
            n_sw[r] += np.where(m, np.nan_to_num(S), 0.0).sum(axis=0)
            sum_q[r] += np.where(m, Q, 0.0).sum(axis=0)
            n_days[r] += m.sum(axis=0)
    rows = []
    for j, name in enumerate(names):
        for r, state in ((0, "dry"), (1, "wet")):
            rows.append({
                "station": name, "state": state, "n_days": int(n_days[r, j]),
                "n_switches": int(n_sw[r, j]), "expected_switches": sum_q[r, j],
                "ratio": n_sw[r, j] / sum_q[r, j] if sum_q[r, j] > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def _pair_class_negloglik(rho, z_j, z_k, s_j, s_k, q_j, q_k, eps: float = 1e-15):
    """Negative log-likelihood of one pair's switch outcomes on one state class.

    Same four-case bivariate-probit factor as the pairwise composite likelihood
    (Eq. rho_direct of main.tex, ``pairwise_loglik_one_step_nanaware``), but with
    a single free correlation ``rho`` instead of the LMC block.
    """
    q_joint = phi2_vec(z_j, z_k, rho)
    p = np.where(
        s_j & s_k, q_joint,
        np.where(
            s_j & ~s_k, q_j - q_joint,
            np.where(~s_j & s_k, q_k - q_joint, 1.0 - q_j - q_k + q_joint),
        ),
    )
    return -float(np.sum(np.log(np.maximum(p, eps))))


def switch_pair_stats(
    histories: List[dict],
    params_by_station: Dict[str, dict],
    min_days_per_class: int = 100,
    fit_rho: bool = True,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Per-pair, per-state-class dependence statistics of the switch indicators.

    For every station pair (j, k) and every class of ``_SWITCH_CLASSES``
    (current states of the two stations: both dry, both wet, discordant),
    pooling the days of all ``histories`` where both stations have valid
    ``(R, D, S)``:

    - ``ratio_emp`` — *joint-switch ratio with empirical margins*
      ``mean(S_j S_k) / (mean(S_j) mean(S_k))`` on the class days: observed
      joint-switch frequency over its value if the two stations switched
      independently at their observed class rates. Fully model-free; > 1 means
      positive dependence of the switches, < 1 negative dependence.
    - ``ratio`` — same numerator standardised by the *fitted single-site
      marginals*, ``sum_n S_j S_k / sum_n q_j q_k``. Ties the diagnostic to the
      single-site model (duration-resolved margins) but inherits its marginal
      calibration bias (cf. :func:`switch_marginal_calibration`), so
      ``ratio_emp`` is the cleaner stage-1 statistic.
    - ``rho_hat`` — *per-pair latent correlation*: MLE of the single free
      correlation of a thresholded bivariate Gaussian, maximising the pair's
      class-restricted likelihood (:func:`_pair_class_negloglik`). This is the
      empirical, unconstrained counterpart of the LMC block
      ``C^(r, r')(s_j, s_k)`` that the spatial model then parametrises.

    Returns a long DataFrame with columns ``station_j, station_k, dist,
    state_class, n_days, n_sw_j, n_sw_k, n_joint, expected_joint, ratio_emp,
    ratio, rho_hat``. Classes with fewer than ``min_days_per_class`` jointly
    valid days are skipped.
    """
    names = histories[0]["station_names"]
    m = len(names)
    dist = station_distance_matrix(params_by_station, names)

    # Pool the per-day quantities over histories once (T_total x m arrays).
    Q = np.vstack([exit_prob_matrix(h, params_by_station) for h in histories])
    S = np.vstack([h["S"] for h in histories])
    R = np.vstack([h["R"][:h["S"].shape[0]] for h in histories])
    Z = norm.ppf(Q)

    records = []
    pairs = [(j, k) for j in range(m - 1) for k in range(j + 1, m)]
    iterator = tqdm(pairs, desc="switch pair stats") if show_progress else pairs
    for j, k in iterator:
        valid = np.isfinite(Q[:, j]) & np.isfinite(Q[:, k])
        if not np.any(valid):
            continue
        rj, rk = R[valid, j].astype(int), R[valid, k].astype(int)
        sj, sk = S[valid, j].astype(bool), S[valid, k].astype(bool)
        qj, qk = Q[valid, j], Q[valid, k]
        zj, zk = Z[valid, j], Z[valid, k]
        labels = _switch_class_labels(rj, rk)
        for cls in _SWITCH_CLASSES:
            sel = labels == cls
            n = int(sel.sum())
            if n < min_days_per_class:
                continue
            n_sw_j, n_sw_k = int(np.sum(sj[sel])), int(np.sum(sk[sel]))
            n_joint = int(np.sum(sj[sel] & sk[sel]))
            expected_joint = float(np.sum(qj[sel] * qk[sel]))
            rho_hat = np.nan
            if fit_rho:
                res = minimize_scalar(
                    _pair_class_negloglik,
                    bounds=(-0.999, 0.999),
                    method="bounded",
                    args=(zj[sel], zk[sel], sj[sel], sk[sel], qj[sel], qk[sel]),
                )
                rho_hat = float(res.x)
            records.append({
                "station_j": names[j], "station_k": names[k],
                "dist": float(dist[j, k]), "state_class": cls, "n_days": n,
                "n_sw_j": n_sw_j, "n_sw_k": n_sw_k,
                "n_joint": n_joint, "expected_joint": expected_joint,
                "ratio_emp": (
                    n_joint * n / (n_sw_j * n_sw_k)
                    if n_sw_j > 0 and n_sw_k > 0 else np.nan
                ),
                "ratio": n_joint / expected_joint if expected_joint > 0 else np.nan,
                "rho_hat": rho_hat,
            })
    return pd.DataFrame(records)


def lmc_block_curves(theta, h: np.ndarray) -> Dict[str, np.ndarray]:
    """Model-implied LMC covariance blocks ``C^(r, r')`` at distances ``h``.

    The parametric curves (Eq. lmc_blocks_inline of main.tex, exponential latent
    correlations) that :func:`switch_pair_stats`'s free ``rho_hat`` estimates
    when the model holds; keys match ``_SWITCH_CLASSES``.
    """
    th = normalize_theta(theta)
    h = np.asarray(h, dtype=float)
    rho_wc = np.exp(-h / th["sigma_wc"])
    rho_w0 = np.exp(-h / th["sigma_w0"])
    rho_w1 = np.exp(-h / th["sigma_w1"])
    return {
        "dry-dry": th["lam0"] * rho_wc + (1.0 - th["lam0"]) * rho_w0,
        "wet-wet": th["lam1"] * rho_wc + (1.0 - th["lam1"]) * rho_w1,
        "dry-wet": -np.sqrt(th["lam0"] * th["lam1"]) * rho_wc,
    }


def _binned_median_line(ax, df, stat, n_bins, color, label):
    """Distance-binned median of per-pair values (equal-count bins)."""
    sub = df[["dist", stat]].dropna()
    if len(sub) == 0:
        return
    edges = np.unique(np.quantile(sub["dist"], np.linspace(0, 1, n_bins + 1)))
    centers, med = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        vals = sub.loc[(sub["dist"] >= a) & (sub["dist"] <= b), stat]
        if len(vals) == 0:
            continue
        centers.append(0.5 * (a + b))
        med.append(np.median(vals))
    ax.plot(centers, med, color=color, lw=1.8, label=label)


_SWITCH_STAT_LABELS = {
    "ratio_emp": "joint switches / expected under independence",
    "ratio": "joint switches / expected under independence (fitted margins)",
    "rho_hat": r"per-pair latent correlation $\hat\rho$",
}
_SWITCH_STAT_REFLINE = {"ratio_emp": 1.0, "ratio": 1.0, "rho_hat": 0.0}


def plot_switch_stat_vs_distance(
    obs_stats: pd.DataFrame,
    stat: str = "ratio_emp",
    sim_stats_list: Optional[List[pd.DataFrame]] = None,
    theta=None,
    n_bins: int = 8,
    save: bool = True,
    filename: str = "diag_switch_ratio_vs_distance.pdf",
):
    """Per-pair switch-dependence statistic vs distance, one panel per state class.

    ``obs_stats`` (and each element of ``sim_stats_list``) is the long DataFrame
    of :func:`switch_pair_stats`. Observed pairs are black dots with a
    distance-binned median; ``sim_stats_list`` adds the simulated 2.5-97.5%
    envelope (same statistic on trajectories simulated from the fitted model);
    ``theta`` overlays the model-implied LMC block curves of
    :func:`lmc_block_curves` (meaningful for ``stat="rho_hat"`` only, since the
    free per-pair correlation is what the blocks parametrise).
    """
    fig, axes = plt.subplots(
        1, len(_SWITCH_CLASSES), figsize=(4.8 * len(_SWITCH_CLASSES), 4.0),
        squeeze=False, sharey=True,
    )
    h_grid = np.linspace(0.0, float(obs_stats["dist"].max()), 200)
    curves = lmc_block_curves(theta, h_grid) if theta is not None else None
    for ax, cls in zip(axes[0], _SWITCH_CLASSES):
        obs_c = obs_stats[obs_stats["state_class"] == cls]
        if sim_stats_list is not None:
            sim_c = [df[df["state_class"] == cls] for df in sim_stats_list]
            _ensemble_band(ax, sim_c, stat, n_bins, "steelblue", "simulated")
        ax.scatter(obs_c["dist"], obs_c[stat], s=12, color="black", alpha=0.6,
                   linewidths=0, zorder=3, label="observed (pairs)")
        _binned_median_line(ax, obs_c, stat, n_bins, "crimson", "observed (binned median)")
        if curves is not None:
            ax.plot(h_grid, curves[cls], color="forestgreen", lw=1.6, ls="--",
                    label=r"model $C^{(r,r')}(h)$ at $\hat\theta$")
        ax.axhline(_SWITCH_STAT_REFLINE.get(stat, 0.0), color="grey", lw=1, ls=":")
        ax.set_xlabel("inter-station distance (km)")
        ax.set_title(f"current states: {cls}", fontsize=10)
    axes[0][0].set_ylabel(_SWITCH_STAT_LABELS.get(stat, stat))
    axes[0][0].legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    if save:
        fig.savefig(_ensure_figures_dir() / filename, bbox_inches="tight")
    return fig
