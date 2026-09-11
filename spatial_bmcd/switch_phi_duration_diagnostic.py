"""Temporary diagnostic: duration-stratified Pearson correlation of the exit indicators.

Implements the metric of the temporary (violet) subsection inserted before
"Empirical spatial structure of the exit indicators" in main.tex: for every
station pair, the days are stratified by the *exact* current state pair
(r, r') AND duration pair (d, d'), so that within a cell the exit margins are
constant and the independence baseline of the Pearson (phi) correlation of the
two exit indicators is exactly 0 — no leakage from co-varying day-to-day
margins. Per-cell phi's are then combined per pair and state class with
day-count weights, which preserves the exact baseline.

Side computation: the same statistic with durations coarsened into bins
({1}, {2,3}, {4..7}, {8+}) — approximate baseline, more data per cell — and a
printed comparison of the two aggregations.

The article figure (Figure 1 of main.tex) shows the per-cell statistic at the
single admissibility threshold N >= 10, with one row per meteorological season
(spring, summer, autumn, winter) and one column per state-pair class.

The same statistic is also computed on trajectories simulated from the fitted
latent Gaussian model (``simulate_phi_cells``); the simulated record is masked
and processed exactly like the observed one, so the two dot clouds carry the
same estimation noise. Two model-vs-data figures follow from it:
``plot_phi_obs_vs_sim_paired_rows`` (two sub-rows per season, observed cloud on
top, simulated below, each with its own median curve) and
``plot_phi_quantile_fan_season_grid`` (no dots: median and top-decile curves of
both estimations on the same axes).

A last section reads the same statistic along the *equal-duration* diagonal
d = d' <= 20 -- all station pairs, only the state pairs carrying the same
duration at the two stations, at the article's threshold N >= 10 -- with that
duration given to the colour of the dot (``equal_duration_estimates``,
``plot_psi_equal_duration_grid``). The same cells are then read through the
fitted model: eq. (12) gives a value of psi for each of them at theta_hat
(``psi_model_equal_duration``), which is put against the observed estimate of
eq. (9) on the two axes of one scatter (``plot_psi_model_vs_obs_grid``,
``psi_model_vs_obs_summary``). Both are driven by
``spatial_bmcd_data_analysis.ipynb``, not by the script below.

Outputs:
- experiment_outputs/switch_phi_duration_cells_<season>.csv  (exact cells)
- experiment_outputs/switch_phi_duration_pairs_<season>.csv  (per-pair, exact + binned)
- figure diag_switch_phi_duration_per_cell_seasons.pdf (season x class grid),
  plus the spring-only per-pair and per-cell scatters, saved both to the
  article figures directory and to figures/spatial (+ .png copies in the
  article directory for quick inspection).

Run from the repo root:  python -m spatial_bmcd.switch_phi_duration_diagnostic
"""

from __future__ import annotations

import pathlib
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

if __name__ == "__main__":  # script use: no display needed (kept out of notebooks)
    matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spatial_bmcd.spatial_diagnostics import (  # noqa: E402
    _binned_median_line,
    station_distance_matrix,
)

_CLASSES = ("dry-dry", "wet-wet", "dry-wet")
_SEASONS = ("spring", "summer", "autumn", "winter")

# Duration bins of the binned variant: {1}, {2,3}, {4..7}, {8+}.
_BIN_EDGES = np.array([2, 4, 8])
_BIN_LABELS = ("1", "2-3", "4-7", "8+")


def _class_of(rj: np.ndarray, rk: np.ndarray) -> np.ndarray:
    return np.where(
        (rj == 0) & (rk == 0), "dry-dry",
        np.where((rj == 1) & (rk == 1), "wet-wet", "dry-wet"),
    )


def _pool_histories(histories: List[dict]):
    """Stack per-year (R, D, S) into day-pooled arrays aligned with S."""
    S = np.vstack([h["S"] for h in histories])
    R = np.vstack([h["R"][: h["S"].shape[0]] for h in histories])
    D = np.vstack([h["D"][: h["S"].shape[0]] for h in histories])
    return R, D, S


def switch_phi_duration_cells(
    histories: List[dict],
    params_by_station: Dict[str, dict],
    min_days_per_cell: int = 10,
    binned: bool = False,
    d_cap: int = 4000,
) -> pd.DataFrame:
    """Per-pair, per-(r, r', d, d')-cell Pearson correlation of the exit indicators.

    A cell of pair (j, k) collects the days n with exact current states
    (r_{n,j}, r_{n,k}) = (r, r') and durations (d_{n,j}, d_{n,k}) = (d, d')
    (both stations valid on days n and n+1). Within a cell the margins are
    constant, so the plug-in Pearson correlation of (b_{n,j}, b_{n,k}),

        phi_hat = (N * n11 - e_j * e_k)
                  / sqrt(e_j (N - e_j) e_k (N - e_k)),

    with N the cell size, e_j = sum b_{n,j}, e_k = sum b_{n,k} and
    n11 = sum b_{n,j} b_{n,k}, has an *exact* independence baseline of 0.
    Cells with N < ``min_days_per_cell`` or a degenerate margin (e = 0 or N)
    are returned with phi = NaN (they are excluded from any aggregation but
    kept for the day-coverage accounting).

    ``binned=True`` replaces the exact durations by their bin index
    ({1}, {2,3}, {4..7}, {8+}): approximate baseline, larger cells.
    """
    names = histories[0]["station_names"]
    m = len(names)
    dist = station_distance_matrix(params_by_station, names)
    R, D, S = _pool_histories(histories)
    valid_all = np.isfinite(R) & np.isfinite(D) & np.isfinite(S)

    records = []
    for j in range(m - 1):
        for k in range(j + 1, m):
            valid = valid_all[:, j] & valid_all[:, k]
            if not np.any(valid):
                continue
            rj, rk = R[valid, j].astype(np.int64), R[valid, k].astype(np.int64)
            dj, dk = D[valid, j].astype(np.int64), D[valid, k].astype(np.int64)
            sj, sk = S[valid, j].astype(bool), S[valid, k].astype(bool)
            if binned:
                dj, dk = np.digitize(dj, _BIN_EDGES), np.digitize(dk, _BIN_EDGES)
            # One integer key per (r, r', d, d') cell.
            key = ((rj * 2 + rk) * d_cap + np.minimum(dj, d_cap - 1)) * d_cap \
                + np.minimum(dk, d_cap - 1)
            uniq, inv = np.unique(key, return_inverse=True)
            n = np.bincount(inv)
            ej = np.bincount(inv, weights=sj)
            ek = np.bincount(inv, weights=sk)
            n11 = np.bincount(inv, weights=sj & sk)
            var_term = ej * (n - ej) * ek * (n - ek)
            ok = (n >= min_days_per_cell) & (var_term > 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                phi = np.where(ok, (n * n11 - ej * ek) / np.sqrt(var_term), np.nan)
            rr = uniq // (d_cap * d_cap)
            cd_j = (uniq // d_cap) % d_cap
            cd_k = uniq % d_cap
            r_j, r_k = rr // 2, rr % 2
            cls = _class_of(r_j, r_k)
            for i in range(len(uniq)):
                records.append({
                    "station_j": names[j], "station_k": names[k],
                    "dist": float(dist[j, k]), "state_class": cls[i],
                    "r_j": int(r_j[i]), "r_k": int(r_k[i]),
                    "d_j": _BIN_LABELS[cd_j[i]] if binned else int(cd_j[i]),
                    "d_k": _BIN_LABELS[cd_k[i]] if binned else int(cd_k[i]),
                    "n_days": int(n[i]), "n_exit_j": int(ej[i]),
                    "n_exit_k": int(ek[i]), "n_joint": int(n11[i]),
                    "phi": float(phi[i]),
                })
    return pd.DataFrame(records)


def aggregate_phi_per_pair(cells: pd.DataFrame) -> pd.DataFrame:
    """Day-count-weighted mean of per-cell phi, per (pair, state class).

    phi_bar = sum_cells N * phi_hat / sum_cells N over the admissible cells
    (phi defined). Each admissible cell has population phi = 0 under
    conditionally independent exits, so phi_bar keeps the exact 0 baseline.
    Also returns the day-coverage accounting: ``n_class_days`` (all valid
    class days of the pair) vs ``n_used`` (days inside admissible cells).
    """
    cells = cells.copy()
    cells["used"] = np.isfinite(cells["phi"])
    cells["w_phi"] = np.where(cells["used"], cells["n_days"] * cells["phi"], 0.0)
    cells["n_used"] = np.where(cells["used"], cells["n_days"], 0)
    g = cells.groupby(["station_j", "station_k", "state_class"], as_index=False).agg(
        dist=("dist", "first"),
        n_class_days=("n_days", "sum"),
        n_used=("n_used", "sum"),
        n_cells=("used", "sum"),
        w_phi=("w_phi", "sum"),
    )
    g["phi_bar"] = np.where(g["n_used"] > 0, g["w_phi"] / g["n_used"], np.nan)
    return g.drop(columns="w_phi")


def _sqrtn_size_alpha(n: np.ndarray, n_ref: float, s_min=3.0, s_max=55.0,
                      a_min=0.15, a_max=0.85):
    """Dot area and opacity both proportional to sqrt(N) (normalised, floored)."""
    u = np.sqrt(np.asarray(n, dtype=float)) / np.sqrt(n_ref)
    u = np.clip(u, 0.0, 1.0)
    return s_min + (s_max - s_min) * u, a_min + (a_max - a_min) * u


# Quantile pairs drawn around the central curve, from the outer to the inner one.
_QUANTILE_PAIRS = ((0.10, 0.90, ":"), (0.20, 0.80, "--"))
# One-sided levels, kept for a top-only variant of the fan figure.
_TOP_QUANTILES = (
    (0.90, ":", "top 10%"),
    (0.80, (0, (1, 1)), "top 20%"),
    (0.70, "--", "top 30%"),
    (0.60, (0, (3, 1, 1, 1)), "top 40%"),
)
# Levels of the dots-free fan figure: top *and* bottom 10%, 20%, 30% and 40%,
# i.e. the quantile pairs 10/90, 20/80, 30/70 and 40/60 around the median.
_DECILE_PAIRS = (
    (0.10, 0.90, ":"),
    (0.20, 0.80, (0, (1, 1))),
    (0.30, 0.70, "--"),
    (0.40, 0.60, (0, (3, 1, 1, 1))),
)


def bin_edges(cells: pd.DataFrame, n_bins: int, stat: str = "phi") -> np.ndarray:
    """Equal-count distance bins over the admissible estimates of ``cells``."""
    d = cells.loc[np.isfinite(cells[stat]), "dist"]
    return np.unique(np.quantile(d, np.linspace(0, 1, n_bins + 1)))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Quantile of ``values`` under the day-count weights ``weights``."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return np.nan
    # Weighted plotting positions, so equal weights give the usual quantile.
    p = (cw - 0.5 * w) / cw[-1]
    return float(np.interp(q, p, v))


def _binned_weighted_stats(df: pd.DataFrame, stat: str, n_col: str,
                           edges: np.ndarray,
                           quantiles: Sequence[float] = ()) -> Dict[str, np.ndarray]:
    """N-weighted mean (and quantiles) of ``stat`` in each distance bin.

    Each bin $b$ gets the day-count weighted mean
    ``phi_bar_b = sum_b N_c phi_c / sum_b N_c``, so an estimate resting on twice
    as many days weighs twice as much; the requested quantiles are taken under
    the same weights. Empty bins give ``NaN``.
    """
    sub = df[["dist", stat, n_col]].replace([np.inf, -np.inf], np.nan).dropna()
    n_b = len(edges) - 1
    out = {"centers": 0.5 * (edges[:-1] + edges[1:]),
           "mean": np.full(n_b, np.nan)}
    for q in quantiles:
        out[f"q{q:g}"] = np.full(n_b, np.nan)
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        m = (sub["dist"] >= a) & (sub["dist"] <= b)
        w = sub.loc[m, n_col].to_numpy(dtype=float)
        v = sub.loc[m, stat].to_numpy(dtype=float)
        if w.sum() <= 0:
            continue
        out["mean"][i] = float(w @ v / w.sum())
        for q in quantiles:
            out[f"q{q:g}"][i] = _weighted_quantile(v, w, q)
    return out


def _weighted_bin_means(df: pd.DataFrame, stat: str, n_col: str,
                        edges: np.ndarray) -> np.ndarray:
    """N-weighted mean of ``stat`` in each distance bin defined by ``edges``."""
    return _binned_weighted_stats(df, stat, n_col, edges)["mean"]


def _draw_binned_curves(ax, df, stat, n_col, edges, color, label,
                        zorder=8, center="mean", pairs=_QUANTILE_PAIRS,
                        singles=(), quantile_label=True):
    """Central curve of ``stat`` per distance bin, plus quantile curves around it.

    ``center`` is ``"mean"`` (the $N$-weighted bin mean of the article),
    ``"median"`` (its weighted 50% quantile) or ``None``. ``pairs`` are
    symmetric quantile pairs ``(q_lo, q_hi, linestyle)`` and ``singles`` are
    one-sided levels ``(q, linestyle, label)``, e.g. the top-decile curves of
    the fan figure. All curves are day-count weighted (see
    :func:`_binned_weighted_stats`) and drawn with a white outline so they stay
    readable over a dot cloud; the quantiles describe the *spread of the dots*
    inside each distance bin, not the uncertainty of the central curve.
    """
    qs = [q for p in pairs for q in p[:2]] + [q for q, *_ in singles]
    if center == "median":
        qs = qs + [0.5]
    st = _binned_weighted_stats(df, stat, n_col, edges, quantiles=qs)
    centers = st["centers"]
    stroke = [pe.Stroke(linewidth=3.4, foreground="white"), pe.Normal()]
    for q_lo, q_hi, ls in pairs:
        lab = (f"{label}, {q_lo:.0%}/{q_hi:.0%} quantiles"
               if quantile_label else None)
        ax.plot(centers, st[f"q{q_lo:g}"], color=color, lw=1.1, ls=ls,
                zorder=zorder - 1, path_effects=stroke, label=lab)
        ax.plot(centers, st[f"q{q_hi:g}"], color=color, lw=1.1, ls=ls,
                zorder=zorder - 1, path_effects=stroke)
    for q, ls, q_name in singles:
        ax.plot(centers, st[f"q{q:g}"], color=color, lw=1.1, ls=ls,
                zorder=zorder - 1, path_effects=stroke,
                label=f"{label}, {q_name}" if quantile_label else None)
    if center is not None:
        ax.plot(centers, st["mean" if center == "mean" else "q0.5"], color=color,
                lw=1.8, marker="o", ms=3.5, label=label, zorder=zorder,
                path_effects=[pe.Stroke(linewidth=4.0, foreground="white"),
                              pe.Normal()])
    return st


def plot_phi_vs_distance(
    df: pd.DataFrame,
    value_col: str,
    n_col: str,
    ylabel: str,
    n_bins: int = 8,
    filenames: Sequence[pathlib.Path] = (),
    title_note: str = "",
):
    """3-panel (state classes) scatter vs distance, sqrt(N) size/opacity encoding."""
    fig, axes = plt.subplots(1, len(_CLASSES), figsize=(4.8 * len(_CLASSES), 4.0),
                             squeeze=False, sharey=True)
    n_ref = float(df[n_col].max())
    for ax, cls in zip(axes[0], _CLASSES):
        sub = df[(df["state_class"] == cls) & np.isfinite(df[value_col])]
        s, a = _sqrtn_size_alpha(sub[n_col].to_numpy(), n_ref)
        ax.scatter(sub["dist"], sub[value_col], s=s, alpha=a, color="black",
                   linewidths=0, zorder=3)
        _binned_median_line(ax, sub.rename(columns={value_col: "stat"}), "stat",
                            n_bins, "crimson", "binned median")
        ax.axhline(0.0, color="grey", lw=1, ls=":")
        ax.set_xlabel("inter-station distance (km)")
        ax.set_title(f"current states: {cls}{title_note}", fontsize=10)
    axes[0][0].set_ylabel(ylabel)
    # sqrt(N)-encoded legend dots
    for n_leg in _legend_ns(n_ref):
        s, a = _sqrtn_size_alpha(np.array([n_leg]), n_ref)
        axes[0][0].scatter([], [], s=s[0], alpha=a[0], color="black",
                           label=f"N = {n_leg:d}")
    axes[0][0].legend(fontsize=7, loc="upper right", title="days used",
                      title_fontsize=7)
    plt.tight_layout()
    for f in filenames:
        fig.savefig(f, bbox_inches="tight")
    return fig


def _legend_ns(n_ref: float) -> List[int]:
    top = int(n_ref)
    return sorted({max(10, top // 100), max(10, top // 10), top})


def plot_phi_per_cell_threshold_grid(
    cells: pd.DataFrame,
    thresholds: Sequence[int] = (20, 30, 50),
    n_bins: int = 8,
    filenames: Sequence[pathlib.Path] = (),
):
    """Per-cell phi vs distance for several minimum cell sizes, one row per threshold.

    Same encoding as the per-cell figure (dot area and opacity ~ sqrt(N), with a
    single normalisation shared by all rows so dots are comparable across
    thresholds); each row keeps only the admissible cells with N >= threshold.
    """
    cells = cells[np.isfinite(cells["phi"])]
    n_ref = float(cells["n_days"].max())
    n_rows = len(thresholds)
    fig, axes = plt.subplots(n_rows, len(_CLASSES),
                             figsize=(4.8 * len(_CLASSES), 3.4 * n_rows),
                             squeeze=False, sharex=True, sharey=True)
    for row, thr in enumerate(thresholds):
        sub_t = cells[cells["n_days"] >= thr]
        for col, cls in enumerate(_CLASSES):
            ax = axes[row][col]
            sub = sub_t[sub_t["state_class"] == cls]
            s, a = _sqrtn_size_alpha(sub["n_days"].to_numpy(), n_ref)
            ax.scatter(sub["dist"], sub["phi"], s=s, alpha=a, color="black",
                       linewidths=0, zorder=3)
            _binned_median_line(ax, sub.rename(columns={"phi": "stat"}), "stat",
                                n_bins, "crimson", "binned median")
            ax.axhline(0.0, color="grey", lw=1, ls=":")
            if row == 0:
                ax.set_title(f"current states: {cls}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"$N \\geq {thr}$\n"
                              r"per-cell $\hat\varphi(d,d')$", fontsize=9)
            if row == n_rows - 1:
                ax.set_xlabel("inter-station distance (km)")
    for n_leg in _legend_ns(n_ref):
        s, a = _sqrtn_size_alpha(np.array([n_leg]), n_ref)
        axes[0][0].scatter([], [], s=s[0], alpha=a[0], color="black",
                           label=f"N = {n_leg:d}")
    axes[0][0].legend(fontsize=7, loc="upper right", title="cell days",
                      title_fontsize=7)
    plt.tight_layout()
    for f in filenames:
        fig.savefig(f, bbox_inches="tight")
    return fig


def plot_phi_per_cell_season_grid(
    cells_by_season: Dict[str, pd.DataFrame],
    seasons: Sequence[str] = _SEASONS,
    n_bins: int = 8,
    filenames: Sequence[pathlib.Path] = (),
    legend_ns: Optional[Sequence[int]] = None,
    dot_color: str = "black",
    trend_color: str = "crimson",
    dot_label: str = "observed",
):
    """Per-cell phi vs distance, one row per season and one column per state class.

    Figure 1 of the article. A single admissibility rule is applied, the
    ``N >= min_days_per_cell`` (10) of :func:`switch_phi_duration_cells`, so
    every panel shows all the admissible dots of its (season, class) and no
    further thinning is done.

    All rows share the same station set, the same y-axis and a single sqrt(N)
    size/opacity normalisation taken over the four seasons at once, so the dots
    and the trend curves are directly comparable from one season to the next.
    Each panel prints its number of dots. Over the scatter, the N-weighted bin
    mean and, around it, the 10/90% and 20/80% weighted quantiles of the dots in
    each distance bin (:func:`_draw_binned_curves`).
    ``legend_ns`` fixes the reference cell sizes shown in the legend; ``None``
    falls back to the data-driven :func:`_legend_ns`.
    ``dot_color`` / ``trend_color`` / ``dot_label`` let the same figure be drawn
    for a simulated replicate (blue) instead of the observations (black).
    No visible label of this figure uses the word "cell".
    """
    cells_by_season = {s: df[np.isfinite(df["phi"])] for s, df in cells_by_season.items()}
    n_ref = float(max(df["n_days"].max() for df in cells_by_season.values()))
    n_rows = len(seasons)
    fig, axes = plt.subplots(n_rows, len(_CLASSES),
                             figsize=(4.8 * len(_CLASSES), 3.4 * n_rows),
                             squeeze=False, sharex=True, sharey=True)
    for row, season in enumerate(seasons):
        cells = cells_by_season[season]
        for col, cls in enumerate(_CLASSES):
            ax = axes[row][col]
            sub = cells[cells["state_class"] == cls]
            s, a = _sqrtn_size_alpha(sub["n_days"].to_numpy(), n_ref)
            ax.scatter(sub["dist"], sub["phi"], s=s, alpha=a, color=dot_color,
                       linewidths=0, zorder=3)
            _draw_binned_curves(
                ax, sub, "phi", "n_days", bin_edges(sub, n_bins), trend_color,
                rf"{dot_label} $\bar\varphi_b$",
            )
            ax.axhline(0.0, color="grey", lw=1, ls=":")
            ax.text(0.02, 0.03, f"{len(sub)} dots", transform=ax.transAxes,
                    fontsize=8, color="dimgrey", zorder=5,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none",
                              pad=1.5))
            if row == 0:
                ax.set_title(f"current states: {cls}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{season}\n" r"$\hat\varphi_{jj'}(x,x')$",
                              fontsize=9)
            if row == n_rows - 1:
                ax.set_xlabel("inter-station distance (km)")
    for n_leg in (legend_ns if legend_ns is not None else _legend_ns(n_ref)):
        s, a = _sqrtn_size_alpha(np.array([n_leg]), n_ref)
        axes[0][0].scatter([], [], s=s[0], alpha=a[0], color=dot_color,
                           label=f"$N_{{jj'}}(x,x')$ = {n_leg:d} days")
    # Lower right: the upper right of the dry-dry panels is saturated by the
    # phi = 1 cells, which would hide the legend.
    axes[0][0].legend(fontsize=7, loc="lower right", framealpha=0.9)
    plt.tight_layout()
    for f in filenames:
        fig.savefig(f, bbox_inches="tight")
    return fig


def plot_gamma_state_pair_season_grid(
    gamma_by_season: Dict[str, pd.DataFrame],
    seasons: Sequence[str] = _SEASONS,
    state_classes: Sequence[str] = ("dry-dry", "wet-wet"),
    value_col: str = "gamma",
    min_n: int = 10,
    filenames: Sequence[pathlib.Path] = (),
    dot_color: str = "black",
    dot_size: float = 2.5,
    dot_alpha: float = 0.35,
    shared_y: bool = True,
    plot_quantiles: bool = False,
    n_bins: int = 8,
    trend_color: str = "crimson",
    y_log = False,
):
    """Spell-age semi-variogram vs distance, one row per season, one column per class.

    One dot per (station pair, spell-type pair) estimate of gamma_hat, kept when
    at least ``min_n`` days feed it. Unlike the phi figures, N is not encoded:
    every dot has the same ``dot_size`` and ``dot_alpha``, small and faint since
    the figure is read as a cloud, not dot by dot. ``shared_y`` puts all panels
    on a single log y-axis, so the levels compare across seasons and across
    classes; with ``shared_y=False`` each panel is scaled to its own dots, which
    shows the shape of each cloud but not the level differences. A perfectly
    synchronised pair sits at gamma = 0, off a log axis, and is reported on
    stdout rather than drawn.

    With ``plot_quantiles`` each panel also carries, over its cloud, the
    N-weighted median of the gamma_hat in each of ``n_bins`` equal-count distance
    bins and, around it, the 10/90% and 20/80% weighted quantiles of the dots of
    the bin (:func:`_draw_binned_curves`). Those curves describe the *spread of
    the estimates* inside a distance bin, not the uncertainty of the median.
    """
    seasons, state_classes = list(seasons), list(state_classes)
    n_rows = len(seasons)
    fig, axes = plt.subplots(n_rows, len(state_classes),
                             figsize=(4.8 * len(state_classes), 3.0 * n_rows),
                             squeeze=False, sharex=True, sharey=shared_y)
    dropped = []
    for row, season in enumerate(seasons):
        table = gamma_by_season[season]
        for col, cls in enumerate(state_classes):
            ax = axes[row][col]
            sub = table[(table["state_class"] == cls)
                        & np.isfinite(table[value_col])
                        & (table["n_days"] >= min_n)]
            dropped += [f"{r.station_j}-{r.station_k} ({cls}, {season}, "
                        f"N={r.n_days})"
                        for r in sub[sub[value_col] <= 0].itertuples()]
            ax.scatter(sub["dist"], sub[value_col], s=dot_size, alpha=dot_alpha,
                       color=dot_color, linewidths=0, zorder=3)
            if plot_quantiles and len(sub) > 1:
                _draw_binned_curves(
                    ax, sub, value_col, "n_days",
                    bin_edges(sub, n_bins, stat=value_col), trend_color,
                    r"binned median $\hat\gamma_b$",
                    center="median",
                )
            # ax.set_yscale("log")
            if row == 0:
                ax.set_title(f"current spell types: {cls}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{season}\n" r"$\hat\gamma_{jj'}(r,r')$",
                              fontsize=9)
            if row == n_rows - 1:
                ax.set_xlabel("inter-station distance (km)")
            if y_log:
                ax.set_yscale("log")
    if plot_quantiles:
        axes[0][0].legend(fontsize=7, loc="lower right", framealpha=0.9)
    if dropped:
        print(f"note: {len(dropped)} estimate(s) at {value_col} <= 0 left off the "
              "log y-axis: " + ", ".join(dropped))

    plt.tight_layout()
    for f in filenames:
        fig.savefig(f, bbox_inches="tight")
    return fig


def gamma_station_pairs(table: pd.DataFrame, order: Sequence[str],
                        min_n: int = 10) -> pd.DataFrame:
    """One row per drawn cell of the station-by-station map: two ranks + gamma_hat.

    ``order`` is the spatial seriation of the stations
    (:func:`spatial_bmcd.spatial_diagnostics.spatial_station_order`), so
    adjacent ranks are neighbouring stations. Each admissible estimate with
    ``gamma_hat > 0`` (a log colour scale cannot show 0) is written twice, once
    per orientation of the pair, except for the ``dry-wet`` class whose two
    orientations are distinct cells: there the dry station goes on x.
    """
    rank_of = {c: i + 1 for i, c in enumerate(order)}
    df = table[np.isfinite(table["gamma"]) & (table["gamma"] > 0)].copy()
    if min_n is not None:
        df = df[df["n_days"] >= min_n]
    df["rank_j"] = df["station_j"].map(rank_of)
    df["rank_k"] = df["station_k"].map(rank_of)
    if df[["rank_j", "rank_k"]].isna().any(axis=None):
        raise ValueError("station of the table absent from `order`")
    straight = df.rename(columns={"station_j": "station_x", "station_k": "station_y",
                                  "rank_j": "x", "rank_k": "y"})
    flipped = df.rename(columns={"station_k": "station_x", "station_j": "station_y",
                                 "rank_k": "x", "rank_j": "y"})
    dw = df["state_class"] == "dry-wet"   # only class whose two orientations differ
    return pd.concat([straight[~dw | (df["r_j"] == 0)],
                      flipped[~dw | (df["r_j"] == 1)]], ignore_index=True)


def plot_gamma_distance_and_station_matrix(
    gamma_table: pd.DataFrame,
    order: Sequence[str],
    state_class: str = "dry-dry",
    season: str = "summer",
    min_n: int = 10,
    filenames: Sequence[pathlib.Path] = (),
    dot_color: str = "black",
    dot_size: float = 15,
    dot_alpha: float = 0.75,
    plot_quantiles: bool = True,
    n_bins: int = 10,
    trend_color: str = "crimson",
    y_log: bool = True,
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (11.0, 4.4),
):
    """The two readings of one season's gamma_hat side by side, on one row.

    Left: the cloud of :func:`plot_gamma_state_pair_season_grid` for the single
    ``(season, state_class)`` panel -- one dot per station pair, gamma_hat
    against inter-station distance, with the N-weighted binned median and
    quantile curves over it. Right: the *same* estimates laid on the two
    station axes of :func:`gamma_station_pairs`, stations in spatial order, dot
    colour on a log scale. The distance view shows the trend, the matrix view
    shows which pairs carry it -- the ranges of the colour bar and of the left
    y-axis are the same numbers.
    """
    sub = gamma_table[(gamma_table["state_class"] == state_class)
                      & np.isfinite(gamma_table["gamma"])
                      & (gamma_table["n_days"] >= min_n)]
    fig, (ax_d, ax_m) = plt.subplots(1, 2, figsize=figsize,
                                     gridspec_kw=dict(width_ratios=[1.0, 1.0]))

    # --- left: gamma_hat vs inter-station distance -------------------------
    ax_d.scatter(sub["dist"], sub["gamma"], s=dot_size, alpha=dot_alpha,
                 color=dot_color, linewidths=0, zorder=3)
    if plot_quantiles and len(sub) > 1:
        _draw_binned_curves(
            ax_d, sub, "gamma", "n_days", bin_edges(sub, n_bins, stat="gamma"),
            trend_color, r"binned median $\hat\gamma_b$", center="median",
        )
        ax_d.legend(fontsize=7, loc="lower right", framealpha=0.9)
    if y_log:
        ax_d.set_yscale("log")
    ax_d.set_xlabel("inter-station distance (km)")
    ax_d.set_ylabel(r"$\hat\gamma_{jj'}(r,r')$", fontsize=9)
    ax_d.set_title(f"{season}, current spell types: {state_class}", fontsize=10)

    # --- right: the same estimates on the two station axes ------------------
    pairs = gamma_station_pairs(sub, order, min_n=min_n)
    n_st = len(order)
    grid = np.full((n_st, n_st), np.nan)
    grid[pairs["y"].to_numpy(int) - 1, pairs["x"].to_numpy(int) - 1] = \
        pairs["gamma"].to_numpy(float)
    dropped = int((~(gamma_table["gamma"] > 0)).sum())
    if dropped:   # a perfectly synchronised pair sits at 0, off any log scale
        print(f"note: {dropped} estimate(s) with gamma_hat <= 0 left off the log "
              "colour scale")
    g = pairs["gamma"].to_numpy(float)
    im = ax_m.imshow(
        np.ma.masked_invalid(grid), origin="lower", interpolation="nearest",
        extent=(0.5, n_st + 0.5, 0.5, n_st + 0.5), cmap=cmap,
        norm=mcolors.LogNorm(vmin=g.min(), vmax=g.max()),
    )
    ax_m.set_xlabel("station rank (spatial order)")
    ax_m.set_ylabel("station rank (spatial order)")
    ax_m.set_title(f"{season}, {state_class}: all station pairs", fontsize=10)
    cb = fig.colorbar(im, ax=ax_m, fraction=0.046, pad=0.03)
    cb.set_label(r"$\hat\gamma_{jj'}(r,r')$ (days$^2$)", fontsize=9)

    plt.tight_layout()
    for f in filenames:
        fig.savefig(f, bbox_inches="tight")
    return fig


def simulate_histories_like_obs(
    theta,
    params_by_station: Dict[str, dict],
    Rbin: pd.DataFrame,
    seed: Optional[int] = None,
    n_burn: int = 500,
    apply_obs_mask: bool = True,
) -> List[dict]:
    """One simulated replicate processed exactly like the observed record.

    A single stationary trajectory of the fitted LMC model (Section 5 of
    main.tex, :func:`spatial_bmcd.spatial_model.simulate_cholesky`) is drawn
    with the length of the observed occurrence frame ``Rbin`` and written into
    a copy of it, so it inherits its dates and station columns. With
    ``apply_obs_mask`` the days a station does not record are blanked in the
    simulation as well. The frame is then pushed through the *same*
    :func:`history_from_Rbin_drop_ambiguous_spell_after_nan` and
    :func:`split_history_by_year` as the observations, so the durations are
    reconstructed under the same ambiguous-spell and season-boundary rules and
    a simulated cell of :func:`switch_phi_duration_cells` rests on the same
    number of days as its observed counterpart: the two dot clouds carry the
    same estimation noise and pass the same admissibility filter.
    """
    from spatial_bmcd.spatial_model import (
        history_from_Rbin_drop_ambiguous_spell_after_nan,
        simulate_cholesky,
        split_history_by_year,
    )

    names = list(Rbin.columns)
    T = len(Rbin)
    sim = simulate_cholesky(
        theta, params_by_station, n_steps=T, n_burn=n_burn,
        station_names=names, seed=seed,
    )
    R_sim = np.asarray(sim["R"][:T], dtype=float)
    if apply_obs_mask:
        R_sim[~np.isfinite(Rbin.to_numpy(dtype=float))] = np.nan
    sim_Rbin = pd.DataFrame(R_sim, index=Rbin.index, columns=names)
    return split_history_by_year(
        history_from_Rbin_drop_ambiguous_spell_after_nan(sim_Rbin)
    )


def simulate_phi_cells(
    theta,
    params_by_station: Dict[str, dict],
    Rbin: pd.DataFrame,
    n_rep: int = 5,
    min_days_per_cell: int = 10,
    n_burn: int = 500,
    seed0: int = 100,
    apply_obs_mask: bool = True,
    verbose: bool = True,
) -> List[pd.DataFrame]:
    """``n_rep`` replicates of :func:`switch_phi_duration_cells` on simulated data.

    Each replicate is one call to :func:`simulate_histories_like_obs` pushed
    through the *same* estimator as the observations, so the simulated cloud
    carries the model's latent correlation of~\\eqref{eq:lmc_corr_blocks} seen
    through the thresholding relation~\\eqref{eq:phi_from_rho}, plus the
    finite-sample noise of the observed design.
    """
    out = []
    for rep in range(n_rep):
        sim_blocks = simulate_histories_like_obs(
            theta, params_by_station, Rbin, seed=seed0 + rep,
            n_burn=n_burn, apply_obs_mask=apply_obs_mask,
        )
        cells = switch_phi_duration_cells(
            sim_blocks, params_by_station, min_days_per_cell=min_days_per_cell,
        )
        cells["rep"] = rep
        out.append(cells)
        if verbose:
            adm = int(np.isfinite(cells["phi"]).sum())
            print(f"  replicate {rep + 1}/{n_rep}: {adm} admissible cells")
    return out


def _bottom_legend(fig, ax_source, ncol: int, inches: float = 0.55):
    """Legend of ``ax_source`` in a strip under the grid, not over the dots."""
    handles, labels = ax_source.get_legend_handles_labels()
    # Free the strip *in addition* to the current bottom margin, which already
    # holds the x labels of the last row.
    strip = inches / fig.get_figheight()
    fig.subplots_adjust(bottom=fig.subplotpars.bottom + strip)
    fig.legend(handles, labels, fontsize=8, ncol=ncol, loc="lower center",
               bbox_to_anchor=(0.5, 0.002), framealpha=0.9)


def plot_phi_obs_vs_sim_paired_rows(
    cells_by_season: Dict[str, pd.DataFrame],
    sim_cells_by_season: Dict[str, List[pd.DataFrame]],
    seasons: Sequence[str] = _SEASONS,
    n_bins: int = 8,
    rep_index: int = 0,
    filenames: Sequence[pathlib.Path] = (),
    legend_ns: Optional[Sequence[int]] = None,
):
    """Observed and simulated dot clouds side by side, two sub-rows per season.

    Each season occupies a block of two sub-rows sharing the same axes: on top
    the observed $\\hat\\varphi_{jj'}(x,x')$ (black dots, crimson curve), below
    the same statistic on the simulated replicate ``rep_index`` (steel blue
    dots, blue curve), one column per state-pair class. The curve is the
    $N$-weighted **median** of the dots in each of the ``n_bins`` equal-count
    distance bins of the panel — the level of the dependence, read without the
    two clouds hiding each other. Dot area and opacity encode $\\sqrt{N}$ with
    a single normalisation shared by every panel of the figure.
    """
    obs_by_season = {s: df[np.isfinite(df["phi"])] for s, df in cells_by_season.items()}
    sim_by_season = {
        s: v[rep_index][np.isfinite(v[rep_index]["phi"])]
        for s, v in sim_cells_by_season.items() if len(v) > rep_index
    }
    n_ref = float(max([df["n_days"].max() for df in obs_by_season.values()]
                      + [df["n_days"].max() for df in sim_by_season.values()]))
    n_seasons = len(seasons)
    fig = plt.figure(figsize=(4.8 * len(_CLASSES), 2.6 * 2 * n_seasons))
    outer = fig.add_gridspec(n_seasons, 1, hspace=0.30, left=0.07, right=0.99,
                             top=0.97, bottom=0.06)
    ax0 = None
    for i, season in enumerate(seasons):
        inner = outer[i].subgridspec(2, len(_CLASSES), hspace=0.08, wspace=0.06)
        sources = ((obs_by_season[season], "black", "crimson", "observed"),
                   (sim_by_season.get(season), "steelblue", "mediumblue", "simulated"))
        for sub, (cells, dot_color, curve_color, tag) in enumerate(sources):
            for col, cls in enumerate(_CLASSES):
                ax = fig.add_subplot(inner[sub, col], sharex=ax0, sharey=ax0)
                if ax0 is None:
                    ax0 = ax
                if cells is not None:
                    sub_c = cells[cells["state_class"] == cls]
                    s, a = _sqrtn_size_alpha(sub_c["n_days"].to_numpy(), n_ref)
                    ax.scatter(sub_c["dist"], sub_c["phi"], s=s, alpha=a,
                               color=dot_color, linewidths=0, zorder=3)
                    _draw_binned_curves(
                        ax, sub_c, "phi", "n_days", bin_edges(sub_c, n_bins),
                        curve_color, rf"{tag}, median $\hat\varphi$",
                        center="median", pairs=(),
                    )
                    ax.text(0.02, 0.03, f"{len(sub_c)} dots", transform=ax.transAxes,
                            fontsize=8, color="dimgrey", zorder=5,
                            bbox=dict(facecolor="white", alpha=0.75,
                                      edgecolor="none", pad=1.5))
                ax.axhline(0.0, color="grey", lw=1, ls=":")
                if i == 0 and sub == 0:
                    ax.set_title(f"current states: {cls}", fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"{season} — {tag}\n" r"$\hat\varphi_{jj'}(x,x')$",
                                  fontsize=9)
                else:
                    ax.tick_params(labelleft=False)
                if i == n_seasons - 1 and sub == 1:
                    ax.set_xlabel("inter-station distance (km)")
                else:
                    ax.tick_params(labelbottom=False)
    first = fig.axes[0]
    for n_leg in (legend_ns if legend_ns is not None else _legend_ns(n_ref)):
        s, a = _sqrtn_size_alpha(np.array([n_leg]), n_ref)
        first.scatter([], [], s=s[0], alpha=a[0], color="black",
                      label=f"$N_{{jj'}}(x,x')$ = {n_leg:d} days")
    # The simulated median lives in another axes: add its proxy by hand.
    first.plot([], [], color="mediumblue", lw=1.8, marker="o", ms=3.5,
               label=r"simulated, median $\hat\varphi$")
    _bottom_legend(fig, first, ncol=5)
    for f in filenames:
        fig.savefig(f, bbox_inches="tight")
    return fig


def plot_phi_quantile_fan_season_grid(
    cells_by_season: Dict[str, pd.DataFrame],
    sim_cells_by_season: Dict[str, List[pd.DataFrame]],
    seasons: Sequence[str] = _SEASONS,
    n_bins: int = 8,
    filenames: Sequence[pathlib.Path] = (),
    pairs: Sequence[tuple] = _DECILE_PAIRS,
    singles: Sequence[tuple] = (),
):
    """Observed and simulated distributions of phi, curves only — no dots.

    One row per season, one column per state-pair class, both estimations on
    the same axes: in crimson the observations, in steel blue the simulated
    replicates pooled. Per distance bin, the $N$-weighted median (solid, with
    markers) and the quantile curves of ``pairs`` — by default the top *and*
    bottom 10%, 20%, 30% and 40% levels, i.e. the weighted 10/90, 20/80, 30/70
    and 40/60 quantile pairs of the dots of that bin, the two curves of a pair
    sharing a line style and one legend entry. ``singles`` adds one-sided
    levels instead (e.g. :data:`_TOP_QUANTILES` with ``pairs=()``).
    """
    n_rows = len(seasons)
    obs_by_season = {s: df[np.isfinite(df["phi"])] for s, df in cells_by_season.items()}
    fig, axes = plt.subplots(n_rows, len(_CLASSES),
                             figsize=(4.8 * len(_CLASSES), 3.0 * n_rows),
                             squeeze=False, sharex=True, sharey=True)
    for row, season in enumerate(seasons):
        obs = obs_by_season[season]
        sims = sim_cells_by_season.get(season, [])
        for col, cls in enumerate(_CLASSES):
            ax = axes[row][col]
            obs_c = obs[obs["state_class"] == cls]
            edges = bin_edges(obs_c, n_bins)
            sims_c = [df[(df["state_class"] == cls) & np.isfinite(df["phi"])]
                      for df in sims]
            if sims_c:
                _draw_binned_curves(
                    ax, pd.concat(sims_c, ignore_index=True), "phi", "n_days",
                    edges, "steelblue", "simulated", zorder=7,
                    center="median", pairs=pairs, singles=singles,
                )
            _draw_binned_curves(
                ax, obs_c, "phi", "n_days", edges, "crimson", "observed",
                zorder=9, center="median", pairs=pairs, singles=singles,
            )
            ax.axhline(0.0, color="grey", lw=1, ls=":")
            if row == 0:
                ax.set_title(f"current states: {cls}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{season}\n" r"$\hat\varphi_{jj'}(x,x')$", fontsize=9)
            if row == n_rows - 1:
                ax.set_xlabel("inter-station distance (km)")
    plt.tight_layout()
    _bottom_legend(fig, axes[0][0], ncol=5)
    for f in filenames:
        fig.savefig(f, bbox_inches="tight")
    return fig


def obs_vs_sim_trend_summary(
    cells_by_season: Dict[str, pd.DataFrame],
    sim_cells_by_season: Dict[str, List[pd.DataFrame]],
    seasons: Sequence[str] = _SEASONS,
    n_bins: int = 8,
) -> pd.DataFrame:
    """Observed vs simulated median phi in the nearest and farthest bins.

    Per (season, class), on the ``n_bins`` equal-count distance bins of the
    observed dots, and matching what the figures draw: the $N$-weighted median
    of the observations and of the pooled replicates in the closest and the
    farthest bin, their bottom-10% and top-10% levels (weighted 10% and 90%
    quantiles) in the closest bin, and ``bins_inside_replicate_band``, the share
    of bins in which the observed median falls inside the 2.5-97.5%
    replicate-to-replicate range of the simulated median — the sampling
    uncertainty of that curve, which is not what the quantile curves show.
    """
    qs = (0.1, 0.5, 0.9)
    rows = []
    for season in seasons:
        obs = cells_by_season[season]
        obs = obs[np.isfinite(obs["phi"])]
        sims = sim_cells_by_season.get(season, [])
        for cls in _CLASSES:
            obs_c = obs[obs["state_class"] == cls]
            edges = bin_edges(obs_c, n_bins)
            obs_st = _binned_weighted_stats(obs_c, "phi", "n_days", edges, quantiles=qs)
            obs_curve = obs_st["q0.5"]
            sims_c = [df[(df["state_class"] == cls) & np.isfinite(df["phi"])]
                      for df in sims]
            row = {"season": season, "class": cls,
                   "obs_near": obs_curve[0], "obs_far": obs_curve[-1],
                   "obs_near_bot10": obs_st["q0.1"][0],
                   "obs_near_top10": obs_st["q0.9"][0]}
            if sims_c:
                sim_st = _binned_weighted_stats(pd.concat(sims_c, ignore_index=True),
                                                "phi", "n_days", edges, quantiles=qs)
                # Replicate-to-replicate spread of the bin median: the reference
                # against which a gap between the two median curves is judged.
                curves = np.vstack([
                    _binned_weighted_stats(df, "phi", "n_days", edges,
                                           quantiles=(0.5,))["q0.5"]
                    for df in sims_c
                ])
                with np.errstate(invalid="ignore"):
                    lo = np.nanpercentile(curves, 2.5, axis=0)
                    hi = np.nanpercentile(curves, 97.5, axis=0)
                inside = (obs_curve >= lo) & (obs_curve <= hi)
                row.update({
                    "sim_near": sim_st["q0.5"][0], "sim_far": sim_st["q0.5"][-1],
                    "sim_near_bot10": sim_st["q0.1"][0],
                    "sim_near_top10": sim_st["q0.9"][0],
                    "bins_inside_replicate_band": f"{int(np.nansum(inside))}/{len(obs_curve)}",
                })
            rows.append(row)
    return pd.DataFrame(rows)


def season_trend_summary(cells_by_season: Dict[str, pd.DataFrame],
                         seasons: Sequence[str] = _SEASONS,
                         n_bins: int = 8) -> pd.DataFrame:
    """Per (season, class): dot count, max N, and the trend curve's end values.

    Reports what the article text quotes about Figure 1: the number of
    admissible dots, the largest cell size, and the N-weighted bin mean
    ``phi_bar_b`` of :eq:`tmp_phi_binned_mean` in the nearest and the farthest
    of the ``n_bins`` equal-count distance bins.
    """
    rows = []
    for season in seasons:
        cells = cells_by_season[season]
        cells = cells[np.isfinite(cells["phi"])]
        for cls in _CLASSES:
            sub = cells[cells["state_class"] == cls]
            edges = np.unique(np.quantile(sub["dist"], np.linspace(0, 1, n_bins + 1)))
            means = []
            for a, b in zip(edges[:-1], edges[1:]):
                m = (sub["dist"] >= a) & (sub["dist"] <= b)
                w = sub.loc[m, "n_days"].to_numpy(dtype=float)
                v = sub.loc[m, "phi"].to_numpy(dtype=float)
                if w.sum() > 0:
                    means.append(float(w @ v / w.sum()))
            rows.append({
                "season": season, "class": cls, "n_dots": len(sub),
                "max_N": int(sub["n_days"].max()),
                "phi_bar_near": means[0], "phi_bar_far": means[-1],
                "phi_bar_min": min(means), "phi_bar_max": max(means),
                "monotone_decay": bool(np.all(np.diff(np.abs(means)) <= 1e-12)),
            })
    return pd.DataFrame(rows)


def print_threshold_retention(cells: pd.DataFrame, thresholds: Sequence[int]) -> None:
    """Share of admissible cells and days retained by each minimum cell size."""
    adm = cells[np.isfinite(cells["phi"])]
    print("\n=== per-cell retention vs minimum cell size (baseline N >= 10) ===")
    for thr in thresholds:
        kept = adm[adm["n_days"] >= thr]
        print(f"N >= {thr:>2}: {len(kept):>5} cells ({len(kept) / len(adm):5.1%}), "
              f"{kept['n_days'].sum():>6} days ({kept['n_days'].sum() / adm['n_days'].sum():5.1%})")


def _compare_exact_binned(pairs_exact: pd.DataFrame, pairs_binned: pd.DataFrame) -> pd.DataFrame:
    m = pairs_exact.merge(
        pairs_binned, on=["station_j", "station_k", "state_class"],
        suffixes=("_ex", "_bin"),
    )
    rows = []
    for cls, sub in m.groupby("state_class"):
        both = sub[np.isfinite(sub["phi_bar_ex"]) & np.isfinite(sub["phi_bar_bin"])]
        rows.append({
            "class": cls,
            "n_pairs": len(both),
            "corr(exact, binned)": both["phi_bar_ex"].corr(both["phi_bar_bin"]),
            "median |diff|": (both["phi_bar_ex"] - both["phi_bar_bin"]).abs().median(),
            "days used exact": both["n_used_ex"].sum() / both["n_class_days_ex"].sum(),
            "days used binned": both["n_used_bin"].sum() / both["n_class_days_bin"].sum(),
            "median phi exact": both["phi_bar_ex"].median(),
            "median phi binned": both["phi_bar_bin"].median(),
        })
    return pd.DataFrame(rows).set_index("class")


# ---------------------------------------------------------------------------
# The same statistic read along the equal-duration diagonal d = d'
# ---------------------------------------------------------------------------
#
# Driver: spatial_bmcd_data_analysis.ipynb. The article writes the statistic of
# its eq. (9) as psi_hat_{jj'}(x, x'), whereas the older functions above name it
# phi; it is the same quantity, and :func:`equal_duration_estimates` returns it
# in a ``psi`` column beside the cached ``phi`` one. What changes here is the
# slice of the state pairs being looked at -- only x = (r, d), x' = (r', d),
# i.e. the *same* duration at the two stations -- and the fact that this common
# duration, constant over a dot, is given to the colour of the dot, the channel
# Figure 1 of the article leaves free.


def load_Rbin_by_season(
    params_by_season: Dict[str, Dict[str, dict]],
    seasons: Sequence[str] = _SEASONS,
    year_min: int = 1980,
    year_max: int = 2020,
) -> Dict[str, pd.DataFrame]:
    """Per-season raw occurrence frames ``Rbin`` of the ECAD record.

    The daily 0/1 occurrences of the stations of ``params_by_season``,
    restricted to the season and to ``[year_min, year_max]``, carrying the
    dates and the missing-data pattern. Same pipeline as
    :func:`_load_season_data`, except that the single-site parameters are taken
    from ``params_by_season`` instead of being rebuilt. It is the slow step of
    the notebooks (one pass over the raw files), needed both by
    :func:`compute_cells_from_raw` and by any simulation reusing the observed
    design (:func:`simulate_phi_cells`).
    """
    from article_code.util_files import config
    from spatial_bmcd.spatial_model import (
        build_joint_df_occurrence_from_raw_data,
        load_all_station_rr,
    )

    seasons = list(seasons)
    stations_union = sorted(set().union(*(params_by_season[s].keys() for s in seasons)))
    dfs_by_city = load_all_station_rr(
        config.ECAD_RAW_DIR, stations_to_get=stations_union, verbose=False,
    )
    out = {}
    for season in seasons:
        stations = sorted(c for c in params_by_season[season] if c in dfs_by_city)
        Rbin = build_joint_df_occurrence_from_raw_data(
            dfs_by_city, season=season, station_names=stations,
            wet_day_threshold=float(config.WET_DAY_THRESHOLD),
        )
        out[season] = Rbin.loc[(Rbin.index.year >= year_min)
                               & (Rbin.index.year <= year_max)]
    return out


def compute_cells_from_raw(
    params_by_season: Dict[str, Dict[str, dict]],
    seasons: Sequence[str] = _SEASONS,
    year_min: int = 1980,
    year_max: int = 2020,
    min_days_per_cell: int = 10,
    Rbin_by_season: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """Per-season cell tables of eq. (9), rebuilt from the raw ECAD record.

    Slow path of the notebooks, for when the cached
    ``experiment_outputs/switch_phi_duration_cells_<season>.csv`` is missing:
    daily occurrences (:func:`load_Rbin_by_season`, or ``Rbin_by_season`` when
    already loaded) -> per-year histories (R, D, S) with the ambiguous-spell
    NaN masking of the likelihood -> :func:`switch_phi_duration_cells`.
    """
    from spatial_bmcd.spatial_model import (
        history_from_Rbin_drop_ambiguous_spell_after_nan,
        split_history_by_year,
    )

    seasons = list(seasons)
    if Rbin_by_season is None:
        Rbin_by_season = load_Rbin_by_season(params_by_season, seasons,
                                             year_min, year_max)
    out = {}
    for season in seasons:
        params = params_by_season[season]
        history = history_from_Rbin_drop_ambiguous_spell_after_nan(Rbin_by_season[season])
        out[season] = switch_phi_duration_cells(
            split_history_by_year(history),
            {c: params[c] for c in history["station_names"]},
            min_days_per_cell=min_days_per_cell,
        )
    return out


def equal_duration_estimates(cells: pd.DataFrame, d_max: int = 20) -> pd.DataFrame:
    """Diagonal d = d' <= ``d_max`` of a cell table, with psi of eq. (9) recomputed.

    The ``phi`` column of :func:`switch_phi_duration_cells` is NaN below the
    ``min_days_per_cell`` threshold it was computed with, whereas the ``psi``
    column returned here is recomputed from the stored counts and is therefore
    defined on *every* cell with non-degenerate margins, whatever
    N_{jj'}(x,x') -- so the admissibility threshold is applied downstream, at
    the figure, instead of being inherited from whatever the cache was built
    with. Cells with e_j in {0, N} or e_{j'} in {0, N} carry no information on
    the dependence (eq. (9) has a vanishing denominator there) and are dropped;
    this is what removes nearly every very small cell, N = 1 in particular,
    which can only be degenerate. The common duration is returned in a ``d``
    column.
    """
    sub = cells[(cells["d_j"] == cells["d_k"]) & cells["d_j"].between(1, d_max)].copy()
    n = sub["n_days"].to_numpy(float)
    e_j, e_k = sub["n_exit_j"].to_numpy(float), sub["n_exit_k"].to_numpy(float)
    e_jk = sub["n_joint"].to_numpy(float)
    var = e_j * (n - e_j) * e_k * (n - e_k)
    with np.errstate(invalid="ignore", divide="ignore"):
        psi = np.where(var > 0, (n * e_jk - e_j * e_k) / np.sqrt(var), np.nan)
    sub["d"] = sub["d_j"].astype(int)
    sub["psi"] = psi
    return sub[np.isfinite(sub["psi"])].reset_index(drop=True)


def equal_duration_coverage(
    diag_by_season: Dict[str, pd.DataFrame],
    min_days: int = 10,
    d_max: int = 20,
):
    """Two coverage tables of the equal-duration estimates.

    Returns ``(by_class, by_duration)``. ``by_class`` has one row per (season,
    state-pair class): the number of estimates with and without the
    ``min_days`` threshold, the median and the largest cell size, and the share
    of saturated estimates |psi| = 1. ``by_duration`` has one row per duration
    and one column per season, holding the string
    ``"<estimates> | <estimates with N >= min_days>"``.
    """
    rows = []
    for season, diag in diag_by_season.items():
        for cls in _CLASSES:
            sub = diag[diag["state_class"] == cls]
            kept = sub[sub["n_days"] >= min_days]
            rows.append({
                "season": season, "class": cls,
                "dots (any N)": len(sub), f"dots (N >= {min_days})": len(kept),
                "median N": sub["n_days"].median(), "max N": sub["n_days"].max(),
                "share |psi| = 1": float(np.mean(np.abs(sub["psi"]) > 0.999)),
            })
    by_class = pd.DataFrame(rows).set_index(["season", "class"])

    index = range(1, d_max + 1)
    counts = pd.DataFrame({
        season: diag.groupby("d").size() for season, diag in diag_by_season.items()
    }).reindex(index).fillna(0).astype(int)
    counts_kept = pd.DataFrame({
        season: diag[diag["n_days"] >= min_days].groupby("d").size()
        for season, diag in diag_by_season.items()
    }).reindex(index).fillna(0).astype(int)
    return by_class, counts.astype(str) + " | " + counts_kept.astype(str)


def plot_psi_equal_duration_grid(
    diag_by_season: Dict[str, pd.DataFrame],
    value_col: str = "psi",
    n_col: str = "n_days",
    min_n: Optional[int] = None,
    d_max: int = 20,
    seasons: Optional[Sequence[str]] = None,
    n_bins: int = 8,
    n_ref: Optional[float] = None,
    legend_ns: Optional[Sequence[int]] = (10, 80, 150),
    alpha_range: Sequence[float] = (0.15, 0.85),
    cmap: str = "viridis",
    color_scale: str = "log",
    ylabel: str = r"$\widehat{\psi}_{jj'}(x,x')$",
    trend_label: str = r"$N$-weighted bin mean",
    cbar_label: str = r"common duration $d = d'$ (days)",
    filenames: Sequence[pathlib.Path] = (),
):
    """Figure-1 grid on the equal-duration estimates, dots coloured by d = d'.

    One row per season, one column per state-pair class, ``value_col`` against
    the inter-station distance in km. Dot area *and* opacity both
    follow ``sqrt(N_{jj'}(x,x'))`` under the mapping of the article
    (:func:`_sqrtn_size_alpha`), normalised by ``n_ref``, which defaults to the
    largest cell being drawn -- as in Figure 1, so a dot's weight is read the
    same way in both. Note that opacity here competes with the duration carried
    by the colour: a faint dot has its hue washed towards the background, and
    ``alpha_range`` (``(a_min, a_max)`` at N = 0 and N = ``n_ref``) is there to
    trade the two off. ``min_n`` filters on ``n_col``: ``min_n=10`` is the
    admissibility rule of the article, ``None`` keeps every cell with
    non-degenerate margins. Over the dots, the ``N``-weighted mean per
    equal-count distance bin (:func:`_draw_binned_curves`) and the independence
    baseline.

    ``color_scale="log"`` spaces the colours logarithmically in ``d``, which is
    where the estimates are: about four fifths of them carry d <= 3, and a
    linear scale packs those into the dark end of the colormap. The colorbar
    keeps integer ticks, unevenly spaced. ``color_scale="linear"`` gives instead
    one flat colour band per duration. Returns the figure.
    """
    seasons = list(diag_by_season) if seasons is None else list(seasons)
    data = {}
    for season in seasons:
        df = diag_by_season[season]
        df = df[np.isfinite(df[value_col]) & df["d"].between(1, d_max)]
        if min_n is not None:
            df = df[df[n_col] >= min_n]
        data[season] = df.sort_values(n_col, ascending=False)  # small dots drawn last
    if n_ref is None:
        n_ref = float(max(df[n_col].max() for df in data.values()))

    if color_scale == "log":
        cmap_d, norm_d = plt.get_cmap(cmap), mcolors.LogNorm(vmin=0.85, vmax=d_max + 0.5)
        cbar_ticks = [t for t in (1, 2, 3, 4, 5, 6, 8, 10, 14, 20) if t <= d_max]
    else:
        cmap_d = plt.get_cmap(cmap, d_max)
        norm_d = mcolors.BoundaryNorm(np.arange(0.5, d_max + 1.5), d_max)
        cbar_ticks = np.arange(1, d_max + 1)

    fig, axes = plt.subplots(len(seasons), len(_CLASSES),
                             figsize=(4.8 * len(_CLASSES), 3.4 * len(seasons)),
                             squeeze=False, sharex=True, sharey=True)
    for row, season in enumerate(seasons):
        for col, cls in enumerate(_CLASSES):
            ax = axes[row][col]
            sub = data[season][data[season]["state_class"] == cls]
            if len(sub):
                s, a = _sqrtn_size_alpha(sub[n_col].to_numpy(), n_ref,
                                         a_min=alpha_range[0], a_max=alpha_range[1])
                ax.scatter(sub["dist"], sub[value_col], s=s, c=sub["d"],
                           cmap=cmap_d, norm=norm_d, alpha=a, linewidths=0,
                           zorder=3, rasterized=True)
                _draw_binned_curves(
                    ax, sub, value_col, n_col, bin_edges(sub, n_bins, stat=value_col),
                    "crimson", trend_label, center="mean", pairs=(),
                )
            ax.axhline(0.0, color="grey", lw=1, ls=":")
            ax.text(0.02, 0.03, f"{len(sub)} dots", transform=ax.transAxes,
                    fontsize=8, color="dimgrey", zorder=5,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5))
            if row == 0:
                ax.set_title(f"current states: {cls}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{season}\n" + ylabel, fontsize=9)
            if row == len(seasons) - 1:
                ax.set_xlabel("inter-station distance (km)")

    # Size/opacity legend in grey: colour is taken by the duration.
    for n_leg in (legend_ns if legend_ns is not None else _legend_ns(n_ref)):
        s, a = _sqrtn_size_alpha(np.array([n_leg]), n_ref,
                                 a_min=alpha_range[0], a_max=alpha_range[1])
        axes[0][0].scatter([], [], s=s[0], color="dimgrey", alpha=a[0],
                           label=f"$N_{{jj'}}(x,x')$ = {n_leg:d} days")
    axes[0][0].legend(fontsize=7, loc="lower right", framealpha=0.9)

    fig.tight_layout(rect=(0.0, 0.0, 0.92, 1.0))
    cax = fig.add_axes([0.935, 0.12, 0.012, 0.76])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_d, norm=norm_d), cax=cax,
                        ticks=cbar_ticks)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.set_yticklabels([str(t) for t in cbar_ticks])  # no minor log labels
    cbar.ax.tick_params(labelsize=7, which="both")
    cbar.ax.minorticks_off()
    for f in filenames:
        fig.savefig(f, bbox_inches="tight")
    return fig


def zero_joint_exit_locus(diag: pd.DataFrame) -> np.ndarray:
    """Value of eq. (9) on a cell where the two stations never exit together.

    Setting e_{jj'} = 0 in eq. (9) leaves
    ``-sqrt(e_j e_{j'} / ((N - e_j)(N - e_{j'})))``: a function of the three
    marginal counts alone, carrying no information on the dependence and none
    on the distance. It is a small negative number whenever the exits are rare.
    """
    n = diag["n_days"].to_numpy(float)
    e_j, e_k = diag["n_exit_j"].to_numpy(float), diag["n_exit_k"].to_numpy(float)
    return -np.sqrt(e_j * e_k / ((n - e_j) * (n - e_k)))


def attainable_psi_count(diag: pd.DataFrame) -> np.ndarray:
    """How many distinct values eq. (9) can take on each cell, margins fixed.

    Only e_{jj'} is free, an integer in ``[max(0, e_j + e_{j'} - N),
    min(e_j, e_{j'})]``, so a cell holding few exits admits only a handful of
    psi's -- often just two: the locus of :func:`zero_joint_exit_locus`, and 1.
    """
    n = diag["n_days"].to_numpy()
    e_j, e_k = diag["n_exit_j"].to_numpy(), diag["n_exit_k"].to_numpy()
    return np.minimum(e_j, e_k) - np.maximum(0, e_j + e_k - n) + 1


def plot_zero_joint_exit_diagnostic(
    diag_by_season: Dict[str, pd.DataFrame],
    state_class: str = "dry-dry",
    min_n: int = 10,
    d_max: int = 20,
    seasons: Sequence[str] = _SEASONS,
    few: int = 2,
    many: int = 5,
    filenames: Sequence[pathlib.Path] = (),
):
    """Three panels reading the flat band at psi ~ 0 of the equal-duration figure.

    Seasons pooled, one state-pair class (the band is a dry-dry feature: dry
    spells are long, so a cell holds few exits). Left: the band is exactly the
    cells with e_{jj'} = 0, and it spans every distance. Middle: on those cells
    eq. (9) sits on :func:`zero_joint_exit_locus`, a function of the margins
    only -- the points fall on the identity line. Right: why they are so
    numerous -- with ``min(e_j, e_{j'}) <= few`` exits eq. (9) can barely reach
    anything except that band and 1 (:func:`attainable_psi_count`), against a
    smooth spread once ``min(e_j, e_{j'}) >= many``.

    Prints the counts behind the three panels and returns the figure.
    """
    diag = pd.concat([diag_by_season[s] for s in seasons], ignore_index=True)
    diag = diag[(diag["n_days"] >= min_n) & (diag["state_class"] == state_class)
                & diag["d"].between(1, d_max)].copy()
    diag["locus"] = zero_joint_exit_locus(diag)
    diag["n_attainable"] = attainable_psi_count(diag)
    e_min = diag[["n_exit_j", "n_exit_k"]].min(axis=1)
    no_joint = diag["n_joint"] == 0

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))

    ax = axes[0]
    ax.scatter(diag.loc[~no_joint, "dist"], diag.loc[~no_joint, "psi"], s=7,
               color="0.6", alpha=0.45, linewidths=0, rasterized=True,
               label=r"$e_{jj'} \geq 1$")
    ax.scatter(diag.loc[no_joint, "dist"], diag.loc[no_joint, "psi"], s=9,
               color="crimson", alpha=0.65, linewidths=0, rasterized=True,
               label=r"$e_{jj'} = 0$  (no joint exit)")
    ax.axhline(0.0, color="grey", lw=1, ls=":")
    ax.set_xlabel("inter-station distance (km)")
    ax.set_ylabel(r"$\widehat{\psi}_{jj'}(x,x')$")
    ax.set_title("the band is the cells with no joint exit", fontsize=10)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)

    ax = axes[1]
    ax.scatter(diag.loc[no_joint, "locus"], diag.loc[no_joint, "psi"], s=9,
               color="crimson", alpha=0.55, linewidths=0, rasterized=True)
    lims = [diag.loc[no_joint, "locus"].min() - 0.03, 0.02]
    ax.plot(lims, lims, color="black", lw=1, ls="--", label="identity")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(r"$-\sqrt{e_j e_{j'} / ((N - e_j)(N - e_{j'}))}$")
    ax.set_ylabel(r"$\widehat{\psi}_{jj'}(x,x')$")
    ax.set_title(r"there, eq. (9) is fixed by the margins", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    ax = axes[2]
    edges = np.linspace(-1, 1, 61)
    for mask, color, label in (
        (e_min <= few, "crimson", rf"$\min(e_j, e_{{j'}}) \leq {few}$"),
        (e_min >= many, "steelblue", rf"$\min(e_j, e_{{j'}}) \geq {many}$"),
    ):
        ax.hist(diag.loc[mask, "psi"], bins=edges, density=True, histtype="step",
                lw=1.6, color=color,
                label=f"{label}  ({int(mask.sum())} cells)")
    ax.axhline(0.0, color="grey", lw=1, ls=":")
    ax.set_xlabel(r"$\widehat{\psi}_{jj'}(x,x')$")
    ax.set_ylabel("density")
    ax.set_title("few exits: only the band and 1 are reachable", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    for f in filenames:
        fig.savefig(f, bbox_inches="tight")

    residual = np.abs(diag.loc[no_joint, "psi"] - diag.loc[no_joint, "locus"]).max()
    q1, q3 = diag.loc[no_joint, "psi"].quantile([0.25, 0.75])
    print(f"{state_class}, seasons pooled, N >= {min_n}, d = d' <= {d_max}: "
          f"{len(diag)} cells")
    print(f"  e_jj' = 0            : {int(no_joint.sum())} cells "
          f"({no_joint.mean():.1%}), psi in [{q1:.3f}, {q3:.3f}] (quartiles), "
          f"all of them < 0")
    print(f"  max |psi - locus|    : {residual:.1e}  (eq. (9) with e_jj' = 0, exact)")
    print(f"  <= 3 reachable psi's : "
          f"{(diag['n_attainable'] <= 3).mean():.1%} of the cells; "
          f"median reachable = {int(np.median(diag['n_attainable']))}")
    print(f"  psi = 1 (the mirror) : {(diag['psi'] > 0.999).mean():.1%} of the cells")
    return fig


#: Hover columns of the interactive equal-duration figure, in ``customdata`` order.
_EQUAL_DURATION_HOVER_COLS = [
    "station_j", "station_k", "d", "n_days", "n_exit_j", "n_exit_k", "n_joint",
]


def _equal_duration_frames(
    diag_by_season: Dict[str, pd.DataFrame],
    state_class: str,
    min_n: int,
    d_max: int,
    seasons: Sequence[str],
    psi_range: Optional[Tuple[float, float]] = None,
    n_ref: Optional[float] = None,
):
    """Per-season cells of one state class on the equal-duration diagonal.

    ``psi_range`` (inclusive) further restricts the cells to a band of eq. (9);
    ``n_ref`` is always taken from the *unfiltered* selection unless given, so
    that the ``sqrt(N)`` size/opacity law does not rescale when the band moves.
    """
    frames, n_max = {}, 1.0
    for season in seasons:
        df = diag_by_season[season]
        df = df[(df["state_class"] == state_class) & (df["n_days"] >= min_n)
                & df["d"].between(1, d_max) & np.isfinite(df["psi"])]
        if len(df):
            n_max = max(n_max, float(df["n_days"].max()))
        if psi_range is not None:
            df = df[df["psi"].between(*psi_range)]
        frames[season] = df.sort_values("n_days", ascending=False)
    return frames, (n_max if n_ref is None else float(n_ref))


def _equal_duration_marker(df: pd.DataFrame, n_ref: float):
    """Dot diameter (px) and opacity of one season's cells.

    Same ``sqrt(N)`` law as the static figure; matplotlib's ``s`` is an area in
    pt^2, plotly's ``size`` a diameter in px, hence the square root.
    """
    s_mpl, alpha = _sqrtn_size_alpha(df["n_days"].to_numpy(), n_ref)
    return np.sqrt(s_mpl) * 2.2, alpha


def plot_psi_equal_duration_interactive(
    diag_by_season: Dict[str, pd.DataFrame],
    state_class: str = "dry-dry",
    min_n: int = 10,
    d_max: int = 20,
    seasons: Sequence[str] = _SEASONS,
    psi_range: Optional[Tuple[float, float]] = None,
    n_ref: Optional[float] = None,
    height: int = 860,
    cmap: str = "Viridis",
):
    """Hover-inspectable equal-duration figure, one state-pair class at a time.

    Same axes, colour law and ``sqrt(N)`` size/opacity law as
    :func:`plot_psi_equal_duration_grid`, restricted to ``state_class`` and laid
    out as a 2x2 grid of seasons so it fits a screen. Hovering a dot gives the
    two cities, the distance, the common duration and the four counts behind
    eq. (9) -- ``N_{jj'}(x,x')``, ``e_j``, ``e_{j'}``, ``e_{jj'}`` -- which is
    what tells an estimate apart from an artifact of the margins.

    A second, legend-only trace outlines the cells with ``e_{jj'} = 0``: click
    it in the legend to see the flat band of
    :func:`plot_zero_joint_exit_diagnostic` light up in place.

    ``psi_range`` keeps only the cells whose estimate falls in that (inclusive)
    band, e.g. ``psi_range=(-0.05, 0.05)`` to read the flat band alone;
    :func:`explore_psi_equal_duration` drives it from a slider instead.

    Needs ``plotly`` (imported here, so the rest of the module does not depend
    on it). Returns the figure; call ``.show()`` on it in a notebook.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    seasons = list(seasons)
    frames, n_ref = _equal_duration_frames(diag_by_season, state_class, min_n,
                                           d_max, seasons, psi_range, n_ref)

    ticks = [t for t in (1, 2, 3, 4, 5, 6, 8, 10, 14, 20) if t <= d_max]
    hover = (
        "<b>%{customdata[0]} &mdash; %{customdata[1]}</b><br>"
        "distance = %{x:.1f} km &nbsp; duration d = d' = %{customdata[2]}<br>"
        "<b>psi = %{y:.3f}</b><br>"
        "N = %{customdata[3]} days &nbsp; e_j = %{customdata[4]} &nbsp; "
        "e_j' = %{customdata[5]} &nbsp; e_jj' = %{customdata[6]}"
        "<extra></extra>"
    )
    cols = _EQUAL_DURATION_HOVER_COLS

    n_cols = 2
    n_rows = int(np.ceil(len(seasons) / n_cols))
    fig = make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=seasons,
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.07, vertical_spacing=0.09,
    )
    for i, season in enumerate(seasons):
        row, col = i // n_cols + 1, i % n_cols + 1
        df = frames[season]
        size, alpha = _equal_duration_marker(df, n_ref)
        fig.add_trace(
            go.Scattergl(
                x=df["dist"], y=df["psi"], mode="markers",
                customdata=df[cols].to_numpy(dtype=object),
                hovertemplate=hover, name=season, showlegend=False,
                marker=dict(
                    size=size, opacity=alpha, color=np.log10(df["d"].to_numpy(float)),
                    colorscale=cmap, cmin=np.log10(0.85), cmax=np.log10(d_max + 0.5),
                    line=dict(width=0),
                    colorbar=dict(
                        title=dict(text="d = d'", side="right"),
                        tickvals=np.log10(ticks), ticktext=[str(t) for t in ticks],
                        len=0.9, thickness=14,
                    ) if i == 0 else None,
                    showscale=(i == 0),
                ),
            ),
            row=row, col=col,
        )
        no_joint = df[df["n_joint"] == 0]
        size0, _ = _equal_duration_marker(no_joint, n_ref)
        fig.add_trace(
            go.Scattergl(
                x=no_joint["dist"], y=no_joint["psi"], mode="markers",
                customdata=no_joint[cols].to_numpy(dtype=object),
                hovertemplate=hover, name="e_jj' = 0 (no joint exit)",
                legendgroup="nojoint", showlegend=(i == 0), visible="legendonly",
                marker=dict(size=size0 + 3, color="rgba(0,0,0,0)",
                            line=dict(width=1.3, color="crimson")),
            ),
            row=row, col=col,
        )
        fig.add_hline(y=0.0, line=dict(color="grey", width=1, dash="dot"),
                      row=row, col=col)

    fig.update_xaxes(title_text="inter-station distance (km)",
                     row=n_rows)
    fig.update_yaxes(title_text="psi_hat", col=1)
    fig.update_layout(
        height=height, hovermode="closest",
        title=dict(text=_equal_duration_title(state_class, min_n, d_max, psi_range,
                                              frames), x=0.5),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
        margin=dict(l=70, r=90, t=110, b=60),
    )
    return fig


def _equal_duration_title(state_class, min_n, d_max, psi_range, frames):
    """Figure title, with the psi band and the cell count it keeps when filtering."""
    text = (f"equal-duration estimates, {state_class}, "
            f"N &ge; {min_n}, d = d' &le; {d_max}")
    if psi_range is None:
        return text
    lo, hi = psi_range
    n_sel = int(sum(len(df) for df in frames.values()))
    return (f"{text}<br><span style='font-size:0.8em'>"
            f"psi &isin; [{lo:.3f}, {hi:.3f}] &mdash; {n_sel} cells</span>")


def explore_psi_equal_duration(
    diag_by_season: Dict[str, pd.DataFrame],
    state_class: str = "dry-dry",
    min_n: int = 10,
    d_max: int = 20,
    seasons: Sequence[str] = _SEASONS,
    psi_range: Optional[Tuple[float, float]] = None,
    step: float = 0.005,
    zoom_y: bool = False,
    **kwargs,
):
    """:func:`plot_psi_equal_duration_interactive` with a psi range slider on top.

    Drag the two handles to keep only the cells whose eq. (9) estimate falls in
    the band -- the flat band at psi ~ 0, the pile-up at psi = 1, or anything in
    between -- and hover what is left to read the counts behind it. The dot
    sizes, the colour scale and the axes stay fixed as the band moves, so two
    settings are comparable; tick "zoom y" to have the vertical axis follow the
    band instead. ``psi_range`` sets where the handles start (default: the full
    range of the selection).

    Needs ``ipywidgets`` on top of ``plotly``; with ``anywidget`` also installed
    the dots are updated in place (the pan/zoom of the panels survives a move of
    the slider), otherwise the figure is redrawn at each move. Returns a widget:
    it is the display value of the notebook cell, so end the cell with it (no
    ``.show()``).
    """
    import ipywidgets as widgets
    import plotly.graph_objects as go

    seasons = list(seasons)
    full, n_ref = _equal_duration_frames(diag_by_season, state_class, min_n,
                                         d_max, seasons)
    psi_all = np.concatenate([df["psi"].to_numpy(float) for df in full.values()
                              if len(df)] or [np.zeros(1)])
    lo_all = float(np.floor(psi_all.min() / step) * step)
    hi_all = float(np.ceil(psi_all.max() / step) * step)
    value = (lo_all, hi_all) if psi_range is None else (
        max(lo_all, float(psi_range[0])), min(hi_all, float(psi_range[1])))

    def _build(band):
        return plot_psi_equal_duration_interactive(
            diag_by_season, state_class=state_class, min_n=min_n, d_max=d_max,
            seasons=seasons, psi_range=band, n_ref=n_ref, **kwargs)

    def _y_range(band, on):
        pad = 0.02 * max(band[1] - band[0], step)
        return [band[0] - pad, band[1] + pad] if on else None

    slider = widgets.FloatRangeSlider(
        value=value, min=lo_all, max=hi_all, step=step,
        description="psi range", readout_format=".3f", continuous_update=False,
        layout=widgets.Layout(width="70%"), style={"description_width": "initial"},
    )
    zoom = widgets.Checkbox(value=zoom_y, description="zoom y on the range",
                            indent=False, layout=widgets.Layout(width="200px"))
    reset = widgets.Button(description="reset", layout=widgets.Layout(width="80px"))

    try:
        fig = go.FigureWidget(_build(value))
    except ImportError:  # plotly >= 6 wants anywidget for FigureWidget
        fig = None

    if fig is not None:
        def _refresh(*_):
            band = tuple(slider.value)
            sel, _ = _equal_duration_frames(diag_by_season, state_class, min_n,
                                            d_max, seasons, band, n_ref)
            with fig.batch_update():
                for i, season in enumerate(seasons):
                    df = sel[season]
                    size, alpha = _equal_duration_marker(df, n_ref)
                    trace = fig.data[2 * i]  # cloud, then its e_jj' = 0 outline
                    trace.x = df["dist"].to_numpy(float)
                    trace.y = df["psi"].to_numpy(float)
                    trace.customdata = (
                        df[_EQUAL_DURATION_HOVER_COLS].to_numpy(dtype=object))
                    trace.marker.size = size
                    trace.marker.opacity = alpha
                    trace.marker.color = np.log10(df["d"].to_numpy(float))

                    no_joint = df[df["n_joint"] == 0]
                    size0, _a = _equal_duration_marker(no_joint, n_ref)
                    outline = fig.data[2 * i + 1]
                    outline.x = no_joint["dist"].to_numpy(float)
                    outline.y = no_joint["psi"].to_numpy(float)
                    outline.customdata = (
                        no_joint[_EQUAL_DURATION_HOVER_COLS].to_numpy(dtype=object))
                    outline.marker.size = size0 + 3
                fig.layout.title.text = _equal_duration_title(
                    state_class, min_n, d_max, band, sel)
                fig.update_yaxes(range=_y_range(band, zoom.value),
                                 autorange=not zoom.value)
        view = fig
    else:
        view = widgets.Output()

        def _refresh(*_):
            band = tuple(slider.value)
            drawn = _build(band)
            drawn.update_yaxes(range=_y_range(band, zoom.value),
                               autorange=not zoom.value)
            view.clear_output(wait=True)
            with view:
                drawn.show()

    slider.observe(_refresh, names="value")
    zoom.observe(_refresh, names="value")
    reset.on_click(lambda _b: setattr(slider, "value", (lo_all, hi_all)))
    _refresh()
    return widgets.VBox([widgets.HBox([slider, zoom, reset]), view])


# ---------------------------------------------------------------------------
# The same cells read through the fitted model: eq. (12) against eq. (9)
# ---------------------------------------------------------------------------
#
# Everything above estimates psi from the counts alone (eq. (9)); the fitted
# model gives a *value* for the same cell through eq. (12),
#
#     psi_{jj'}(x,x') = [Phi_2(z_j, z_{j'}; rho^{(r,r')}_{jj'}) - q_j q_{j'}]
#                       / sqrt(q_j (1 - q_j) q_{j'} (1 - q_{j'})),
#
# with q_j = q^{(r)}_{s_j}(d) the single-site exit probability, z_j its Gaussian
# threshold of eq. (10), and rho^{(r,r')}_{jj'} the LMC block of eq. (15) at
# theta_hat. Both are then one number per cell, so they can be put on the two
# axes of one scatter -- :func:`plot_psi_model_vs_obs_grid`.


def psi_model_equal_duration(
    diag: pd.DataFrame,
    params_by_station: Dict[str, dict],
    theta,
    clip_q: float = 1e-12,
) -> pd.DataFrame:
    """Add the model value of eq. (12) to an equal-duration table of eq. (9).

    ``diag`` is the output of :func:`equal_duration_estimates` for one season
    (columns ``station_j``, ``station_k``, ``r_j``, ``r_k``, ``d``, ``psi``),
    or any table with those cell keys -- including one whose cells are *not* on
    the diagonal, where the two spell ages are read from the ``d_j``/``d_k``
    columns if the table carries them and from ``d`` otherwise,
    ``params_by_station`` the single-site parameter dict of that season (the
    fitted exit-probability callables and the station coordinates), and
    ``theta`` the fitted LMC parameter, in any form accepted by
    ``normalize_theta`` -- ``(lam0, lam1, sigma_wc, sigma_w0, sigma_w1)`` for
    the state-dependent model of the article.

    A copy of ``diag`` is returned with four added columns: ``q_j`` and ``q_k``
    (the two exit probabilities q^{(r)}_{s_j}(d), q^{(r')}_{s_{j'}}(d), clipped
    away from 0 and 1 as in the likelihood), ``rho_model`` (the LMC block of
    eq. (15) selected by the state pair, at ``theta``) and ``psi_model``
    (eq. (12)). ``Phi_2`` is the fast approximation of Appendix A, the one the
    pairwise likelihood itself uses.

    Note that eq. (12) is deterministic given the cell, whereas ``psi`` is a
    plug-in estimate on N_{jj'}(x,x') days: the two columns are *not* two
    estimates of the same accuracy, and the spread of the scatter is dominated
    by the sampling noise of the observed side on the small cells.
    """
    from spatial_bmcd.spatial_model import build_lmc_blocks, phi2_vec  # noqa: E402
    from scipy.stats import norm  # noqa: E402

    names = sorted(set(diag["station_j"]).union(diag["station_k"]))
    missing = [c for c in names if c not in params_by_station]
    if missing:
        raise KeyError(
            f"{len(missing)} station(s) of the cell table absent from "
            f"params_by_station: {missing[:5]}"
        )
    pos = {c: i for i, c in enumerate(names)}
    blocks = build_lmc_blocks(names, params_by_station, theta)

    j = diag["station_j"].map(pos).to_numpy()
    k = diag["station_k"].map(pos).to_numpy()
    r_j, r_k = diag["r_j"].to_numpy(int), diag["r_k"].to_numpy(int)
    # One spell age per station. They are the single ``d`` of the cell on an
    # equal-duration table; a table that also carries the off-diagonal cells
    # (d != d') gives each side its own through ``d_j``/``d_k``. Nothing else
    # of the model side changes off the diagonal: rho is selected by the state
    # pair alone, so only the two q's have to be read at their own duration.
    d_of_j = (diag["d_j"] if "d_j" in diag.columns else diag["d"]).to_numpy(int)
    d_of_k = (diag["d_k"] if "d_k" in diag.columns else diag["d"]).to_numpy(int)

    # Eq. (13): the block is selected by the two spell types, as in
    # build_state_selected_cov -- C00 dry-dry, C11 wet-wet, C01 discordant.
    rho = np.where(
        (r_j == 0) & (r_k == 0), blocks["C00"][j, k],
        np.where((r_j == 1) & (r_k == 1), blocks["C11"][j, k], blocks["C01"][j, k]),
    )

    q_j = _exit_prob_of_cells(diag["station_j"].to_numpy(), r_j, d_of_j,
                              params_by_station, clip_q)
    q_k = _exit_prob_of_cells(diag["station_k"].to_numpy(), r_k, d_of_k,
                              params_by_station, clip_q)
    z_j, z_k = norm.ppf(q_j), norm.ppf(q_k)

    num = phi2_vec(z_j, z_k, rho) - q_j * q_k
    den = np.sqrt(q_j * (1.0 - q_j) * q_k * (1.0 - q_k))

    out = diag.copy()
    out["q_j"], out["q_k"] = q_j, q_k
    out["rho_model"] = rho
    with np.errstate(invalid="ignore", divide="ignore"):
        out["psi_model"] = np.where(den > 0, num / den, np.nan)
    return out


def p00_model_equal_duration(
    diag: pd.DataFrame,
    params_by_station: Dict[str, dict],
    theta,
    obs_col: str = "p00",
    clip_q: float = 1e-12,
) -> pd.DataFrame:
    """Add the model co-persistence probability p^stay to an equal-duration table.

    Companion of :func:`psi_model_equal_duration` for the second metric read on
    the same cells, the probability that *neither* station leaves its spell,

        p^stay_{jj'}(x,x') = 1 - q^{(r)}_{s_j}(d) - q^{(r')}_{s_{j'}}(d')
                             + Phi_2(z_j, z_{j'}; rho^{(r,r')}_{jj'}),

    with d = d' on an equal-duration table and the two ages taken from the
    ``d_j``/``d_k`` columns on a table that also holds the off-diagonal cells
    (see :func:`psi_model_equal_duration`).

    This is not an analogy with eq. (12): it is verbatim the (0, 0) term of the
    pairwise composite likelihood the model was fitted on
    (``1 - qj - qk + q_joint`` in
    :func:`spatial_bmcd.spatial_model.pairwise_loglik_one_step_nanaware_vec`),
    so the figure it feeds reads the fit on its own scale.

    ``diag`` needs the cell keys ``station_j``, ``station_k``, ``r_j``, ``r_k``,
    ``d`` -- the equal-duration table of :func:`equal_duration_estimates` or the
    one the notebook builds from the day record. The returned copy carries the
    columns of :func:`psi_model_equal_duration` plus

    ``p00_model``
        the expression above;
    ``p00_indep``
        the independence baseline (1 - q_j)(1 - q_{j'}), the value the same
        cell would take with the two exits drawn independently;
    ``delta_p00_model``
        ``p00_model - p00_indep`` = Phi_2(z_j, z_{j'}; rho) - q_j q_{j'}, the
        part of p^stay the *spatial* layer is responsible for;
    ``delta_p00``
        the same excess on the observed side, ``diag[obs_col] - p00_indep``,
        added only when ``obs_col`` is present.

    The last two matter for how much the p^stay scatter can be asked to say:
    p^stay is dominated by the margins, and the dependence enters it only through
    the small perturbation ``delta``. A ``p00_model`` vs ``p00_hat`` cloud
    therefore hugs the identity line largely because the single-site q's are
    right, nearly whatever ``theta`` is; the ``delta`` pair is the same
    comparison with the margins taken out, so its reference value 0 carries the
    spatial statement -- eq. (9) without the sqrt(q(1-q)q'(1-q')) normalisation,
    and so without its +-1 saturation or its degenerate-margin NaNs.
    """
    from spatial_bmcd.spatial_model import phi2_vec  # noqa: E402
    from scipy.stats import norm  # noqa: E402

    out = psi_model_equal_duration(diag, params_by_station, theta, clip_q=clip_q)
    q_j, q_k = out["q_j"].to_numpy(float), out["q_k"].to_numpy(float)
    p11 = np.asarray(phi2_vec(norm.ppf(q_j), norm.ppf(q_k),
                              out["rho_model"].to_numpy(float)), dtype=float)

    out["p00_model"] = 1.0 - q_j - q_k + p11
    out["p00_indep"] = (1.0 - q_j) * (1.0 - q_k)
    out["delta_p00_model"] = out["p00_model"] - out["p00_indep"]
    if obs_col in out.columns:
        out["delta_p00"] = out[obs_col].to_numpy(float) - out["p00_indep"]
    return out


def simulated_psi_equal_duration(
    theta,
    params_by_station: Dict[str, dict],
    Rbin: pd.DataFrame,
    d_max: int = 20,
    n_rep: int = 1,
    min_days_per_cell: int = 1,
    n_burn: int = 500,
    seed0: int = 100,
    clip_q: float = 1e-12,
    verbose: bool = False,
) -> pd.DataFrame:
    """The equal-duration table of eq. (9), estimated on data simulated at ``theta``.

    Same three steps as the simulated cloud of
    ``switch_dependence_diagnostics.ipynb``: :func:`simulate_phi_cells` draws
    ``n_rep`` trajectories from ``theta`` with the algorithm of the article,
    blanks the days the corresponding station does not record (the observed NaN
    pattern of ``Rbin``) and rebuilds (R, D, S) with the same masking as the
    likelihood, then pushes the result through the *same*
    :func:`switch_phi_duration_cells` as the observations. The diagonal
    d = d' <= ``d_max`` is then taken with :func:`equal_duration_estimates` and
    given its model value by :func:`psi_model_equal_duration`.

    The returned frame pools the replicates (column ``rep``) and has the
    columns of :func:`psi_model_equal_duration`: ``psi`` is now eq. (9) read on
    a record the model *did* generate, so its departure from ``psi_model`` is
    pure estimation noise -- the null against which the observed scatter of
    :func:`plot_psi_model_vs_obs_grid` is to be read. Note that the cells
    themselves are redrawn by the simulation: a simulated cell carries its own
    N_{jj'}(x,x'), close to but not equal to the observed one for that state
    pair. ``min_days_per_cell`` is left at 1 so that, as on the observed side,
    the admissibility threshold is applied downstream at the figure.
    """
    sims = simulate_phi_cells(
        theta, params_by_station, Rbin, n_rep=n_rep,
        min_days_per_cell=min_days_per_cell, n_burn=n_burn, seed0=seed0,
        verbose=verbose,
    )
    out = [
        psi_model_equal_duration(
            equal_duration_estimates(cells, d_max=d_max),
            params_by_station, theta, clip_q=clip_q,
        )
        for cells in sims
    ]
    return pd.concat(out, ignore_index=True)


def _exit_prob_of_cells(stations, r, d, params_by_station, clip_q):
    """q^{(r)}_{s}(d) per cell, memoised over the distinct (station, r, d)."""
    cache: Dict[Tuple[str, int, int], float] = {}
    q = np.empty(len(r), dtype=float)
    for i, (city, ri, di) in enumerate(zip(stations, r, d)):
        key = (city, int(ri), int(di))
        val = cache.get(key)
        if val is None:
            f = params_by_station[city][
                "q_d_dry_function" if ri == 0 else "q_d_wet_function"
            ]
            val = cache[key] = float(f(int(di)))
        q[i] = val
    return np.clip(q, clip_q, 1.0 - clip_q)


def _weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Pearson correlation of ``x`` and ``y`` under the weights ``w``."""
    if w.sum() <= 0:
        return np.nan
    mx, my = w @ x / w.sum(), w @ y / w.sum()
    vx, vy = w @ (x - mx) ** 2, w @ (y - my) ** 2
    if vx <= 0 or vy <= 0:
        return np.nan
    return float((w @ ((x - mx) * (y - my))) / np.sqrt(vx * vy))


def psi_model_vs_obs_summary(
    diag_by_season: Dict[str, pd.DataFrame],
    obs_col: str = "psi",
    model_col: str = "psi_model",
    n_col: str = "n_days",
    min_n: Optional[int] = 10,
    d_max: int = 20,
    seasons: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Per (season, state class) agreement between eq. (12) and eq. (9).

    One row per panel of :func:`plot_psi_model_vs_obs_grid`: the number of
    dots, the ``N``-weighted means of the two columns, the weighted mean signed
    departure ``model - observed``, the weighted RMSE and the weighted
    correlation of the two. The weights are the cell sizes N_{jj'}(x,x'), as
    everywhere else in this module.
    """
    seasons = list(diag_by_season) if seasons is None else list(seasons)
    rows = []
    for season in seasons:
        df = _psi_model_panel_data(diag_by_season[season], obs_col, model_col,
                                   n_col, min_n, d_max)
        for cls in _CLASSES:
            sub = df[df["state_class"] == cls]
            if not len(sub):
                continue
            x = sub[obs_col].to_numpy(float)
            y = sub[model_col].to_numpy(float)
            w = sub[n_col].to_numpy(float)
            rows.append({
                "season": season, "class": cls, "n_dots": len(sub),
                "obs mean (N-w.)": w @ x / w.sum(),
                "model mean (N-w.)": w @ y / w.sum(),
                "bias (model - obs)": w @ (y - x) / w.sum(),
                "rmse": float(np.sqrt(w @ (y - x) ** 2 / w.sum())),
                "corr (N-w.)": _weighted_corr(x, y, w),
            })
    return pd.DataFrame(rows).set_index(["season", "class"])


def _psi_model_panel_data(df, obs_col, model_col, n_col, min_n, d_max):
    """Rows of one season entering the model-vs-observed figure."""
    out = df[np.isfinite(df[obs_col]) & np.isfinite(df[model_col])
             & df["d"].between(1, d_max)]
    if min_n is not None:
        out = out[out[n_col] >= min_n]
    return out


def plot_psi_model_vs_obs_grid(
    diag_by_season: Dict[str, pd.DataFrame],
    obs_col: str = "psi",
    model_col: str = "psi_model",
    n_col: str = "n_days",
    min_n: Optional[int] = None,
    d_max: int = 20,
    seasons: Optional[Sequence[str]] = None,
    n_ref: Optional[float] = None,
    lims: Optional[Sequence[float]] = None,
    legend_ns: Optional[Sequence[int]] = (10, 80, 150),
    alpha_range: Sequence[float] = (0.15, 0.85),
    panel_size: Sequence[float] = (2.4, 2.2),
    s_range: Sequence[float] = (1.0, 22.0),
    cmap: str = "viridis",
    color_scale: str = "log",
    xlabel: str = r"observed $\widehat{\psi}_{jj'}(x,x')$   (eq. 9)",
    ylabel: str = r"model $\psi_{jj'}(x,x')$   (eq. 12)",
    cbar_label: str = r"common duration $d = d'$ (days)",
    identity_label: str = r"$\psi_{\mathrm{model}} = \widehat{\psi}$",
    filenames: Sequence[pathlib.Path] = (),
):
    """Eq. (12) against eq. (9), cell by cell, on the equal-duration diagonal.

    Same grid, same slice of the state pairs and the same dot encoding as
    :func:`plot_psi_equal_duration_grid` -- one row per season, one column per
    state-pair class, the dot colour carrying the common duration d = d' on a
    log scale and both the dot area and the dot opacity carrying
    N_{jj'}(x,x') through the sqrt(N) mapping of the article
    (:func:`_sqrtn_size_alpha`, normalised by ``n_ref``) -- but the
    inter-station distance leaves the x-axis to the observed estimate: a dot is
    at ``(psi_hat, psi_model)`` for its cell, and the grey line is the identity
    ``psi_model = psi_hat``. ``diag_by_season`` must already carry ``model_col``
    (:func:`psi_model_equal_duration`). Panels share one square set of axes, so
    the four seasons and the three classes are directly comparable, and each
    carries its number of dots; the ``N``-weighted correlation and RMSE are
    left to the table of :func:`psi_model_vs_obs_summary` and no longer printed
    on the panels. ``min_n=10`` applies the article's
    admissibility rule, ``lims`` overrides the shared square window (which
    defaults to the data, i.e. to the +-1 saturation of the observed side).

    ``panel_size`` is the (width, height) of one panel in inches -- half that of
    :func:`plot_psi_equal_duration_grid`, this figure being read as a cloud
    around a line rather than dot by dot -- and ``s_range`` the ``(s_min,
    s_max)`` of the dot areas, scaled down with it so the clouds keep the same
    density. ``x`` need not be the observed estimate: passing the simulated
    tables of :func:`simulated_psi_equal_duration` (and the matching
    ``xlabel``) draws the same figure under a true model. Nor need the metric
    be psi -- ``obs_col``/``model_col`` pick the pair of columns and
    ``xlabel``, ``ylabel``, ``identity_label`` the wording, which is how
    :func:`p00_model_equal_duration` feeds it p^stay and its excess over
    independence. Returns the figure.
    """
    seasons = list(diag_by_season) if seasons is None else list(seasons)
    data = {}
    for season in seasons:
        df = _psi_model_panel_data(diag_by_season[season], obs_col, model_col,
                                   n_col, min_n, d_max)
        data[season] = df.sort_values(n_col, ascending=False)  # small dots last
    if n_ref is None:
        n_ref = float(max(df[n_col].max() for df in data.values()))

    if color_scale == "log":
        cmap_d, norm_d = plt.get_cmap(cmap), mcolors.LogNorm(vmin=0.85, vmax=d_max + 0.5)
        cbar_ticks = [t for t in (1, 2, 3, 4, 5, 6, 8, 10, 14, 20) if t <= d_max]
    else:
        cmap_d = plt.get_cmap(cmap, d_max)
        norm_d = mcolors.BoundaryNorm(np.arange(0.5, d_max + 1.5), d_max)
        cbar_ticks = np.arange(1, d_max + 1)

    # One square window shared by the whole grid, so a departure from the
    # identity line has the same meaning in every panel. It defaults to the
    # data, hence to the +-1 saturation of the observed side, which leaves the
    # model side in a narrow band -- pass ``lims`` to zoom on it.
    if lims is None:
        lo = min(min(df[obs_col].min(), df[model_col].min()) for df in data.values())
        hi = max(max(df[obs_col].max(), df[model_col].max()) for df in data.values())
        pad = 0.05 * (hi - lo)
        lims = (lo - pad, hi + pad)
    lims = tuple(float(v) for v in lims)

    fig, axes = plt.subplots(len(seasons), len(_CLASSES),
                             figsize=(panel_size[0] * len(_CLASSES),
                                      panel_size[1] * len(seasons)),
                             squeeze=False, sharex=True, sharey=True)
    for row, season in enumerate(seasons):
        for col, cls in enumerate(_CLASSES):
            ax = axes[row][col]
            sub = data[season][data[season]["state_class"] == cls]
            ax.axhline(0.0, color="grey", lw=0.6, ls=":", zorder=1)
            ax.axvline(0.0, color="grey", lw=0.6, ls=":", zorder=1)
            ax.plot(lims, lims, color="dimgrey", lw=0.9, ls="--", zorder=2,
                    label=identity_label)
            if len(sub):
                s, a = _sqrtn_size_alpha(sub[n_col].to_numpy(), n_ref,
                                         s_min=s_range[0], s_max=s_range[1],
                                         a_min=alpha_range[0], a_max=alpha_range[1])
                ax.scatter(sub[obs_col], sub[model_col], s=s, c=sub["d"],
                           cmap=cmap_d, norm=norm_d, alpha=a, linewidths=0,
                           zorder=3, rasterized=True)
                note = f"{len(sub)} dots"
            else:
                note = "0 dots"
            ax.text(0.03, 0.97, note, transform=ax.transAxes, va="top",
                    fontsize=5.5, color="dimgrey", zorder=5,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.0))
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.set_title(f"current states: {cls}", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{season}\n" + ylabel, fontsize=6.5)
            if row == len(seasons) - 1:
                ax.set_xlabel(xlabel, fontsize=6.5)

    # Size/opacity legend in grey (the identity line is already a handle of
    # that axis): colour is taken by the duration.
    for n_leg in (legend_ns if legend_ns is not None else _legend_ns(n_ref)):
        s, a = _sqrtn_size_alpha(np.array([n_leg]), n_ref,
                                 s_min=s_range[0], s_max=s_range[1],
                                 a_min=alpha_range[0], a_max=alpha_range[1])
        axes[0][0].scatter([], [], s=s[0], color="dimgrey", alpha=a[0],
                           label=f"$N_{{jj'}}(x,x')$ = {n_leg:d} days")
    axes[0][0].legend(fontsize=5, loc="lower right", framealpha=0.9,
                      handletextpad=0.4, borderpad=0.4, labelspacing=0.35)

    fig.tight_layout(rect=(0.0, 0.0, 0.90, 1.0))
    cax = fig.add_axes([0.925, 0.12, 0.014, 0.76])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_d, norm=norm_d), cax=cax,
                        ticks=cbar_ticks)
    cbar.set_label(cbar_label, fontsize=7)
    cbar.ax.set_yticklabels([str(t) for t in cbar_ticks])  # no minor log labels
    cbar.ax.tick_params(labelsize=5.5, which="both")
    cbar.ax.minorticks_off()
    for f in filenames:
        fig.savefig(f, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Script: same data pipeline as switch_dependence_diagnostics.ipynb, per season
# ---------------------------------------------------------------------------


def _load_season_data(season: str = "spring"):
    """Load the Spain--Portugal 1980--2020 history of one meteorological season."""
    import json

    from article_code.util_files import config
    from spatial_bmcd.spatial_model import (
        build_dict_model_params,
        build_joint_df_occurrence_from_raw_data,
        filter_stations_by_nan_fraction,
        history_from_Rbin_drop_ambiguous_spell_after_nan,
        load_all_station_rr,
        prepare_station_fit_table,
        split_history_by_year,
    )

    base_filename = (
        f"ecad_data_south_europe_filtered_after_{config.START_YEAR}"
        f"_wet_day_thresh_{config.WET_DAY_THRESHOLD}.json"
    )
    with open(config.EXPORTS_JSON_DIR / base_filename) as fh:
        spells = json.load(fh)
    stations_metadata = pd.read_csv(config.STATION_METADATA_CSV)
    stations_used = stations_metadata[stations_metadata["city"].isin(spells.keys())]
    fit_folder = (
        config.RESULTS_FIT_DIR
        / f"fit_south_europe_subset_excess_over_{config.WET_DAY_THRESHOLD}"
    )
    df_fit_dry = pd.read_csv(fit_folder / "dry_spell_fit_egpd1_excess_over_1result_fit_parameters.csv")
    df_fit_wet = pd.read_csv(fit_folder / "wet_spell_fit_mixt_geomresult_fit_parameters.csv")
    df_fit_dry["city"] = df_fit_dry["data_source"].map(lambda s: s.split()[0])
    df_fit_dry["season"] = df_fit_dry["data_source"].map(lambda s: s.split()[-1])

    df_fit_dry_sub, df_fit_wet_sub = prepare_station_fit_table(
        df_fit_dry, df_fit_wet, stations_used, countries=["PT", "ES"], season=season,
    )
    params = build_dict_model_params(
        df_fit_dry_sub, df_fit_wet_sub, spells, season=season, show_progress=False,
    )
    # The raw record is needed to know how incomplete each station is, so the
    # station set is settled in two passes: a provisional one to know which
    # files to read, then the definitive one on the stations that pass the
    # missing-day filter. Only the first pass touches the disk.
    dfs_by_city = load_all_station_rr(
        config.ECAD_RAW_DIR, stations_to_get=sorted(params.keys()), verbose=False,
    )
    stations_used = filter_stations_by_nan_fraction(stations_used, dfs_by_city)
    df_fit_dry_sub, df_fit_wet_sub = prepare_station_fit_table(
        df_fit_dry, df_fit_wet, stations_used, countries=["PT", "ES"], season=season,
    )
    params = build_dict_model_params(
        df_fit_dry_sub, df_fit_wet_sub, spells, season=season, show_progress=False,
    )
    stations = sorted(c for c in params if c in dfs_by_city)
    Rbin = build_joint_df_occurrence_from_raw_data(
        dfs_by_city, season=season, station_names=stations,
        wet_day_threshold=float(config.WET_DAY_THRESHOLD),
    )
    Rbin = Rbin.loc[(Rbin.index.year >= 1980) & (Rbin.index.year <= 2020)]
    history = history_from_Rbin_drop_ambiguous_spell_after_nan(Rbin)
    histories = split_history_by_year(history)
    params_by_station = {c: params[c] for c in history["station_names"]}
    return histories, params_by_station, Rbin


def figure_targets(stem: str) -> List[pathlib.Path]:
    """Where a figure of this module is written: article dir (pdf + png) and figures/spatial."""
    from article_code.util_files import config

    article_fig_dir = _HERE / "Spatialisation-Rain-Occurrence-Generator-article" / "figures"
    spatial_fig_dir = config.ROOT / "figures" / "spatial"
    spatial_fig_dir.mkdir(parents=True, exist_ok=True)
    return [article_fig_dir / f"{stem}.pdf", article_fig_dir / f"{stem}.png",
            spatial_fig_dir / f"{stem}.pdf"]


def main():
    from article_code.util_files import config

    out_dir = _HERE / "experiment_outputs"
    article_fig_dir = _HERE / "Spatialisation-Rain-Occurrence-Generator-article" / "figures"
    spatial_fig_dir = config.ROOT / "figures" / "spatial"
    spatial_fig_dir.mkdir(parents=True, exist_ok=True)

    _targets = figure_targets

    cells_by_season, pairs_by_season, stations_by_season = {}, {}, {}
    for season in _SEASONS:
        print(f"\n########## {season} ##########")
        print("loading data (same pipeline as switch_dependence_diagnostics.ipynb)...")
        histories, params_by_station, _ = _load_season_data(season)
        stations_by_season[season] = list(histories[0]["station_names"])
        print(f"{len(params_by_station)} stations, {len(histories)} year-blocks")

        print("computing exact (d, d') cells...")
        cells_exact = switch_phi_duration_cells(
            histories, params_by_station, min_days_per_cell=10, binned=False,
        )
        print("computing binned duration cells...")
        cells_binned = switch_phi_duration_cells(
            histories, params_by_station, min_days_per_cell=10, binned=True,
        )
        pairs_exact = aggregate_phi_per_pair(cells_exact)
        pairs_binned = aggregate_phi_per_pair(cells_binned)

        cells_exact.to_csv(out_dir / f"switch_phi_duration_cells_{season}.csv", index=False)
        pairs = pairs_exact.merge(
            pairs_binned, on=["station_j", "station_k", "state_class"],
            suffixes=("_exact", "_binned"),
        )
        pairs.to_csv(out_dir / f"switch_phi_duration_pairs_{season}.csv", index=False)

        print("\n=== coverage (exact cells, N >= 10 and non-degenerate margins) ===")
        cov = pairs_exact.groupby("state_class").apply(
            lambda g: pd.Series({
                "n_pairs": len(g),
                "median days used / pair": g["n_used"].median(),
                "median cells used / pair": g["n_cells"].median(),
                "fraction of class days used": g["n_used"].sum() / g["n_class_days"].sum(),
            }), include_groups=False,
        )
        print(cov.round(3).to_string())

        print("\n=== exact vs binned aggregation ===")
        print(_compare_exact_binned(pairs_exact, pairs_binned).round(3).to_string())

        cells_by_season[season] = cells_exact
        pairs_by_season[season] = pairs_exact

    # The four rows of Figure 1 must rest on the same stations to be comparable.
    common = set.intersection(*(set(v) for v in stations_by_season.values()))
    for season, names in stations_by_season.items():
        if set(names) != common:
            raise ValueError(
                f"station set of {season} differs from the common one "
                f"({len(names)} vs {len(common)}); the seasonal rows of Figure 1 "
                "would not be comparable"
            )
    n_pairs = len(common) * (len(common) - 1) // 2
    print(f"\n=== Figure 1: {len(common)} stations common to the four seasons, "
          f"{n_pairs} station pairs ===")

    summary = season_trend_summary(cells_by_season)
    print(summary.round(3).to_string(index=False))
    summary.to_csv(out_dir / "switch_phi_duration_season_summary.csv", index=False)
    print(f"\ntotal dots over the grid: {summary['n_dots'].sum()}, "
          f"largest cell size N = {summary['max_N'].max()}")

    plot_phi_per_cell_season_grid(
        cells_by_season, seasons=_SEASONS, legend_ns=(10, 80, 150),
        filenames=_targets("diag_switch_phi_duration_per_cell_seasons"),
    )

    # Spring-only companion scatters, kept for inspection (not in the article).
    cells_spring, pairs_spring = cells_by_season["spring"], pairs_by_season["spring"]
    plot_phi_vs_distance(
        pairs_spring, value_col="phi_bar", n_col="n_used",
        ylabel=r"duration-stratified Pearson correlation $\bar\varphi$",
        filenames=_targets("diag_switch_phi_duration_vs_distance"),
    )
    plot_phi_vs_distance(
        cells_spring[np.isfinite(cells_spring["phi"])],
        value_col="phi", n_col="n_days",
        ylabel=r"per-cell Pearson correlation $\hat\varphi(d,d')$",
        filenames=_targets("diag_switch_phi_duration_per_cell_vs_distance"),
    )
    print("\nfigures written to", article_fig_dir, "and", spatial_fig_dir)


if __name__ == "__main__":
    main()
