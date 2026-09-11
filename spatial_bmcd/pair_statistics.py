"""Model-free pair statistics of a daily occurrence record, observed or simulated.

The two pair statistics that ``spatial_bmcd_data_analysis.ipynb`` reads off the
raw ECAD record, lifted out of its cells so that
``data_analysis_fitted_model.ipynb`` can read them off simulated data from
fitted model with the *same* code:

- the rain-state agreement pi_hat_{jj'}, one number per station pair
  (:func:`state_agreement_table`, :func:`plot_agreement_vs_distance`, and the
  observed-vs-simulated overlay :func:`plot_agreement_obs_vs_sim`);
- the co-persistence probability p_hat^stay_{jj'}(x,x') on the equal-duration
  cells d = d' (:func:`day_record_from_Rbin`, :func:`equal_duration_metric_table`,
  :func:`p00_cell_metrics`, :func:`plot_equal_duration_metric_grid`).

Every function takes a days x stations 0/1/NaN frame ``Rbin`` holding the days
of *one* season (what ``build_joint_df_occurrence_from_raw_data(season=...)``
returns) and, for the coordinates, the single-site parameter dict of that
season. :func:`season_frames` cuts an all-season frame -- the observed
``Rbin_obs`` of the fit notebook, or a simulated trajectory on the same
calendar -- into those per-season frames, so a simulated record goes through
exactly the pipeline of the observed one.

The statistics and the two plotters are verbatim copies of the notebook's
cells (their module-level defaults made explicit); the notebook still holds
its own copies, and switching it to these imports is part of T-015.
"""

from __future__ import annotations

import pathlib
from typing import Callable, Dict, Optional, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spatial_bmcd.spatial_model import (
    history_from_Rbin_drop_ambiguous_spell_after_nan,
    season_of_dates,
    split_history_by_year,
    station_distance_matrix,
)
from spatial_bmcd.switch_phi_duration_diagnostic import (
    _QUANTILE_PAIRS,
    _draw_binned_curves,
    _sqrtn_size_alpha,
    bin_edges,
)

_SEASONS = ("spring", "summer", "autumn", "winter")
_CLASSES = ("dry-dry", "wet-wet", "dry-wet")

MIN_DAYS = 10   # admissibility rule of the article (N_{jj'} >= 10)
D_MAX = 20      # equal durations d = d' = 1 ... D_MAX


# ---------------------------------------------------------------------------
# From an occurrence frame to the day-level record (R, D, B)
# ---------------------------------------------------------------------------


def season_frames(Rbin: pd.DataFrame, seasons: Sequence[str] = _SEASONS) -> Dict[str, pd.DataFrame]:
    """``{season: the rows of Rbin whose day falls in that season}``.

    Same cut as ``build_joint_df_occurrence_from_raw_data(season=s)`` applied to
    the full calendar, so an all-season frame -- observed, or a simulated
    trajectory on the observed calendar -- yields the per-season frames the
    statistics below expect.
    """
    labels = season_of_dates(pd.DatetimeIndex(Rbin.index))
    return {s: Rbin.loc[labels == s] for s in seasons}


def day_record_from_Rbin(Rbin: pd.DataFrame, params_by_station: Dict[str, dict]) -> dict:
    """Day-level record (R, D, B) of one season, plus names and distances.

    ``R`` is the rain state and ``D`` the spell age of day n, ``B`` the exit
    indicator of day n (needs day n+1), all aligned on the rows of ``B``: the
    per-year histories of ``history_from_Rbin_drop_ambiguous_spell_after_nan``
    (ambiguous spells masked as in the likelihood), stacked without chaining a
    spell across a year boundary. ``dist`` is the km distance matrix in the
    order of ``station_names``.
    """
    history = history_from_Rbin_drop_ambiguous_spell_after_nan(Rbin)
    years = split_history_by_year(history)   # no spell chained across a year boundary
    B = np.vstack([h["S"] for h in years])   # exit indicator of day n (needs day n+1)
    R = np.vstack([h["R"][: h["S"].shape[0]] for h in years])
    D = np.vstack([h["D"][: h["S"].shape[0]] for h in years])
    names = history["station_names"]
    return {
        "R": R, "D": D, "B": B, "station_names": names,
        "dist": station_distance_matrix(params_by_station, names),
    }


def stack_day_records(day_records: Sequence[dict]) -> dict:
    """Day records of several independent trajectories, pooled into one.

    The statistics of this module read the rows of (R, D, B) one at a time --
    the transition n -> n+1 is already inside ``B`` -- so the day records of
    independent trajectories, each cut at its year boundaries by
    :func:`day_record_from_Rbin`, can be stacked without chaining a spell
    across two of them. The station order must agree; ``dist`` is taken from
    the first record.
    """
    first = day_records[0]
    names = list(first["station_names"])
    for rec in day_records[1:]:
        if list(rec["station_names"]) != names:
            raise ValueError("day records with different station orders cannot be stacked")
    return {
        "R": np.vstack([rec["R"] for rec in day_records]),
        "D": np.vstack([rec["D"] for rec in day_records]),
        "B": np.vstack([rec["B"] for rec in day_records]),
        "station_names": names,
        "dist": first["dist"],
    }


# ---------------------------------------------------------------------------
# Co-persistence p^stay on the equal-duration cells d = d'
# ---------------------------------------------------------------------------


def equal_duration_metric_table(day_record: dict, cell_metrics: Callable,
                                d_max: int = D_MAX) -> pd.DataFrame:
    """One row per (station pair, r, r', d = d' <= d_max): day count + metrics."""
    R, D, B = day_record["R"], day_record["D"], day_record["B"]
    names, dist = day_record["station_names"], day_record["dist"]
    valid = np.isfinite(R) & np.isfinite(D) & np.isfinite(B)
    rows = []
    for j in range(len(names) - 1):
        for k in range(j + 1, len(names)):
            ok = valid[:, j] & valid[:, k]           # both stations on days n, n+1
            d = D[ok, j].astype(int)
            same = (D[ok, k].astype(int) == d) & (d >= 1) & (d <= d_max)
            r_j, r_k = R[ok, j].astype(int)[same], R[ok, k].astype(int)[same]
            b_j, b_k = B[ok, j].astype(int)[same], B[ok, k].astype(int)[same]
            d = d[same]
            for r, rp, dd in np.unique(np.stack([r_j, r_k, d], axis=1), axis=0):
                sel = (r_j == r) & (r_k == rp) & (d == dd)
                rows.append({
                    "station_j": names[j], "station_k": names[k],
                    "dist": float(dist[j, k]),
                    "state_class": ("dry-dry" if r == rp == 0 else
                                    "wet-wet" if r == rp == 1 else "dry-wet"),
                    "r_j": int(r), "r_k": int(rp), "d": int(dd),
                    "n_days": int(sel.sum()),
                    **cell_metrics(b_j[sel], b_k[sel]),
                })
    return pd.DataFrame(rows)


def p00_cell_metrics(b_j: np.ndarray, b_k: np.ndarray) -> dict:
    """Joint persistence: share of the cell's days on which neither station exits."""
    no_exit = (b_j == 0) & (b_k == 0)
    return {"p00": no_exit.mean(), "n_no_exit": int(no_exit.sum())}


def plot_equal_duration_metric_grid(
    table_by_season: Dict[str, pd.DataFrame],
    value_col: str,
    value_label: Optional[str] = None,
    state_classes: Sequence[str] = _CLASSES,
    min_n: Optional[int] = MIN_DAYS,
    d_max: int = D_MAX,
    seasons: Sequence[str] = _SEASONS,
    n_ref: Optional[float] = None,
    cmap: str = "viridis",
    alpha_range: Sequence[float] = (0.15, 0.85),
    y_range: Optional[Sequence[float]] = None,
    zero_line: bool = False,
    trend: bool = True,
    trend_color: str = "crimson",
    trend_label: str = r"$N$-weighted bin mean",
    n_bins: int = 8,
    legend_ns: Sequence[int] = (10, 80, 150),
    cbar_label: str = r"common duration $d = d'$ (days)",
    suptitle: Optional[str] = None,
    filenames: Sequence[pathlib.Path] = (),
):
    """Metric-vs-distance grid, one row per season and one column per state class.

    Matplotlib figure, drawn with the layout and the styling of the other
    article figures (`plot_gamma_state_pair_season_grid`,
    `plot_phi_per_cell_season_grid`): shared axes, `current states: <class>`
    column titles, `<season>` + metric row labels, distance in km on x, the
    dot count of each panel, and the same `crimson` N-weighted bin mean over
    the cloud. Dot area *and* opacity both follow sqrt(N_{jj'}(x,x'))
    (`_sqrtn_size_alpha`, normalised by `n_ref`, the largest cell drawn unless
    given), and the colour carries the common duration d = d' on a log scale,
    where the estimates are -- most of them sit at d <= 3, which a linear
    scale would pack into the dark end of the colormap. `alpha_range` trades
    the two off: a faint dot has its hue washed towards the background.

    `min_n` filters on `n_days` (`None` keeps every row of the table),
    `zero_line` adds the psi = 0 independence baseline, `y_range` fixes the
    shared y-axis, `suptitle` names the record the table was read on, and
    `filenames` are the targets the figure is written to. Returns the figure.
    """
    seasons, state_classes = list(seasons), list(state_classes)
    value_label = value_col if value_label is None else value_label

    data = {}
    for season in seasons:
        df = table_by_season[season]
        df = df[df["d"].between(1, d_max) & np.isfinite(df[value_col])]
        if min_n is not None:
            df = df[df["n_days"] >= min_n]
        data[season] = df.sort_values("n_days", ascending=False)  # small dots on top
    if n_ref is None:
        n_seen = [float(df["n_days"].max()) for df in data.values() if len(df)]
        n_ref = max(n_seen) if n_seen else 1.0

    cmap_d = plt.get_cmap(cmap)
    norm_d = mcolors.LogNorm(vmin=0.85, vmax=d_max + 0.5)
    cbar_ticks = [t for t in (1, 2, 3, 4, 5, 6, 8, 10, 14, 20) if t <= d_max]

    fig, axes = plt.subplots(len(seasons), len(state_classes),
                             figsize=(4.8 * len(state_classes), 3.4 * len(seasons)),
                             squeeze=False, sharex=True, sharey=True)
    for row, season in enumerate(seasons):
        for col, cls in enumerate(state_classes):
            ax = axes[row][col]
            sub = data[season][data[season]["state_class"] == cls]
            if len(sub):
                s, a = _sqrtn_size_alpha(sub["n_days"].to_numpy(), n_ref,
                                         a_min=alpha_range[0], a_max=alpha_range[1])
                ax.scatter(sub["dist"], sub[value_col], s=s, c=sub["d"],
                           cmap=cmap_d, norm=norm_d, alpha=a, linewidths=0,
                           zorder=3, rasterized=True)
                if trend and len(sub) > 1:
                    _draw_binned_curves(
                        ax, sub, value_col, "n_days",
                        bin_edges(sub, n_bins, stat=value_col),
                        trend_color, trend_label, center="mean", pairs=(),
                    )
            if zero_line:   # independence baseline of eq. (9)
                ax.axhline(0.0, color="grey", lw=1, ls=":")
            ax.text(0.02, 0.03, f"{len(sub)} dots", transform=ax.transAxes,
                    fontsize=8, color="dimgrey", zorder=5,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5))
            if row == 0:
                ax.set_title(f"current states: {cls}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{season}\n" + value_label, fontsize=9)
            if row == len(seasons) - 1:
                ax.set_xlabel("inter-station distance (km)")
    if y_range is not None:
        axes[0][0].set_ylim(*y_range)   # shared y-axis

    # Size/opacity legend in grey: colour is taken by the duration.
    for n_leg in legend_ns:
        s, a = _sqrtn_size_alpha(np.array([n_leg]), n_ref,
                                 a_min=alpha_range[0], a_max=alpha_range[1])
        axes[0][0].scatter([], [], s=s[0], color="dimgrey", alpha=a[0],
                           label=f"$N_{{jj'}}(x,x')$ = {n_leg:d} days")
    axes[0][0].legend(fontsize=7, loc="lower right", framealpha=0.9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)

    fig.tight_layout(rect=(0.0, 0.0, 0.92, 1.0))
    cax = fig.add_axes([0.935, 0.12, 0.012, 0.76])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_d, norm=norm_d), cax=cax,
                        ticks=cbar_ticks)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.set_yticklabels([str(t) for t in cbar_ticks])   # no minor log labels
    cbar.ax.tick_params(labelsize=7, which="both")
    cbar.ax.minorticks_off()
    for f in filenames:
        fig.savefig(f, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Rain-state agreement pi_hat_{jj'}
# ---------------------------------------------------------------------------


def state_agreement_table(Rbin: pd.DataFrame, params: Dict[str, dict]) -> pd.DataFrame:
    """One row per station pair: N_{jj'}, pi_hat_{jj'} and its independence baseline.

    Read off the raw occurrence frame and not off the day record: no spell age
    and no day n+1 are involved, so every day on which both stations are
    observed contributes, including the days the spell-age pipeline masks.
    """
    names = list(Rbin.columns)
    R = Rbin.to_numpy(dtype=float)
    obs = np.isfinite(R)
    dist = station_distance_matrix(params, names)
    rows = []
    for j in range(len(names) - 1):
        for k in range(j + 1, len(names)):
            ok = obs[:, j] & obs[:, k]           # both stations observed on day n
            if not ok.any():
                continue
            r_j, r_k = R[ok, j], R[ok, k]
            p_j, p_k = float(r_j.mean()), float(r_k.mean())
            rows.append({
                "station_j": names[j], "station_k": names[k],
                "dist": float(dist[j, k]), "n_days": int(ok.sum()),
                "agree": float(np.mean(r_j == r_k)),
                "agree_indep": p_j * p_k + (1.0 - p_j) * (1.0 - p_k),
                "wet_frac_j": p_j, "wet_frac_k": p_k,
            })
    return pd.DataFrame(rows)


def plot_agreement_vs_distance(
    tables_by_panel: Dict[str, pd.DataFrame],
    min_n: Optional[int] = MIN_DAYS,
    n_bins: int = 10,
    ncols: Optional[int] = None,
    dot_size: float = 15,
    dot_alpha: float = 0.75,
    show_independence: bool = True,
    trend: bool = True,
    plot_quantiles: bool = True,
    trend_color: str = "crimson",
    y_range: Optional[Sequence[float]] = None,
    filenames: Sequence[pathlib.Path] = (),
):
    """pi_hat_{jj'} against inter-station distance, one panel per entry of the dict.

    One black dot per station pair kept when at least `min_n` common days feed
    it. Over the cloud, `trend` draws the N-weighted median of the dots in each
    of `n_bins` equal-count distance bins and `plot_quantiles` the 20%/80% and
    10%/90% weighted quantiles of the dots of that bin around it
    (`_draw_binned_curves`); the two switch independently, and both off leaves
    the bare cloud. The grey curve of `show_independence` is the binned median
    of the independence baseline of the same pairs: the gap between the two
    curves is the share of the agreement the marginal wet fractions do not
    explain.
    """
    panels = list(tables_by_panel.items())
    ncols = len(panels) if ncols is None else min(ncols, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows),
                             squeeze=False, sharex=True, sharey=True)
    flat = [ax for row in axes for ax in row]
    for i, (ax, (name, df)) in enumerate(zip(flat, panels)):
        sub = df[np.isfinite(df["agree"])
                 & (df["n_days"] >= (0 if min_n is None else min_n))]
        ax.scatter(sub["dist"], sub["agree"], s=dot_size, alpha=dot_alpha,
                   color="black", linewidths=0, zorder=3)
        if len(sub) > 1:
            if show_independence:
                _draw_binned_curves(
                    ax, sub, "agree_indep", "n_days",
                    bin_edges(sub, n_bins, stat="agree_indep"), "dimgrey",
                    "independence baseline", center="median", pairs=(), zorder=6,
                )
            if trend or plot_quantiles:
                _draw_binned_curves(
                    ax, sub, "agree", "n_days",
                    bin_edges(sub, n_bins, stat="agree"), trend_color,
                    r"binned median $\widehat{\pi}_b$",
                    center="median" if trend else None,
                    pairs=_QUANTILE_PAIRS if plot_quantiles else (),
                )
        ax.set_title(name, fontsize=10)
        ax.text(0.02, 0.03, f"{len(sub)} pairs", transform=ax.transAxes,
                fontsize=8, color="dimgrey", zorder=5,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5))
        if i >= len(panels) - ncols:
            ax.set_xlabel("inter-station distance (km)")
    for ax in flat[len(panels):]:
        ax.set_visible(False)
    for row in axes:
        row[0].set_ylabel(r"$\widehat{\pi}_{jj'}$", fontsize=9)
    if y_range is not None:
        axes[0][0].set_ylim(*y_range)   # shared y-axis
    if axes[0][0].get_legend_handles_labels()[0]:
        axes[0][0].legend(fontsize=7, loc="best", framealpha=0.9)

    fig.tight_layout()
    for f in filenames:
        fig.savefig(f, bbox_inches="tight")
    return fig


def plot_agreement_obs_vs_sim(
    tables_obs: Dict[str, pd.DataFrame],
    tables_sim: Dict[str, pd.DataFrame],
    labels: Sequence[str] = ("observed data", "simulated data from fitted model"),
    colors: Sequence[str] = ("black", "steelblue"),
    min_n: Optional[int] = MIN_DAYS,
    n_bins: int = 10,
    ncols: Optional[int] = 2,
    dot_size: float = 15,
    dot_alpha: float = 0.7,
    trend: bool = True,
    y_range: Optional[Sequence[float]] = None,
    suptitle: Optional[str] = None,
    filenames: Sequence[pathlib.Path] = (),
):
    """pi_hat_{jj'} of two records on the same axes, one panel per season.

    The panels are the keys of `tables_obs` (`tables_sim` must hold the same
    keys); in each, the pairs of the first record are the dots of the first
    colour and those of the second record the dots of the second colour, both
    against the same inter-station distance, so a pair sits on one vertical
    line in the two colours. `trend` adds the N-weighted binned median of each
    cloud in its own colour (`_draw_binned_curves`, `n_bins` equal-count
    distance bins, no quantile fan). `min_n` keeps the pairs with at least that
    many common days, applied to each record separately. Returns the figure.
    """
    panels = list(tables_obs)
    ncols = len(panels) if ncols is None else min(ncols, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows),
                             squeeze=False, sharex=True, sharey=True)
    flat = [ax for row in axes for ax in row]
    for i, (ax, name) in enumerate(zip(flat, panels)):
        counts = []
        for tables, label, color, z in ((tables_obs, labels[0], colors[0], 3),
                                        (tables_sim, labels[1], colors[1], 4)):
            df = tables[name]
            sub = df[np.isfinite(df["agree"])
                     & (df["n_days"] >= (0 if min_n is None else min_n))]
            counts.append(len(sub))
            ax.scatter(sub["dist"], sub["agree"], s=dot_size, alpha=dot_alpha,
                       color=color, linewidths=0, zorder=z, label=label)
            if trend and len(sub) > 1:
                _draw_binned_curves(
                    ax, sub, "agree", "n_days", bin_edges(sub, n_bins, stat="agree"),
                    color, f"{label}, binned median", center="median", pairs=(),
                    zorder=z + 4,
                )
        ax.set_title(name, fontsize=10)
        ax.text(0.02, 0.03, f"{counts[0]} / {counts[1]} pairs", transform=ax.transAxes,
                fontsize=8, color="dimgrey", zorder=9,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5))
        if i >= len(panels) - ncols:
            ax.set_xlabel("inter-station distance (km)")
    for ax in flat[len(panels):]:
        ax.set_visible(False)
    for row in axes:
        row[0].set_ylabel(r"$\widehat{\pi}_{jj'}$", fontsize=9)
    if y_range is not None:
        axes[0][0].set_ylim(*y_range)   # shared y-axis
    axes[0][0].legend(fontsize=7, loc="upper right", framealpha=0.9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)

    fig.tight_layout()
    for f in filenames:
        fig.savefig(f, bbox_inches="tight")
    return fig


def agreement_obs_vs_sim_summary(
    tables_obs: Dict[str, pd.DataFrame],
    tables_sim: Dict[str, pd.DataFrame],
    min_n: Optional[int] = MIN_DAYS,
) -> pd.DataFrame:
    """Per season, pi_hat of the second record against the first, pair by pair.

    The two tables are joined on the station pair, and the pairs with at least
    `min_n` common days in both records kept. The row gives the number of
    pairs, the two medians, the N-weighted (first record's day counts) mean
    signed departure `sim - obs` and RMSE, the correlation of the two
    estimates across the pairs, and the Spearman correlation of each with the
    inter-station distance -- the decay-with-distance the figure shows.
    """
    rows = []
    for season in tables_obs:
        m = tables_obs[season].merge(tables_sim[season], on=["station_j", "station_k"],
                                     suffixes=("_obs", "_sim"))
        if min_n is not None:
            m = m[(m["n_days_obs"] >= min_n) & (m["n_days_sim"] >= min_n)]
        x, y = m["agree_obs"].to_numpy(float), m["agree_sim"].to_numpy(float)
        w = m["n_days_obs"].to_numpy(float)
        rows.append({
            "season": season, "n_pairs": len(m),
            "obs median": float(np.median(x)) if len(m) else np.nan,
            "sim median": float(np.median(y)) if len(m) else np.nan,
            "bias (sim - obs, N-w.)": w @ (y - x) / w.sum() if w.sum() else np.nan,
            "rmse (N-w.)": float(np.sqrt(w @ (y - x) ** 2 / w.sum())) if w.sum() else np.nan,
            "corr (pairs)": m["agree_obs"].corr(m["agree_sim"]),
            "spearman(obs, h)": m["agree_obs"].corr(m["dist_obs"], method="spearman"),
            "spearman(sim, h)": m["agree_sim"].corr(m["dist_obs"], method="spearman"),
        })
    return pd.DataFrame(rows).set_index("season")
