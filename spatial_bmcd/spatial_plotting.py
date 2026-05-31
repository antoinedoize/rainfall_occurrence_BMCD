"""Visualisation helpers for the spatial BMCD analysis.

Figures saved here back the ``spatial_analysis_article.tex`` document.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.lines import Line2D
from scipy.stats import norm

from article_code.util_files import config


SPATIAL_FIGURES_DIR = config.ROOT / "figures" / "spatial"


def _ensure_figures_dir() -> Path:
    SPATIAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return SPATIAL_FIGURES_DIR


# ---------------------------------------------------------------------------
# Station map (Plotly + Matplotlib)
# ---------------------------------------------------------------------------


def plot_stations_geomap(stations_used: pd.DataFrame, marker_size: int = 4) -> go.Figure:
    """Plotly geo-scatter of every station in ``stations_used`` (requires ``lon``, ``lat``)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scattergeo(
            lon=stations_used["lon"],
            lat=stations_used["lat"],
            mode="markers",
            marker=dict(size=marker_size, color="black"),
            name="stations",
        )
    )
    fig.update_geos(scope="europe", showcountries=True, resolution=50)
    fig.update_layout(margin=dict(l=0, r=0, t=20, b=0))
    return fig


def plot_stations_map(
    df_subset: pd.DataFrame,
    color_col: str = "xi",
    cmap: str = "coolwarm",
    vmin: float = -0.5,
    vmax: float = 0.5,
    ax: Optional[plt.Axes] = None,
):
    """Matplotlib scatter of normalised station coordinates, coloured by ``color_col``.

    ``df_subset`` must carry ``x_norm`` / ``y_norm`` and ``color_col``. The
    original notebook used this for a sanity-check of the dry-spell ``xi``
    distribution across stations.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    norm_ = colors.Normalize(vmin=vmin, vmax=vmax)
    sc = ax.scatter(
        df_subset["x_norm"], df_subset["y_norm"],
        c=df_subset[color_col], cmap=cmap, norm=norm_,
    )
    plt.colorbar(sc, ax=ax, label=color_col)
    ax.set_xlabel("x_norm (lon)")
    ax.set_ylabel("y_norm (lat)")
    ax.set_title(f"Stations coloured by {color_col}")
    return ax.figure


# ---------------------------------------------------------------------------
# Figure: tail-probability illustration (article Fig. caption ref)
# ---------------------------------------------------------------------------


def plot_tail_probability_illustration(
    q_dry: float = 0.20,
    q_wet: float = 0.12,
    save: bool = True,
    filename: str = "tail_probability_illustration.pdf",
):
    """Two-panel illustration of the transition rule on N(0,1) tails."""
    z_dry = norm.ppf(q_dry)
    z_wet = norm.ppf(q_wet)
    x = np.linspace(-4, 4, 1200)
    pdf = norm.pdf(x)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharey=True)
    ax = axes[0]
    ax.plot(x, pdf, lw=2, color="black")
    ax.fill_between(x, 0, pdf, where=(x <= z_dry), alpha=0.35)
    ax.fill_between(x, 0, pdf, where=(x > z_dry), alpha=0.35)
    ax.axvline(z_dry, linestyle="--", lw=2, color="black")
    ax.text(z_dry - 0.2, 0.09, r"$z_{dry}(s_j)$", rotation=90, verticalalignment="center")
    ax.set_title(r"Station $j$ currently in DRY spell ($R_n(s_j)=0$)")
    ax.set_ylabel(r"density $Y(s_j)$")
    ax.text(-3.9, 0.25,
            r"Switch to WET ($R_{n+1}(s_j)=1$)" "\n"
            r"if $Z_n(s_j) \leq z_n(s_j)$" "\n"
            r"i.e. $Y_n(s_j) \leq z_n(s_j)$",
            color="blue")
    ax.text(1.4, 0.25, r"Stay DRY ($R_{n+1}(s_j)=0$)", color="tab:orange")

    ax = axes[1]
    threshold_y = -z_wet
    ax.plot(x, pdf, lw=2, color="black")
    ax.fill_between(x, 0, pdf, where=(x < threshold_y), alpha=0.35)
    ax.fill_between(x, 0, pdf, where=(x >= threshold_y), alpha=0.35)
    ax.axvline(threshold_y, linestyle="--", lw=2, color="black")
    ax.text(threshold_y + 0.2, 0.09, r"$-z_{wet}(s_{j'})$", rotation=90,
            verticalalignment="center")
    ax.set_title(r"Station $j'$ currently in WET spell ($R_n(s_{j'})=1$)")
    ax.set_ylabel(r"density $Y(s_{j'})$")
    ax.text(1.4, 0.25,
            r"Switch to DRY ($R_{n+1}(s_{j'})=0$)" "\n"
            r"if $Z_n(s_{j'}) \leq z_n(s_{j'})$" "\n"
            r"i.e. $Y_n(s_{j'}) \geq -z_n(s_{j'})$",
            color="tab:orange")
    ax.text(-3.9, 0.25, r"Stay WET ($R_{n+1}(s_{j'})=1$)", color="blue")

    plt.tight_layout()
    if save:
        plt.savefig(_ensure_figures_dir() / filename)
    return fig


# ---------------------------------------------------------------------------
# Figure: Gaussian field + thresholds (article Fig. show_spatial_thresholds_rule.pdf)
# ---------------------------------------------------------------------------


def rbf_covariance(coords: np.ndarray, length_scale: float = 0.25, nugget: float = 1e-8) -> np.ndarray:
    """Squared-exponential covariance for a GP on 2-D coordinates of shape ``(N, 2)``."""
    diffs = coords[:, None, :] - coords[None, :, :]
    d2 = np.sum(diffs ** 2, axis=-1)
    K = np.exp(-0.5 * d2 / (length_scale ** 2))
    return K + nugget * np.eye(K.shape[0])


def plot_gaussian_field_with_thresholds(
    nx: int = 25,
    ny: int = 25,
    length_scale: float = 0.18,
    seed: int = 7,
    q_dry: Optional[dict] = None,
    q_wet: Optional[dict] = None,
    dry_idx: Optional[Sequence] = None,
    wet_idx: Optional[Sequence] = None,
    save: bool = True,
    filename: str = "show_spatial_thresholds_rule.pdf",
):
    """3-D illustration of the spatial transition rule.

    Reproduces the figure referenced as ``show_spatial_thresholds_rule.pdf``
    in ``spatial_analysis_article.tex`` (line 95).
    """
    if q_dry is None:
        q_dry = {1: 0.10, 3: 0.25, 5: 0.40}
    if q_wet is None:
        q_wet = {1: 0.06, 2: 0.12, 4: 0.20}
    if dry_idx is None:
        dry_idx = [(6, 8), (14, 18), (20, 5)]
    if wet_idx is None:
        wet_idx = [(8, 20), (16, 7), (22, 18)]

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    coords = np.column_stack([X.ravel(), Y.ravel()])

    K = rbf_covariance(coords, length_scale=length_scale)
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(K)
    Y_field = (L @ rng.standard_normal(K.shape[0])).reshape(ny, nx)

    dry_durations = sorted(q_dry.keys())
    wet_durations = sorted(q_wet.keys())
    z_dry = np.array([norm.ppf(q_dry[d]) for d in dry_durations])
    z_wet = np.array([norm.ppf(q_wet[d]) for d in wet_durations])

    def gather_points(idxs):
        xs = np.array([X[iy, ix] for (iy, ix) in idxs])
        ys = np.array([Y[iy, ix] for (iy, ix) in idxs])
        yvals = np.array([Y_field[iy, ix] for (iy, ix) in idxs])
        return xs, ys, yvals

    x_d, y_d, yv_d = gather_points(dry_idx)
    x_w, y_w, yv_w = gather_points(wet_idx)
    Z_d = +yv_d
    Z_w = -yv_w
    trans_d = Z_d <= z_dry
    trans_w = Z_w <= z_wet

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Y_field, color="tab:blue", alpha=0.55, linewidth=0)
    ax.plot_surface(X, Y, -Y_field, color="tab:orange", alpha=0.55, linewidth=0)

    def _vline(ax, x0, y0, z0, z1, color):
        ax.plot([x0, x0], [y0, y0], [z0, z1], color=color, linewidth=2)

    for k in range(len(dry_idx)):
        ax.scatter(x_d[k], y_d[k], z_dry[k], color="tab:blue", s=70, depthshade=False)
        marker = "x" if trans_d[k] else "o"
        ax.scatter(x_d[k], y_d[k], Z_d[k], color="tab:blue", s=70, marker=marker, depthshade=False)
        _vline(ax, x_d[k], y_d[k], Z_d[k], z_dry[k], "tab:blue")
        ax.text(x_d[k], y_d[k], z_dry[k],
                f" dry d={dry_durations[k]}\nq={q_dry[dry_durations[k]]:.2f}",
                color="tab:blue")

    for k in range(len(wet_idx)):
        ax.scatter(x_w[k], y_w[k], z_wet[k], color="tab:orange", s=70, depthshade=False)
        marker = "x" if trans_w[k] else "o"
        ax.scatter(x_w[k], y_w[k], Z_w[k], color="tab:orange", s=70, marker=marker, depthshade=False)
        _vline(ax, x_w[k], y_w[k], Z_w[k], z_wet[k], "tab:orange")
        ax.text(x_w[k], y_w[k], z_wet[k],
                f" wet d={wet_durations[k]}\nq={q_wet[wet_durations[k]]:.2f}",
                color="tab:orange")

    ax.set_title(
        r"Illustration of Gaussian field sign-flip: $Y_n$ (dry) vs $-Y_n$ (wet)" "\n"
        r"Transition occurs when $Z_n(s) \leq z_n(s)$ (x-marker indicates transition)"
    )
    ax.set_xlabel("s1")
    ax.set_ylabel("s2")
    ax.set_zlabel("value")

    legend_elems = [
        Line2D([0], [0], color="tab:blue", lw=6, label=r"$Y_n(s)$ surface (dry)"),
        Line2D([0], [0], color="tab:orange", lw=6, label=r"$-Y_n(s)$ surface (wet)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="k", markersize=8,
               label=r"$Z_n(s)$ (no transition)"),
        Line2D([0], [0], marker="x", color="k", markersize=8,
               label=r"$Z_n(s)$ (transition)"),
    ]
    ax.legend(handles=legend_elems, loc="upper left")

    plt.tight_layout()
    if save:
        plt.savefig(_ensure_figures_dir() / filename)
    return fig
