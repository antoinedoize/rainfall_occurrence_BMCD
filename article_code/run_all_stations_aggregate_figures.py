"""Aggregate-station batch driver — Figs. 1, 4, 5, 13, 14, A."""
import argparse
import json
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats
from plotly.subplots import make_subplots
from scipy.stats import chi2
from tqdm import tqdm

from .util_files import config, plotting
from .util_files.spell_models import (
    get_spell_length_degenerate_mixture_order_1_extgpd1,
    make_cdf_fitted_hdeGPD_from_params,
)
from .util_files.statistics import (
    adaptive_D,
    build_gof_results_df,
    build_Sigma_matrix_new,
    build_T_matrix,
    get_proba_leaving_by_day,
    get_proba_leaving_state_n_kozu,
    goodness_of_fit_true_all_cities_seasons,
    goodness_of_fit_true_all_cities_seasons_geometric,
    make_approx_mean_excess,
    mean_excess_markov_order_1,
)


# ---------------------------------------------------------------------------
# Figure builders. Each returns the figure (matplotlib or plotly).
# ---------------------------------------------------------------------------

def make_fig1_map(spells, df_stations):
    """Notebook 02 Fig. 1 — Map of southern-Europe stations, PALERMO highlighted."""
    list_cities = sorted(spells.keys())
    stations_used = df_stations[df_stations.city.isin(list_cities)]
    mask_palermo = df_stations["city"].str.upper().str.contains(config.STATION_EXAMPLE, na=False)

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=stations_used.loc[~mask_palermo, "lon"],
        lat=stations_used.loc[~mask_palermo, "lat"],
        mode="markers", marker=dict(size=4, color="black"),
        name="stations"))
    fig.add_trace(go.Scattergeo(
        lon=stations_used.loc[mask_palermo, "lon"],
        lat=stations_used.loc[mask_palermo, "lat"],
        mode="markers", marker=dict(size=10, color="orange"),
        name=config.STATION_EXAMPLE.title()))
    fig.update_geos(showcountries=True, showland=True, landcolor="lightgray",
                    lataxis_range=[34, 46], lonaxis_range=[-10, 30])
    fig.update_layout(width=900, height=500, margin=dict(l=0, r=0, t=0, b=0))
    return fig


def make_fig4_dry_param_hist(df_fit_dry):
    """Notebook 03 Fig. 4 — cross-station histograms of dry-spell hdeGPD params."""
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    params = ["f_1", "xi", "sigma", "kappa"]
    labels = [r"$\hat f_1$", r"$\hat \xi$", r"$\hat \sigma$", r"$\hat \kappa$"]
    xlims = {"f_1": (0, 0.4), "xi": (-0.5, 0.5), "sigma": (0, 110), "kappa": (0, 2)}

    bins_dict, ylims = {}, {}
    for p in params:
        xmin, xmax = xlims[p]
        bins = np.linspace(xmin, xmax, 21)
        bins_dict[p] = bins
        ymax = 0
        for season in config.SEASONS:
            vals = df_fit_dry.loc[df_fit_dry["season"] == season, p].dropna().values
            counts, _ = np.histogram(vals, bins=bins)
            if counts.size:
                ymax = max(ymax, counts.max())
        ylims[p] = (0, ymax * 1.05)

    fig, axes = plt.subplots(
        nrows=len(config.SEASONS), ncols=len(params),
        figsize=(9, 7), constrained_layout=True,
    )
    for i, season in enumerate(config.SEASONS):
        sub = df_fit_dry[df_fit_dry["season"] == season]
        for j, (p, lab) in enumerate(zip(params, labels)):
            ax = axes[i, j]
            ax.hist(sub[p].dropna().values, bins=bins_dict[p], edgecolor="black")
            ax.set_xlim(*xlims[p])
            ax.set_ylim(*ylims[p])
            if i == 0:
                ax.set_title(lab, fontsize=14)
            if j == 0:
                ax.set_ylabel(season.capitalize(), fontsize=13)
            if i == len(config.SEASONS) - 1:
                ax.set_xlabel("Value", fontsize=11)
            ax.grid(alpha=0.3)
    return fig


def make_fig5_wet_param_hist(df_fit_wet):
    """Notebook 03 Fig. 5 — cross-station histograms of wet-spell mixture params."""
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    params = ["pi", "p1", "p2"]
    labels = [r"$\hat \pi$", r"$\hat p_1$", r"$\hat p_2$"]
    xlims = {"pi": (0, 1), "p1": (0, 1), "p2": (0, 1)}

    bins_dict, ylims = {}, {}
    for p in params:
        xmin, xmax = xlims[p]
        bins = np.linspace(xmin, xmax, 21)
        bins_dict[p] = bins
        ymax = 0
        for season in config.SEASONS:
            vals = df_fit_wet.loc[df_fit_wet["season"] == season, p].dropna().values
            counts, _ = np.histogram(vals, bins=bins)
            ymax = max(ymax, counts.max())
        ylims[p] = (0, ymax * 1.05)

    fig, axes = plt.subplots(nrows=len(config.SEASONS), ncols=len(params),
                             figsize=(9, 7), constrained_layout=True)
    for i, season in enumerate(config.SEASONS):
        sub = df_fit_wet[df_fit_wet["season"] == season]
        for j, (p, lab) in enumerate(zip(params, labels)):
            ax = axes[i, j]
            ax.hist(sub[p].dropna().values, bins=bins_dict[p], edgecolor="black")
            ax.set_xlim(*xlims[p])
            ax.set_ylim(*ylims[p])
            if i == 0:
                ax.set_title(lab, fontsize=14)
            if j == 0:
                ax.set_ylabel(season.capitalize(), fontsize=13)
            if i == len(config.SEASONS) - 1:
                ax.set_xlabel("Value", fontsize=11)
            ax.grid(alpha=0.3)
    return fig


def make_fig13_pvalue_maps(spells, df_fit_dry, df_info):
    """Notebook 05 Fig. 13 — spatial maps of p-values per season (hdeGPD vs geometric)."""
    print("[fig13] running GoF on all stations (hdeGPD)...")
    dict_chi2_true, dict_p_true, dict_D_true, _ = goodness_of_fit_true_all_cities_seasons(
        data=spells, df_fit=df_fit_dry, nb_days_min_for_D=20, force_D=None, spell_type="dry",
    )
    df_gof_true_hdegpd = build_gof_results_df(dict_chi2_true, dict_p_true, dict_D_true, df_info)

    print("[fig13] running GoF on all stations (geometric baseline)...")
    dict_chi2_g, dict_p_g, dict_D_g, _, _ = goodness_of_fit_true_all_cities_seasons_geometric(
        data=spells, nb_days_min_for_D=20, force_D=None, spell_type="dry",
    )
    df_gof_true_geom = build_gof_results_df(dict_chi2_g, dict_p_g, dict_D_g, df_info)

    fig = plotting.plot_pvalue_maps_by_season(
        df_gof_bmcd=df_gof_true_hdegpd,
        df_gof_baseline=df_gof_true_geom,
        seasons=("spring", "summer", "autumn", "winter"),
        label_model="hdeGPD model",
        label_baseline="Geometric model",
        vmin=1e-7,
        zoom=3.5,
    )
    return fig


def make_figA_gof_simulation(df_fit_dry, *, n_rep=1000, n_per_rep=500, D_max=10):
    """Notebook 05 Fig. A (appendix) — null-distribution check on simulated data."""
    row = df_fit_dry.iloc[0]

    def gof_for_simulated(simulated_durations, f_1, xi_gpd, sigma_gpd, kappa_gpd, D):
        days, probas_emp, nb_days_state_list = get_proba_leaving_by_day(simulated_durations)
        cdf_fitted = make_cdf_fitted_hdeGPD_from_params(f_1, xi_gpd, sigma_gpd, kappa_gpd)
        fitted_probas_extgpd = np.array(
            [get_proba_leaving_state_n_kozu(cdf_fitted, d) for d in range(1, D)], dtype=float)
        probas_emp = np.asarray(probas_emp, dtype=float)[: D - 1]
        T = build_T_matrix(D, cdf_fitted)
        Sigma = build_Sigma_matrix_new(D, cdf_fitted)
        inv_matrix_mul = np.linalg.inv(T @ Sigma @ T.T)
        diff = probas_emp - fitted_probas_extgpd
        Q_n = float(len(simulated_durations) * diff.T @ inv_matrix_mul @ diff)
        p_val = 1.0 - scipy.stats.chi2.cdf(Q_n, df=D - 1)
        return Q_n, p_val

    Qs, ps = [], []
    print(f"[figA] running {n_rep} simulations of {n_per_rep} spells each...")
    for _ in tqdm(range(n_rep)):
        sim = [
            get_spell_length_degenerate_mixture_order_1_extgpd1(
                xi_gpd=row["xi"], sigma_gpd=row["sigma"], kappa_gpd=row["kappa"], f_1=row["f_1"]
            )
            for _ in range(n_per_rep)
        ]
        Q, p = gof_for_simulated(
            sim, row["f_1"], row["xi"], row["sigma"], row["kappa"], D=D_max + 1,
        )
        Qs.append(Q)
        ps.append(p)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7))
    x = np.linspace(0, 30, 200)
    ax1.hist(Qs, bins=40, density=True, color="grey", edgecolor="k")
    ax1.plot(x, chi2.pdf(x, D_max - 1), "k--")
    ax1.set_title(r"$Q_n$ histogram vs $\chi^2_{%d}$" % (D_max - 1))
    ax2.hist(ps, bins=20, density=True, color="grey", edgecolor="k")
    ax2.axhline(1, color="k", ls="--")
    ax2.set_title("p-values histogram")
    fig.tight_layout()
    return fig


def make_fig14_mean_residual_map(spells, df_fit_dry, df_info):
    """Notebook 06 Fig. 14 — mean residual duration maps (long dry spells, spring)."""
    list_cities = sorted(spells.keys())
    df_fit_dry_spring = df_fit_dry[df_fit_dry.season == "spring"].copy()
    list_d_thresh = [20, 40, 60]

    # hdeGPD mean excess at each threshold
    for d_thresh in list_d_thresh:
        list_mean_excess = []
        print(f"[fig14] hdeGPD mean-excess at d={d_thresh}...")
        for city in tqdm(list_cities):
            city_short = city.split()[0]
            sub = df_fit_dry_spring[df_fit_dry_spring.city == city_short]
            if sub.empty:
                list_mean_excess.append(np.nan)
                continue
            row = sub.iloc[0]
            list_mean_excess.append(make_approx_mean_excess(
                d_thresh=d_thresh,
                sigma=row["sigma"], kappa=row["kappa"], xi=row["xi"], f1=row["f_1"],
                target_precision=1e-5,
            ))
        # rebuild a column aligned to df_fit_dry_spring rows: only available for known city_short
        # match per-row by city_short
        col = []
        for _, frow in df_fit_dry_spring.iterrows():
            try:
                idx = list_cities.index(frow.city) if frow.city in list_cities else None
            except ValueError:
                idx = None
            col.append(list_mean_excess[idx] if idx is not None else np.nan)
        df_fit_dry_spring[f"mean_excess_{d_thresh}"] = col

    # Markov baseline mean excess at each threshold
    markov_by_city = {}
    month_season = config.LIST_MONTH_SEASONS[1]  # spring entry
    spring_months = set(month_season[:3])
    for d_thresh in list_d_thresh:
        markov_by_city[d_thresh] = {}
        print(f"[fig14] geometric Markov baseline at d={d_thresh}...")
        for city in tqdm(list_cities):
            full_year_data = spells[city]["dry_spell"]["duration_spell"]
            full_year_dates = spells[city]["dry_spell"]["start_date_spell"]
            months = np.array([str(s)[4:6] for s in full_year_dates])
            mask = np.array([m in spring_months for m in months])
            spring_data = np.array(full_year_data)[mask]
            if len(spring_data) == 0:
                markov_by_city[d_thresh][city] = np.nan
                continue
            p_geom_dry = 1 / np.mean(spring_data)
            markov_by_city[d_thresh][city] = mean_excess_markov_order_1(p_geom_dry, d_thresh)

    # Map markov baseline back onto df_fit_dry_spring (matched by city short name)
    short_to_full = {c.split()[0]: c for c in list_cities}
    for d_thresh in list_d_thresh:
        df_fit_dry_spring[f"mean_excess_{d_thresh}_markov_baseline"] = [
            markov_by_city[d_thresh].get(short_to_full.get(c), np.nan)
            for c in df_fit_dry_spring["city"]
        ]

    # Build the map.
    df_info_local = df_info.copy()
    df_info_local["city"] = df_info_local["city"].map(lambda s: s.split()[0])
    needed_cols = ["city", "lat", "lon"]
    for d in list_d_thresh:
        needed_cols += [f"mean_excess_{d}", f"mean_excess_{d}_markov_baseline"]
    df = df_fit_dry_spring.merge(df_info_local, on="city")
    df = df[needed_cols].dropna(subset=["lat", "lon"]).copy()

    all_vals = []
    for d in list_d_thresh:
        all_vals.extend(df[f"mean_excess_{d}"].dropna().values)
        all_vals.extend(df[f"mean_excess_{d}_markov_baseline"].dropna().values)
    all_vals = np.array(all_vals, dtype=float)
    all_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]

    label_model = "BMCD model"
    label_baseline = "Geometric model"
    zoom = 3.5
    lat_min, lat_max = df["lat"].min(), df["lat"].max()
    lon_min, lon_max = df["lon"].min(), df["lon"].max()
    fig = make_subplots(
        rows=len(list_d_thresh), cols=1,
        specs=[[{"type": "scattermapbox"}] for _ in list_d_thresh],
        subplot_titles=[f"Mean residual duration after {d} dry days" for d in list_d_thresh],
        vertical_spacing=0.02,
    )
    for i, d_thresh in enumerate(list_d_thresh, start=1):
        col_model = f"mean_excess_{d_thresh}"
        col_base = f"mean_excess_{d_thresh}_markov_baseline"
        model_vals = df[col_model].astype(float)
        base_vals = df[col_base].astype(float)
        fig.add_trace(go.Scattermapbox(
            lon=df["lon"], lat=df["lat"], mode="markers",
            name=label_baseline if i == 1 else None, showlegend=(i == 1),
            marker=dict(size=16, symbol="circle", color=base_vals,
                        coloraxis="coloraxis", opacity=1),
            text=[f"City: {city}<br>{label_baseline}: {val:.2f} days<br>Threshold: {d_thresh}"
                  for city, val in zip(df["city"], base_vals)],
            hovertemplate="%{text}<extra></extra>",
        ), row=i, col=1)
        fig.add_trace(go.Scattermapbox(
            lon=df["lon"], lat=df["lat"], mode="markers",
            name=label_model if i == 1 else None, showlegend=(i == 1),
            marker=dict(size=8, symbol="circle", color=model_vals,
                        coloraxis="coloraxis", opacity=1),
            text=[f"City: {city}<br>{label_model}: {val:.2f} days<br>Threshold: {d_thresh}"
                  for city, val in zip(df["city"], model_vals)],
            hovertemplate="%{text}<extra></extra>",
        ), row=i, col=1)

    center_lat, center_lon = (lat_min + lat_max * 1.1) / 2, (lon_min + lon_max) / 2
    for i in range(1, len(list_d_thresh) + 1):
        map_name = "mapbox" if i == 1 else f"mapbox{i}"
        fig.update_layout({map_name: dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon), zoom=zoom,
        )})
    vmin = 0.1
    fig.update_layout(
        height=1200, width=900, margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=0.001, xanchor="center", x=0.81),
        coloraxis=dict(
            cmin=vmin, cmax=100,
            colorscale=[[0, "blue"], [0.5, "yellow"], [0.75, "orange"], [1, "red"]],
            colorbar=dict(title="Mean<br>residual<br>duration <br>(days)"),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_appendix_figA(fig) -> "config.Path":
    """Fig. A is the only figure with a non-integer ID; save it with a
    parallel layout (figures/figure_A/<filename>.pdf) without changing the
    int-typed `_save_per_figure` helper."""
    out_dir = config.ROOT / "figures" / "figure_A"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "gof_Qn_and_pvalues_simulated_data.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    return out


# (figure_id, builder, save_fn, kwargs-keys-needed)
FIGURE_TASKS = [
    ( 1,  make_fig1_map,
          lambda fig: plotting._save_plotly_per_figure(fig,  1, "map_palermo.pdf"),
          ("spells", "df_stations")),
    ( 4,  make_fig4_dry_param_hist,
          lambda fig: plotting._save_per_figure(fig,        4, "histogram_dry_spell_distrib_params.pdf"),
          ("df_fit_dry",)),
    ( 5,  make_fig5_wet_param_hist,
          lambda fig: plotting._save_per_figure(fig,        5, "histogram_wet_spell_distrib_params.pdf"),
          ("df_fit_wet",)),
    (13,  make_fig13_pvalue_maps,
          lambda fig: plotting._save_plotly_per_figure(fig, 13, "pvalue_maps_by_season_geom_vs_bmcd.pdf"),
          ("spells", "df_fit_dry", "df_info")),
    ("A", make_figA_gof_simulation,
          _save_appendix_figA,
          ("df_fit_dry",)),
    (14,  make_fig14_mean_residual_map,
          lambda fig: plotting._save_plotly_per_figure(fig, 14, "mean_residual_duration_long_dry_spells.pdf"),
          ("spells", "df_fit_dry", "df_info")),
]


def _output_path(fig_id) -> "config.Path":
    """Mirrors the path each save_fn writes to (used by --skip-existing)."""
    suffix = {
        1:   "map_palermo.pdf",
        4:   "histogram_dry_spell_distrib_params.pdf",
        5:   "histogram_wet_spell_distrib_params.pdf",
        13:  "pvalue_maps_by_season_geom_vs_bmcd.pdf",
        "A": "gof_Qn_and_pvalues_simulated_data.pdf",
        14:  "mean_residual_duration_long_dry_spells.pdf",
    }[fig_id]
    return config.ROOT / "figures" / f"figure_{fig_id}" / suffix


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _load_inputs():
    json_filename = (f"ecad_data_south_europe_filtered_after_{config.START_YEAR}"
                     f"_wet_day_thresh_{config.WET_DAY_THRESHOLD}.json")
    with open(config.EXPORTS_JSON_DIR / json_filename) as fh:
        spells = json.load(fh)

    fit_folder = config.RESULTS_FIT_DIR / f"fit_south_europe_subset_excess_over_{config.WET_DAY_THRESHOLD}"
    df_fit_dry = pd.read_csv(fit_folder / "dry_spell_fit_egpd1_excess_over_1result_fit_parameters.csv")
    df_fit_wet = pd.read_csv(fit_folder / "wet_spell_fit_mixt_geomresult_fit_parameters.csv")
    df_fit_dry["city"] = df_fit_dry["data_source"].map(lambda s: s.split()[0])
    df_fit_dry["season"] = df_fit_dry["data_source"].map(lambda s: s.split()[-1])
    df_fit_wet["city"] = df_fit_wet["data_source"].map(lambda s: s.split()[0])
    df_fit_wet["season"] = df_fit_wet["data_source"].map(lambda s: s.split()[-1])

    df_stations = pd.read_csv(config.STATION_METADATA_CSV)
    df_info = df_stations[df_stations.city.isin(sorted(spells.keys()))]

    print(f"[load] {len(spells)} stations from JSON, {len(df_fit_dry)} dry-fit rows, "
          f"{len(df_fit_wet)} wet-fit rows")

    return {
        "spells": spells,
        "df_fit_dry": df_fit_dry,
        "df_fit_wet": df_fit_wet,
        "df_stations": df_stations,
        "df_info": df_info,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", default=None,
                    help="Build only these figure IDs (e.g. --only 4 5 A).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip a figure if its PDF already exists.")
    args = ap.parse_args()

    # Normalise: integer IDs stay int, "A" stays str
    selected = None
    if args.only:
        def _norm(s):
            return int(s) if s.lstrip("-").isdigit() else s
        selected = [_norm(s) for s in args.only]
        known = {fig_id for fig_id, *_ in FIGURE_TASKS}
        unknown = [s for s in selected if s not in known]
        if unknown:
            print(f"[error] unknown figure id(s): {unknown}", file=sys.stderr)
            return 2

    ctx = _load_inputs()

    fail_count = 0
    skip_count = 0
    for fig_id, builder, save_fn, kw_keys in FIGURE_TASKS:
        if selected is not None and fig_id not in selected:
            continue
        if args.skip_existing and _output_path(fig_id).exists():
            print(f"[skip] figure_{fig_id}: already exists")
            skip_count += 1
            continue
        print(f"[build] figure_{fig_id} ...")
        try:
            kwargs = {k: ctx[k] for k in kw_keys}
            fig = builder(**kwargs)
            out = save_fn(fig)
            print(f"[save]  -> {out}")
            if hasattr(fig, "savefig"):  # matplotlib only
                plt.close(fig)
        except Exception:
            fail_count += 1
            print(f"[FAIL] figure_{fig_id}:\n{traceback.format_exc()}", file=sys.stderr)
            plt.close("all")

    print(f"[done] failures: {fail_count}, skipped: {skip_count}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
