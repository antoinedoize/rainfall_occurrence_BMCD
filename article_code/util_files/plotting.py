"""Figure helpers.

Re-exports existing plotters from `rainfall_article_clean/figures.py` and
provides a small set of thin wrappers that save to `config.FIGURES_DIR` with
the exact filename expected by the article's LaTeX source.

Each wrapper:
- takes a single-station / single-season input,
- builds the matplotlib or Plotly figure,
- saves to `config.FIGURES_DIR / <exact article filename>`.

These wrappers are intentionally thin so the notebooks read top-to-bottom.
"""
from __future__ import annotations

from pathlib import Path
from . import config
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# qqplots
def make_qq_plot_dry_spell(
    ax,
    data_name,
    concatenation_neg_exc,
    generator_one_exc_dry,
    params_vals,
    param_names,
    n_vectors_simulated_data=50,
    alpha_over_2_uncertainty_1=0.025):
    
    real_data = np.array(concatenation_neg_exc)
    list_data_simulated = [[generator_one_exc_dry() for i in range(len(real_data))] for i in range(n_vectors_simulated_data)]
    sorted_data_simulated = [sorted(data_list) for data_list in list_data_simulated]
    list_simulated_per_rank = [[sorted_data_list[i] for sorted_data_list in sorted_data_simulated] for i in range(len(real_data))]
    quantile_alpha_over_2_lower1 = np.array([np.quantile(list_values_constant_rank, alpha_over_2_uncertainty_1) 
                                    for list_values_constant_rank in list_simulated_per_rank])
    quantile_alpha_over_2_higher1 = np.array([np.quantile(list_values_constant_rank, 1 - alpha_over_2_uncertainty_1) 
                                     for list_values_constant_rank in list_simulated_per_rank])
    simu_to_be_plot = sorted([generator_one_exc_dry() for i in range(len(real_data))])
    real_data = sorted(real_data)
    min_tot = np.min([simu_to_be_plot[0], real_data[0]])
    max_tot = np.max([simu_to_be_plot[-1], real_data[-1]])
    ax.plot([min_tot, max_tot], [min_tot, max_tot], color='black')
    pairs = np.column_stack((real_data, simu_to_be_plot))          # shape (n, 2)
    uniq_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
    base_area = 6.0          # area (points^2) for a count of 1
    max_area  = 400.0        # optional cap to avoid huge bubbles
    sizes = base_area * np.sqrt(counts )                              # area proportional to counts
    sizes = np.minimum(sizes, max_area)                      # optional cap
    ax.scatter(
        uniq_pairs[:, 0], uniq_pairs[:, 1],
        s=sizes,
    color="orange",
    # alpha=0.9,
    linewidths=1.0)
    ax.fill_between(real_data, quantile_alpha_over_2_lower1, quantile_alpha_over_2_higher1, alpha=0.3,
                    color="orange")
    textstr = '\n'.join((name + r'=%.2f' % (val, ) for (name, val) in zip(param_names,params_vals)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=18, verticalalignment='top', bbox=props)
    ax.set_xlabel("Real durations (days)")
    ax.set_ylabel("Simulated durations (days)")
    ax.set_title(f"{data_name}"+r" $\tau^{(0)}$")


def make_qq_plot_with_uncertainty_areas_wet_days_duration_vs_simu(
    ax,
    data_name,
    wet_durations, 
    generator_one_exc_wet,   
    params_distribution, 
    n_vectors_simulated_data=50,
    alpha_over_2_uncertainty_1=0.025):
    real_data = np.asarray(wet_durations, dtype=float)
    list_data_simulated = [[generator_one_exc_wet() for _ in range(len(real_data))]
        for _ in range(n_vectors_simulated_data)]
    sorted_data_simulated = [sorted(v) for v in list_data_simulated]
    list_simulated_per_rank = [[sorted_vec[i] for sorted_vec in sorted_data_simulated]
        for i in range(len(real_data))]
    quantile_alpha_over_2_lower1 = np.array([
        np.quantile(vals, alpha_over_2_uncertainty_1) for vals in list_simulated_per_rank])
    quantile_alpha_over_2_higher1 = np.array([
        np.quantile(vals, 1 - alpha_over_2_uncertainty_1) for vals in list_simulated_per_rank])
    simu_to_be_plot = sorted([generator_one_exc_wet() for _ in range(len(real_data))])
    real_sorted = sorted(real_data)
    min_tot = np.min([simu_to_be_plot[0], real_sorted[0]])
    max_tot = np.max([simu_to_be_plot[-1], real_sorted[-1]])
    ax.plot([min_tot, max_tot], [min_tot, max_tot], color="black")
    pairs = np.column_stack((real_sorted, simu_to_be_plot))          # shape (n, 2)
    uniq_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
    base_area = 6.0          # area (points^2) for a count of 1
    max_area  = 400.0        # optional cap to avoid huge bubbles
    sizes = base_area * np.sqrt(counts)                              # area proportional to counts
    sizes = np.minimum(sizes, max_area)                      # optional cap
    ax.scatter(uniq_pairs[:, 0], uniq_pairs[:, 1],s=sizes,color="blue",linewidths=1.0)
    ax.fill_between(real_sorted,quantile_alpha_over_2_lower1,quantile_alpha_over_2_higher1,alpha=0.4)
    txt = "\n".join([r"$\hat{\pi}=%.2f$" % params_distribution["pi"],
                r"$\hat{p}_1=%.2f$" % params_distribution["p1"],
                r"$\hat{p}_2=%.2f$" % params_distribution["p2"]])
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(0.05, 0.95, txt,transform=ax.transAxes,fontsize=18,verticalalignment="top",bbox=props)
    ax.set_xlabel("Real durations (days)")
    ax.set_ylabel("Simulated durations (days)")
    ax.set_title(f"{data_name}" + r" $\tau^{(1)}$")


# map_pvalues

def plot_pvalue_maps_by_season(
    df_gof_bmcd, df_gof_baseline, seasons=config.SEASONS,
    label_model="BMCD model", label_baseline="Geometric model",
    vmin=0.0, vmax=1.0, zoom=3.5, big_size=16, small_size=8,
    vertical_spacing=0.03, height_per_row=280, width=900,):
    cols = ["city", "season", "lat", "lon", "p_value"]
    df_model = df_gof_bmcd[cols].rename(columns={"p_value": "p_value_model"}).copy()
    df_base = df_gof_baseline[cols].rename(columns={"p_value": "p_value_baseline"}).copy()
    for df in (df_model, df_base):
        df["city"] = df["city"].astype(str).str.strip()
        df["season"] = df["season"].astype(str).str.lower().str.strip()
    df = df_model.merge(
        df_base[["city", "season", "p_value_baseline"]],
        on=["city", "season"],
        how="inner",
        validate="one_to_one")
    df = df.dropna(subset=["lat", "lon"]).copy()
    # Keep values in the valid p-value range
    df["p_value_model"] = df["p_value_model"].astype(float).clip(lower=vmin, upper=vmax)
    df["p_value_baseline"] = df["p_value_baseline"].astype(float).clip(lower=vmin, upper=vmax)
    center_lat, center_lon = df["lat"].mean(), df["lon"].mean()
    fig = make_subplots(
        rows=len(seasons), cols=1,
        specs=[[{"type": "scattermapbox"}] for _ in seasons],
        subplot_titles=[s.capitalize() for s in seasons],
        vertical_spacing=vertical_spacing)
    for i, season in enumerate(seasons, start=1):
        sub = df[df["season"] == season].copy()
        if sub.empty:
            continue
        model_vals,base_vals = sub["p_value_model"], sub["p_value_baseline"]
        fig.add_trace(
            go.Scattermapbox(
                lon=sub["lon"], lat=sub["lat"], mode="markers",
                name=label_baseline if i == 1 else None,
                showlegend=(i == 1),
                marker=dict(size=big_size,symbol="circle",color=base_vals,           # raw p-values
                    coloraxis="coloraxis",opacity=0.95),
                text=[
                    f"City: {c}<br>Season: {season.capitalize()}<br>{label_baseline}: {v:.4g}"
                    for c, v in zip(sub["city"], base_vals)],hovertemplate="%{text}<extra></extra>",),row=i, col=1,)
        fig.add_trace(
            go.Scattermapbox(
                lon=sub["lon"], lat=sub["lat"], mode="markers",name=label_model if i == 1 else None,
                showlegend=(i == 1),
                marker=dict(size=small_size,symbol="circle",color=model_vals,          # raw p-values
                    coloraxis="coloraxis",opacity=1),
                text=[
                    f"City: {c}<br>Season: {season.capitalize()}<br>{label_model}: {v:.4g}"
                    for c, v in zip(sub["city"], model_vals)],
                hovertemplate="%{text}<extra></extra>",),row=i, col=1,)

    for i in range(1, len(seasons) + 1):
        fig.update_layout({
            ("mapbox" if i == 1 else f"mapbox{i}"): dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=zoom)})

    # Position 0.05 on the normalized [0,1] colorscale
    red_threshold = 0.05
    red_threshold_pos = (red_threshold - vmin) / (vmax - vmin)

    # Custom colorscale:
    # - red from 0 to 0.05
    # - then transition red -> orange -> yellow -> light green -> green up to 1
    pvalue_colorscale = [[0.00, "brown"],[0.01, "red"],[0.05, "orange"],
        [0.50, "yellow"],[0.75, "yellowgreen"],[1.00, "green"],]
    fig.update_layout(
        height=height_per_row * len(seasons),width=width,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h",yanchor="bottom",y=0.001,xanchor="center",x=0.81),
        coloraxis=dict(cmin=vmin,cmax=vmax,colorscale=pvalue_colorscale,
            colorbar=dict(title="p-value",tickvals=[0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
                ticktext=["0", "0.01", "0.05", "0.1", "0.25", "0.5", "0.75", "1"]),),)
    return fig


# Saving functions


def _save(fig, filename: str, *, dpi: int = 200) -> Path:
    """Save a matplotlib Figure to config.FIGURES_DIR/filename, and returns the Path."""
    out = config.FIGURES_DIR / filename
    fig.savefig(out, bbox_inches="tight", dpi=dpi)
    return out


def _save_plotly(fig, filename: str) -> Path:
    """Save a Plotly Figure to config.FIGURES_DIR/filename (PDF via kaleido)."""
    out = config.FIGURES_DIR / filename
    fig.write_image(str(out))
    return out


# Explicit article-filename wrappers. Each one takes a prebuilt figure (so the
# notebook keeps all plotting logic inline and readable) and saves with the
# canonical name. Notebook cells compute the figure, then call these.

def save_map_palermo(fig_plotly) -> Path:
    return _save_plotly(fig_plotly, "map_palermo.pdf")

def save_palermo_survival_overlay(fig) -> Path:
    return _save(fig, f"{config.STATION_EXAMPLE}_comparison_dry_spell_egpd_geom_4seasons.pdf")

def save_toy_data(fig) -> Path:
    return _save(fig, "toy_data_fig_with_N.pdf")

def save_dry_param_hist(fig) -> Path:
    return _save(fig, "histogram_dry_spell_distrib_params.pdf")

def save_wet_param_hist(fig) -> Path:
    return _save(fig, "histogram_wet_spell_distrib_params.pdf")

def save_palermo_bivariate_acf(fig) -> Path:
    return _save(fig, f"{config.STATION_EXAMPLE}_bivariate_acf_plot.pdf")

def save_palermo_dry_hist(fig) -> Path:
    return _save(fig, f"{config.STATION_EXAMPLE}_histogram_dry_spell_fit.pdf")

def save_palermo_wet_hist(fig) -> Path:
    return _save(fig, f"{config.STATION_EXAMPLE}_histogram_wet_spell_fit.pdf")

def save_palermo_dry_qq(fig) -> Path:
    return _save(fig, f"{config.STATION_EXAMPLE}_qqplot_dry_spell_fit.pdf")

def save_palermo_wet_qq(fig) -> Path:
    return _save(fig, f"{config.STATION_EXAMPLE}_qqplot_wet_spell_fit.pdf")

def save_palermo_exit_prob(fig) -> Path:
    return _save(fig, f"{config.STATION_EXAMPLE}_proba_leaving_state.pdf")

def save_belgrade_exit_prob(fig) -> Path:
    return _save(fig, f"BELGRADE_proba_leaving_state.pdf")

def save_pvalue_hist(fig, dmax: int = 20) -> Path:
    return _save(fig, f"hist_p_values_south_europe_geom_vs_egpd_dmax_{dmax}.pdf")

def save_pvalue_maps_by_season(fig_plotly) -> Path:
    return _save_plotly(fig_plotly, "pvalue_maps_by_season_geom_vs_bmcd.pdf")

def save_gof_simulation(fig) -> Path:
    return _save(fig, "gof_Qn_and_pvalues_simulated_data.pdf")

def save_mean_residual_map(fig_plotly) -> Path:
    return _save_plotly(fig_plotly, "mean_residual_duration_long_dry_spells.pdf")


# --- Per-station batch save helpers -----------------------------------------
# These save into `figures/figure_{N}/{station}_{name}.pdf` rather than the
# single-station `figures/{STATION}/...` layout used above. Used by the
# per-station driver `run_all_stations_figures.py` to write one PDF per
# (figure, station) pair across the whole ECAD subset.

def _safe_filename(station: str) -> str:
    """Replace path separators in a station name so it's usable as a filename component."""
    return station.replace("/", "_").replace("\\", "_")


def _save_per_figure(fig, fig_num: int, filename: str, *, dpi: int = 200) -> Path:
    out_dir = config.ROOT / "figures" / f"figure_{fig_num}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / filename
    fig.savefig(out, bbox_inches="tight", dpi=dpi)
    return out


def _save_plotly_per_figure(fig_plotly, fig_num: int, filename: str) -> Path:
    """Plotly counterpart of `_save_per_figure` (PDF via kaleido)."""
    out_dir = config.ROOT / "figures" / f"figure_{fig_num}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / filename
    fig_plotly.write_image(str(out))
    return out

def save_survival_overlay_for_station(fig, station: str) -> Path:
    return _save_per_figure(fig, 2, f"{_safe_filename(station)}_comparison_dry_spell_egpd_geom_4seasons.pdf")

def save_bivariate_acf_for_station(fig, station: str) -> Path:
    return _save_per_figure(fig, 6, f"{_safe_filename(station)}_bivariate_acf_plot.pdf")

def save_dry_hist_for_station(fig, station: str) -> Path:
    return _save_per_figure(fig, 7, f"{_safe_filename(station)}_histogram_dry_spell_fit.pdf")

def save_wet_hist_for_station(fig, station: str) -> Path:
    return _save_per_figure(fig, 8, f"{_safe_filename(station)}_histogram_wet_spell_fit.pdf")

def save_dry_qq_for_station(fig, station: str) -> Path:
    return _save_per_figure(fig, 9, f"{_safe_filename(station)}_qqplot_dry_spell_fit.pdf")

def save_wet_qq_for_station(fig, station: str) -> Path:
    return _save_per_figure(fig, 10, f"{_safe_filename(station)}_qqplot_wet_spell_fit.pdf")

def save_exit_prob_for_station(fig, station: str) -> Path:
    return _save_per_figure(fig, 11, f"{_safe_filename(station)}_proba_leaving_state.pdf")


def save_stationarity_for_station(fig, station: str, spell_type: str, *, dpi: int = 200) -> Path:
    """Save the stationarity-check figure to figures/stationnarity/<station>_<spell_type>_duration_stationarity.pdf."""
    out_dir = config.ROOT / "figures" / "stationnarity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{_safe_filename(station)}_{spell_type}_duration_stationarity.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=dpi)
    return out
