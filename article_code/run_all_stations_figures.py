"""Per-station batch driver — Figs. 2, 6-11 for every station in the JSON export.

Output layout:
    figures/figure_2/{STATION}_comparison_dry_spell_egpd_geom_4seasons.pdf
    figures/figure_6/{STATION}_bivariate_acf_plot.pdf
    figures/figure_7/{STATION}_histogram_dry_spell_fit.pdf
    figures/figure_8/{STATION}_histogram_wet_spell_fit.pdf
    figures/figure_9/{STATION}_qqplot_dry_spell_fit.pdf
    figures/figure_10/{STATION}_qqplot_wet_spell_fit.pdf
    figures/figure_11/{STATION}_proba_leaving_state.pdf

Usage (from the repo root):
    python -m article_code.run_all_stations_figures                   # all stations
    python -m article_code.run_all_stations_figures --first 5         # first 5 stations
    python -m article_code.run_all_stations_figures --only PALERMO    # one station (smoke test)
    python -m article_code.run_all_stations_figures --skip-existing   # don't overwrite

The figure-building cells are lifted near-verbatim from
`notebooks_article_pipeline/02_introduction_plots.ipynb` (Fig. 2) and
`notebooks_article_pipeline/04_palermo_diagnostics.ipynb` (Figs. 6-11), with
the hardcoded `NAME_STATION_EXAMPLE` swapped for the `station` argument.
"""
import argparse
import json
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .util_files import config, plotting
from .util_files.spell_models import (
    S_hat,
    fit_geometric_p_mle,
    get_spell_length_degenerate_mixture_order_1_extgpd1,
    make_cdf_fitted_hdeGPD_from_params,
    mix_geom_pmf,
    sample_geom_mix,
)
from .util_files.statistics import (
    build_data_per_city_per_season_per_year_couple_vector_duration_vector_date,
    get_proba_leaving_by_day,
    get_proba_leaving_state_n_kozu,
    pooled_bivariate_autocorr,
    split_spells_by_season_simple,
)


# ---------------------------------------------------------------------------
# Figure builders. Each returns (fig, save_fn) so the caller can save+close.
# Faithful port of the corresponding notebook cell.
# ---------------------------------------------------------------------------

def _fit_row_dry(df_fit_dry, station, season):
    """Look up the dry-spell fit row for (station, season). Raises if missing."""
    data_name = f"{station} {season}"
    sub = df_fit_dry[df_fit_dry["data_source"] == data_name]
    if sub.empty:
        raise LookupError(f"no dry fit row for '{data_name}'")
    return sub.iloc[0]


def _fit_row_wet(df_fit_wet, station, season):
    data_name = f"{station} {season}"
    sub = df_fit_wet[df_fit_wet["data_source"] == data_name]
    if sub.empty:
        raise LookupError(f"no wet fit row for '{data_name}'")
    return sub.iloc[0]


def make_fig2_survival_overlay(station, spells, df_fit_dry):
    """Notebook 02 Fig. 2 — empirical vs geometric vs hdeGPD survival overlay."""
    dry_durations = spells[station]["dry_spell"]["duration_spell"]
    dry_dates = spells[station]["dry_spell"]["start_date_spell"]
    months = np.array([str(s)[4:6] for s in dry_dates])

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    for ax, vec_month_seasons in zip(axes.flat, config.LIST_MONTH_SEASONS[1:]):
        season = vec_month_seasons[3]
        month_set = set(vec_month_seasons)
        season_mask = np.array([m in month_set for m in months])
        concatenation_neg_exc = np.array(dry_durations)[season_mask]

        row = _fit_row_dry(df_fit_dry, station, season)
        cdf_egpd = make_cdf_fitted_hdeGPD_from_params(
            row["f_1"], row["xi"], row["sigma"], row["kappa"],
        )

        min_season = int(np.min(concatenation_neg_exc))
        max_season = int(np.max(concatenation_neg_exc))
        x_min_plot = max(1, min_season)
        x_full = np.arange(x_min_plot, max_season + 1)
        x_ext = np.arange(max_season, 3 * max_season + 1)

        f_emp_array = np.array([S_hat(concatenation_neg_exc, int(x))[0] for x in x_full])
        errors_f_emp_array = np.array([S_hat(concatenation_neg_exc, int(x))[1] for x in x_full])

        p_geom = fit_geometric_p_mle(concatenation_neg_exc)
        q_leave = 1 - p_geom
        left_x = x_min_plot

        ax.plot(x_full, f_emp_array, label="Empirical survival function", color="orange")
        ax.plot(x_full, [1 - cdf_egpd(int(x) - 1) for x in x_full],
                label="Suggested survival function", color="black")
        ax.plot(x_ext, [1 - cdf_egpd(int(x) - 1) for x in x_ext],
                linestyle="dashed", color="black")
        ax.plot(x_full, [q_leave ** (int(x) - 1) for x in x_full],
                label="Geometric survival function", color="red", alpha=0.8)
        ax.plot(x_ext, [q_leave ** (int(x) - 1) for x in x_ext],
                linestyle="dashed", color="red", alpha=0.8)

        mask = x_full >= left_x
        ax.fill_between(
            x_full[mask],
            (f_emp_array - errors_f_emp_array)[mask],
            (f_emp_array + errors_f_emp_array)[mask],
            alpha=0.25, linewidth=0.1, color="orange",
        )

        ax.set_title(f"{station} {season}")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("Duration of dry spell (days)")
        ax.set_ylabel("Survival probability")
        ax.set_xlim((max(0.9 * left_x, 0.9), int(3 * max_season)))
        ax.set_ylim((1e-8, 1.1))
        if season == "spring":
            ax.legend()

    fig.tight_layout()
    return fig


def make_fig6_bivariate_acf(station, data_per_city):
    """Notebook 04 Fig. 6 — pooled bivariate ACF of (dry, wet) durations."""
    L = 20
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size=14)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)

    components = [
        (r"$\tau_0$ vs $\tau_0$", (0, 0), "o", "-",  "orange"),
        (r"$\tau_0$ vs $\tau_1$", (0, 1), "s", "--", "black"),
        (r"$\tau_1$ vs $\tau_0$", (1, 0), "^", "-.", "black"),
        (r"$\tau_1$ vs $\tau_1$", (1, 1), "D", ":",  "blue"),
    ]

    for j, season in enumerate(config.SEASONS):
        ax = axes[j // 2, j % 2]
        data_city_season_by_year = data_per_city[station][season]
        try:
            R, Gamma, N_l, bands, Vbar = pooled_bivariate_autocorr(data_city_season_by_year, L)
        except ValueError as exc:
            ax.text(0.5, 0.5, f"Insufficient data\n({exc})",
                    transform=ax.transAxes, ha="center", va="center", fontsize=12)
            ax.set_title(f"{station} {season}")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        lags = np.arange(R.shape[0])
        upper = bands
        ax.fill_between(lags, -upper, upper, color="grey", alpha=0.25,
                        label=r"$\pm 2/\sqrt{N_\ell}$" if j == 1 else None)
        for label, (a, b), marker, ls, col in components:
            y = R[:, a, b]
            ax.plot(lags, y, linestyle=ls, linewidth=1.0, color="black", alpha=0.7)
            ax.scatter(lags, y, marker=marker, s=30, facecolors=col, edgecolors=col,
                       label=label if j == 1 else None)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_title(f"{station} {season}")
        ax.set_xlabel("Lag (number of spell-cycle steps)")
        ax.set_ylabel("Correlation")
    axes[0, 1].legend(loc="right", fontsize=12, ncol=1, frameon=True)
    fig.tight_layout()
    return fig


def make_fig7_dry_hist(station, spells, df_fit_dry):
    """Notebook 04 Fig. 7 — dry-spell duration histogram with hdeGPD pmf overlay."""
    dry_spells = spells[station]["dry_spell"]["duration_spell"]
    dry_dates = spells[station]["dry_spell"]["start_date_spell"]
    months = np.array([str(s)[4:6] for s in dry_dates])

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    for ax, vec_month_seasons in zip(axes.flat, config.LIST_MONTH_SEASONS[1:]):
        season = vec_month_seasons[3]
        month_set = set(vec_month_seasons)
        season_mask = np.array([m in month_set for m in months])
        concatenation_neg_exc = np.array(dry_spells)[season_mask]
        data_name = f"{station} {season}"

        row = _fit_row_dry(df_fit_dry, station, season)
        xi, sigma, kappa = row["xi"], row["sigma"], row["kappa"]
        min_season, max_season = int(np.min(concatenation_neg_exc)), int(np.max(concatenation_neg_exc))
        days, counts = np.unique(concatenation_neg_exc, return_counts=True)
        freqs = counts / np.sum(counts)
        ax.bar(days + 0.5, freqs, width=1, label="Recorded durations histogram",
               color="orange", alpha=0.8)
        f_1 = sum(s == 1 for s in concatenation_neg_exc) / len(concatenation_neg_exc)
        cdf_fitted = make_cdf_fitted_hdeGPD_from_params(f_1, xi, sigma, kappa)

        list_k, list_vals = [], []
        for k in range(1, max_season - 1):
            list_k += [k, k + 1]
            current_val = cdf_fitted(k) - cdf_fitted(k - 1)
            list_vals += [current_val, current_val]

        params_vals = [f_1, xi, sigma, kappa]
        param_names = [r"$\hat f_1$", r"$\hat \xi$", r"$\hat \sigma$", r"$\hat \kappa$"]
        ax.plot(list_k, list_vals, label="hdeGPD pmf", color="black")
        textstr = "\n".join(name + r"=%.2f" % (val,) for name, val in zip(param_names, params_vals))
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(0.65, 0.65, textstr, transform=ax.transAxes, fontsize=18,
                verticalalignment="top", bbox=props)
        if season == "spring":
            ax.legend(loc="upper center", fontsize=13.5)
        ax.set_xlabel("Duration (days)")
        ax.set_title(data_name + r" $\tau{(0)}$")

    fig.tight_layout()
    return fig


def make_fig8_wet_hist(station, spells, df_fit_wet):
    """Notebook 04 Fig. 8 — wet-spell duration histogram with mixture-geom pmf overlay."""
    wet_spells = spells[station]["wet_spell"]["duration_spell"]
    wet_dates = spells[station]["wet_spell"]["start_date_spell"]
    months = np.array([str(s)[4:6] for s in wet_dates])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, vec_month_seasons in zip(axes.flat, config.LIST_MONTH_SEASONS[1:]):
        season = vec_month_seasons[3]
        month_set = set(vec_month_seasons)
        season_mask = np.array([m in month_set for m in months])
        concatenation_neg_exc = np.array(wet_spells)[season_mask]
        data_name = f"{station} {season}"

        days, counts = np.unique(concatenation_neg_exc, return_counts=True)
        freqs = counts / np.sum(counts)
        ax.bar(days + 0.5, freqs, width=1, label="Recorded durations histogram",
               color="blue", alpha=0.8)

        est = _fit_row_wet(df_fit_wet, station, season).to_dict()

        k_max = int(np.max(wet_spells))
        ks = np.arange(1, k_max + 1)
        pmf_fit = mix_geom_pmf(ks, est["pi"], est["p1"], est["p2"])

        list_k, list_vals = [], []
        for k, v in zip(ks, pmf_fit):
            list_k += [k, k + 1]
            list_vals += [v, v]

        ax.plot(list_k, list_vals, label="Geometric mixture pmf", color="black", linewidth=2)

        txt = "\n".join([
            r"$\hat{\pi}=%.2f$" % est["pi"],
            r"$\hat{p}_1=%.2f$" % est["p1"],
            r"$\hat{p}_2=%.2f$" % est["p2"],
        ])
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(0.45, 0.65, txt, transform=ax.transAxes, fontsize=18,
                verticalalignment="top", bbox=props)

        if season == "spring":
            ax.legend(loc="upper center", fontsize=13.5)
        ax.set_xlabel("Duration (days)")
        ax.set_title(data_name + r" $\tau^{(1)}$")
    fig.tight_layout()
    return fig


def make_fig9_dry_qq(station, spells, df_fit_dry):
    """Notebook 04 Fig. 9 — dry-spell Q-Q plot with bootstrap envelope."""
    dry_spells = spells[station]["dry_spell"]["duration_spell"]
    dry_dates = spells[station]["dry_spell"]["start_date_spell"]
    months = np.array([str(s)[4:6] for s in dry_dates])

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    for ax, vec_month_seasons in zip(axes.flat, config.LIST_MONTH_SEASONS[1:]):
        season = vec_month_seasons[3]
        month_set = set(vec_month_seasons)
        season_mask = np.array([m in month_set for m in months])
        concatenation_neg_exc = np.array(dry_spells)[season_mask]
        data_name = f"{station} {season}"

        row = _fit_row_dry(df_fit_dry, station, season)
        xi, sigma, kappa = row["xi"], row["sigma"], row["kappa"]
        f_1 = sum(s == 1 for s in concatenation_neg_exc) / len(concatenation_neg_exc)
        generator_one_exc_dry = (lambda xi=xi, sigma=sigma, kappa=kappa, proba1=f_1:
                                 get_spell_length_degenerate_mixture_order_1_extgpd1(xi, sigma, kappa, proba1))
        params_vals = [f_1, xi, sigma, kappa]
        param_names = [r"$\hat f_1$", r"$\hat \xi$", r"$\hat \sigma$", r"$\hat \kappa$"]
        plotting.make_qq_plot_dry_spell(
            ax, data_name, concatenation_neg_exc, generator_one_exc_dry,
            params_vals, param_names,
            n_vectors_simulated_data=50, alpha_over_2_uncertainty_1=0.025,
        )

    fig.tight_layout()
    return fig


def make_fig10_wet_qq(station, spells, df_fit_wet):
    """Notebook 04 Fig. 10 — wet-spell Q-Q plot with bootstrap envelope."""
    wet_spells = spells[station]["wet_spell"]["duration_spell"]
    wet_dates = spells[station]["wet_spell"]["start_date_spell"]
    months = np.array([str(s)[4:6] for s in wet_dates])

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    for ax, vec_month_seasons in zip(axes.flat, config.LIST_MONTH_SEASONS[1:]):
        season = vec_month_seasons[3]
        month_set = set(vec_month_seasons)
        season_mask = np.array([m in month_set for m in months])
        concatenation_neg_exc = np.array(wet_spells)[season_mask]
        data_name = f"{station} {season}"

        row = _fit_row_wet(df_fit_wet, station, season)
        pi, p1, p2 = row["pi"], row["p1"], row["p2"]
        rng = np.random.default_rng(12345)
        generator_one_exc_wet = (lambda pi=pi, p1=p1, p2=p2, rng=rng:
                                 int(sample_geom_mix(1, pi, p1, p2, rng=rng)[0]))
        params_distribution = {"pi": pi, "p1": p1, "p2": p2}
        plotting.make_qq_plot_with_uncertainty_areas_wet_days_duration_vs_simu(
            ax, data_name, concatenation_neg_exc, generator_one_exc_wet,
            params_distribution,
            n_vectors_simulated_data=50, alpha_over_2_uncertainty_1=0.025,
        )

    fig.tight_layout()
    return fig


def make_fig11_exit_prob(station, spells, df_fit_dry):
    """Notebook 04 Fig. 11 — empirical vs hdeGPD-implied exit probability q_d^(0)."""
    dry_spells = spells[station]["dry_spell"]["duration_spell"]
    dry_dates = spells[station]["dry_spell"]["start_date_spell"]
    months = np.array([str(s)[4:6] for s in dry_dates])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    for j, (ax, vec_month_seasons) in enumerate(zip(axes.flat, config.LIST_MONTH_SEASONS[1:])):
        season = vec_month_seasons[3]
        month_set = set(vec_month_seasons)
        season_mask = np.array([m in month_set for m in months])
        season_durations = np.array(dry_spells)[season_mask]

        days, probas, nb_days_state_list = get_proba_leaving_by_day(season_durations)
        days = np.asarray(days)
        probas = np.asarray(probas)
        nb_days_state_list = np.asarray(nb_days_state_list)
        ax.scatter(days, probas, s=np.sqrt(nb_days_state_list), color="orange",
                   label=r"Empirical fit $\hat q_{d,emp}^{(0)}$")

        row = _fit_row_dry(df_fit_dry, station, season)
        xi_gpd, sigma_gpd, kappa_gpd = row["xi"], row["sigma"], row["kappa"]
        f_1 = sum(s == 1 for s in season_durations) / len(season_durations)
        cdf_fitted = make_cdf_fitted_hdeGPD_from_params(f_1, xi_gpd, sigma_gpd, kappa_gpd)

        fitted_probas_extgpd = np.array([get_proba_leaving_state_n_kozu(cdf_fitted, d) for d in days])
        errors = fitted_probas_extgpd * (1 - fitted_probas_extgpd) / np.sqrt(nb_days_state_list)
        ax.plot(days, fitted_probas_extgpd, "x", color="black",
                label=r"hdeGPD fit $\hat q_{d,\hat{\theta}}^{(0)}$", alpha=0.6)
        ax.fill_between(days,
                        fitted_probas_extgpd - 2 * errors,
                        fitted_probas_extgpd + 2 * errors,
                        color="gray", alpha=0.6)

        params_vals = [f_1, xi_gpd, sigma_gpd, kappa_gpd]
        param_names = [r"$\hat f_1$", r"$\hat \xi$", r"$\hat \sigma$", r"$\hat \kappa$"]
        items = list(zip(param_names, params_vals))
        mid = (len(items) + 1) // 2
        left_col = items[:mid]
        right_col = items[mid:]
        left_text = "\n".join(f"{k}={v:.2f}" for k, v in left_col)
        right_text = "\n".join(f"{k}={v:.2f}" for k, v in right_col)
        left_lines = left_text.split("\n")
        right_lines = right_text.split("\n") + [""] * (len(left_lines) - len(right_text.split("\n")))
        textstr = "\n".join(f"{l:<20}{r}" for l, r in zip(left_lines, right_lines))

        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(0.6, 0.95, textstr, transform=ax.transAxes, fontsize=16,
                verticalalignment="top", bbox=props)
        ax.set_title(f"{station} {season}")
        ax.set_xlabel(r"$d$: Duration since spell start (days)")
        if j % 2 == 0:
            ax.set_ylabel(r"$q_d^{(0)}$: Exit probability")
        if j == 0:
            ax.legend(loc="upper left", fontsize=13)

    for a in axes.ravel():
        a.set_xlim(-1, 50)

    plt.tight_layout()
    plt.ylim(-1e-2, 0.5)
    return fig


# ---------------------------------------------------------------------------
# Per-station orchestration
# ---------------------------------------------------------------------------

# (figure_number, builder, save_helper, kwargs-keys-needed)
FIGURE_TASKS = [
    ( 2, make_fig2_survival_overlay, plotting.save_survival_overlay_for_station, ("spells", "df_fit_dry")),
    ( 6, make_fig6_bivariate_acf,    plotting.save_bivariate_acf_for_station,    ("data_per_city",)),
    ( 7, make_fig7_dry_hist,         plotting.save_dry_hist_for_station,         ("spells", "df_fit_dry")),
    ( 8, make_fig8_wet_hist,         plotting.save_wet_hist_for_station,         ("spells", "df_fit_wet")),
    ( 9, make_fig9_dry_qq,           plotting.save_dry_qq_for_station,           ("spells", "df_fit_dry")),
    (10, make_fig10_wet_qq,          plotting.save_wet_qq_for_station,           ("spells", "df_fit_wet")),
    (11, make_fig11_exit_prob,       plotting.save_exit_prob_for_station,        ("spells", "df_fit_dry")),
]


def _output_path(fig_num: int, station: str) -> "config.Path":
    # Mirrors the filename built inside each save_*_for_station helper.
    suffix = {
        2: "comparison_dry_spell_egpd_geom_4seasons.pdf",
        6: "bivariate_acf_plot.pdf",
        7: "histogram_dry_spell_fit.pdf",
        8: "histogram_wet_spell_fit.pdf",
        9: "qqplot_dry_spell_fit.pdf",
        10: "qqplot_wet_spell_fit.pdf",
        11: "proba_leaving_state.pdf",
    }[fig_num]
    return config.ROOT / "figures" / f"figure_{fig_num}" / f"{plotting._safe_filename(station)}_{suffix}"


def process_station(station: str, ctx: dict, *, skip_existing: bool = False) -> dict:
    """Build and save all 7 figures for one station. Returns {fig_num: status}."""
    results: dict = {}
    for fig_num, builder, save_fn, kw_keys in FIGURE_TASKS:
        if skip_existing and _output_path(fig_num, station).exists():
            results[fig_num] = "skipped (exists)"
            continue
        try:
            kwargs = {k: ctx[k] for k in kw_keys}
            fig = builder(station, **kwargs)
            save_fn(fig, station)
            plt.close(fig)
            results[fig_num] = "ok"
        except Exception as exc:
            plt.close("all")
            results[fig_num] = f"FAIL ({type(exc).__name__}: {exc})"
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _load_inputs():
    """Load JSON spells + the two fit CSVs once. Returns a dict ready to pass
    into `process_station` as the `ctx` argument."""
    json_filename = (f"ecad_data_south_europe_filtered_after_{config.START_YEAR}"
                     f"_wet_day_thresh_{config.WET_DAY_THRESHOLD}.json")
    with open(config.EXPORTS_JSON_DIR / json_filename) as fh:
        spells = json.load(fh)

    fit_folder = config.RESULTS_FIT_DIR / f"fit_south_europe_subset_excess_over_{config.WET_DAY_THRESHOLD}"
    df_fit_dry = pd.read_csv(fit_folder / "dry_spell_fit_egpd1_excess_over_1result_fit_parameters.csv")
    df_fit_wet = pd.read_csv(fit_folder / "wet_spell_fit_mixt_geomresult_fit_parameters.csv")

    print(f"[load] {len(spells)} stations from JSON")
    print(f"[load] dry-fit rows: {len(df_fit_dry)}, wet-fit rows: {len(df_fit_wet)}")

    # Pre-compute the per-city per-season per-year structure once (used by Fig. 6).
    data_by_season = split_spells_by_season_simple(
        spells, start_key="start_date_spell", dur_key="duration_spell",
    )
    data_per_city = build_data_per_city_per_season_per_year_couple_vector_duration_vector_date(data_by_season)

    return {
        "spells": spells,
        "df_fit_dry": df_fit_dry,
        "df_fit_wet": df_fit_wet,
        "data_per_city": data_per_city,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--first", type=int, default=None,
                    help="Process only the first N stations (sorted alphabetically).")
    ap.add_argument("--only", action="append", default=None,
                    help="Process only the named station(s). May be repeated.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip a (station, figure) pair if its PDF already exists.")
    args = ap.parse_args()

    ctx = _load_inputs()

    stations = sorted(ctx["spells"].keys())
    if args.only:
        unknown = [s for s in args.only if s not in ctx["spells"]]
        if unknown:
            print(f"[error] unknown station(s) in --only: {unknown}", file=sys.stderr)
            return 2
        stations = list(args.only)
    elif args.first is not None:
        stations = stations[: args.first]

    print(f"[run] processing {len(stations)} station(s) -> figures/figure_*/")
    fail_count = 0
    skip_count = 0
    for station in tqdm(stations):
        try:
            res = process_station(station, ctx, skip_existing=args.skip_existing)
        except Exception:
            print(f"[FATAL] unexpected error for '{station}':\n{traceback.format_exc()}",
                  file=sys.stderr)
            fail_count += 1
            continue
        bad = {k: v for k, v in res.items() if v.startswith("FAIL")}
        skipped = {k: v for k, v in res.items() if v.startswith("skipped")}
        if bad:
            fail_count += len(bad)
            tqdm.write(f"[partial] {station}: " + ", ".join(f"fig{k}={v}" for k, v in bad.items()))
        if skipped:
            skip_count += len(skipped)

    print(f"[done] stations: {len(stations)}, "
          f"figure failures: {fail_count}, "
          f"skipped (existing): {skip_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
