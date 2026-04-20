# article_code/ — companion code

Reproduction pipeline and re-usable implementation for the article
*"A duration-augmented binary Markov chain for rainfall occurrence with long
dry spells"* (Doizé, Allard, Naveau, Wintenberger).

## What this code accompanies

Finite-order Markov chains are the workhorse of rainfall-occurrence
modelling, but they impose a *geometric* tail on spell durations. In
climates with long dry spells — most of the Mediterranean domain studied
here — this systematically under-predicts severe drought (see
Fig. 2 of the article, Palermo-spring).

The article introduces a **Binary Markov Chain with Duration (BMCD)** on
state space $\{0,1\} \times \mathbb N^*$, and formally links it to an
alternating renewal chain. This link (Eq. 7, after Kozubowski 2025) lets
the exit probability $q^{(r)}_d$ be recovered from *any* spell-duration
law, so the model class is as expressive as the choice of that law.

The article then picks two such laws from the extended Generalized Pareto
Distribution (eGPD) family:

- **Dry spells** — a hurdle-discretised eGPD (**hdeGPD**, Eq. 3), fit by
  Probability Weighted Moments. The shape parameter $\xi$ ranges from
  bounded to heavy-tailed across stations and seasons.
- **Wet spells** — a two-component geometric mixture (Eq. 5), fit by EM.

Two headline findings motivate the provided tooling:

- A chi-squared goodness-of-fit test on exit probabilities
  (Proposition 4) rejects the geometric baseline on roughly **twice as
  many** southern-European stations as hdeGPD.
- At 40- and 60-day thresholds, the **mean residual dry-spell duration**
  along the Mediterranean coast is materially larger under hdeGPD than
  under the geometric model — a concrete revision of drought exposure.

Definitions, proofs, and proposition/equation numbers cited throughout
the code match the article's LaTeX source (`main-oup-template.tex`).

## Reproducing the article

Thin, notebook-driven layer that regenerates every figure. All paths live
in [config.py](config.py). The numerical core (ECAD loading, hdeGPD
cdf/pmf, wet-spell EM mixture, empirical exit probabilities, chi-squared
GOF test) is **imported from [../rainfall_article_clean/](../rainfall_article_clean/)**
to avoid duplication — see [_legacy.py](_legacy.py). The modules below
are thin re-export facades.

### Modules

| File | Role |
|---|---|
| [config.py](config.py) | Paths (`FIGURES_DIR = ../figures`), Palermo station id, seasons |
| [data_load.py](data_load.py) | ECAD raw → per-station spell JSON |
| [spell_models.py](spell_models.py) | hdeGPD cdf/pmf, geometric-mixture EM, samplers |
| [statistics.py](statistics.py) | empirical exit probabilities, bivariate ACF, Appendix-I bounds |
| [gof.py](gof.py) | Proposition 4 chi-squared GOF test, across-station loops |
| [plotting.py](plotting.py) | Figure helpers; one `save_<name>` per article figure |

### Notebooks (run in order)

| # | File | Figures produced |
|---|---|---|
| 01 | [notebooks/01_prepare_data.ipynb](notebooks/01_prepare_data.ipynb) | (data only — JSON export) |
| 02 | [notebooks/02_toy_and_map.ipynb](notebooks/02_toy_and_map.ipynb) | `map_palermo.pdf`, `PALERMO_comparison_dry_spell_egpd_geom_4seasons.pdf`, `toy_data_fig_with_N.pdf` |
| 03 | [notebooks/03_fit_and_params.ipynb](notebooks/03_fit_and_params.ipynb) | `histogram_dry_spell_distrib_params.pdf`, `histogram_wet_spell_distrib_params.pdf` |
| 04 | [notebooks/04_palermo_diagnostics.ipynb](notebooks/04_palermo_diagnostics.ipynb) | `PALERMO_{bivariate_acf,histogram_dry,histogram_wet,qqplot_dry,qqplot_wet,proba_leaving_state}.pdf` |
| 05 | [notebooks/05_gof_and_maps.ipynb](notebooks/05_gof_and_maps.ipynb) | `hist_p_values_south_europe_geom_vs_egpd_dmax_20.pdf`, `pvalue_maps_by_season_geom_vs_bmcd.pdf`, `gof_Qn_and_pvalues_simulated_data.pdf` |
| 06 | [notebooks/06_risk_maps.ipynb](notebooks/06_risk_maps.ipynb) | `mean_residual_duration_long_dry_spells.pdf` |

All outputs land in [../figures/](../figures/), matching the
`\includegraphics{figures/...}` paths in the article's LaTeX source.

### Setup

```bash
pip install -r requirements.txt
```

The raw ECAD and the R-fit CSVs already sit in the parent repo; no
download is needed. If `ECAD_RAW_DIR` moves, update [config.py](config.py).

### Notes

- To refit the hdeGPD parameters yourself, run the R scripts in
  [../one_sided_reflected_geometric_brownian_motion/R_files/](../one_sided_reflected_geometric_brownian_motion/R_files/)
  (entrypoints: `fit_dry_spells_extgpd.R`,
  `fit_ext_gpd_spell_length_v2_with_different_threshold_wet_spell_occurrence.R`).
  Without refitting, the committed CSVs in
  `R_files/fit_europe_spells_1945_threshold_JSON_ext_gpd/thr_6/` are used.
- The wet-spell mixture (Eq. 5 in the article) is fit in Python directly
  from the JSON exports — no R step needed.
- The Palermo station is resolved by name in [config.py](config.py)
  (`PALERMO_NAME = "PALERMO"`); adapt to run the diagnostics on another
  city.

## Applying the model to your own data

The BMCD / hdeGPD machinery in this package is not ECAD-specific. To fit
it to another rainfall record, follow the seven steps below — every
function cited already lives in the modules above.

### 1. Shape your data

Provide a daily binary wet/dry sequence per site and season. The article
uses **0.6 mm** as the wet-day threshold (Section 3) to absorb
low-intensity measurement error, tags each spell by the season of its
*start date*, and drops the first and last spell of every continuous
recorded segment to avoid censored spells. Re-use those conventions
unless you have a reason to deviate.

### 2. Extract spell durations

From a tidy dataframe of daily precipitation with dates, use
[data_load.py](data_load.py):

- `process_and_extract_excursions_from_raw_input_df_with_dates(df)` —
  segment into wet/dry spells,
- `from_concat_with_dates_to_concat_by_season(...)` — split by season,
- or the lower-level `extract_longer_dry_spells_w_dates_list_df` if you
  only need dry spells.

Output: lists of $\tau^{(0)}$ (dry) and $\tau^{(1)}$ (wet) durations per
site × season.

### 3. Fit the spell-duration distributions

- **Dry spells (hdeGPD, Eq. 3).** The PWM fit is run from R
  (`fit_dry_spells_extgpd.R` under
  [../one_sided_reflected_geometric_brownian_motion/R_files/](../one_sided_reflected_geometric_brownian_motion/R_files/)),
  using the `mev` package. Drop the resulting CSVs into the layout
  expected by [config.py](config.py) (`RESULTS_FIT_DIR`, `R_FIT_DIR`), or
  repoint `config.py` at your own layout. Parameters are then read via
  `load_param` / `load_xi` / `load_kappa` / `load_sigma` from
  [spell_models.py](spell_models.py).
- **Wet spells (geometric mixture, Eq. 5).** Pure Python:
  `fit_geometric_mixture_em_support1(durations)` from
  [spell_models.py](spell_models.py) runs the EM of Appendix D.

### 4. Recover exit probabilities $q^{(r)}_d$

Given fitted parameters $\hat\theta$, the BMCD exit probability is

$$q^{(r)}_d = \mathbb P_{\hat\theta}(\tau^{(r)} = d) / \mathbb P_{\hat\theta}(\tau^{(r)} \ge d)$$

(Eq. 7, the Kozubowski identity). Build the cdfs via
`make_cdf_fitted_extgpd_from_params(f1, xi, sigma, kappa)` and
`make_cdf_fitted_geometric_from_p(p)` from [spell_models.py](spell_models.py).

### 5. Validate the fit

- **Independence of successive spells (Appendix E)** —
  `pooled_bivariate_autocorr` in [statistics.py](statistics.py).
- **Tail agreement** — simulation-based Q-Q plot, with
  `simulate_season_durations_from_fit` for the fitted-model sample.
- **Exit-probability curve** — empirical
  $\hat q^{(r)}_{d,\mathrm{emp}}$ via `get_proba_leaving_by_day`
  ([statistics.py](statistics.py)) against the model-implied
  $\hat q^{(r)}_{d,\hat\theta}$ from the cdf factory.
- **Chi-squared GOF test (Proposition 4)** —
  `goodness_of_fit_for_city_season` and
  `goodness_of_fit_for_sample_geometric` in [gof.py](gof.py). Use
  `adaptive_D` to pick `d_max` so at least 20 spells exceed it (the
  CLT-regime safety rule of Section 3.2).

### 6. Simulate and quantify drought risk

- Sample paths: `simulate_season_durations_from_fit` and
  `rvs_duration_from_fitted_extgpd` from [spell_models.py](spell_models.py).
- **Mean residual dry-spell duration (Appendix H)** —
  `compute_bounds_refined` in [statistics.py](statistics.py). The bounds
  of Eq. H.3 give an arbitrarily tight approximation as the cut-off $u$
  grows.
- **Proportion of time in long dry spells (Example 1, Appendix I)** —
  `make_approx_share_dry_days_longer_dthresh` and the closed-form
  geometric baseline `share_dry_days_longer_dthresh_markov_order_1_approx`
  in [statistics.py](statistics.py).

### 7. What to edit

In most cases only [config.py](config.py): `ECAD_RAW_DIR`,
`STATION_METADATA_CSV`, `SEASONS`, `DEFAULT_THRESHOLD`, and `PALERMO_NAME`
(→ your reference station). The notebooks assume ECA\&D-style inputs;
adapting to another source usually means rewriting
[notebooks/01_prepare_data.ipynb](notebooks/01_prepare_data.ipynb) only —
downstream notebooks consume the JSON export it produces.
