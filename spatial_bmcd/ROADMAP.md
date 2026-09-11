# Roadmap — spatial BMCD (article + code)

Project-level task memory for the spatialisation of the BMCD rainfall-occurrence generator:
the article [main.tex](Spatialisation-Rain-Occurrence-Generator-article/main.tex), the analysis
notebooks ([spatial_bmcd_data_analysis.ipynb](spatial_bmcd_data_analysis.ipynb),
[spatial_lin_mod_coregion_fit_spain_portugal.ipynb](spatial_lin_mod_coregion_fit_spain_portugal.ipynb),
[switch_dependence_diagnostics.ipynb](switch_dependence_diagnostics.ipynb)) and the library code
([spatial_model.py](spatial_model.py), [spatial_diagnostics.py](spatial_diagnostics.py),
[config_spatial.py](config_spatial.py)).

This file is read and maintained by both the human owner and AI assistant sessions. It tracks
*project-level* work at three horizons (Now / Next / Later). Notebook-level open issues for the
Iberia fit live in [NOTES_spatial_diagnostics.md](NOTES_spatial_diagnostics.md) §0 — reference
them from here by item number (e.g. "NOTES §0.3"), do not duplicate their text.

---

## How to use this file (session protocol)

**Working session (AI or human):**

1. Read **Now** top to bottom. Pick the first unblocked `[ ]` task, or the task the user names.
2. Mark it `[~]` (in progress) before starting.
3. Do the work. If it turns out bigger than its size tag, stop, split it, and say so.
4. In the **same session** as the work: move the finished task to the **Done log** (newest first,
   with date and a one-line outcome), and if Now has fewer than ~3 tasks, promote from Next.

**Grooming session** (a valid session type of its own — "populate the roadmap" means this):

- Decompose one **Later** epic into 2–5 **Next** tasks, each written with the task template.
- Walk the **Intake** checklist for new items; check off a source once it has been fully mined.
- Re-order Now/Next to match current priorities (the human owner has the last word on ordering).

**Rules:**

- Statuses: `[ ]` todo · `[~]` in progress · `[!]` blocked (state on what) · done → Done log.
- Never delete a task — move it to the Done log; use outcome `dropped:` if abandoned.
- IDs are permanent and never reused. Next free IDs: **T-035**, **E-008** (update these counters
  whenever you create a task).
- Record non-obvious direction choices in the **Decisions log** (one dated line each).

## Task template

Copy-paste for every short task; each entry must be workable from its own text alone.

```
- [ ] **T-000 — imperative one-line goal** (S/M/L)
  - Why: one line of motivation.
  - Done when: observable acceptance criterion.
  - Where: file(s) / notebook section / main.tex section.
```

Epics in Later use `E-###`, need only a goal, a "Why", and a "Break down into:" hint line —
acceptance criteria appear when they are decomposed into `T-###` tasks.

---

## Now (≤ ~5 short tasks — the current focus)

*Focus (owner, 2026-08-26): model explanation, testing, evaluation and results analysis —
article writing is deferred until the results stabilize (see the deferred block at the end of
Next).*

- [ ] **T-005 — Git housekeeping: triage and commit the branch backlog** (M)
  - Why: `spatial-bmcd` carries ~60 untracked/modified paths across repo *and* submodule
    (figures, `experiment_outputs/*.csv`, NOTES, ROADMAP, KNMI dirs); un-snapshotted work is
    at risk and the noise hides real changes.
  - Done when: each untracked path is either committed (durable result) or ignored (scratch
    output) with `.gitignore` updated; repo and submodule committed separately (submodule
    first, then the pointer bump); `git status` clean in both.
  - Where: repo root, `.gitignore`, the article submodule.
  - Note 2026-08-26: urgency confirmed — HEAD's fit CSVs are still 35-station / normalized-unit
    versions, off by 1000× on the σ's against the working tree's 26-station / km versions; the
    repo's last commit is 2026-07-13, the submodule's 2026-08-04.

- [ ] **T-024 — Repair and refresh the stale experiment outputs** (M)
  - Why: disk results are inconsistent with the current 26-station/km scope: duplicated `summer`
    row in `theta_fit_spain_portugal_by_season_state_dep_lambda.csv` (feeds every diagnostic and
    both `sim_ensemble_cache` fingerprints); `switch_pair_stats_{season}.csv` predate the
    26-station rescope (Jul 13 vs Aug 5); `spatial_bmcd_data_analysis.ipynb` has 24/34 cells
    unexecuted with stored outputs from older code and two of its declared summary CSVs
    (`switch_psi_model_vs_{obs,sim}_*`) missing while their PDFs exist;
    `switch_dependence_diagnostics.ipynb`'s two halves were run in different kernel sessions.
  - Done when: the duplicate row is removed; `switch_pair_stats_*` are regenerated on the current
    geometry (or confirmed current, and dated); both analysis notebooks run top-to-bottom on the
    current fit CSV and rewrite their CSVs/figures; the ensemble cache is confirmed to still hit.
  - Where: `experiment_outputs/`,
    [spatial_bmcd_data_analysis.ipynb](spatial_bmcd_data_analysis.ipynb),
    [switch_dependence_diagnostics.ipynb](switch_dependence_diagnostics.ipynb).

- [ ] **T-001 — Apply the small correctness fixes of NOTES §0 items 2, 5, 6** (M)
  - Why: item 2 is *blocking* (running Part 2 from the ensemble cell raises `NameError`); items
    5–6 are silent inconsistencies in the paper-bound diagnostic figures.
  - Done when: Part 2 runs from the ensemble cell onward without error; Q-Q bands are invariant to
    `N_QQ_STATIONS` (per-(station, season, ensemble) seeding); the 1.c legend states each
    ensemble's own size. Tick the three items off in NOTES §0.
  - Where: [spatial_lin_mod_coregion_fit_spain_portugal.ipynb](spatial_lin_mod_coregion_fit_spain_portugal.ipynb),
    [spatial_diagnostics.py](spatial_diagnostics.py); details per item in NOTES §0.
  - Scope change 2026-08-20: NOTES §0 item 4 (the 1.e plotters ignoring `BAND_QUANTILES`) left
    this task — section 1.e is not in this notebook, see T-018 — and is now handled there.

- [ ] **T-013 — Diagnose the σ̂₁ / λ̂ instability on the Iberian fit** (M)
  - Why: on real data σ̂₁ runs 3.4 km (spring), 287 km (summer), 122 km (autumn), 84 km (winter)
    while the nested shared-λ fit stays at 286–340 km, and λ̂⁽⁰⁾/λ̂⁽¹⁾ swap roles between seasons
    (spring 0.29/0.71, summer 0.86/0.29). Either a flat/multimodal composite likelihood or an
    optimiser failure — not four different climates. T-004 tests this on *simulated* data only.
  - Done when: the profile composite log-likelihood in σ₁ (and along λ⁽⁰⁾−λ⁽¹⁾) is computed per
    season, the optimiser re-run from ≥5 starts per season, and a verdict recorded in the
    Decisions log — genuine seasonal signal, weak identifiability, or optimiser failure — with
    the paper's claims adjusted accordingly.
  - Where: [spatial_model.py](spatial_model.py) `mle_theta_pairwise`, new experiment notebook,
    main.tex §`sec:model_evaluation`.
  - Evidence 2026-08-26: the λ-swap already appears on *simulated* data — the smoke fit
    `experiment_outputs/mle_smoke_state_dep_lambda.csv` returned λ̂₀=0.955, λ̂₁=0.249 against
    true (0.35, 0.75), boundary-pinned — and the estimator is a single-start L-BFGS-B (fixed
    `THETA_X0_STATE_DEP`, finite-difference gradient over the ~1e-3-accurate `phi2_vec`, closed
    bounds), so multi-start + profile ℓ are the first moves.

- [ ] **T-022 — Finalize and smoke-test the headless θ-consistency runner** (S)
  - Why: the rerun (T-004) is prepared in
    [tests_parameter_estimation.ipynb](tests_parameter_estimation.ipynb) (2 models × 20 seeds ×
    6 lengths = 240 fits, resume + on-disk log) but has produced nothing: the single launch
    (2026-08-05) died before the first fit and `theta_consistency_by_model.csv` does not exist.
    Owner (2026-08-26): code-prep and execution are separate tasks.
  - Done when: a headless launch path is verified end-to-end at toy size (1 model × 1 seed ×
    shortest length), the resume/skip logic is demonstrated on an interrupted run, the exact
    launch command is recorded in the notebook and in T-023, and the stale one-line
    `theta_consistency_run.log` is archived.
  - Where: [tests_parameter_estimation.ipynb](tests_parameter_estimation.ipynb) (run cell),
    `experiment_outputs/`.
  - Note: coordinate with T-013 — if multi-start becomes the retained estimator, the runner
    adopts it before T-023 launches.

## Next (groomed, ready to start)

*Ordered by recommended priority; the first unblocked ones are promoted into Now as it empties.
The deferred block at the end holds the article-writing tasks (owner, 2026-08-26: resume once
the results stabilize).*

- [!] **T-023 — Execute the 240-fit θ-consistency experiment and archive the results** (M)
  - Why: settles the paper's red TODOs — whether σ₁ has a genuine estimation problem or was
    noise — on both variants at once; no results exist for the current 26-station/km design.
  - Done when: `theta_consistency_by_model.csv` holds all 240 rows, the MPE plot cell runs with
    its decreasing-error assert, the run log is archived, and the headline numbers (σ₁ error at
    n=10000, any λ-swaps across seeds) are recorded in the Decisions log. Analysis write-up and
    the paper-side figure swap stay in T-004.
  - Where: [tests_parameter_estimation.ipynb](tests_parameter_estimation.ipynb),
    `experiment_outputs/`.
  - Blocked on: T-022.

- [ ] **T-012 — Godambe (sandwich) standard errors for the pairwise composite MLE** (M)
  - Why: no uncertainty is attached to θ̂ anywhere. T-009's table needs it, and the CLAIC/CLBIC
    criterion of T-006/T-014 needs the same object, tr(J H⁻¹). Cheap first step of E-004.
  - Done when: H (numerical Hessian of the pairwise log-likelihood at θ̂) and J (across-day score
    covariance, or a block jackknife over years) are estimated per season, SEs stored beside θ̂ in
    the fit CSV, and one season cross-checked against a small parametric bootstrap.
  - Where: [spatial_model.py](spatial_model.py) (estimator module), `experiment_outputs/`.
  - Note 2026-08-26: per-day log-likelihood terms are already exposed
    (`pairwise_loglik_one_step_nanaware_vec`), so J is cheap; the numerical Hessian must not sit
    on `phi2_vec`'s ~1e-3 error — use the exact `phi2` or tuned central differences;
    `run_simulation_mle_experiment` is the ready-made bootstrap harness.

- [ ] **T-014 — Δℓ table for the nested variants, with the adjusted CL ratio test** (S)
  - Why: half of T-006 is already computable. Δℓ (state-dependent − shared λ) = +196.3 spring,
    +4.0 summer, +172.9 autumn, +199.0 winter: summer barely pays for its extra parameter. This
    also runs the NOTES §1 sanity check Δℓ ≥ 0 (currently passing).
  - Done when: the table is produced from the two fit CSVs and the comparison uses the adjusted
    composite-likelihood ratio test (Chandler–Bate / `varin2011overview`, already in the bib) —
    plain Δℓ is not χ² here; needs T-012 for the adjustment.
  - Where: main.tex §`subsec:model_comparison`;
    `experiment_outputs/theta_fit_spain_portugal_by_season*.csv`.

- [ ] **T-007 — Make the ensemble bands quantitative: raise N_REPLICATES or add IQR bands** (M)
  - Why: NOTES §0.3 — at 30 replicates the 5–95 % band edges are near-extreme order statistics;
    the §7 band-*width* claim is quantitative and needs better resolution for a paper figure.
  - Done when: either `N_REPLICATES` raised (cost linear, ≈ 15 s/replicate/ensemble) or an
    interquartile band quoted alongside; the 1.c spread-ratio table re-generated; NOTES §0.3
    ticked.
  - Where: ensemble cell of
    [spatial_lin_mod_coregion_fit_spain_portugal.ipynb](spatial_lin_mod_coregion_fit_spain_portugal.ipynb).

- [ ] **T-027 — Add an interannual-variability (overdispersion) diagnostic** (S)
  - Why: the classic failure mode of Markov-type generators — year-to-year variance of seasonal
    wet-day counts too small — is measured nowhere: 1.e reads single days, 1.c/1.d read spells,
    no diagnostic reads the year axis. Cheap: the 30-replicate ensembles and the per-year
    `blocks_from_Rbin` split already exist.
  - Done when: per station × season, the dispersion of yearly wet-day counts (variance or IQR)
    is compared obs vs model-(5) vs independent ensembles (per-station obs value against the
    ensemble band), plus one regional aggregate (yearly wet-day total over the 26 stations); the
    figure is in `../figures/spatial/`, a reading paragraph in NOTES, and the verdict (matched /
    underdispersed) recorded for §`sec:model_evaluation`.
  - Where: fit-notebook Part 2 (new cell after 1.e),
    [spatial_diagnostics.py](spatial_diagnostics.py) (helper beside `daily_wet_fraction`).
  - Note: run on the T-024-refreshed ensembles.

- [ ] **T-028 — Add a lagged cross-correlation (space-time) diagnostic** (M)
  - Why: the latent fields are drawn i.i.d. per day, so cross-station *lagged* dependence exists
    only through state persistence and is direction-blind by construction — while Atlantic
    systems cross Iberia W→E. Nothing measures how much the model misses; even a negative result
    is a needed limitations statement for the paper.
  - Done when: lag-0 and lag-1 cross-correlations of the wet indicators per station pair, obs vs
    both ensembles, plotted against distance; the lag-1 asymmetry
    (corr(R_j,n, R_k,n+1) − corr(R_k,n, R_j,n+1)) plotted against the pair's E–W separation; the
    verdict (adequate / limitation) in NOTES and the Decisions log, and a limitations sentence
    drafted for the paper.
  - Where: [spatial_diagnostics.py](spatial_diagnostics.py) (helper beside
    `_pairwise_binary_stats`), fit-notebook Part 2, `../figures/spatial/`.
  - Note: run on the T-024-refreshed ensembles.

- [ ] **T-018 — Port section 1.e to the current model-(5) pipeline and regenerate its figures** (M)
  - Why: `plot_pairwise_stat_vs_distance` and `plot_fraction_distribution` exist in
    [spatial_diagnostics.py](spatial_diagnostics.py) but their only caller in the repo is
    `archive_old_single_latent_field/diagnostics_fitted_model.ipynb`. Section 1.e is therefore
    absent from the current notebook (working copy *and* HEAD), and
    `../figures/spatial/diag_pairwise_vs_distance_sim_vs_independent.pdf` and
    `diag_heavy_dry_fraction_sim_vs_independent.pdf` were produced by the **superseded**
    single-latent-field model.
  - Done when: 1.e cells run in the fit notebook on the model-(5) and null ensembles, both
    plotters thread `BAND_QUANTILES` (NOTES §0.4, moved here from T-001), the two figures are
    regenerated and the stale ones replaced, `HEAVY_DRY_MIN_DAYS` sits with the other diagnostic
    constants, and NOTES §§0.4/0.9/0.10, §3 and §8 are re-anchored to what the notebook holds.
  - Where: fit notebook (new 1.e cells), [spatial_diagnostics.py](spatial_diagnostics.py),
    `../figures/spatial/`.

- [!] **T-008 — Add the non-parametric null (per-station circular shifts) to 1.e** (S)
  - Why: NOTES §0.10 / §8 — reviewer-proofing: separates "spatial structure right" from
    "marginals right" with no model assumption; a few lines (`np.roll` per column on `Rbin_obs`).
  - Done when: the shifted-observed curve appears on the 1.e panels (or a variant figure), the
    single spurious junction per column is stated in the caption, NOTES §0.10 ticked.
  - Where: 1.e cells of the fit notebook,
    [spatial_diagnostics.py](spatial_diagnostics.py) if a helper is worth extracting.
  - Blocked on: T-018 — those cells do not exist in the current notebook.

- [ ] **T-015 — Extract the notebook-resident driver code into modules** (M/L)
  - Why: `fit_theta_by_season`, `build_params_by_season`, `simulate_ensemble`,
    `spell_time_fraction` and the Q-Q helpers are defined in cells of the fit notebook, so the
    pipeline cannot be run from a script (Codespaces), reused on another dataset, or tested.
    This is what makes T-004, T-007 and all of E-003 expensive.
  - Done when: those functions live in [spatial_model.py](spatial_model.py) /
    [spatial_diagnostics.py](spatial_diagnostics.py) (or a new `spatial_pipeline.py`), the
    notebook imports them, one season refits to the same θ̂, and the append-cache skip-on-rerun
    guard that let a `summer` row be written twice is restored and covered by a test.
  - Where: [spatial_lin_mod_coregion_fit_spain_portugal.ipynb](spatial_lin_mod_coregion_fit_spain_portugal.ipynb)
    cells 2/4/16/18/20 → modules.
  - Note 2026-08-26: `build_params_by_season` is defined ×4 across 3 notebooks (drift hazard);
    `count_valid_pairs` is notebook-only though its `n_pairs` lands in the fit CSVs; the fit-CSV
    writer reads its `done` set once per run and has no schema room for the SE columns of T-012.
  - Note 2026-09-10: [pair_statistics.py](pair_statistics.py) now holds the pair statistics of
    the data-analysis notebook (`state_agreement_table`, `equal_duration_metric_table`,
    `p00_cell_metrics`, both plotters, plus `season_frames` / `day_record_from_Rbin`), imported by
    [data_analysis_fitted_model.ipynb](data_analysis_fitted_model.ipynb); the data-analysis
    notebook still carries its own copies — switch it to those imports here. `build_params_by_season`
    is now defined ×5 (the new notebook's setup cell is the fit notebook's).

- [ ] **T-016 — One cached, dataset-agnostic loading and cleaning pipeline** (M)
  - Why: the red note in main.tex §Data ("pipeline à nettoyer dans le code (fait et refait sur
    chaque notebook et chargement des données trop long)"); E-003 needs it too, ECAD being a CSV
    directory and KNMI Rotterdam a pre-extracted spells JSON.
  - Done when: a loader interface with an ECAD backend and a JSON-spells backend, a parquet cache
    keyed by (dataset, window, wet-day threshold, NaN filter) with a documented invalidation
    rule, every notebook loading through it, and the load time before/after recorded.
  - Where: [spatial_model.py](spatial_model.py) `read_ecad_rr_file` / `load_all_station_rr`,
    [config_spatial.py](config_spatial.py), new cache module.
  - Note 2026-08-26: `read_ecad_rr_file` parses `Q_RR` but no caller filters on it — suspect
    days enter as valid; the flag policy is decided here (counting the affected days), see E-007.

- [!] **T-025 — One-command fit→evaluation pipeline for a new station set** (L)
  *(first slice of E-003, with T-016)*
  - Why: owner (2026-08-26) wants the model easily testable on new data (another set of cities).
    Today the fit driver and ensemble/diagnostic machinery live in notebook cells wired to the
    Iberia dataset, and the single-site marginal fits are a separate manual `article_code` step
    whose outputs are read from `results_fit/fit_south_europe_subset_excess_over_6/`.
  - Done when: a single driver (e.g. `spatial_pipeline.py`, `run_region(region_config)`) goes
    from a station selection + window + wet-day threshold to (i) cleaned occurrence/history data
    per season, (ii) per-station single-site marginal fits invoking the `article_code` utilities,
    written to `results_fit/fit_<region>/`, (iii) θ̂ per season appended to a region-keyed fit
    CSV, (iv) the standard diagnostics (raster, 1.c/1.d bands, co-persistence model-vs-obs)
    written under `figures/<region>/`; and re-running Iberia through it reproduces the current θ̂.
  - Where: new `spatial_pipeline.py`; [config_spatial.py](config_spatial.py); builds on T-015
    (module extraction) and T-016 (dataset-agnostic cached loader).
  - Blocked on: T-015, T-016.

- [ ] **T-017 — Minimal pytest suite on the model invariants** (M)
  - Why: there is no automated test anywhere in the repo (`tests_*.ipynb` are manual notebooks),
    and the invariants that would break silently are exactly the paper's claims.
  - Done when: tests cover marginal invariance (the per-station BMCD law is unchanged by λ and
    σ), the null (σ → 0 ⇒ identity correlation ⇒ independent stations), `phi2_vec` against
    `scipy` on a (z, z′, ρ) grid, positive-definiteness of the LMC blocks over a θ grid,
    seriation determinism, and Δℓ ≥ 0 on the nested fits; all run in well under a minute and the
    command to run them is in the README.
  - Where: new `spatial_bmcd/tests/`, `requirements.txt`.
  - Note 2026-08-26: `haversine` is imported by [spatial_model.py](spatial_model.py) but absent
    from `requirements.txt` — fix alongside.

- [ ] **T-019 — Map figure provenance and script the copy into the article** (S)
  - Why: 46 files in `../figures/spatial/` and 40 in the article submodule's `figures/`, copied by
    hand, and at least two are stale (see T-018). Nothing records which cell produces which file.
  - Done when: a table "figure file → notebook/module + cell that produces it" exists (NOTES or a
    new `figures/README.md`), a copy script replaces the manual step, and orphan/stale figures
    are listed for deletion.
  - Where: [NOTES_spatial_diagnostics.md](NOTES_spatial_diagnostics.md) or a new file, `../figures/spatial/`.

- [ ] **T-006 — Choose the model-comparison criteria and populate the comparison subsection** (L)
  *(first slice of E-002)*
  - Why: main.tex ≈ line 819 (red TODO): the subsection *Comparison of the model variants* is
    empty of quantitative content; the claim about discordant profiles must be "quantified on
    the co-persistence scale" (red note line 818).
  - Done when: criteria chosen (composite-likelihood information criterion and/or
    obs-vs-sim diagnostics — record the choice in the Decisions log), the numbers computed for
    the variants, the discordant-profile claim quantified, and the subsection written.
  - Where: main.tex §`subsec:model_comparison`; fits in
    `experiment_outputs/theta_fit_spain_portugal_by_season*.csv`.
  - Depends on: T-012 and T-014 for the criterion itself.
  - Note 2026-09-10: [data_analysis_fitted_model.ipynb](data_analysis_fitted_model.ipynb) reads
    $\widehat{\pi}_{jj'}$ and $\widehat{p}^{\,\mathrm{stay}}$ on one replicate at $\hat\theta$ and
    puts the model-vs-estimate scatter on the observed and on the simulated record side by side.
    The dry-wet class is where they part: model bias +0.09…+0.12 on the observed cells against
    ≤ +0.03 on the simulated ones, and 3–9× fewer admissible discordant cells ($N \ge 20$) in the
    replicate, none below ≈ 200 km — the co-persistence-scale quantification of the discordant
    claim starts there.

- [!] **T-026 — Fit and evaluate a second ECAD region through the pipeline** (M)
  *(second slice of E-003)*
  - Why: demonstrate applicability beyond Iberia (E-003); Rotterdam cannot exercise the spatial
    model (single station). A second region is also an independent identifiability probe: does
    the σ̂₁ / λ̂ instability of T-013 recur on other data?
  - Done when: a region is chosen from `data/ecad_data/all_stations_europe_metadata.csv` (choice
    in the Decisions log), the pipeline runs end-to-end on it, θ̂ and the standard diagnostics
    are produced, and a short cross-region comparison note (θ̂ ranges, instability recurrence,
    band behaviour) is recorded.
  - Where: T-025 pipeline + a new region config; `results_fit/`, `experiment_outputs/`,
    `figures/`.
  - Blocked on: T-025.

- [!] **T-004 — Rerun the θ-consistency experiment with the retained 5-parameter model** (L)
  - Why: the experiment behind Fig. `theta_consistency` was run with the superseded shared-λ
    variant; the paper's red TODOs (main.tex ≈ lines 903–905) demand a rerun with
    $\theta=(\lambda^{(0)},\lambda^{(1)},\sigma_c,\sigma_0,\sigma_1)$, with more time steps /
    iterations to settle whether $\sigma_1$ has a genuine estimation problem or was noise.
  - Done when: experiment rerun with the state-dependent model at a sample size that settles the
    $\sigma_1$ question; figure regenerated and swapped into the paper; both red TODOs removed
    from `\subsection` *Parameter estimation on simulated data*; conclusions updated in the text.
  - Where: [tests_parameter_estimation.ipynb](tests_parameter_estimation.ipynb),
    main.tex §`subsec:estimation_simulated_data`.
  - Scope change 2026-08-26: runner finalization → T-022, execution of the run → T-023; this
    task keeps the analysis of the results and the paper-side updates (figure swap, red-TODO
    removal, conclusions).
  - Note 2026-09-03: the paper now writes the ranges $a_c,a_0,a_1$ (Decisions log). Two unit
    inconsistencies to clear with the rerun: the appendix still reports
    $\theta_{\text{true}}=(\lambda,a_c,a_0,a_1)=(0.6,\,0.30,\,0.15,\,0.45)$, which are the old
    normalized units and not the kilometres §3.6 now announces; and Figure `theta_consistency`
    carries $\sigma$ component labels baked into the PDF, so the figure is stale until regenerated.
  - Blocked on: T-023.

*Deferred while the results may still change (owner, 2026-08-26) — article-writing tasks, kept
groomed:*

- [ ] **T-021 — Literature review of multi-site occurrence generators, and positioning** (M)
  *(first slice of E-006)*
  - Why: `references.bib` holds 27 entries of which ~2 are domain references
    (`benoit2018stochastic`, `kleintank2002daily`); nothing in the paper states what already
    exists nor what this model does differently. Nothing else in E-006 can be written first.
  - Done when: 10–20 references on multi-site rainfall-occurrence generation (Wilks-type
    multi-site chains, latent-Gaussian / copula occurrence fields, spell-length and
    duration-based models, weather-generator reviews) are added to `references.bib`, each with a
    one-line note on what it does and how it relates; a half-page positioning paragraph (what is
    new here: durations *and* a spatial latent field, state-dependent mixing) is drafted for
    reuse in the introduction.
  - Where: `references.bib`; draft notes for the introduction created by E-006.

- [ ] **T-009 — Report θ̂ in the paper: table of the fitted parameters by season** (M)
  - Why: §5 evaluates the model *at* θ̂ without ever printing θ̂ — the paper carries no table of
    estimates. The numbers exist, and one of them is a headline: σ̂₁ = 3.4 km in spring against
    287 km in summer.
  - Done when: the duplicated `summer` row of
    `experiment_outputs/theta_fit_spain_portugal_by_season_state_dep_lambda.csv` is removed; a
    table (season × λ̂⁽⁰⁾, λ̂⁽¹⁾, σ̂_c, σ̂₀, σ̂₁, ℓ, n_pairs) is in §5 with a paragraph reading the
    ranges in km and the dry/wet contrast; an SE column is added if T-012 has landed.
  - Where: main.tex §`sec:model_evaluation`; `experiment_outputs/theta_fit_*_state_dep_lambda.csv`.
  - Note 2026-08-26: the duplicate-row removal is now part of T-024 (Now); this task keeps the
    paper-side table and its reading.
  - Note 2026-09-03: the table's range columns are headed $\hat a_c,\hat a_0,\hat a_1$ in the
    paper (Decisions log) while the CSV columns stay `sigma_*`; the headline number is therefore
    $\hat a_1$ = 3.4 km in spring against 287 km in summer.
  - Note 2026-09-03 (T-029 follow-up, **resolved** same day): the statement that the marginal exit
    probabilities are fitted station by station (red `ref XYZ`) and treated as known inputs, lost
    when §3.6 was compacted, was re-inserted by the owner's choice at the **top of
    §`subsec:likelihood`** — the point where the likelihood starts depending on $\theta$ alone —
    and §5's pointer was retargeted there. Nothing left to decide for this table.

- [ ] **T-010 — Add a simulation-based validation subsection to §5** (L)
  - Why: §5 compares a model probability with an empirical frequency and never simulates; the
    paper shows nothing generated by the fitted generator, although Part 2 of the notebook
    produces occurrence rasters, spell-time fractions and dry-spell Q-Q against an
    independent-station null. It is the first question a referee asks.
  - Done when: 3–4 figures are selected from Part 2 (raster observed vs simulated, spell time
    fraction, dry-spell Q-Q, each with the no-spatial-structure null), copied into the article
    submodule, and written up with the ensemble/band construction stated once.
  - Where: main.tex §`sec:model_evaluation`; fit notebook Part 2; wording material in
    [NOTES_spatial_diagnostics.md](NOTES_spatial_diagnostics.md) §§3–4.

- [ ] **T-011 — Add the station map and a data table to §2** (S)
  - Why: a spatial paper with no map of its domain is a reviewer flag, and the figure already
    exists (`../figures/spatial/diag_station_order_map.pdf`). Clears the red note at main.tex
    ≈ line 99 as well: window, country filter and NaN policy are stated nowhere in the text.
  - Done when: the map figure is in §2 with a caption; a short table or paragraph gives the 26
    stations, the 1980–2020 window, the wet-day threshold and `MAX_NAN_FRACTION_STATION` = 0.50
    with its effect (35 candidates → 26 kept); the red note is removed.
  - Where: main.tex §`sec:data`; [config_spatial.py](config_spatial.py) for the numbers.
  - Note 2026-09-02: the French red note of §2 was collapsed to a two-line English TODO pointing
    here; its open questions are now carried by this task and must be answered before §2 can be
    written — (i) station set: keep the Spain+Portugal city filter, or widen/change it (see
    T-025/T-026); (ii) window: keep 1980–2020, or change it; (iii) missing days: state the
    `MAX_NAN_FRACTION_STATION` = 0.50 policy and settle the unused `Q_RR` flag (T-016, E-007).

- [ ] **T-032 — State that the paper's pair statistics are computed within season** (S)
  - Why: Figure `fig:state_agreement_seasons` has one panel per season and the
    `fig:pstay_equal_duration_seasons` grid has one row per season, so $\widehat{\pi}_{jj'}$ and
    $\widehat{p}^{\,\mathrm{stay}}_{jj'}(x,x')$ are both estimated season by season — but
    `eq:agreement_estimator` and `eq:pstay_estimator` carry no season index and no sentence says
    so, which reads as an all-year statistic. Owner's decision 2026-09-07: keep the season out of
    the notation, say it once in prose.
  - Done when: one sentence in §2 states that every pair statistic of the paper is computed within
    season, over the days of that season only; no season superscript is added to $\widehat{\pi}$,
    $\widehat{p}^{\,\mathrm{stay}}$ or $\mathcal{T}_j$.
  - Where: main.tex §`sec:data` (next to the $\mathcal{T}_j$ definition), §`subsec:data_synchronicity`,
    §`subsec:switch_diagnostics`.

- [ ] **T-033 — Rename the time index $n \to t$ throughout main.tex** (M) *(slice of E-001)*
  - Why: owner (2026-09-08) — $t$ is the conventional index for a time step, $n$ reads as a count;
    the paper's day sets are already $\mathcal{T}_j$, so the letter and the sets stop disagreeing.
    Cheap and mechanical, but it touches ~149 lines, so it must land before the other notation and
    structure passes rather than be merged into them.
  - Done when: every math-mode time index is $t$ ($R_t, D_t, X_t, B_t, Z_t, Y_t$; $r_{t,j}, d_{t,j},
    x_{t,j}, b_{t,j}, q_{t,j}, z^{*}_{t,j}$; $\mathbf{x}_{t+1}\mid\mathbf{x}_t$;
    $\prod_{t=0}^{t_{\text{obs}}-1}$; the convention $\cdot_{t,j}:=\cdot_t(\mathbf{s}_j)$ and
    Table~`tab:station_notation`); $n_{\text{obs}} \to t_{\text{obs}}$ at its 12 sites (§`sec:data`
    ×2, the likelihood block ×7, §`subsec:estimation_simulated_data` ×3); the simulated
    sample size $T$ of the simulation appendix becomes $t_{\text{sim}}$ and its loop reads
    $t=0,\dots,t_{\text{sim}}-1$; one sentence in §1 *Notations* states that the model is a
    **discrete-time** chain, $t=0,1,2,\ldots$ indexing days, not a continuous time; nothing else is
    swept ($\mathbb{N}$, $\mathbb{N}^*$, $\mathcal{N}^{+}_{jj'}$, $N^{+}_{jj'}$, `\top`, prose `n`);
    the document compiles at 22 pp with the same 3 pre-existing undefined refs (T-020) and no new one.
  - Where: [main.tex](Spatialisation-Rain-Occurrence-Generator-article/main.tex), whole file
    (~141 lines carry a math-$n$ token), **including** the `\hide{...}` appendix block and the text
    inside the `\added`/`\removed` markup, so nothing resurrects with the old index. Line numbers
    are not pinned here: §3 is being merged (T-030 territory) and the file moves under them.
  - Conflicts checked 2026-09-08 (no blocker): $t$ is currently unused as a symbol in main.tex (the
    only `t_n` matches are `\cdot_n`). Owner's rulings, same day:
    (i) the simulated sample size is $t_{\text{sim}}$, **not** $T$ — one lowercase-$t$ family for the
    two horizons ($t_{\text{obs}}$ observed, $t_{\text{sim}}$ simulated), and no capital $T$ competing
    with the day sets $\mathcal{T}_j$; (ii) $\tau$ (spell duration) is **kept** beside $t$ (day
    index) — a duration in days vs an index, no action; (iii) the discrete-time sentence goes in
    §1 *Notations*, so the reader knows from the start, not at the index's first use in §2.
  - Open (non-blocking): the generic sequence $(u_n)_{n\ge 0}$ of §1 line 58 is a dummy index, not a
    time. Default: rename it to $(u_t)_{t\ge 0}$ — its only instantiation in the paper is the
    time-indexed $\mathbf{x}_{0:t_{\text{obs}}}$ — unless the owner wants a neutral dummy letter.
  - Note: after the rename the $N$ of $\mathcal{N}^{+}_{jj'}(x,x')$ / $N^{+}_{jj'}(x,x')$ matches no
    index in the paper, and the natural target for T-031's $\mathcal{N}$-collision becomes
    $\mathcal{T}^{+}_{jj'}$, matching $\mathcal{T}_j$ — record it there, do not do it here.
  - Note: figures — only `figures/theta_consistency_mpe.pdf` bakes the index into a label
    ("history length nobs (days)"); it is already stale (T-004), whose regeneration must emit
    $t_{\text{obs}}$. The other four included figures carry no index label.
  - Note: the **code is not renamed** (`n_obs`, `n` loops stay) — same deliberate paper/code
    divergence as $p^{\mathrm{stay}}$/`p00` and $a_k$/`sigma_*`. A `n_pairs` column in T-009's table
    is a count, not an index, and is unaffected.
  - Ordering: run **before** T-002 and T-031 (their audit tables should record the final symbols)
    and before T-030 (a section merge stacked on a global rename is much harder to review);
    orthogonal to T-003, but commit it alone in the submodule so the diff stays readable.

- [ ] **T-002 — Notation consistency pass over main.tex** (M) *(first slice of E-001)*
  - Why: notations were flagged by the owner as the top writing pain point; symbols drift
    between the Notations section, the model sections and the diagnostics sections.
  - Done when: every symbol used in the paper is defined once (Notations section or first use),
    with no conflicting duplicate definitions and no orphan notation; known cases handled
    (e.g. co-persistence is $p^{\mathrm{stay}}$ in the paper vs `p00` in code/filenames — code
    stays, paper must be internally consistent). Produce a short symbol audit table in the
    session summary, not in the paper.
  - Where: [main.tex](Spatialisation-Rain-Occurrence-Generator-article/main.tex), whole file.
  - Scope 2026-09-03: consistency and the symbol inventory only; the *economy* of the notations
    and *where* they are introduced is T-031, which consumes this task's audit table.

- [~] **T-030 — Restructure the Section-3 plan: merge its nine subsections into four** (M)
  *(slice of E-001)*
  - Why: §3 carries nine subsections for a single construction, so the reader crosses a heading
    every ~30 lines and the model reads as nine topics rather than one argument. The three-way
    forward chain at main.tex ≈ line 217 ("studied in Section~\ref{subsec:switch_diagnostics},
    built in \ref{subsec:latent_gaussian_lmc}, given a parametric family in
    \ref{subsec:exponential_cov_function}") is the symptom: three pointers that a merge turns
    into one. Owner, 2026-09-03.
  - Done when: §3 holds the target plan — (1) *Single site model*; (2) *Bernoulli random fields*
    + *Spatial extension of the BMCD*; (3) *Empirical spatial structure of the exit indicators*
    + *From the empirical structure to a latent Gaussian construction* + *Exponential correlation
    function*; (4) *Likelihood* + *Pairwise likelihood approximation* + *Likelihood with
    non-simultaneous station records* — each merge either applied or refused with its reason in
    the Decisions log; every `\label` the merge absorbs is kept (on a `\subsubsection` or
    `\paragraph`) or its call sites retargeted (≈34 `\ref` sites today:
    `subsec:switch_diagnostics` ×13, `subsec:exponential_cov_function` ×7, `subsec:spatial_model`
    ×5, `subsec:likelihood` ×5, `subsec:pairwise_likelihood` ×2, `subsec:latent_gaussian_lmc` ×2,
    `subsec:nan_handling` ×0 — some sit inside the review comment blocks T-003 removes); the
    junction sentences are rewritten so each merged block opens on one lead and no orphan
    transition survives; the document compiles with no *new* undefined reference (3 are
    pre-existing, see T-020).
  - Where: [main.tex](Spatialisation-Rain-Occurrence-Generator-article/main.tex) §3
    (≈ lines 149–548).
  - Note: run after T-003 — restructuring §3 while it still carries unvalidated
    `\added`/`\removed`/`\orange` markup mixes two passes.
  - Note 2026-09-03: T-029 already shortened the third block's last member — §3.6 *Exponential
    correlation function* is now two paragraphs plus two displays — which makes the 3.4+3.5+3.6
    merge cheaper than when it was proposed. The plan *outside* §3 (§2, the Evaluation section,
    Appendix order) is out of scope: state what the merge forces, do not do it.
  - Progress 2026-09-08: the merge is **applied in the file as a blue/orange review pass**, awaiting
    the owner's read-through (same two-step protocol as T-029). §3 now holds the four blocks; the
    five absorbed headings are marked by orange `[merge seam --- ...]` lines; block 3 carries the
    new label `subsec:spatial_specification` and all 13 live call sites of the five absorbed labels
    were retargeted; blue lead-in paragraphs open blocks 2 and 3; the collapsed three-way forward
    chain, the orphan transition at the head of the former §3.5 and the two "specified in
    Section~X" pointers that became intra-block are marked. Document compiles, 22 pp, the same 3
    pre-existing undefined refs and no new one. **Block 3's title is still open** — it carries two
    candidates, blue (*Construction of the latent field*) and purple (*From the data to the latent
    field*); exactly one is to be kept. Blocks 2 and 4 are settled (*Spatial extension of the
    BMCD*, *Likelihood*).
  - Progress 2026-09-08 (second pass, owner's request): a **compaction pass on block 2** was added
    in a new colour, magenta `\trim` (macro defined in the preamble), distinct from the merge's
    orange. Five candidates marked: the repeated "extend the model to a domain $\mathcal{D}$"
    (introduced once now, in the block's opening paragraph, carrying the "usually some geographical
    region" gloss); the unused $\tau^{(r)}_{\mathbf{s}}$, introduced and never used again in the
    paper; the starred-$z^{*}$ convention, which belongs in §1 *Notations* (→ T-031); the
    three-line multivariate-binary detour, compressed to one sentence; and the sentence restating
    `eq:gaussian_thresholding_sugg` just before the display that states it precisely. The broken
    sentence "Let us rewrite the specific case of the latent Gaussian construction is explicitly
    suggested in\ldots" was repaired in blue. Table~`tab:station_notation` is the largest remaining
    compaction target in block 2 and stays with T-031.
  - Progress 2026-09-08 (block 2 accepted): §3.2 *Spatial extension of the BMCD* is **collapsed** —
    blue to plain, orange deleted, merge seam removed. It is the first of the four blocks finished.
    The appendix was reordered by order of first reference in the body at the same time: distance
    (§2, l. 81), co-persistence on all duration pairs (l. 279), LMC formalism (l. 322), exponential
    correlation (l. 342), fast bivariate cdf (l. 427), then the `\hide` block, seriation and
    simulation. **Five appendix subsections have no reference from the body at all** —
    `model_variants` and `model_comparison` (inside `\hide`, T-020), `seriation`,
    `sec:simulation` (cited only from inside the appendix) and `estimation_simulated_data` — so
    their order is the pre-existing one; give them body call sites or drop them (T-020 / T-004).
  - Progress 2026-09-08 (owner's read-through of block 2): four of the five compaction candidates
    accepted and **collapsed**, $\tau^{(r)}_{\mathbf{s}}$ kept, and the whole passage reordered on
    the owner's plan (see the Decisions log). No `\trim` call site is left in the file; the macro
    stays in the preamble for the next compaction pass. Block 2 is now settled apart from
    Table~`tab:station_notation` (T-031). Remaining markup in §3 is the merge's own orange/blue
    plus the blue reordered passage, awaiting the same collapse. 22 pp, same 3 undefined refs.
  - Note 2026-09-08 (ordering, resolved by fact): T-033 was written the same day with a "run before
    T-030" line, but the merge had already been applied when it landed. Kept in this order
    deliberately — see the Decisions log: collapsing T-030 first means the rename sweeps less text,
    since the orange blocks it would otherwise have to rename are deleted at collapse.
  - Progress 2026-09-10 (block 3 compaction, owner-driven): §3.3 *Construction of the latent
    field* is compacted in rounds — orange = delete, blue = insert, the owner rules on each round
    and it is collapsed the same day; the T-030/T-034 baseline markup in the block is left as is
    (blue read as black, orange as deleted). Rounds A and B collapsed: the Table-1 paragraph
    rewritten to four sentences on the two conventions of §1 (the $z^{*}_{\mathbf{s}}(x)$
    convention is now stated nowhere while the table's last row uses it → T-031); the merge-seam
    recap before "Let us now build $Z_n$" deleted; the `eq:cond_corr` explanation replaced by a
    three-sentence *modelling choice* statement (the $(r,r')$ dependence is read on Figure 2,
    dropping the durations is a simplification); the monotonicity of $p^{\mathrm{stay}}$ in $\rho$
    stated once after `eq:pstay_def` with the purpose folded in; the figure description cut to
    what the caption does not say; the LMC gloss, the range gloss, the duplicate appendix pointers
    and three repeats of the state-pair notation deleted. Open for rounds C and D: the
    physical-intuition sentence, the "second observation" on long durations (feeds nothing
    downstream), the first line of `eq:cond_corr`, and whether `eq:common_days_plus` (eq. 6)
    becomes an inline definition (evaluation given to the owner 2026-09-10). 21 pp, 2 undefined
    refs (`eq:spatialized_markov_model_option2` is no longer reported).
  - Progress 2026-09-10 (block 3 **flow round**, owner's request: fewer paragraphs, one simple
    flow): written as orange/blue markup, read and **accepted by the owner, collapsed the same
    day** (outcome at the end of this note). A paragraph break
    proposed for deletion is shown as an orange ¶ (`\orange{\P}`) at the join, the blank line
    already removed in the source; the one proposed for insertion as a blue ¶ at the end of the
    paragraph it closes. Moves: the day set `eq:common_days_plus` (old eq. 6) relocated from
    before `eq:pstay_def` to just before `eq:pstay_estimator`, its only use (orange `equation*` at
    the old place, blue numbered display at the new one — numbering becomes co-persistence (6),
    day set (7), estimator (8), all downstream `\eqref`s follow); the sentence justifying the
    conditioning on full states dropped (derivable from the second line of (6)); the
    restriction/appendix/threshold sentences reordered so Figure 2 is named before its appendix
    counterpart, and the figure reading opens a paragraph of its own; the one-sentence paragraph
    "The correlation entering (6) is thus $\rho^{(r,r')}$" folded into the modelling-choice
    sentence so that constraint (9), its justification and the three properties form one
    paragraph; the lead-in and the Table-1 paragraph merged; the concordant/discordant reading
    sentence and the closing Markov sentence repaired (wording). 10 → 7 paragraphs. Untouched on
    purpose: the three round-C sentences (physical intuition, second observation, first line of
    (9)) and the Table-1 paragraph's content — noted to the owner that it duplicates the table
    caption apart from the definition of $d_{n,j}$. 21 pp, the same 3 undefined refs as the log
    reports (`eq:spatialized_markov_model_option2` is back in the log).
    **Collapsed** on the owner's acceptance: orange deleted, blue unwrapped, ¶ marks removed. The
    owner had meanwhile dropped one blue clause themselves (the one naming $\rho^{(r,r')}$ on every
    cell after "only through the margins"), so the modelling-choice sentence ends there and the
    three-properties sentence follows directly — re-read-before-collapse confirmed again. §3.3 is
    now 7 paragraphs. The T-030/T-034 baseline markup of the block (the `\sugg{below}` pointer and
    the merge seam before the isotropy paragraph) was collapsed too on the owner's instruction the
    same day, so **§3.3 carries no colour markup any more**; T-030's remaining markup is the two
    merge seams of §3.4, the orange fragment in its non-simultaneous-records paragraph and the
    retargeted `\sugg{\ref{...}}` call sites elsewhere. Still open for rounds C/D: the three
    sentences listed above and the Table-1 paragraph vs caption duplication. 21 pp, same 3
    undefined refs.
  - Progress 2026-09-10 (block 3, **LMC-paragraph regrouping**, owner's request — the displays
    were spread through the paragraph): written as orange/blue, **accepted by the owner and
    collapsed the same day**. Before accepting, the owner joined the isotropy paragraph ("We work
    with stationary and isotropic…") onto the new paragraph; the join was kept at collapse.
    Old paragraph kept in orange with its three displays as unnumbered `equation*`; new blue
    paragraph = one lead sentence, then a single `align` carrying (10) state selection and (11)
    the two LMC fields, an `\intertext` link, and (12) the correlation blocks — labels and numbers
    unchanged — then the definitions of $W^{(c)}_n,W^{(0)}_n,W^{(1)}_n$ and $\lambda^{(r)}$ as a
    "where" clause, the LMC citation, and the reading of properties (i)–(iii). One claim changed:
    "property (i) requires one further process per state" became "the state-specific components
    let the two concordant blocks differ", since (i) already holds at $\lambda^{(r)}=1$. Compiles,
    21 pp, (9)–(12) keep their numbers, same 3 undefined refs.
  - Progress 2026-09-10 (block 4 **collapsed**, owner's instruction): §3.4 *Likelihood* is free
    of review markup — the two orange merge seams deleted, the orange repeat "of
    Section~\ref{sec:data}" after $\mathcal{T}_j$ deleted (the pointer already opens that
    sentence), the retargeted `\sugg{\ref{subsec:spatial_specification}}` in the θ sentence
    unwrapped. **All four §3 blocks are now collapsed**; the red `ref XYZ` citation note at the
    head of §3.4 is a missing reference, not review markup, and stays. T-030's only remaining
    markup is outside §3: the retargeted `\sugg{\ref{...}}` call sites in §`sec:model_evaluation`
    (itself still blue, awaiting its own validation), Appendix §`subsec:all_duration_appendix`
    (×2), §`subsec:model_variants` (inside `\hide`) and §`subsec:estimation_simulated_data`.

- [~] **T-034 — Move the distance convention and the exponential kernel to the appendix** (S)
  *(slice of E-001, follow-up of T-030)*
  - Progress 2026-09-08: **applied the same session it was created**, on the owner's instruction to
    treat the haversine formula like the exponential kernel — use them in the body, point to the
    appendix for the formulas. New appendix subsection `subsec:corr_functions_appendix` (blue),
    placed after §`subsec:lmc_formalism` and **outside** the `\hide` block, carrying
    `eq:haversine` and `eq:exp_cov`; the body keeps one paragraph naming the isotropy assumption,
    the range $a_k$ and $\theta$, which the likelihood needs immediately. The homogeneous-Markov
    paragraph stayed in the body. Call sites retargeted: §2 opening → appendix,
    §`subsec:model_variants` → appendix, §`subsec:seriation` → §`sec:data` (where $h_{jj'}$ is
    defined). Collapses together with T-030's markup; document compiles, 22 pp, no duplicate
    label, same 3 pre-existing undefined refs.
  - Left open: the two `\eqref{eq:haversine}` in §2.1's prose and figure caption now point into the
    appendix. Harmless, but since §2 defines $h_{jj'}$ in words they could simply read
    "$h_{jj'}$" — decide at collapse.
  - **Settled 2026-09-08 (owner):** the distance convention and the exponential kernel do *not*
    share one appendix subsection — the distance is used from §2 onwards and is a property of the
    data, the kernel is a modelling choice. Of the four options put to the owner, **the split into
    two appendix subsections** was chosen (over moving the haversine display into §2, giving it
    inline/in a footnote, or keeping one subsection with two paragraphs). Applied in blue:
    §`subsec:distance_appendix` *Great-circle distance between stations* carrying `eq:haversine`,
    and §`subsec:exp_corr_appendix` *Exponential correlation function of the latent fields*
    carrying `eq:exp_cov` and pointing back to the first for $h_{jj'}$. Call sites split
    accordingly: §2 → distance, the body paragraph → both, §`subsec:model_variants` → kernel.
  - Why: owner (2026-09-08), ruling on the §3 merge — the haversine formula~`eq:haversine` and the
    exponential correlation function~`eq:exp_cov` are machinery, not argument. §3's third block
    should end on *what* the latent correlation has to satisfy (the three properties read off the
    data), leaving the parametric family and the distance convention to the appendix, where the
    LMC formalism already sits.
  - Done when: `eq:haversine`, `eq:exp_cov` and their two paragraphs live in a new appendix
    subsection with its own label, placed beside §`subsec:lmc_formalism`; the sentence fixing
    $\theta=(\lambda^{(0)},\lambda^{(1)},a_c,a_0,a_1)$ and the paragraph arguing that $(X_t)$ is a
    homogeneous Markov chain are each placed deliberately (body or appendix — the θ sentence is
    needed in the body, the likelihood uses it immediately); the 5 call sites that point at that
    content are retargeted; the document compiles with no new undefined reference.
  - Where: [main.tex](Spatialisation-Rain-Occurrence-Generator-article/main.tex), §3 block 3
    (the segment marked by the third orange merge seam) and §Appendix.
  - Note: §2's **opening paragraph** (line ≈ 77) defines the inter-station distance by a forward
    `\eqref{eq:haversine}` into §3, and §2.1's figure caption cites it twice more. Moving the
    formula to the appendix turns those into pointers into the appendix — decide there whether §2
    keeps the pointer or simply says "great-circle distance in kilometres" and lets the appendix
    carry the formula. The two other sites are in §`subsec:likelihood` (θ) and the appendix
    §`subsec:seriation`.
  - Depends on: T-030 — the merge leaves this content in place and marks its seam, so this task
    starts by lifting exactly that segment.

- [ ] **T-031 — Reduce and regroup the places where notations are introduced** (M)
  *(slice of E-001, second half of T-002)*
  - Why: owner (2026-09-03) — the *number* of introduction sites is itself the reading cost:
    ~23 "denote / we write / we call / we define" sites plus 10 `:=` across the paper. Two are
    anti-patterns the paper's own review comments already record (quoted below). T-002 asks
    whether each symbol is defined *consistently*; this asks whether it needs to exist at all and
    whether it is introduced in the right place.
  - Done when: every introduction site is listed (symbol · line · line of first use) with a
    verdict — (i) drop the symbol (spell it out, or reuse an existing one), (ii) move it to its
    first use, (iii) group it into a neighbouring introduction block, (iv) keep as is — and the
    accepted moves are applied; no symbol is introduced at a place where it is not used, the
    Notations section being the only allowed exception; the site count before/after is recorded
    in the Decisions log; the document compiles. The audit table goes in the session summary,
    not in the paper.
  - Where: [main.tex](Spatialisation-Rain-Occurrence-Generator-article/main.tex), whole file;
    input: the symbol audit table of T-002.
  - Carried over from the §3 notation-review comment blocks (main.tex ≈ 88–111, 238–254,
    committed in the submodule at 0f62e87; T-003 deletes the comments, these decisions live here
    now):
    (i) $\mathcal{N}$ names both the common-day sets $\mathcal{N}_{jj'}$
    of~\eqref{eq:common_days}/\eqref{eq:pstay_estimator} and the Gaussian law
    $\mathcal{N}(\mathbf{0},\mathbf{I}_J)$ of the simulation appendix — a rename of the day sets
    to $\mathcal{T}_{jj'}$ (keeping $N_{jj'}$ for the cardinality) was drafted and left
    unapplied;
    (ii) `tab:station_notation` (§`subsec:spatial_model`) introduces $q_{n,j}$ and $z^{*}_{n,j}$,
    first used in §`subsec:likelihood` — "the price of exhaustiveness" of an exhaustive table;
    this is exactly the anti-pattern the task targets, so rule on it explicitly.
    Already settled, do not re-propose: the generic subsets
    $A,A'\subseteq\{0,1\}\times\mathbb{N}^*$ (only two of the four instantiations were ever used)
    and the two-display variant of $\mathcal{N}^{+}_{jj'}$ through an intermediate set.
  - Depends on: T-002 (inventory) and T-030 (a merge relocates text, and with it the sites).
  - Note 2026-09-08: two §1 *Notations* sites were already cleared ahead of this task, on the
    owner's call — the generic sequence $(u_n)_{n\ge 0}$ / $u_{p_1:p_2}$ (deleted outright: the
    colon notation has a single point of use, §`subsec:likelihood`, where
    $\mathbf{x}_{0:n_{\text{obs}}}:=(\mathbf{x}_0,\ldots,\mathbf{x}_{n_{\text{obs}}})$ already
    spells it out) and the probability sequence $\mathbf{a}=(a_i)_{i\in\mathbb{N}^*}$ (used
    nowhere; its removal also clears a latent clash with the ranges $a_c,a_0,a_1$ of T-029, and
    $\mathbf{q}^{(r)}$ is self-defining where it appears). §1 now opens on $\mathbb{N}^*$ alone.

- [ ] **T-020 — Collapse the review markup outside Section 3** (S) *(second slice of T-003/E-001)*
  - Why: T-003 covers Section 3 only; `\removed`/`\added`/`\orange` call sites also sit in the
    Appendix subsections `lmc_formalism`, `model_variants` and `simulation` (main.tex ≈ lines
    742, 771, 777, 812, 818, 891), plus the `\cut`/`\sugg` pair in §2.
  - Done when: no colour-markup macro call site remains outside a deliberate one, the macros
    themselves are kept (still used by future passes) or removed with their definitions, and the
    document compiles. Owner reads before collapsing, as in T-003.
  - Where: [main.tex](Spatialisation-Rain-Occurrence-Generator-article/main.tex), Appendix and §2.
  - Note 2026-08-26: the Appendix subsections `model_variants` and `model_comparison` are
    currently wrapped in `\hide{...}` (main.tex ≈ lines 774–820), so they vanish from the
    compiled PDF while §`subsec:estimation_simulated_data` still references
    `\eqref{eq:variant_shared_lambda}` from inside them — decide the `\hide`'s fate here.
  - Note 2026-09-02: the **§2 half is done**. The `\cut`/`\sugg` notation pair was collapsed
    (blue version kept), the $x_{n,j}$ sentence of §`subsec:data_synchronicity` moved into the
    §2 notation paragraph and its back-reference beside `tab:station_notation` retargeted to
    `sec:data`, and the two stale review-comment blocks of §2 deleted; §2.1 now opens on
    "Two stations may share a rain state…". Document compiles (22 pp, same 3 pre-existing
    undefined refs from the `\hide`). Remaining sites: ~~the `\orange` fragment in
    §`subsec:exponential_cov_function`~~ (gone 2026-09-03: the sentence it sat in was dropped when
    that subsection was compacted, see T-029), the `\added`/`\removed` pair in
    §`subsec:nan_handling` (**cleared 2026-09-08 with T-003**, the `\added` accepted; that
    subsection is now the third block of the merged §3) and the Appendix ones incl. the `\hide`
    decision above.
  - Note 2026-09-03: one Appendix site was cleared early with T-029 — the `\removed{}` sentence of
    §`subsec:lmc_formalism` whose only content was a reference to the deleted
    `sec:model_specification` label; its plain replacement was already sitting on the next line.
    The 3 undefined refs are unchanged: `eq:spatialized_markov_model_option2` (simulation
    appendix, line ≈ 764) and `eq:variant_shared_lambda` / `subsec:model_variants` (line ≈ 785).

## Later (long-term enhancements / epics — E-###)

- **E-001 — Article writing quality: organisation, flow, notations, wording, structure, repetitions**
  - Why: standing top priority (owner, 2026-08-20): the paper must read well end-to-end;
    the draft carries restructure markup, notation drift and repetitions.
  - Break down into: notation audit (→ T-002, first slice); collapse of pending review passes
    (→ T-003); then per-section organisation/flow passes (one task per section), a repetition
    sweep across sections, a wording pass, and a final full read-through.

- **E-006 — Write the missing article skeleton: abstract, introduction, related work, conclusion**
  - Why: `main.tex` runs `\maketitle` → Notations → Data → BMCD → Model specification →
    Evaluation → Appendix → bibliography. There is no abstract, no introduction, no related-work
    section and no conclusion, and the bibliography is almost entirely probability/numerics — so
    a reader cannot tell what problem is solved nor how it differs from existing generators.
    Gating item for any submission; complements E-001, which polishes what is already written.
  - Break down into: literature review and positioning (→ T-021, first slice); target journal +
    template/format decision (affects length, abstract style, section order); introduction with
    motivation and an explicit contribution list; conclusion, limitations and perspectives
    (the discordant-block constraint of §5, the fixed marginals, E-004's uncertainty caveat).

- **E-002 — Model comparison: implement competing models on the same data and argue the advantages**
  - Why: the paper needs to show *why* the retained state-dependent LMC-BMCD beats alternatives —
    on criteria, diagnostics and analysis, not by assertion.
  - Absorbs the NOTES §0.1 decision: the fate of the commented-out shared-λ model (4) code
    (re-enable as baseline vs delete) is decided *here*, when the comparison work starts.
  - Break down into: criteria choice + subsection (→ T-006, first slice); decide model (4) code
    fate; pick and implement 1–2 external baselines (e.g. a classical latent-Gaussian occurrence
    generator without durations); run the comparison; write the analysis of advantages.

- **E-003 — Multi-dataset application: fit, figures and evaluation on other European datasets**
  - Why: owner (2026-08-20): the model must be demonstrably applicable beyond Iberia — fitting,
    figures and fit results on other cities/regions, with an evaluation of how the model behaves
    across climates. The KNMI Rotterdam work (`data/knmi_rotterdam_data/`,
    `figures/knmi_rotterdam/`, `results_fit/fit_knmi_rotterdam/`) is the started first instance.
  - Re-scoped 2026-08-20: `data/knmi_rotterdam_data/exports_json/` holds a **single** station's
    spells JSON, and `results_fit/fit_knmi_rotterdam/` only single-site spell fits — Rotterdam
    cannot exercise the spatial model at all. The cheap first *spatial* instance is a second
    ECAD region (France, Alps…): same loader, and the candidate metadata is already in
    `data/ecad_data/all_stations_europe_metadata.csv`.
  - Decomposed 2026-08-26: the dataset-agnostic pipeline and the second-region fit are now
    concrete Next tasks → T-025 (one-command fit→evaluation pipeline, absorbing the single-site
    marginal step) and T-026 (second ECAD region).
  - Break down into: make the pipeline dataset-agnostic (→ T-016, plus the module extraction of
    T-015); one-command fit→evaluation driver (→ T-025); fit a second ECAD region end-to-end
    (→ T-026); keep Rotterdam as a single-site marginal check (or extend it to the surrounding
    KNMI stations); add a further dataset; cross-dataset evaluation and write-up.

- **E-004 — Propagate the estimation uncertainty on θ̂ into the diagnostic bands**
  - Why: NOTES §0.7 / §6 — every band holds θ̂ fixed, so all bands are lower bounds on the total
    spread; a parametric bootstrap over the pairwise MLE would fix it. Expensive; explicitly
    "not currently planned" — revisit if a reviewer asks.
  - The cheap first step was split out as T-012 (Godambe sandwich SEs on θ̂): it gives the
    uncertainty *on the parameter* for a few Hessian/score evaluations; only the propagation
    *into the bands* needs the bootstrap and stays here.
  - Break down into: bootstrap design (replicates × refits budget); implementation on one season;
    decision on full rollout.

- **E-005 — Matérn correlation function variant**
  - Why: an exponential kernel is a Matérn special case, so the generalisation is natural if the
    fit warrants it, and a Matérn model would additionally let the dry and wet fields differ in
    smoothness (sentence available in `main_old.tex` ≈ line 431).
  - Status 2026-09-03: the "full treatment vs. remove the stub" branch is **settled** — the stub
    subsection was deleted from main.tex with the §3.6 restructure (T-029) and the idea is demoted
    to a perspectives sentence to be written by E-006. The paper now claims nothing about Matérn;
    `gneiting2010matern` stays in `references.bib` unused until it is either cited there or
    dropped. This epic is what to do if a fit or a referee makes the variant worth having.
  - Break down into: perspectives sentence in the conclusion (with E-006); Matérn kernel in
    [spatial_model.py](spatial_model.py) (kernel + Cholesky path); refit one season and compare
    with exponential.
  - Note 2026-08-26: `lmc_block_curves` in [spatial_diagnostics.py](spatial_diagnostics.py)
    duplicates the exponential kernel inline (×3) — refactor to a single kernel definition
    before any Matérn work, or the swap will silently diverge.

- **E-007 — Evaluation robustness: out-of-sample, data sensitivity, diagnostic calibration**
  - Why: every current diagnostic is in-sample, at one fixed preprocessing, with an informal
    band reading — the three standard referee attacks not covered by any existing task. Parked
    at epic level (owner, 2026-08-26): groomed only after the cheap diagnostics (T-027/T-028)
    and the identifiability verdict (T-013) land.
  - Break down into: temporal-split fit (e.g. 1980–2000 vs 2001–2020) as a joint holdout +
    stationarity check; a held-out-stations prediction pilot (one refit/season on a subset,
    pairwise stats predicted at the left-out stations — the natural test of a distance-kernel
    model); a `Q_RR` quality-flag policy in the loader (flag parsed but never used — count the
    suspect days, decide with T-016); a one-season sensitivity refit on `WET_DAY_THRESHOLD` and
    `MAX_NAN_FRACTION_STATION` (the NaN-aware likelihood also implicitly assumes ignorable
    missingness — state it in §`subsec:nan_handling`); a band-calibration (rank-of-observed /
    PIT-style) check of the 90 % bands; minor: a wet-spell Q-Q beside 1.d, an
    anisotropy/orography residual check on the isotropic kernel.
  - Note: every refit slice adopts the estimator retained by T-013.

---

## Intake — known sources to mine at grooming

- [x] [NOTES_spatial_diagnostics.md](NOTES_spatial_diagnostics.md) §0 — mined 2026-08-20:
      items 2,4,5,6 → T-001; item 3 → T-007; item 10 → T-008; item 1 → E-002; item 7 → E-004;
      items 8–9 are documented advisories, no task.
- [x] main.tex review pass — mined 2026-08-20: Section-3 markup → T-003.
- [x] TODO/FIXME markers in the notebooks — checked 2026-08-20: none found (clean).
- [x] `git status` backlog — mined 2026-08-20 → T-005; KNMI dirs classified under E-003.
- [x] Article completeness — mined 2026-08-20: red TODOs → T-004 (θ-consistency), T-006
      (comparison subsection), E-005 (Matérn stub).
- [x] Article structure against a submittable paper — mined 2026-08-20 (2nd pass): no abstract,
      introduction, related work or conclusion → E-006 + T-021; §5 reports no θ̂ table and shows
      nothing simulated → T-009, T-010; no map of the domain → T-011.
- [x] `experiment_outputs/theta_fit_*.csv` — mined 2026-08-20: duplicated `summer` row → T-009 /
      T-015; σ̂₁ and λ̂ instability across seasons → T-013; Δℓ between the nested variants → T-014.
- [x] Code organisation — mined 2026-08-20: fit driver and ensemble machinery live in notebook
      cells → T-015; ECAD-only loader, no cache → T-016; no automated test anywhere → T-017;
      figure provenance untracked → T-019.
- [x] Current notebook vs `archive_old_single_latent_field/` — mined 2026-08-20: section 1.e and
      its two `*_sim_vs_independent` figures belong to the archived single-latent-field model
      → T-018; T-001 and T-008 re-scoped accordingly.
- [x] Review markup coverage — re-mined 2026-08-20: 20 call sites, those outside Section 3 → T-020.
- [x] Code + outputs survey (AI session) — mined 2026-08-26: stale/inconsistent outputs → T-024;
      prepared-but-never-run θ-consistency design → T-022/T-023; single-start estimator and the
      smoke-fit λ-swap → T-013 evidence note; SE building blocks → T-012 note; new-data pipeline
      request → T-025/T-026; `\hide` on the variants/comparison appendix → T-020 note.

- [x] Evaluation blind-spot review (AI session) — mined 2026-08-26: no interannual-variance or
      lagged cross-correlation diagnostic → T-027/T-028; in-sample-only evaluation, Q_RR flag
      unused, threshold/NaN-policy sensitivity and band calibration → E-007; `Q_RR` note on T-016.

*(add new sources here as they appear — e.g. referee reports, co-author feedback)*

## Decisions log (dated, one line each — why a direction was chosen)

- 2026-09-10 · Owner request: the data-analysis figures are re-read on a record simulated at
  $\hat\theta$ in a new notebook, [data_analysis_fitted_model.ipynb](data_analysis_fitted_model.ipynb)
  (raster, $\widehat{\pi}_{jj'}$ overlay, $\widehat{p}^{\,\mathrm{stay}}$ vs distance and model vs
  estimate on both records). The statistics were lifted verbatim into
  [pair_statistics.py](pair_statistics.py) rather than copied into a third notebook, so both records
  go through one code path (T-015 direction); the observed-side numbers reproduce the data-analysis
  notebook's exactly. Its figures go to `figures/spatial/diag_fitted_model_*` only — nothing is
  written into the article submodule and no existing figure is overwritten. A last section re-reads
  $\widehat{\pi}_{jj'}$ and both $\widehat{p}^{\,\mathrm{stay}}$ figures on 25 independent 41-year
  trajectories (1025 years, no gaps, cached in `sim_ensemble_cache/`), to separate the model's
  departure from the sampling noise of one 41-year draw; the trajectories are stacked because one
  continuous run that long does not fit pandas' timestamp range (year 2262).
- 2026-09-07 · §2.1 rebuilt on the rain-state agreement $\widehat{\pi}_{jj'}$ and the spell-age
  semi-variogram $\widehat{\gamma}_{jj'}(r,r')$ deleted: same conclusion (spatial structure exists,
  decaying with distance), no model and far less notation. The day sets $\mathcal{N}_{jj'}(r,r')$
  went with it; $\mathcal{T}_j$ moves from §3.9 up to §2 and now carries both sections.
- 2026-09-07 · The season stays out of the notation (no superscript on $\widehat{\pi}$ or
  $\widehat{p}^{\,\mathrm{stay}}$): one prose sentence in §2 instead → T-032.
- 2026-09-08 · Owner: a notation introduced in §1 *Notations* but used nowhere is deleted, not
  relocated — applied to $(u_n)/u_{p_1:p_2}$ and to $\mathbf{a}=(a_i)$; the colon sub-sequence
  notation needs no definition because its single use spells the tuple out. Cleared ahead of T-031,
  noted there.
- 2026-09-08 · Owner: the time index of the paper becomes $t$ (and $n_{\text{obs}}\to t_{\text{obs}}$),
  with an explicit sentence **in §1 *Notations*** that the chain is discrete-time — $t$ indexes days,
  it is not a continuous time → T-033. The simulated sample size follows the same family,
  $T\to t_{\text{sim}}$, so no capital $T$ competes with the day sets $\mathcal{T}_j$; $\tau$ (spell
  duration) is kept as is. Code keeps `n`/`n_obs`: third instance of the deliberate paper/code
  divergence, after $p^{\mathrm{stay}}$/`p00` and $a_k$/`sigma_*`. The rename runs ahead of T-002,
  T-031 and T-030 so those passes work on the final symbols.

- 2026-09-08 · T-030 merge accepted as planned, all three merges applied, none refused. The
  absorbed `\label`s are **retargeted**, not kept on `\subsubsection`s or `\paragraph`s: keeping
  them would give §3 its sub-headings back, which is the very reading cost the merge exists to
  remove. Block 2 keeps the title *Spatial extension of the BMCD* and the label
  `subsec:spatial_model`, the Bernoulli-field material being reframed as a tool by a new opening
  paragraph rather than by reordering the two halves; block 3 takes a **new** label
  `subsec:spatial_specification` (13 live call sites retargeted) because the surviving lead label
  `subsec:switch_diagnostics` would have named a block that is no longer about diagnostics alone;
  block 4 keeps *Likelihood* / `subsec:likelihood`.
- 2026-09-08 · The 3.4+3.5+3.6 merge is accepted on the argument it makes explicit — we read the
  spatial structure on the data, then build a latent field to match — which the owner will sharpen
  in the block's opening paragraph. Section titles must make the progression legible: single site
  → spatial extension → full specification → likelihood.
- 2026-09-08 · §3.6's content (the haversine distance convention and the exponential kernel) leaves
  the body for the appendix → T-034, **executed the same day** rather than after T-030's collapse:
  the owner's rule is the same for both formulas — use them in the body, point to the appendix —
  the exponential kernel because it is elementary, the haversine because it is not central to the
  application.
- 2026-09-08 · Block 3's title drops the word "specification": the block *builds* an object, and
  the paper already spends "specification" on §4's dissolved title. **Settled: *Construction of the
  latent field*.** §3 now reads Single site model → Spatial extension of the BMCD → Construction of
  the latent field → Likelihood.
- 2026-09-08 · Block 2's opening was **reordered** on the owner's plan: Bernoulli random field →
  multivariate binary and why it is intractable → *therefore* parsimonious subfamilies, thresholded
  Gaussian fields, given as two examples (the fixed-level excursion set of spatial statistics, then
  the varying-level construction of Joe/Emrich–Piedmonte that the paper uses). The previous order
  introduced the thresholding example before the problem it solves. Compaction verdicts, same pass:
  the starred-$z^{*}$ convention **moved** to §1 *Notations* (it is global, not local to the
  subsection); the multivariate-binary detour compressed to the owner's two-sentence version; the
  sentence restating `eq:gaussian_thresholding_sugg` deleted; the duplicated "we couple the sites
  only through…" deleted; the broken "Let us rewrite the specific case…" sentence repaired.
  $\tau^{(r)}_{\mathbf{s}}$ is **kept** despite being unused after its introduction (owner).
- 2026-09-08 · Owner: a display equation is reserved for what the model actually uses. The generic
  varying-level thresholding $B(\mathbf{s})=\mathbbm{1}(Z(\mathbf{s})\le z^{*}_{\mathbf{s}})$, which
  only illustrates the construction borrowed from the literature, becomes inline text and loses its
  number and label (`eq:gaussian_thresholding_sugg`, formerly eq. (4)); the state-indexed threshold
  `eq:new_z_threshold`, which the model uses, stays a numbered display. **All equation numbers from
  the old (5) onwards shift down by one.**
- 2026-09-08 · T-030 ran **before** T-033 (the $n\to t$ rename), against T-033's stated ordering:
  the merge was already applied when the rename task was written. Kept in that order rather than
  reverting — the ordering rule exists for review legibility, which is preserved as long as the two
  passes are validated and collapsed in sequence, and collapsing T-030 first strictly reduces the
  rename's surface, since the orange text it would otherwise have to sweep is deleted at collapse.

- 2026-08-20 · Article writing quality (organisation, flow, notations, wording, repetitions) is
  the standing top priority → E-001; notation pass (T-002) goes first.
- 2026-08-20 · Model (4) dead code (NOTES §0.1): not decided in isolation — folded into the
  model-comparison epic E-002 and settled when that work starts.
- 2026-08-20 · KNMI Rotterdam work classified as the first instance of the multi-dataset
  application epic E-003, not a side experiment.
- 2026-08-20 · The article's missing skeleton (abstract, introduction, related work, conclusion)
  is its own epic E-006, not a slice of E-001: E-001 polishes text that exists, E-006 writes text
  that does not.
- 2026-08-20 · Uncertainty on θ̂: the Godambe sandwich (T-012) is done first, ahead of E-004's
  parametric bootstrap — it is cheap and it supplies the tr(J H⁻¹) that the CLAIC/CLBIC of
  T-006/T-014 needs anyway.
- 2026-08-20 · Section 1.e belongs to the archived single-latent-field pipeline, not to the
  current fit notebook (its plotters' only caller is `archive_old_single_latent_field/`) → port it
  forward (T-018) before T-008; NOTES §0.4 moved out of T-001 into T-018.
- 2026-08-20 · E-003's first *spatial* instance switched from KNMI Rotterdam — a single station,
  which cannot exercise a spatial model — to a second ECAD region; Rotterdam kept as a
  single-site marginal check.
- 2026-08-20 · The σ̂₁ / λ̂ instability of the Iberian fit is tracked separately (T-013) from the
  simulated-data consistency rerun (T-004): one asks whether this dataset's likelihood surface is
  identified, the other whether the estimator is consistent.
- 2026-08-26 · Owner: current focus is model explanation, testing, evaluation and results
  analysis; article writing deferred until results stabilize — Now/Next reordered, writing tasks
  kept in Next as a deferred block.
- 2026-08-26 · T-004 split per owner: runner finalization (T-022) and execution (T-023) are
  separate tasks; T-004 keeps the results analysis and the paper-side updates.
- 2026-08-26 · T-013 promoted to Now: the λ-swap already appears on simulated data in the smoke
  fit and the estimator is single-start — identifiability is the gating scientific question
  behind every number the paper would report.
- 2026-08-26 · New T-024 from the code survey: stale-outputs repair (duplicate `summer` row,
  pre-rescope `switch_pair_stats`, stale notebook outputs) is its own Now task, ahead of any new
  analysis run.
- 2026-08-26 · Owner: the model must be easy to fit and evaluate on new city sets — E-003
  decomposed into T-025 (one-command fit→evaluation pipeline, absorbing the single-site marginal
  step) and T-026 (second ECAD region), raising the priority of T-015/T-016 which gate them.
- 2026-09-02 · Owner: unfinished article material stays **visible in the PDF as short red notes**
  (no TODOs buried in LaTeX comments); the detail and the open questions live in this file,
  referenced from the note by task id. Applied to §2, whose French red note became a two-line
  English TODO pointing at T-011/T-016.
- 2026-09-03 · Owner: §4 *Model specification* is dissolved — its exponential-kernel content
  becomes §3.6 *Exponential correlation function*, placed **before** the likelihood subsections so
  that the correlation blocks are fully specified before the likelihood uses them, and compacted
  to two paragraphs plus the two displays. The Matérn stub subsection is deleted and demoted to a
  perspective (E-005 updated accordingly). The pass is in the file as orange/blue markup → T-029.
- 2026-09-03 · Owner: the latent-field range is written $a_k$ ($a_c,a_0,a_1$ — the classical
  geostatistics *range*) in the paper instead of $\sigma_k$, which read as a standard deviation
  although the latent fields are standardised. Code and fit-CSV names (`sigma_c/sigma_0/sigma_1`)
  are **not** renamed: same deliberate paper/code divergence as $p^{\mathrm{stay}}$ vs `p00`.
  Candidates rejected: $\ell_k$ (clashes with the log-likelihood ℓ that T-009/T-014 will report),
  $\kappa_k$ (a rate in km⁻¹, inverts the reading of every fitted value), $\phi_k$ (clashes with
  $\varphi$/$\Phi$).
- 2026-09-03 · Owner: §3's nine subsections merge to four (3.2+3.3, 3.4+3.5+3.6, 3.7+3.8+3.9)
  → T-030; the merge runs *after* T-003 so the restructure does not sit on unvalidated review
  markup, and absorbed `\label`s are kept on subsubsections or their ≈34 `\ref` sites retargeted.
- 2026-09-03 · Notation work split in two: T-002 keeps *consistency* (each symbol defined once,
  no conflicting duplicates, no orphans), T-031 takes *economy and placement* (fewer symbols,
  fewer introduction sites, none introduced where it is not used — the Notations section being
  the only exception). T-031 consumes T-002's audit table, and carries the two open notation
  decisions previously parked in §3's LaTeX comment blocks, so T-003 can delete those freely.
- 2026-08-26 · Evaluation blind-spot review: owner grooms only the cheap ensemble-reuse
  diagnostics now (interannual variance → T-027, lagged cross-correlation → T-028); holdout /
  sensitivity / calibration work parked as epic E-007 until T-013's verdict and the T-024
  refresh land.

## Done log (append-only, newest first: date · ID · one-line outcome)

- 2026-09-08 · T-003 · §3 review markup cleared, as the gate to T-030. The owner accepted the
  `\added` of the former §`subsec:nan_handling` (its now-duplicated `Section~\ref{sec:data}` marked
  orange in the T-030 pass) and ordered the deletion of the two remaining review comment blocks —
  the one above `tab:station_notation` and the one at the head of the co-persistence subsection —
  whose open notation decisions T-031 already carries. No `\added`/`\removed`/`\orange` call site
  from the restructure pass is left in §3; the colour macros now in §3 belong to the T-030 merge
  pass. Document compiles, 22 pp, the same 3 pre-existing undefined refs (T-020).

- 2026-09-03 · T-029 · §4 *Model specification* dissolved into a compacted §3.6 *Exponential
  correlation function* sitting before the likelihood subsections; Matérn stub deleted (→ E-005
  perspective); range renamed $\sigma_k \to a_k$ at all 14 sites; review markup written, validated
  by the owner (who further compacted the new subsection) and collapsed the same day. main.tex
  22 pp, no `sec:model_specification` left, same 3 pre-existing undefined refs.
  Follow-up closed the same day: the "fitted marginally per station, treated as known inputs"
  statement (with its red `ref XYZ`) was dropped from §3.6 in the compaction and re-inserted, on
  the owner's choice between two proposed phrasings, at the top of §`subsec:likelihood`; §5's
  pointer retargeted from `subsec:exponential_cov_function` to `subsec:likelihood`.
