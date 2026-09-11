# Notes — spatial BMCD fit and diagnostics (Iberia, PT + ES)

Companion to [`spatial_lin_mod_coregion_fit_spain_portugal.ipynb`](spatial_lin_mod_coregion_fit_spain_portugal.ipynb).
The notebook keeps only what is needed to *read* the figures; this file holds the reasoning, the
construction details, the caveats, the numbers and the open issues that would otherwise flood the
markdown cells. Written to be read by a later session (human or agent) picking the notebook back up.

Scope: the Iberian (PT + ES) fit, 26 stations, 1980–2020, 14 976 days, model (5) with
state-dependent mixing weights $\lambda^{(0)}, \lambda^{(1)}$.

---

## 0. Open issues and TODO

Ordered roughly by how likely they are to bite. Items marked *(blocking)* break a run; the rest are
correctness caveats or clean-ups.

1. **Model (4) is dead code, and the notebook still half-claims otherwise.** The 4-D shared-$\lambda$
   fit (`fit_theta_by_season(RESULTS_PATH, ...)`), its `load_theta_by_season` call and its
   `Rbin_sim` simulation are all commented out, yet the Part-2 loading cell still prints a bare
   `"model (4), shared lambda:"` header with nothing under it. **Decide and act:** either re-enable
   the three commented blocks (≈ 10 min/season of MLE, and every diagnostic of 1.c–1.e would then
   need a fourth curve or a second figure), or delete them and the header outright. Leaving them
   commented is the worst of the three.
2. *(blocking)* **`HEAVY_DRY_MIN_DAYS` is defined in the 1.b raster cell but consumed in 1.e.**
   Running Part 2 from the ensemble cell onwards — the natural thing to do once the raster has been
   drawn — raises `NameError`. Move the constant up to the ensemble cell (or the Setup cell) with
   the other diagnostic constants.
3. **`BAND_QUANTILES = (0.05, 0.95)` at `N_REPLICATES = 30` puts the band edges at order statistics
   ≈ 1.5 and ≈ 28.5 out of 30** — essentially the extremes, and therefore noisy. Tolerable for a
   qualitative read, but the band-*width* comparison of §7 is a quantitative claim: for a paper
   figure, either raise `N_REPLICATES` (cost is linear) or quote an interquartile band alongside.
4. **The 1.e plotters hard-code a 2.5 %–97.5 % envelope.** `plot_pairwise_stat_vs_distance` and
   `plot_fraction_distribution` in [`spatial_diagnostics.py`](spatial_diagnostics.py) do not read
   `BAND_QUANTILES`, so the bands of 1.e are 95 % envelopes while those of 1.c/1.d are 90 %.
   Harmless as long as the captions say so — but **do not compare band widths across those
   sections** without correcting for it. Fix properly by threading `BAND_QUANTILES` through both
   plotters.
5. **1.d shares one `qq_rng` across every panel and both ensembles.** `qq_dry_spell_band` consumes
   `qq_rng` sequentially, so which durations get thinned away depends on the loop order and on
   `N_QQ_STATIONS`: changing the number of plotted stations silently changes the bands of the
   stations that were already there. The figure is reproducible only for a fixed loop layout.
   Fix by seeding per (station, season, ensemble), e.g.
   `np.random.default_rng((QQ_SUBSAMPLE_SEED, hash(station), season_index, tag))`.
6. **The 1.c band legend labels both ensembles with `len(Rbin_sim_reps)`.** True today because the
   two ensembles have the same size by construction, but it is a silent lie the moment they differ.
7. **The estimation uncertainty on $\hat\theta_s$ is not propagated anywhere.** Every band in the
   notebook holds $\hat\theta_s$ fixed, so all of them are *lower bounds* on the total spread
   (see §6). Propagating it would mean a parametric bootstrap over the pairwise MLE — expensive, and
   not currently planned.
8. **`spell_time_fraction`'s default `d_max` is `max(D_MAX_BY_KIND.values())`**, evaluated at
   definition time from a dict defined just above it. So the wet panel computes 120 unused columns
   (trivial cost), and editing `D_MAX_BY_KIND` after the fact does not change the default. Pass
   `d_max` explicitly per kind if that ever matters.
9. **`simulate_block_ensemble` is superseded** — see §9. Do not wire it into the 1.e plotters.
10. **The non-parametric null on the observed side is not implemented** — see §8. Worth a few lines
    if a reviewer questions whether the parametric null is doing the work.

---

## 1. The two models, and what Part 1 fits

**Model (4)** — the original shared-$\lambda$ LMC of
[main.tex](Spatialisation-Rain-Occurrence-Generator-article/main.tex), Eq. (4)
(`lmc_fields_direct`):

$$
Z^{(r)}_n(\mathbf{s})=(-1)^{r}\sqrt{\lambda}\,W^{(c)}_n(\mathbf{s})+\sqrt{1-\lambda}\,W^{(r)}_n(\mathbf{s}),
\qquad r\in\{0,1\},\quad \lambda\in[0,1],
$$

with $\theta=(\lambda,\sigma_{w^c},\sigma_{w^{(0)}},\sigma_{w^{(1)}})$.

**Model (5)** — the alternative suggestion of main.tex (blue): the mixing weight depends on the
state $r$, $\lambda^{(0)}$ for the dry field and $\lambda^{(1)}$ for the wet field, so
$\theta=(\lambda^{(0)},\lambda^{(1)},\sigma_{w^c},\sigma_{w^{(0)}},\sigma_{w^{(1)}})$ and the
covariance blocks become

$$
C^{(r,r')}_n(\mathbf{s},\mathbf{s}')=
\begin{cases}
\lambda^{(r)}\,\rho_{w^{(c)}}+(1-\lambda^{(r)})\,\rho_{w^{(r)}}, & r=r',\\[4pt]
-\,\sqrt{\lambda^{(0)}\lambda^{(1)}}\,\rho_{w^{(c)}}, & r\neq r'.
\end{cases}
$$

Model (4) is the nested special case $\lambda^{(0)}=\lambda^{(1)}$, so the 5-D fit must reach at
least the shared-$\lambda$ log-likelihood ($\Delta\ell \ge 0$) — a cheap sanity check on the
optimiser whenever both CSVs exist.

Estimation is the pairwise composite likelihood of main.tex §3.2 (fast vectorized `approx_cdf`
kernel), validated on simulated data in
[tests_parameter_estimation.ipynb](tests_parameter_estimation.ipynb). Cost ≈ 10 min/season for the
4-D fit, somewhat more for the 5-D one.

**Caching.** Both fits go through `fit_theta_by_season`, which appends each finished season to its
CSV in `experiment_outputs/` *immediately* — so an interrupted run resumes where it stopped — and
skips on re-run any season already present. Delete the CSV (or one row of it) to force a re-fit.
The two models have separate CSVs.

**Station set, settled in two passes.** How incomplete a station is can only be read from the raw
record, so a provisional set says which files to load, `filter_stations_by_nan_fraction` drops the
stations missing more than `MAX_NAN_FRACTION_STATION` of the window (see
[`config_spatial.py`](config_spatial.py)), and the second pass gives the definitive set. Only the
load touches the disk; the second `build_params_by_season` is cheap.

---

## 2. Simulation across successive seasons

One continuous trajectory per model on the observed daily calendar, by the Cholesky algorithm of
main.tex §5 (`simulate_cholesky_seasonal` in [`spatial_model.py`](spatial_model.py)), with
season-dependent parameters. Three rules, and they are not the same rule:

* **Latent fields — switch immediately.** Day $n$ uses $\hat\theta_{s(n)}$ of the current season, so
  the covariance blocks change on the *first day* of each new season.
* **Exit probabilities — keep the spell's starting season.** A dry spell that starts in spring keeps
  the spring $q^{(0)}$ until it ends, even after summer has begun; the next spell is stamped with
  the season of its own first day. Every diagnostic that splits spells by season
  (`dry_spells_by_station_season` in 1.d) reproduces this stamping rule, so the pools match what the
  simulator did.
* **Burn-in.** The calendar is extended 1000 days backwards, following the real season sequence, and
  those days are discarded.

Marginally, each station behaves as the single-site BMCD of its spell's season; the latent fields
carry the spatial dependence and nothing else. That invariance is exactly what 1.d tests.

**Station ordering.** The raster and Q-Q figures use the stations as an ordered axis: the 2-D
station cloud is turned into a 1-D sequence in which list-neighbours are also map-neighbours
(seriation of the inter-station distance matrix, approximated by hierarchical clustering with
optimal leaf ordering — appendix of main.tex). `plot_station_order_map` shows where each column
(left → right) sits geographically. Normalised coordinates are (near-)identical across seasons, so
spring's are used for all of them.

---

## 3. What the diagnostics compare

Every figure of sections 1.c–1.e carries **three** things:

| curve | colour | what it is |
|---|---|---|
| observed | black | the single 1980–2020 ECAD record |
| model (5) | steelblue | ensemble of `N_REPLICATES` seasonal Cholesky trajectories at $\hat\theta_s$ |
| independent stations | darkorange | the same, with the spatial dependence switched off |

The colour convention is fixed in the ensemble cell (`COLOR_OBS`, `COLOR_SIM`, `COLOR_INDEP`) and
matches what `spatial_diagnostics.plot_pairwise_stat_vs_distance` /
`plot_fraction_distribution` hard-code internally (steelblue = primary simulated ensemble,
darkorange = the `*_alt` ensemble they overlay). Section 1.d used to draw model (5) in darkorange;
it was switched to steelblue so that darkorange means "no spatial structure" throughout.

A single replicate cannot tell a genuine misfit from the wandering of one 41-year draw, which is
precisely what matters in the tails: 1.c and 1.d read the far right of the spell-duration
distribution, where a handful of spells decide the curve. Hence the ensembles.

---

## 4. The null model: how it is built, and why no refit is needed

The three latent fields are correlated **only** through the exponential kernel
$\exp(-h/\sigma_k)$ (`spatial_model.build_C_from_sigma`). Collapsing the three ranges to
$\sigma \to 0$ turns each correlation matrix into the identity, so:

* the Cholesky factors `Lc, L0, L1` are `I`, hence $W^{(c)}, W^{(0)}, W^{(1)}$ are white noise
  and every station runs **its own** single-site BMCD;
* the marginals are untouched, because
  $Z^{(r)} = \pm\sqrt{\lambda_r}\,W^{(c)} + \sqrt{1-\lambda_r}\,W^{(r)}$ is still $N(0,1)$ per
  station whatever $\lambda_r$ is. The $\lambda$'s are therefore carried over unchanged from
  $\hat\theta_s$ — they are inert once the fields are independent, and keeping them makes the only
  difference between the two ensembles the three ranges.

In the notebook:

```python
EPS_KM = 1e-6
theta_by_season_indep = {s: (th[0], th[1], EPS_KM, EPS_KM, EPS_KM)
                         for s, th in theta_by_season_sd.items()}
```

**No refit.** This *is* the fitted independent model, not an approximation of one: under an
independence assumption the pairwise composite likelihood of §3.2 is maximised at $\sigma \to 0$ by
construction, and everything marginal — the per-station, per-season exit probabilities
$q^{(0)}, q^{(1)}$ — was fitted upstream by the single-site pipeline and is reused as is. So the
null shares the *exact* marginal specification of model (5); the only thing it lacks is the spatial
coupling. That is what makes the comparison clean.

### Caveat: co-located stations

$\exp(-h/\sigma) \to 0$ needs $h > 0$. Two stations at identical coordinates would stay perfectly
correlated whatever the range, and the null would not be a null for that pair. The ensemble cell
asserts `min(off-diagonal distance) > 0` and prints the closest pair before simulating. On the
current 26-station Iberian set this passes comfortably.

### Numerical note

`exp(-h / 1e-6)` with $h$ in km underflows to exactly `0.0` for any realistic separation
(e.g. $h = 10$ km gives $\exp(-10^7)$), silently and without a warning; the diagonal is
`exp(0) = 1`. The resulting matrix is exactly `I`, and `np.linalg.cholesky(I)` is exact — no jitter
needed.

### Cost

Identity Cholesky factors buy **no** speed-up: the per-day cost is dominated by the Python loop in
`_q_thresholds`, which calls one fitted closure per station per day, not by the `L @ z` matvecs.
Budget the same wall clock as the model-(5) ensemble — measured ≈ 12.6 s per replicate, so ≈ 7.5 min
per ensemble of 30, ≈ 15 min for both. Memory is not a constraint (≈ 50 MB per ensemble, float32).

### Masking

Every replicate goes through the notebook's local `mask_like_obs`: reindexed on the observed
calendar and columns, then `.where(Rbin_obs.notna())`. The simulation has no gaps, so masking it
with the observed pattern makes both sides lose *exactly* the same station-days to the censoring
rule of `_complete_runs`. Values are 0/1/NaN, so `float32` is exact and halves the memory.

---

## 5. Seeds: why the two ensembles use disjoint ranges

Model (5) uses seeds `0 … N-1`, the null uses `1000 … 1000+N-1`.

Both models consume the RNG identically (three `standard_normal(m)` draws per simulated day), so
reusing seeds `0 … N-1` for the null would hand the two ensembles the *same raw white noise* — a
common-random-numbers coupling. That is harmless and mildly variance-reducing on a comparison of
medians, but section 1.c makes a claim about the **relative width of the two bands**, and that claim
is much easier to defend if the two ensembles are independent. Hence the offset. (The coupling would
in any case decay fast: the chains diverge as soon as their states differ, since
`_assemble_selected_field` and `_q_thresholds` are state-dependent per station.)

Seed 0 stays first in the model-(5) ensemble so that `Rbin_sim_reps[0]` is bit-for-bit the
`Rbin_sim_sd` trajectory drawn alone in the raster plot of 1.b, keeping the two views consistent.

---

## 6. What the bands do and do not cover

The band is the **sampling variability of the model at $\hat\theta_s$ fixed**: how far a 41-year run
of the fitted model wanders from one draw to the next. The estimation uncertainty on $\hat\theta_s$
is *not* propagated, so the band is a **lower bound** on the total spread. Reading rule:

* observed curve outside the band → evidence of misfit;
* observed curve inside → compatible with the fit;
* and the band widens exactly where the statistic is decided by a handful of long spells (far right
  of the dry panels), so a departure there is much weaker evidence than one in the body.

Two caveats that live in §0 because they are actionable: the band resolution at 30 replicates
(item 3) and the 90 % / 95 % mismatch between 1.c–1.d and 1.e (item 4).

---

## 7. Which diagnostics actually discriminate (and the band-width argument)

**1.c and 1.d are both "marginal" diagnostics, but the null reads differently in each — the
distinction is pooling, and it is easy to get wrong.**

### 1.c, $F^{(r)}(d)$ — pooled over stations, so the band widths differ

Definition: with $L_n(\mathbf{s})$ the length in days of the spell containing day $n$ at station
$\mathbf{s}$, and $\mathcal{V}$ the set of station-days carried by an uncensored spell,

$$
F^{(r)}(d)\;=\;\frac{\#\bigl\{(n,\mathbf{s})\in\mathcal{V}\;:\;R_n(\mathbf{s})=r,\;\;L_n(\mathbf{s})>d\bigr\}}{\#\,\mathcal{V}}\,,
$$

so a day sitting inside a 7-day wet spell is counted in $F^{(1)}(d)$ for every $d<7$. At $d=0$,
$F^{(1)}(0)$ is the wet-day frequency and $F^{(0)}(0)+F^{(1)}(0)=1$. Compared with the plain
survival curve of the spell durations, this weights each spell by its own length, so the long
spells — the ones the spatial field has to reproduce — dominate the right tail.

`D_MAX_BY_KIND` gives each panel its own horizon: wet spells are short, whereas Iberian dry spells
routinely run for weeks, so the dry panel goes out to $d=150$ against $d=30$ for the wet one. The
$d$-grid is identical for every replicate, so the bands collapse column by column with no
interpolation. The $y$ axis is logarithmic, so the lower edge is floored just below the smallest
positive value drawn (a zero would map to $-\infty$ and tear the polygon), and past the longest
simulated spell the band is dropped to NaN rather than left as a zero-height sliver on that floor.

Now, the reading. The obvious expectation is that the null lands on top of model (5), since both
have the same per-station marginals by construction — which would make the null useless here. That
is only half true, and the other half is the interesting part:

* **The medians should coincide.** This is a genuine check, not a tautology: it confirms that the
  LMC coupling has not distorted the marginal spell law it was supposed to preserve. If the two
  medians *separate*, something is wrong in the simulator, not in the fit.
* **The two bands should not have the same width, and the gap should be large.** $F^{(r)}(d)$ is
  pooled over all 26 stations; under spatial dependence the effective number of independent
  station-years is a small fraction of $26 \times 41$, so at the same 30 replicates and the same
  41-year record the model-(5) band is materially wider than the null's.

That second point reframes what the band is for. A generator without spatial structure would produce
an uncertainty band several times too narrow, and would therefore **declare the observed curve a
misfit where model (5) correctly calls it compatible**. The null is not a curve that overlaps; it is
the calibration of the band, and the ratio printed by the 1.c table

```
width ratio = [(p95 - p05) / p50]_model(5)  /  [(p95 - p05) / p50]_independent
```

is the quantitative form of "how much of the model's own sampling spread is spatial dependence".
A ratio ≫ 1 is the headline number. (`_safe_div` guards the far tail, where both the widths and the
medians go to zero because no replicate carries a spell that long — without it the table fills with
`inf` and a `RuntimeWarning` instead of a blank.)

### 1.d, dry-spell Q-Q — per station, so the two bands should be *identical*

What is being tested: in
[04_palermo_diagnostics.ipynb](../article_code/notebooks_article_pipeline/04_palermo_diagnostics.ipynb)
the simulated axis of the same Q-Q is an i.i.d. draw from the *fitted single-site law*

$$
\tau^{(0)}\;\sim\;\hat f_1\,\delta_1+(1-\hat f_1)\,\bigl(2+\mathrm{extGPD}_1(\hat\xi,\hat\sigma,\hat\kappa)\bigr),
$$

so that plot tests the single-site fit. Here the simulated axis is read off the **spatial**
trajectories instead. Marginally each station of the spatial BMCD is supposed to behave exactly as
its own single-site BMCD, so the question is whether the seasonal Cholesky simulation, run at
$\hat\theta_s$ with state-dependent $\lambda^{(r)}$, has left those marginals intact. A systematic
departure from the diagonal means the spatial layer has distorted the durations it was only supposed
to decorate.

Construction. The marginal is checked *station by station* rather than on a pooled sample: one row
per station, one column per season, so reading across a row compares the four seasons of a single
station and reading down a column compares the stations. `N_QQ_STATIONS` stations are taken evenly
along the spatial ordering `order`, so the rows span the station cloud end to end. Within a panel,
the dry spells are split by the season of their *first day* (§2). Plotting positions
$p_i=(i-\tfrac12)/n$ are fixed by the **observed** sample size, so every replicate contributes a
curve over the same abscissae — the observed order statistics — and the ensemble collapses column by
column. `_order_statistics` uses the inverted-CDF rule rather than an interpolated quantile, which
keeps the result on the integer grid the durations live on so that coincident Q-Q pairs can be
counted and merged into one larger marker.

With `QQ_MATCH_SAMPLE_SIZE`, a replicate pool larger than the observed one is thinned to $n$
durations without replacement, so a band is the spread of a record of the *same length* as the
observed one rather than the narrower spread of a quantile estimated from a larger sample; since
both pools run over the same 41 years the thinning is light. (See §0 item 5 for the RNG caveat.)
With one station contributing ~26 times fewer spells than a pooled sample, the panels are noisy by
construction — which is exactly what the bands make visible.

**The band-width argument of 1.c does NOT carry over here.** These pools are built one station at a
time (`dry_spells_by_station_season`), so the spatial dependence plays no role whatsoever: a single
station's chain has exactly the same law under both models — $Z_j$ is $N(0,1)$ and independent
across days either way — and the two bands are simply two 30-replicate samples of the *same*
distribution. They should coincide in position **and** in width, up to Monte-Carlo error.

So in 1.d the null is a **consistency check, not a contrast**. A visible separation means one of two
things, both worth chasing: either the spatial layer has distorted the marginal it only had to
decorate, or the $\sigma \to 0$ construction is not producing the independence it claims (start with
the co-location check of §4).

### 1.e — the genuine spatial contrast

This is where the null bites hardest, and where the three-way figure carries the argument. All three
statistics go through `blocks_from_Rbin`, which splits each matrix into one block per calendar year
and keeps the raw 0/1/NaN values, so the censoring is re-applied symmetrically to observed and
simulated blocks.

* **$\phi(h)$, `p11`, `p00` vs inter-station distance** — for each of the $\binom{26}{2}=325$ pairs:
  $\phi$ (correlation of the wet indicators), `p11` (both wet), `p00` (both dry), against the
  haversine distance. The null sits flat at $\phi \approx 0$ across all distances with a tight band;
  the vertical gap up to the black observed dots *is* the spatial signal, and what model (5) has to
  reproduce is that gap and its decay with $h$. `phi_dry` is exactly `phi` (the $\phi$ coefficient
  is invariant under $0\leftrightarrow1$), so only three of the four available statistics are worth
  a panel. The dashed $\hat\sigma$ marker is the **median of `sigma_wc` across the four seasons**
  (the four fitted values differ), so read it as an order of magnitude, not as a per-season range.
  Cost: 325 pairs × 61 matrices, ≈ 1 min in total for the $O(m^2)$ loop of `_pairwise_binary_stats`.
* **Daily wet fraction** — the share of stations wet on a given day. The null collapses to a
  near-binomial spike around the mean wet frequency; the observations are strongly overdispersed
  (genuinely region-wide wet days and region-wide dry days). The most legible panel of the three.
* **Heavy-dry daily fraction** (drought extent, `HEAVY_DRY_MIN_DAYS = 10`) — same reading, on the
  statistic the spatial field most needs to get right: how much of the peninsula is simultaneously
  deep inside a dry spell.

The printed standard deviations are the one-number summary of the last two panels: an observed
spread far above the null's, with model (5) in between or on target, is the quantitative statement
of the figures.

---

## 8. Not implemented: a non-parametric null on the observed side

Circularly shifting each observed station column by an independent random lag (`np.roll` per column
on `Rbin_obs`) keeps each station's marginals *and* its temporal dependence exactly as observed
while destroying the spatial dependence. It separates "the model gets the spatial structure right"
from "the model gets the marginals right" with no model assumption at all. A few lines, worth adding
if a reviewer questions whether the parametric null is doing the work. Watch the edge effect: a
circular shift welds December 2020 onto January 1980, creating one spurious junction per column
(negligible over 14 976 days, but it should be stated).

---

## 9. Related code, and one deprecation

* `spatial_diagnostics.simulate_block_ensemble` is **superseded** by the notebook's replicate cell.
  It builds a *stationary* "perpetual spring" ensemble (one fixed $\theta$, block layout cut out of
  a single long run), whereas the notebook's `Rbin_sim_reps` follow the real season sequence with
  one $\hat\theta_s$ per season and the observed NaN mask. Do not wire the weaker ensemble into the
  1.e plotters — pass `blocks_from_Rbin(rep)` over the seasonal replicates instead. A pointer to this
  effect has been added to its docstring.
* `blocks_from_Rbin` splits per calendar year and keeps raw 0/1/NaN; the statistics functions
  re-apply the censoring symmetrically, so observed and simulated blocks lose the same station-days.
* `_complete_runs` (1.c cell) run-length encodes one station column with NaN as a sentinel, then
  drops every run touching a NaN run or a column edge — exactly the rule of `extract_complete_spells`
  (the first and last run of each NaN-free segment have their true length cut short by the gap or by
  the record boundary). Both 1.c and 1.d go through it.
* The notebook's local `mask_like_obs` (ensemble cell) is *not* the `mask_like_obs` of
  `spatial_diagnostics` (which operates on blocks, not on a DataFrame). The setup cell does not
  import the latter, so there is no shadowing — but do not add it to the imports without renaming.

---

## 10. Notebook cell map

| section | contents |
|---|---|
| Setup | spell data, station metadata, single-site fits, per-season PT/ES subsets, raw daily RR |
| Part 1, model (4) | per-year histories (shared by both fits) + 4-D pairwise MLE — **commented out**, §0 item 1 |
| Part 1, model (5) | 5-D pairwise MLE, own CSV |
| load $\hat\theta_s$ | reads the two fit CSVs from `experiment_outputs/` |
| simulation | `Rbin_obs`, single trajectory `Rbin_sim_sd` (seed 0) |
| station ordering | seriation of the distance matrix → `order`, map |
| 1.b | occurrence raster, observed vs model (5) |
| **ensemble** | `Rbin_sim_reps` (seeds 0…29) and `Rbin_indep_reps` (seeds 1000…1029), both masked |
| 1.c | $F^{(r)}(d)$, three-way, + spread-ratio table |
| 1.d | dry-spell Q-Q per station × season, three-way |
| 1.e | pairwise $\phi$/`p11`/`p00` vs distance, daily wet fraction, heavy-dry extent — three-way |

Figures are written to `figures/spatial/` (`SPATIAL_FIGURES_DIR`).
