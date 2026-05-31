# Faster pairwise-MLE: in-house bivariate-normal CDF for `phi2`

## Context

Inference (`mle_sigma_pairwise*`) maximises the pairwise composite likelihood of
§3.3–3.4 of `main.tex`. The hot spot is the bivariate Gaussian CDF
`q_joint = Φ₂(z_{n,j}, z_{n,j'}; ρ*)` (`eq:sigma_Z_pair`), computed by
[`phi2`](../Desktop/code/rainfall_occurrence_BMCD/spatial_bmcd/spatial_model.py#L405-L409),
which builds a fresh `scipy.stats.multivariate_normal` object and runs Genz QMC
on **every** call inside the double `for j,k` loop — tens of millions of
heavyweight calls per fit.

The marginals are already exact/free (`Φ(z[j]) = q[j]` since `z = norm.ppf(q)`),
so only the bivariate term needs accelerating. `phi2` has exactly one caller,
[`pairwise_loglik_one_step_nanaware`](../Desktop/code/rainfall_occurrence_BMCD/spatial_bmcd/spatial_model.py#L412-L460).

Per the user's decisions, this plan does the **minimum, lowest-risk speedup now**
(swap the `phi2` body, keep the loop), **defers** the loop vectorization to a
separate notes file, **adds** a runnable accuracy+timing verification, and
**supplies** a LaTeX paragraph + references for the article.

## Deliverable 1 — APPLY NOW: swap the `phi2` body (`spatial_model.py`)

- Add a module-level **`_bvnd(dh, dk, r)`** implementing Genz's standardized
  bivariate-normal upper-tail `P(X>dh, Y>dk)` (Genz 2004 `BVND`, the
  Drezner–Wesolowsky Gauss–Legendre quadrature with the `|r|<0.925` integral
  branch and `|r|≥0.925` near-boundary expansion). Write it with **numpy
  elementwise ops + `np.where`** for the two branches and a fixed high-order
  Gauss–Legendre node set (≈10 nodes via ± symmetry) so it reaches ~1e-12 for
  all `r` and — as a free bonus — **already accepts arrays** (used by
  Deliverable 2 later). Reuses `scipy.stats.norm` for the `Φ(−h)Φ(−k)` term.
- Rewrite **`phi2`** body to `Φ₂(z1,z2;ρ) = _bvnd(−z1, −z2, ρ)` (lower-tail via
  the `(−X,−Y)` symmetry, which preserves `r`), keeping the existing
  `np.clip(rho, ±0.999999)`. Same scalar signature as today.
- **`pairwise_loglik_one_step_nanaware` is left UNCHANGED** — the `for j,k` loop
  still calls `phi2(z[j], z[k], rho_star)` scalar-wise. Only `phi2`'s internals
  change, so every downstream quantity (`q_joint`, the four cell probabilities,
  `ll`, `sigma_hat`, NaN handling of §3.4) is identical up to ~1e-12.
- Keep the `scipy.stats.multivariate_normal` import (still used by the
  simulation path `step_spatial_markov`).

This single change removes scipy's per-call object construction — the biggest
single win — with no change to the algorithm's structure.

## Deliverable 2 — DO NOT APPLY: write a deferred-work notes file

Create **`spatial_bmcd/vectorize_pairwise_likelihood.md`** (a standalone guide
the user can apply later). It documents *why* (kill the Python pair loop on top
of the per-call cost already removed) and *how*, including:

- The exact vectorized rewrite of `pairwise_loglik_one_step_nanaware` using
  `np.triu_indices(m, 1)`: build `sign = 1 − 2·((Rv[j]+Rv[k]) % 2)`,
  `rho = sign * C_v[j,k]`, one call `q_joint = phi2(z[j], z[k], rho)` (already
  array-capable from Deliverable 1), then the four cases via boolean masks /
  `np.where`, and `ll = np.sum(np.log(np.maximum(p, eps)))`.
- The **Fréchet safety clip** `q_joint = clip(q_joint, max(0,qj+qk−1), min(qj,qk))`
  as an optional robustness net (near-no-op with the accurate `_bvnd`).
- A note that, because `_bvnd`/`phi2` are already vectorized, this is a pure
  loop rewrite with no new numerics, and the same accuracy/verification applies.

## Deliverable 3 — APPLY: runnable verification (acceleration + accuracy)

Create **`spatial_bmcd/validate_fast_phi2.py`** (runnable: `python
spatial_bmcd/validate_fast_phi2.py`):

1. **Accuracy kept** — mesh `z1,z2 ∈ [−3.5,3.5] × ρ ∈ {0,±0.3,±0.6,±0.9,±0.99}`
   (incl. the `≥0.925` branch); compare new `phi2` vs
   `scipy.stats.multivariate_normal(...).cdf`; print and `assert max|Δ| < 1e-5`
   (expect ~1e-12). Since `phi2` is the only thing changed, matching scipy here
   is *sufficient* proof that `sigma_hat` is unaffected.
2. **Acceleration** — time `N` random calls of a reference scipy `phi2` vs the
   new `phi2`; print the speedup factor.

(Self-contained — no ECAD data needed.) Run it after implementing and report the
measured max error and speedup.

## Deliverable 4 — Provide LaTeX description + references (in the reply)

Supply ready-to-paste text for the article (likely a short remark after
`eq:sigma_Z_pair` in §3.3, or a sentence in §5):
- A 2–3 sentence description: each `Φ₂` is evaluated with the
  Drezner–Wesolowsky / Genz Gauss–Legendre quadrature, accurate to ~1e-12, in
  place of a general multivariate routine, making the per-pair cost negligible.
- `\citep` usage + two `references.bib` BibTeX entries:
  - Genz (2004), *Stat. Comput.* 14(3):251–260, doi 10.1023/B:STCO.0000035304.20635.31
  - Drezner & Wesolowsky (1990), *J. Stat. Comput. Simul.* 35(1–2):101–107, doi 10.1080/00949659008811236

Provided in the chat reply (not auto-written into `main.tex`/`references.bib`
unless requested).

## Out of scope
- Vectorizing the pair loop now (Deliverable 2 documents it for later).
- Simulation sampling (`multivariate_normal.rvs`); Matérn covariance (§4.2 TODO).
