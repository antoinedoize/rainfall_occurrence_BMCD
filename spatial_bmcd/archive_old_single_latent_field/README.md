# Archived: single-latent-field spatial BMCD (old model)

This directory is a **frozen, self-contained snapshot** of the original spatial model —
the single-latent-field construction marked `\old{}` (orange) in
`../Spatialisation-Rain-Occurrence-Generator-article/main.tex`.

In that construction a single centred Gaussian field `Y_n` drives every station, and the
latent variable is the sign-adjusted field `Z_n(s) = (-1)^{R_n(s)} Y_n(s)`. The spatial
dependence is a single scalar range parameter `sigma` (exponential correlation
`exp(-h/sigma)`), and the pairwise-likelihood off-diagonal is
`rho = (-1)^{r_j+r_k} (Σ_n)_{jj'}`.

The **live** model at the `spatial_bmcd/` top level has since moved to the direct Linear
Model of Coregionalization (LMC, `\new{}` blue in `main.tex`): three latent fields
`W^c, W^(0), W^(1)`, a mixing weight `λ`, and parameter `θ = (λ, σ_wᶜ, σ_w⁰, σ_w¹)`.

## Contents

- `spatial_model.py`, `spatial_diagnostics.py`, `spatial_plotting.py` — copies of the
  single-field scripts as they were when archived (not shared with the top-level code).
- `spatial_analysis_article.ipynb` — the old clean pipeline.
- `diagnostics_fitted_model.ipynb` — fitted-model diagnostics.
- `tests_parameter_estimation.ipynb`, `tests_simulation_methods.ipynb` — validation.
- `util_speedup_approx_cdf.ipynb` — `approxcdf`/Tsay–Ke kernel benchmark against the old API.

## How to recover / run the old model

These notebooks are self-contained: they import **only** from this archive package
(`spatial_bmcd.archive_old_single_latent_field.spatial_model`, `…spatial_plotting`,
`…spatial_diagnostics`), so they keep running unchanged regardless of how the top-level LMC
code evolves. Open any notebook and run it top to bottom; the `sys.path` bootstrap to the
repo root (which locates `article_code/`) is unchanged.

Nothing at the top level imports this archive.
