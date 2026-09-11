# rainfall_occurrence_BMCD

Spatial extension of the BMCD (Binary Markov Chain with Duration) rainfall-occurrence generator:
research code + article redaction. The active project lives in `spatial_bmcd/`; `article_code/`
is the frozen single-site pipeline of the published companion article.

**Before starting work, read `spatial_bmcd/ROADMAP.md`** — it holds the task lists (Now / Next /
Later), the session protocol for picking up and closing tasks, and the decisions/done logs.

Key references:

- `spatial_bmcd/NOTES_spatial_diagnostics.md` — reasoning, caveats and open issues behind the
  Iberia fit-and-diagnostics notebook; read before touching that notebook.
- `spatial_bmcd/spatial_lin_mod_coregion_fit_spain_portugal.ipynb` — fit + diagnostics (Iberia).
- `spatial_bmcd/spatial_bmcd_data_analysis.ipynb` — exploratory data analysis.
- `spatial_bmcd/Spatialisation-Rain-Occurrence-Generator-article/main.tex` — the article.

Caution: `spatial_bmcd/Spatialisation-Rain-Occurrence-Generator-article/` is a **git submodule**
with its own history — commit inside it deliberately, never as a side effect of a repo-root commit.
