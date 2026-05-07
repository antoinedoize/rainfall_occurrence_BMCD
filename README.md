# A duration-augmented binary Markov chain for rainfall occurrence with long dry spells

Reproduction pipeline and re-usable implementation for the article
*"A duration-augmented binary Markov chain for rainfall occurrence with long
dry spells"* (Doizé, Allard, Naveau, Wintenberger).

The repository (a) reproduces every figure of the paper from the bundled
ECA&D southern-Europe data and (b) packages the model so that it can be
applied to any new daily-precipitation dataset, either through Jupyter
notebooks or via batch CLI scripts.

---

## What this code accompanies

The article introduces a **Binary Markov Chain with Duration (BMCD)** on the
state space $\{0,1\}\times\mathbb{N}^*$, and formally links it to an
alternating renewal chain (Eq. 7, after Kozubowski 2025). That link lets the
exit probability $q^{(r)}_d$ be recovered from *any* spell-duration law, so
the model class is as expressive as the choice of that law.

> **Definition (BMCD).** Let $\mathbf{q}^{(0)},\mathbf{q}^{(1)}$ be two
> sequences of probabilities. A random sequence $(R_n,D_n)_{n\ge 0}$ on
> $\{0,1\}\times\mathbb{N}^*$ is a *Binary Markov Chain with Duration* with
> exit probabilities $(\mathbf{q}^{(0)},\mathbf{q}^{(1)})$ if, for every
> $n\ge 0$,
>
> $$ (R_{n+1}, D_{n+1}) = \begin{cases} (1-R_n,\;1) & \text{w.p. } q^{(R_n)}_{D_n},\\ (R_n,\;D_n+1) & \text{w.p. } 1-q^{(R_n)}_{D_n}. \end{cases} $$
>
> By convention $(R_0,D_0)=(0,1)$.

Spell durations are the hitting times $\tau^{(r)}_k$ and cycle durations
$\tau_k = \tau^{(0)}_k + \tau^{(1)}_k$.

The article picks two duration laws from the extended Generalized Pareto
Distribution (eGPD) family:

- **Dry spells.** A *hurdle-discretised eGPD* (**hdeGPD**, Eq. 3), fit by
  Probability Weighted Moments. The shape $\xi$ ranges from bounded
  ($\xi<0$) to heavy-tailed ($\xi>0$) across stations and seasons.

  $$ \mathbb{P}_{f_1,\kappa,\sigma,\xi}\bigl(\tau^{(0)}=d\bigr) = \begin{cases} f_1, & d=1,\\ (1-f_1)\bigl[F_{\kappa,\sigma,\xi}(d-1)-F_{\kappa,\sigma,\xi}(d-2)\bigr], & d\ge 2, \end{cases} $$

  where $F_{\kappa,\sigma,\xi}$ is the type-1 eGPD cdf.

- **Wet spells.** A two-component geometric mixture (Eq. 5), fit by EM:

  $$ \mathbb{P}_{\pi,p_1,p_2}\bigl(\tau^{(1)}=d\bigr) = \pi\, p_1(1-p_1)^{d-1}+(1-\pi)\, p_2(1-p_2)^{d-1}, \quad d\in\mathbb{N}^*. $$

---

## Repository layout

```
companion_code_article/
├── article_code/
│   ├── notebooks_article_pipeline/         # 6 notebooks, Figs. 1–14 + A
│   ├── notebooks_supplementary_material/   # PWM/stationarity checks + apply_to_new_dataset.ipynb
│   ├── util_files/                         # config, data loading, fitting (PWM, EM), plotting, statistics
│   ├── run_all_stations_figures.py         # per-station Figs. 2, 6–11
│   ├── run_all_stations_aggregate_figures.py   # cross-station Figs. 1, 4, 5, 13, 14, A
│   ├── run_all_stations_stationarity.py    # supplementary stationarity figures
│   └── run_on_new_dataset.py               # end-to-end driver for new datasets
├── data/
│   ├── ecad_data/                          # ECA&D southern-Europe stations + canonical spells JSON
│   └── toy_data/                           # synthetic 5-station toy set
├── figures/                                # all output PDFs (one folder per figure id)
├── results_fit/                            # fitted parameters (CSVs)
└── requirements.txt
```

---

## Installation

Python ≥ 3.12 is required (the codebase uses PEP 604 union syntax).

### Conda (preferred)

```bash
conda create -n bmcd_rainfall python=3.14 pip -y
conda activate bmcd_rainfall
pip install -r requirements.txt
```

### venv (alternative)

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

`requirements.txt` pins `numpy, scipy, pandas, matplotlib, plotly, tqdm,
mplcursors, jupyter, kaleido` to compatible-release ranges (`~=major.minor`)
matching the `bmcd_rainfall` env used to produce the article (Python 3.14).
Installing on a different Python minor or with looser pins may produce
slightly different numerical results.

### LaTeX (default; optional)

By default, every figure builder calls `plt.rc("text", usetex=True)` (or the
`rcParams` equivalent), which produces the exact typography used in the
article. This requires a working LaTeX toolchain on `PATH`:

- **Linux/macOS:** install [TeX Live](https://www.tug.org/texlive/) (the
  `texlive-full` meta-package is the safest).
- **Windows:** install [MiKTeX](https://miktex.org/) and let it install
  missing packages on first use.

Quick sanity check:

```bash
python -c "import matplotlib.pyplot as plt; plt.rc('text', usetex=True); plt.figure(); plt.text(0.5,0.5,r'$x$'); plt.savefig('latex_check.pdf')"
```

If this writes `latex_check.pdf` without raising, you're set.

#### Running without LaTeX

If you'd rather not install a TeX distribution, you can disable LaTeX rendering
and let matplotlib's built-in mathtext fallback handle `$...$` strings. The
figures still build and math labels (e.g. `$\xi$`, `$\tau^{(0)}$`) remain
readable, but fonts and spacing will differ from the published version.

To do so, comment out every `plt.rc("text", usetex=True)` and
`mpl.rcParams["text.usetex"] = True` line in the following files:

| File | Occurrences |
| ---- | ----------- |
| [article_code/run_all_stations_figures.py](article_code/run_all_stations_figures.py) | 6 (lines 80, 142, 190, 288, 322, 355) |
| [article_code/run_all_stations_aggregate_figures.py](article_code/run_all_stations_aggregate_figures.py) | 2 (lines 64, 106) |
| [article_code/run_all_stations_stationarity.py](article_code/run_all_stations_stationarity.py) | 1 (line 50) |
| [article_code/notebooks_article_pipeline/02_introduction_plots.ipynb](article_code/notebooks_article_pipeline/02_introduction_plots.ipynb) | 1 cell |
| [article_code/notebooks_article_pipeline/03_fit_and_params.ipynb](article_code/notebooks_article_pipeline/03_fit_and_params.ipynb) | 2 cells |
| [article_code/notebooks_article_pipeline/04_palermo_diagnostics.ipynb](article_code/notebooks_article_pipeline/04_palermo_diagnostics.ipynb) | 6 cells |
| [article_code/notebooks_supplementary_material/00_utils_check_stationnarity.ipynb](article_code/notebooks_supplementary_material/00_utils_check_stationnarity.ipynb) | 1 cell |

To locate them all in one go:

```bash
# bash / macOS / Linux
grep -rn "usetex" article_code/
```

```powershell
# Windows PowerShell
Select-String -Path article_code\*.py,article_code\**\*.py,article_code\**\*.ipynb -Pattern 'usetex'
```

The companion `plt.rc("font", family="serif")` lines can stay — they are
harmless without LaTeX and just keep a serif font for figure text.

---

## Reproducing the article figures

### Data

The southern-Europe ECA&D subset used in the paper is bundled under
[data/ecad_data/](data/ecad_data/):

- `ecad_data_south_europe_filtered/` — raw `RR_SOUID*.txt` files.
- `all_stations_metadata_filtered.csv` — station metadata (city, lat, lon).
- `exports_json/ecad_data_south_europe_filtered_after_1946_wet_day_thresh_6.json`
  — canonical spells JSON (the artifact every downstream stage reads).

### Option 1 — notebook walk-through

Run the six notebooks in [article_code/notebooks_article_pipeline/](article_code/notebooks_article_pipeline/) in order:

| Notebook                                          | Produces                                  |
| ------------------------------------------------- | ----------------------------------------- |
| `01_prepare_data_and_fit_distributions.ipynb`     | extracts spells, fits hdeGPD (PWM) + EM mixture |
| `02_introduction_plots.ipynb`                     | Figs. 1–3 (map, Palermo survival, renewal toy) |
| `03_fit_and_params.ipynb`                         | Figs. 4–5 (cross-station parameter histograms) |
| `04_palermo_diagnostics.ipynb`                    | Figs. 6–11 (ACF, histograms, Q-Q, exit prob.)  |
| `05_gof_and_maps.ipynb`                           | Fig. 13 + Fig. A (GoF p-value maps, null check) |
| `06_risk_maps.ipynb`                              | Fig. 14 (mean residual duration maps)     |

### Option 2 — direct batch reproduction

From the repo root:

```bash
python -m article_code.run_all_stations_figures            # Figs. 2, 6–11 per station
python -m article_code.run_all_stations_aggregate_figures  # Figs. 1, 4, 5, 13, 14, A
python -m article_code.run_all_stations_stationarity       # supplementary stationarity figures
```

Useful flags shared by all three drivers:

- `--first N` — process only the first `N` stations (sorted alphabetically).
- `--only STATION` — process only the named station(s); repeatable.
- `--skip-existing` — don't overwrite a PDF that already exists.

### Output

All PDFs land under [figures/](figures/), one folder per figure id:

```
figures/figure_1/   figures/figure_2/   figures/figure_4/   figures/figure_5/
figures/figure_6/   figures/figure_7/   figures/figure_8/   figures/figure_9/
figures/figure_10/  figures/figure_11/  figures/figure_13/  figures/figure_14/
figures/figure_A/   figures/stationnarity/
```

---

## Applying the model to a new dataset

[article_code/run_on_new_dataset.py](article_code/run_on_new_dataset.py) is
the end-to-end driver. It runs four selectable stages:

1. **ingest** — CSV directory → spells dict → canonical spells JSON.
2. **fit** — spells JSON → dry-spell hdeGPD CSV + wet-spell mixture-geom CSV.
3. **stationarity** — per-station 5-year rolling mean ± std PDFs.
4. **figures** — per-station Figs. 2, 6–11.

### Input format

Two ingest modes are supported:

- **CSV-directory mode** — a directory containing one `<STATION>.csv` per
  station. Each CSV must have:
  - a **date** column (default name `date`; either `YYYYMMDD` integer or any
    pandas-parseable string),
  - a **precipitation** column (default name `precip`).

  Precipitation defaults match ECA&D: values in **0.1 mm units** with a
  wet-day threshold of `6` (i.e. 0.6 mm). Both are overridable via
  `--wet-threshold`.

- **Pre-extracted spells JSON mode** — a JSON file already in the canonical
  schema produced by
  [article_code/util_files/data_load.py](article_code/util_files/data_load.py)
  (`build_spells_export_filtered`). Use this to skip extraction.

### How to run

#### Notebook

Open [article_code/notebooks_supplementary_material/apply_to_new_dataset.ipynb](article_code/notebooks_supplementary_material/apply_to_new_dataset.ipynb),
set `DATASET_NAME` and the ingest config in the first cells, then run all.

#### CLI

```bash
# CSV-directory mode
python -m article_code.run_on_new_dataset \
    --dataset-name my_dataset \
    --csv-dir data/my_dataset/csv_files \
    --date-col date --precip-col precip \
    --wet-threshold 6 --start-year 1946

# Pre-extracted spells JSON
python -m article_code.run_on_new_dataset \
    --dataset-name my_dataset \
    --spells-json /path/to/spells.json
```

Stage selection: `--skip-stages ingest fit stationarity figures` (any subset).
Station filtering: `--only`, `--first` (same semantics as the article
drivers). `--skip-existing` avoids overwriting figures.

### Output namespacing

All outputs are namespaced under `<DATASET>` so a run never collides with the
southern-Europe pipeline:

```
data/<DATASET>_data/exports_json/<DATASET>_spells.json
results_fit/fit_<DATASET>/dry_spell_fit_egpd1_excess_over_1result_fit_parameters.csv
results_fit/fit_<DATASET>/wet_spell_fit_mixt_geomresult_fit_parameters.csv
figures/<DATASET>/stationnarity/{STATION}_<dry|wet>_spell_duration_stationarity.pdf
figures/<DATASET>/figure_{2,6,7,8,9,10,11}/{STATION}_<suffix>.pdf
```

---

## Toy / example data

A small synthetic 5-station dataset is shipped under
[data/toy_data/](data/toy_data/):

- `toy_csv_data/` — ATHENS, BARCELONA, LISBON, MARSEILLE, ROME (1950–2020,
  daily; columns `date`, `precip` in 0.1 mm units).
- `exports_json/toy_spells.json` — same data already extracted to the
  canonical spells JSON.
- `generate_toy_dataset.py` — regenerator (a seasonal Markov chain mimicking
  Mediterranean rainfall: dry summers, wet winters).

End-to-end smoke test:

```bash
python data/toy_data/generate_toy_dataset.py
python -m article_code.run_on_new_dataset \
    --dataset-name toy \
    --csv-dir data/toy_data/toy_csv_data \
    --first 2
```

This writes fits to `results_fit/fit_toy/` and figures to `figures/toy/`.

---

## Citation

> Doizé, A., Allard, D., Naveau, P., Wintenberger, O. *A duration-augmented
> binary Markov chain for rainfall occurrence with long dry spells.*
