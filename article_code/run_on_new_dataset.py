"""End-to-end driver for applying the BMCD/hdeGPD pipeline to a new dataset.

Two ingest modes:

- CSV-directory mode: a directory containing one CSV per station
  (`<STATION>.csv`), each with a date column and a precipitation column whose
  names are configurable. Columns are renamed to the layout that the existing
  spell extractor expects, then fed through it.
- Spells-JSON mode: a JSON file already in the canonical schema produced by
  `data_load.build_spells_export_filtered`. Skip extraction, jump to fit + plots.

Stages (selectable via --skip-stages):

    ingest        CSV dir -> spells dict -> spells JSON
    fit           spells JSON -> dry hdeGPD CSV + wet mixt-geom CSV
    stationarity  spells JSON -> per-station 5-year rolling mean +/- std PDFs
    figures       spells JSON + fit CSVs -> per-station Figs 2, 6-11

Outputs are namespaced under <DATASET> so a run never collides with the
southern-Europe pipeline:

    data/<DATASET>_data/exports_json/<DATASET>_spells.json
    results_fit/fit_<DATASET>/dry_spell_fit_egpd1_excess_over_1result_fit_parameters.csv
    results_fit/fit_<DATASET>/wet_spell_fit_mixt_geomresult_fit_parameters.csv
    figures/<DATASET>/stationnarity/{STATION}_<dry|wet>_spell_duration_stationarity.pdf
    figures/<DATASET>/figure_{2,6,7,8,9,10,11}/{STATION}_<suffix>.pdf

Usage:

    # CSV-directory mode
    python -m article_code.run_on_new_dataset \\
        --dataset-name my_dataset \\
        --csv-dir /path/to/csv_files \\
        --date-col date --precip-col precip \\
        --wet-threshold 6 --start-year 1946

    # Pre-extracted spells JSON mode
    python -m article_code.run_on_new_dataset \\
        --dataset-name my_dataset \\
        --spells-json /path/to/spells.json
"""
import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from .util_files import config, plotting
from .util_files.data_load import (
    build_spells_export_filtered,
    process_and_extract_excursions_from_raw_input_df_with_dates,
)
from .run_all_stations_figures import (
    make_fig2_survival_overlay,
    make_fig6_bivariate_acf,
    make_fig7_dry_hist,
    make_fig8_wet_hist,
    make_fig9_dry_qq,
    make_fig10_wet_qq,
    make_fig11_exit_prob,
)
from .run_all_stations_stationarity import SPELL_TYPES, make_stationarity_figure
from .util_files.statistics import (
    build_data_per_city_per_season_per_year_couple_vector_duration_vector_date,
    split_spells_by_season_simple,
)
from .util_files.egp_pwm import fit_hdegpd_dry_spell_durations
from .util_files.mixt_geom_em import fit_mixt_geom_wet_spell_durations


STAGES = ("ingest", "fit", "stationarity", "figures")

FIGURE_SUFFIX = {
    2: "comparison_dry_spell_egpd_geom_4seasons.pdf",
    6: "bivariate_acf_plot.pdf",
    7: "histogram_dry_spell_fit.pdf",
    8: "histogram_wet_spell_fit.pdf",
    9: "qqplot_dry_spell_fit.pdf",
    10: "qqplot_wet_spell_fit.pdf",
    11: "proba_leaving_state.pdf",
}

FIGURE_BUILDERS = {
    2: (make_fig2_survival_overlay, ("spells", "df_fit_dry")),
    6: (make_fig6_bivariate_acf, ("data_per_city",)),
    7: (make_fig7_dry_hist, ("spells", "df_fit_dry")),
    8: (make_fig8_wet_hist, ("spells", "df_fit_wet")),
    9: (make_fig9_dry_qq, ("spells", "df_fit_dry")),
    10: (make_fig10_wet_qq, ("spells", "df_fit_wet")),
    11: (make_fig11_exit_prob, ("spells", "df_fit_dry")),
}


# ---------------------------------------------------------------------------
# Path layout
# ---------------------------------------------------------------------------

def dataset_paths(dataset_name: str, *, output_root: Path | None = None) -> dict:
    """Resolve the canonical output paths for a dataset."""
    root = Path(output_root) if output_root is not None else config.ROOT
    spells_json = (root / "data" / f"{dataset_name}_data" / "exports_json"
                   / f"{dataset_name}_spells.json")
    fit_folder = root / "results_fit" / f"fit_{dataset_name}"
    figures_root = root / "figures" / dataset_name
    return {
        "root": root,
        "spells_json": spells_json,
        "fit_folder": fit_folder,
        "dry_fit_csv": fit_folder / "dry_spell_fit_egpd1_excess_over_1result_fit_parameters.csv",
        "wet_fit_csv": fit_folder / "wet_spell_fit_mixt_geomresult_fit_parameters.csv",
        "stationarity_dir": figures_root / "stationnarity",
        "figures_root": figures_root,
    }


# ---------------------------------------------------------------------------
# Stage 1: ingest
# ---------------------------------------------------------------------------

def _adapt_csv_to_extractor_df(df: pd.DataFrame, date_col: str, precip_col: str) -> pd.DataFrame:
    """Rename a clean (date, precip) frame to the column names the extractor expects."""
    if date_col not in df.columns or precip_col not in df.columns:
        raise KeyError(f"CSV must contain columns '{date_col}' and '{precip_col}'; "
                       f"found {list(df.columns)}")
    out = pd.DataFrame()
    raw_dates = df[date_col]
    if pd.api.types.is_numeric_dtype(raw_dates):
        # Integer YYYYMMDD encoding (e.g. 19500101). Without an explicit format,
        # pd.to_datetime would treat these as nanoseconds-since-epoch and collapse
        # every row to 1970-01-01.
        dates = pd.to_datetime(raw_dates.astype("Int64").astype(str),
                               format="%Y%m%d", errors="coerce")
    else:
        dates = pd.to_datetime(raw_dates, errors="coerce")
    keep = dates.notna()
    if not keep.all():
        dates = dates[keep]
        df = df.loc[keep]
    out["    DATE"] = dates.dt.strftime("%Y%m%d").astype(int).to_numpy()
    out["   RR"] = pd.to_numeric(df[precip_col], errors="coerce").to_numpy(dtype=float)
    return out


def ingest_csv_dir(csv_dir: Path | str,
                   *,
                   date_col: str = "date",
                   precip_col: str = "precip",
                   wet_threshold: int | float | None = None,
                   drop_first_last: bool = True,
                   start_year: int | None = None,
                   min_number_spells: int | None = None,
                   verbose: bool = False) -> dict:
    """Iterate CSVs in `csv_dir`, extract spells, return the same dict shape as
    `data_load.load_all_data` so we can pass it straight to
    `build_spells_export_filtered`.
    """
    csv_dir = Path(csv_dir)
    if not csv_dir.is_dir():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")
    csv_paths = sorted(p for p in csv_dir.iterdir()
                       if p.is_file() and p.suffix.lower() == ".csv")
    if not csv_paths:
        raise FileNotFoundError(f"No .csv files in {csv_dir}")

    feat = {
        "wet_day_threshold": wet_threshold,
        "drop_first_last": drop_first_last,
        "start_year": start_year,
        "min_number_spells": min_number_spells,
    }

    dict_data: dict = {}
    rejected = 0
    with tqdm(csv_paths, desc="Ingesting CSVs") as pbar:
        for csv_path in pbar:
            station = csv_path.stem
            pbar.set_postfix(value=station)
            try:
                raw = pd.read_csv(csv_path)
                df = _adapt_csv_to_extractor_df(raw, date_col, precip_col)
                pos_exc, neg_exc = process_and_extract_excursions_from_raw_input_df_with_dates(
                    df, verbose=verbose, dict_feature_eng=feat,
                )
                dict_data[station] = {
                    "concatenation_pos_exc_with_dates": pos_exc,
                    "concatenation_neg_exc_with_dates": neg_exc,
                }
            except Exception as exc:
                rejected += 1
                if verbose:
                    print(f"[reject] {station}: {type(exc).__name__}: {exc}")
    print(f"[ingest] kept {len(dict_data)} stations, rejected {rejected}")
    return dict_data


def write_spells_json(d: dict, out_path: Path | str) -> Path:
    """Write the spells dict to `out_path` (matching the canonical JSON schema)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # `build_spells_export_filtered` writes <out_dir>/<base_filename>.json. We
    # let it write to a tmp filename inside the right folder, then move.
    tmp_base = out_path.stem + "__tmp_run_on_new_dataset"
    build_spells_export_filtered(d, out_dir=str(out_path.parent), base_filename=tmp_base)
    tmp_path = out_path.parent / f"{tmp_base}.json"
    tmp_path.replace(out_path)
    return out_path


def load_spells_json(json_path: Path | str) -> dict:
    with open(json_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Stage 2: fit
# ---------------------------------------------------------------------------

def fit_dry_and_wet(json_path: Path | str, fit_folder: Path | str) -> tuple[Path, Path]:
    """Fit hdeGPD (dry) + mixture-geom (wet); return the two CSV paths."""
    json_path = Path(json_path)
    fit_folder = Path(fit_folder)
    fit_folder.mkdir(parents=True, exist_ok=True)
    fit_hdegpd_dry_spell_durations(json_path=json_path, path_folder_save_results=fit_folder)
    fit_mixt_geom_wet_spell_durations(json_path=json_path, path_folder_save_results=fit_folder)
    dry_csv = fit_folder / "dry_spell_fit_egpd1_excess_over_1result_fit_parameters.csv"
    wet_csv = fit_folder / "wet_spell_fit_mixt_geomresult_fit_parameters.csv"
    return dry_csv, wet_csv


# ---------------------------------------------------------------------------
# Stage 3: stationarity
# ---------------------------------------------------------------------------

def _select_stations(spells: dict, only: Iterable[str] | None, first: int | None) -> list[str]:
    stations = sorted(spells.keys())
    if only:
        only = list(only)
        unknown = [s for s in only if s not in spells]
        if unknown:
            raise KeyError(f"unknown station(s) in --only: {unknown}")
        return only
    if first is not None:
        return stations[:first]
    return stations


def run_stationarity(spells: dict,
                     out_dir: Path | str,
                     *,
                     only: Iterable[str] | None = None,
                     first: int | None = None,
                     skip_existing: bool = False) -> dict:
    """For each station, save dry+wet stationarity-check PDFs into `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stations = _select_stations(spells, only, first)
    results: dict = {}
    print(f"[stationarity] {len(stations)} station(s) -> {out_dir}")
    for station in tqdm(stations):
        results[station] = {}
        for spell_type in SPELL_TYPES:
            out_path = out_dir / f"{plotting._safe_filename(station)}_{spell_type}_duration_stationarity.pdf"
            if skip_existing and out_path.exists():
                results[station][spell_type] = "skipped (exists)"
                continue
            try:
                fig = make_stationarity_figure(station, spells, spell_type)
                fig.savefig(out_path, bbox_inches="tight", dpi=200)
                plt.close(fig)
                results[station][spell_type] = "ok"
            except Exception as exc:
                plt.close("all")
                results[station][spell_type] = f"FAIL ({type(exc).__name__}: {exc})"
                tqdm.write(f"[partial] {station} {spell_type}: {results[station][spell_type]}")
    return results


# ---------------------------------------------------------------------------
# Stage 4: per-station figures
# ---------------------------------------------------------------------------

def _figure_out_path(figures_root: Path, fig_num: int, station: str) -> Path:
    return (figures_root / f"figure_{fig_num}"
            / f"{plotting._safe_filename(station)}_{FIGURE_SUFFIX[fig_num]}")


def run_per_station_figures(spells: dict,
                            df_fit_dry: pd.DataFrame,
                            df_fit_wet: pd.DataFrame,
                            figures_root: Path | str,
                            *,
                            only: Iterable[str] | None = None,
                            first: int | None = None,
                            skip_existing: bool = False) -> dict:
    """Build Figs 2, 6-11 per station and save into <figures_root>/figure_<N>/."""
    figures_root = Path(figures_root)
    stations = _select_stations(spells, only, first)

    data_by_season = split_spells_by_season_simple(
        spells, start_key="start_date_spell", dur_key="duration_spell",
    )
    data_per_city = build_data_per_city_per_season_per_year_couple_vector_duration_vector_date(
        data_by_season,
    )
    ctx = {
        "spells": spells,
        "df_fit_dry": df_fit_dry,
        "df_fit_wet": df_fit_wet,
        "data_per_city": data_per_city,
    }

    results: dict = {}
    print(f"[figures] {len(stations)} station(s) -> {figures_root}/figure_*/")
    for station in tqdm(stations):
        results[station] = {}
        for fig_num, (builder, kw_keys) in FIGURE_BUILDERS.items():
            out_path = _figure_out_path(figures_root, fig_num, station)
            if skip_existing and out_path.exists():
                results[station][fig_num] = "skipped (exists)"
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                kwargs = {k: ctx[k] for k in kw_keys}
                fig = builder(station, **kwargs)
                fig.savefig(out_path, bbox_inches="tight", dpi=200)
                plt.close(fig)
                results[station][fig_num] = "ok"
            except Exception as exc:
                plt.close("all")
                results[station][fig_num] = f"FAIL ({type(exc).__name__}: {exc})"
                tqdm.write(f"[partial] {station} fig{fig_num}: {results[station][fig_num]}")
    return results


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_all(dataset_name: str,
            *,
            csv_dir: Path | str | None = None,
            spells_json: Path | str | None = None,
            output_root: Path | str | None = None,
            date_col: str = "date",
            precip_col: str = "precip",
            wet_threshold: int | float | None = None,
            start_year: int | None = None,
            drop_first_last: bool = True,
            min_number_spells: int | None = None,
            stages: Iterable[str] = STAGES,
            only: Iterable[str] | None = None,
            first: int | None = None,
            skip_existing: bool = False,
            verbose: bool = False) -> dict:
    """Top-level driver. Runs the requested stages in order and returns a summary."""
    stages = tuple(stages)
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise ValueError(f"unknown stage(s): {bad}; expected subset of {STAGES}")

    paths = dataset_paths(dataset_name, output_root=output_root)
    summary: dict = {"dataset_name": dataset_name, "paths": paths, "stages": {}}

    if csv_dir is not None and spells_json is not None:
        raise ValueError("Pass exactly one of --csv-dir / --spells-json, not both.")

    # Stage 1: ingest
    if "ingest" in stages:
        if spells_json is not None:
            # User supplied an external JSON: copy/symlink it to the canonical path
            # so all downstream stages have one source of truth.
            paths["spells_json"].parent.mkdir(parents=True, exist_ok=True)
            d = load_spells_json(spells_json)
            with open(paths["spells_json"], "w", encoding="utf-8") as fh:
                json.dump(d, fh, ensure_ascii=False, indent=2)
            summary["stages"]["ingest"] = {"mode": "spells-json", "stations": len(d)}
        elif csv_dir is not None:
            d = ingest_csv_dir(
                csv_dir,
                date_col=date_col,
                precip_col=precip_col,
                wet_threshold=wet_threshold,
                drop_first_last=drop_first_last,
                start_year=start_year,
                min_number_spells=min_number_spells,
                verbose=verbose,
            )
            write_spells_json(d, paths["spells_json"])
            summary["stages"]["ingest"] = {"mode": "csv-dir", "stations": len(d)}
        else:
            raise ValueError("Either --csv-dir or --spells-json must be set "
                             "(or skip the ingest stage).")
    elif not paths["spells_json"].exists():
        raise FileNotFoundError(
            f"Ingest stage skipped but expected spells JSON not found: "
            f"{paths['spells_json']}",
        )

    # Stage 2: fit
    if "fit" in stages:
        dry_csv, wet_csv = fit_dry_and_wet(paths["spells_json"], paths["fit_folder"])
        summary["stages"]["fit"] = {"dry_csv": dry_csv, "wet_csv": wet_csv}

    # Stage 3: stationarity
    if "stationarity" in stages:
        spells = load_spells_json(paths["spells_json"])
        res = run_stationarity(
            spells, paths["stationarity_dir"],
            only=only, first=first, skip_existing=skip_existing,
        )
        summary["stages"]["stationarity"] = {"per_station": res}

    # Stage 4: per-station figures
    if "figures" in stages:
        spells = load_spells_json(paths["spells_json"])
        if not paths["dry_fit_csv"].exists() or not paths["wet_fit_csv"].exists():
            raise FileNotFoundError(
                f"Figure stage requires fit CSVs at {paths['fit_folder']}; "
                f"run the fit stage first.",
            )
        df_fit_dry = pd.read_csv(paths["dry_fit_csv"])
        df_fit_wet = pd.read_csv(paths["wet_fit_csv"])
        res = run_per_station_figures(
            spells, df_fit_dry, df_fit_wet, paths["figures_root"],
            only=only, first=first, skip_existing=skip_existing,
        )
        summary["stages"]["figures"] = {"per_station": res}

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run the BMCD/hdeGPD pipeline end-to-end on a new dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset-name", required=True,
                    help="Short identifier; namespaces all outputs.")

    src = ap.add_mutually_exclusive_group()
    src.add_argument("--csv-dir", default=None,
                     help="Directory of <station>.csv files (date+precip columns).")
    src.add_argument("--spells-json", default=None,
                     help="JSON in the canonical spells schema (skip extraction).")

    ap.add_argument("--date-col", default="date",
                    help="Date column name in CSV-dir mode (default: date).")
    ap.add_argument("--precip-col", default="precip",
                    help="Precipitation column name in CSV-dir mode (default: precip).")
    ap.add_argument("--wet-threshold", type=float, default=config.WET_DAY_THRESHOLD,
                    help=f"Wet-day threshold (default: {config.WET_DAY_THRESHOLD}).")
    ap.add_argument("--start-year", type=int, default=config.START_YEAR,
                    help=f"Drop spells starting before this year (default: {config.START_YEAR}).")
    ap.add_argument("--no-drop-first-last", action="store_true",
                    help="Keep first/last (potentially censored) spell of each segment.")
    ap.add_argument("--min-number-spells", type=int, default=None,
                    help="Reject stations with fewer total spells than this.")

    ap.add_argument("--output-root", default=None,
                    help="Override the repo root used for output namespacing.")
    ap.add_argument("--skip-stages", nargs="+", default=[], choices=STAGES,
                    help="Stages to skip.")
    ap.add_argument("--only", action="append", default=None,
                    help="Process only the named station(s). May be repeated.")
    ap.add_argument("--first", type=int, default=None,
                    help="Process only the first N stations (sorted alphabetically).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Don't overwrite figures whose PDF already exists.")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    stages = tuple(s for s in STAGES if s not in args.skip_stages)
    try:
        summary = run_all(
            dataset_name=args.dataset_name,
            csv_dir=args.csv_dir,
            spells_json=args.spells_json,
            output_root=args.output_root,
            date_col=args.date_col,
            precip_col=args.precip_col,
            wet_threshold=args.wet_threshold,
            start_year=args.start_year,
            drop_first_last=not args.no_drop_first_last,
            min_number_spells=args.min_number_spells,
            stages=stages,
            only=args.only,
            first=args.first,
            skip_existing=args.skip_existing,
            verbose=args.verbose,
        )
    except Exception:
        print("[FATAL]\n" + traceback.format_exc(), file=sys.stderr)
        return 2

    print(f"[done] dataset='{summary['dataset_name']}' stages={list(summary['stages'].keys())}")
    print(f"       spells JSON : {summary['paths']['spells_json']}")
    print(f"       fit folder  : {summary['paths']['fit_folder']}")
    print(f"       figures root: {summary['paths']['figures_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
