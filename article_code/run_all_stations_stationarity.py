"""Per-station batch driver — stationarity-check figure for every station.

Lifted near-verbatim from
`notebooks_supplementary_material/00_utils_check_stationnarity.ipynb`, with the
hardcoded `NAME_STATION_EXAMPLE` swapped for the `station` argument.

Output layout:
    figures/stationnarity/{STATION}_dry_spell_duration_stationarity.pdf
    figures/stationnarity/{STATION}_wet_spell_duration_stationarity.pdf

Usage (from the repo root):
    python -m article_code.run_all_stations_stationarity
    python -m article_code.run_all_stations_stationarity --first 5
    python -m article_code.run_all_stations_stationarity --only PALERMO
    python -m article_code.run_all_stations_stationarity --skip-existing
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .util_files import config, plotting
from .util_files.statistics import from_date_to_season


SEASON_COLORS = {
    "spring": "tab:olive",
    "summer": "tab:green",
    "autumn": "tab:brown",
    "winter": "tab:cyan",
}

SPELL_TYPES = ("dry_spell", "wet_spell")
SPELL_YLIM = {
    "dry_spell": (-1, 65),
    "wet_spell": None,
}


def make_stationarity_figure(station, spells, spell_type):
    """4-panel (one per season) 5-year rolling mean +/- std of spell durations."""
    mpl.rcParams["text.usetex"] = True
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    all_durations = spells[station][spell_type]["duration_spell"]
    all_dates = spells[station][spell_type]["start_date_spell"]
    ylim = SPELL_YLIM.get(spell_type)
    spell_label = spell_type.replace("_", " ")

    for ax, season in zip(axes, config.SEASONS):
        season_durations = [dur for dur, date in zip(all_durations, all_dates)
                            if from_date_to_season(date) == season]
        season_dates = [date for dur, date in zip(all_durations, all_dates)
                        if from_date_to_season(date) == season]
        season_dates = pd.to_datetime(pd.Series(season_dates).astype(str), format="%Y%m%d")
        df = pd.DataFrame({"date": season_dates, "duration": season_durations})
        df["year"] = df["date"].dt.year
        g = df.groupby("year")["duration"]
        annual = pd.DataFrame({
            "count": g.size(),
            "sum": g.sum(),
            "sumsq": g.apply(lambda s: (s ** 2).sum()),
        })
        years = np.arange(df["year"].min(), df["year"].max() + 1)
        annual = annual.reindex(years)
        r_count = annual["count"].rolling(window=5, center=True, min_periods=1).sum()
        r_sum = annual["sum"].rolling(window=5, center=True, min_periods=1).sum()
        r_sumsq = annual["sumsq"].rolling(window=5, center=True, min_periods=1).sum()
        mean5 = r_sum / r_count
        var5 = (r_sumsq - (r_sum ** 2) / r_count) / (r_count - 1)
        std5 = np.sqrt(var5)
        x = pd.to_datetime(pd.Series(years).astype(str) + "0101", format="%Y%m%d")
        ax.plot(x, mean5.values, label=season, color=SEASON_COLORS[season])
        ax.fill_between(x, (mean5 - std5).values, (mean5 + std5).values,
                        alpha=0.2, color=SEASON_COLORS[season])
        ax.legend()
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean duration (5y)")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.2)
    fig.suptitle(f"{station} - {spell_label} duration stationarity check")
    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def _output_path(station: str, spell_type: str) -> "config.Path":
    return (config.ROOT / "figures" / "stationnarity"
            / f"{plotting._safe_filename(station)}_{spell_type}_duration_stationarity.pdf")


def process_station(station: str, spells, *, skip_existing: bool = False) -> str:
    statuses = []
    for spell_type in SPELL_TYPES:
        if skip_existing and _output_path(station, spell_type).exists():
            statuses.append((spell_type, "skipped"))
            continue
        try:
            fig = make_stationarity_figure(station, spells, spell_type)
            plotting.save_stationarity_for_station(fig, station, spell_type)
            plt.close(fig)
            statuses.append((spell_type, "ok"))
        except Exception as exc:
            plt.close("all")
            statuses.append((spell_type, f"FAIL ({type(exc).__name__}: {exc})"))
    if any(s.startswith("FAIL") for _, s in statuses):
        details = "; ".join(f"{t}={s}" for t, s in statuses)
        return f"FAIL ({details})"
    if all(s == "skipped" for _, s in statuses):
        return "skipped (exists)"
    return "ok"


def _load_inputs():
    json_filename = (f"ecad_data_south_europe_filtered_after_{config.START_YEAR}"
                     f"_wet_day_thresh_{config.WET_DAY_THRESHOLD}.json")
    with open(config.EXPORTS_JSON_DIR / json_filename) as fh:
        spells = json.load(fh)
    print(f"[load] {len(spells)} stations from JSON")
    return spells


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--first", type=int, default=None,
                    help="Process only the first N stations (sorted alphabetically).")
    ap.add_argument("--only", action="append", default=None,
                    help="Process only the named station(s). May be repeated.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip a station if its PDF already exists.")
    args = ap.parse_args()

    spells = _load_inputs()

    stations = sorted(spells.keys())
    if args.only:
        unknown = [s for s in args.only if s not in spells]
        if unknown:
            print(f"[error] unknown station(s) in --only: {unknown}", file=sys.stderr)
            return 2
        stations = list(args.only)
    elif args.first is not None:
        stations = stations[: args.first]

    print(f"[run] processing {len(stations)} station(s) -> figures/stationnarity/")
    fail_count = 0
    skip_count = 0
    for station in tqdm(stations):
        try:
            res = process_station(station, spells, skip_existing=args.skip_existing)
        except Exception:
            print(f"[FATAL] unexpected error for '{station}':\n{traceback.format_exc()}",
                  file=sys.stderr)
            fail_count += 1
            continue
        if res.startswith("FAIL"):
            fail_count += 1
            tqdm.write(f"[partial] {station}: {res}")
        elif res.startswith("skipped"):
            skip_count += 1

    print(f"[done] stations: {len(stations)}, "
          f"failures: {fail_count}, skipped (existing): {skip_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
