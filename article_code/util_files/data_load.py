import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_all_data(data_to_load = "ecad_data_south_europe_filtered",
                  verbose = False,
                  dict_feature_eng = None,
                  stations_to_get = None
                  ):
    """
    Load all station files, process them, and return a dict keyed by city name:
        {city_name: {"concatenation_pos_exc_with_dates": ..., "concatenation_neg_exc_with_dates": ...}, ...}

    `stations_to_get` filters by city names (matches df_info["city"]).
    """
    if dict_feature_eng is None:
        dict_feature_eng = {}
    if stations_to_get is None:
        stations_to_get = []
    if data_to_load == "ecad_data_south_europe_filtered":
        from .config import ECAD_RAW_DIR
        data_dir = ECAD_RAW_DIR
    else:
        raise ValueError(f"Configure data source for data_to_load input: {data_to_load}")

    df_info_path = data_dir / "df_candidates_kept.csv"
    if not df_info_path.exists():
        raise FileNotFoundError(f"Info CSV not found: {df_info_path}")
    df_info = pd.read_csv(df_info_path)

    souid_re = re.compile(r"SOUID(\d+)")
    station_files = list_station_files(data_dir)

    dict_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    nb_rejected = 0
    with tqdm(station_files, desc="Loading stations") as pbar:
        for file_path in pbar:
            file_name = file_path.name
            m = souid_re.search(file_name)
            if not m:
                if verbose:
                    print(f"Skipping (no SOUID): {file_name}")
                continue
            souid = m.group(1)

            city_row = df_info.loc[df_info["souid"] == int(souid)]
            if city_row.empty:
                city_name = f"SOUID_{souid}"
                if verbose:
                    print(f"City not found in df_info for SOUID={souid}; using '{city_name}'.")
            else:
                city_name = city_row["city"].iloc[0]
            pbar.set_postfix(value=f"{city_name} -- {souid}")

            if stations_to_get and city_name not in stations_to_get:
                continue
            if verbose:
                print(city_name)

            try:
                df = load_station_dataframe(file_path, skiprows=18, sep=",", header=0)
            except Exception as e:
                if verbose:
                    print(f"Failed to read {file_name}: {e}")
                continue
            if verbose:
                print(f"Loaded data for city: {city_name} (SOUID: {souid})")

            try:
                pos_exc, neg_exc = process_and_extract_excursions_from_raw_input_df_with_dates(
                    df, verbose=verbose, dict_feature_eng=dict_feature_eng)
                dict_data[city_name] = {
                    "concatenation_pos_exc_with_dates": pos_exc,
                    "concatenation_neg_exc_with_dates": neg_exc,
                }
            except Exception as e:
                nb_rejected += 1
                if verbose:
                    print(f"Could not extract for city: {city_name} (SOUID: {souid}) because of NaNs")
                    print(f"Details: {e}")
    print(f"Finished data load -- Rejected {nb_rejected} dataframes")
    return dict_data


def build_spells_export_filtered(d,
                                 out_dir="data/ecad_data/exports_json",
                                 base_filename="europe_spells_by_threshold"):
    """
    Build the filtered export (start_date >= YYYY0101) in one pass over `d`,
    write the JSON, and return the filtered dict.
    """
    def _duration(first_item):
        if isinstance(first_item, int):
            return int(first_item)
        try:
            return int(len(first_item))
        except Exception:
            return int(first_item)

    # cutoff = int(start_year) * 10000
    spell_kinds = (("dry_spell", "concatenation_neg_exc_with_dates"),
                   ("wet_spell", "concatenation_pos_exc_with_dates"))

    filtered = {}
    for city, spells in d.items():
        city_bucket = {}
        for spell_kind, spell_key in spell_kinds:
            durations, starts = [], []
            for first, dates in spells[spell_key]:
                start = int(dates[0])
                # if start >= cutoff:
                durations.append(_duration(first))
                starts.append(start)
            if durations:
                city_bucket[spell_kind] = {"duration_spell": durations, "start_date_spell": starts}
        if city_bucket:
            filtered[city] = city_bucket
    os.makedirs(out_dir, exist_ok=True)
    # suffix = f"_after_{int(start_year) - 1}_wet_day_thresh_{wet_day_threshold}"
    filtered_path = os.path.join(out_dir, f"{base_filename}.json")
    with open(filtered_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in filtered.items()}, f, ensure_ascii=False, indent=2)
    return filtered


def from_concat_with_dates_to_concat_by_season(concatenation_pos_exc_with_dates,
                                               concatenation_neg_exc_with_dates,
                                               decuplate_when_in_two_seasons=True):
    """
    Split excursions by season. Spells crossing a season boundary go to the
    starting season, and (if decuplate_when_in_two_seasons) also to the ending one.
    """
    def _bucket(items):
        bucket = defaultdict(list)
        for value, dates in items:
            season_first = from_date_to_season(dates[0])
            season_end = from_date_to_season(dates[-1])
            bucket[season_first].append(value)
            if decuplate_when_in_two_seasons and season_end != season_first:
                bucket[season_end].append(value)
        return bucket

    return _bucket(concatenation_pos_exc_with_dates), _bucket(concatenation_neg_exc_with_dates)


def process_and_extract_excursions_from_raw_input_df_with_dates(df,
                                                                verbose=False,
                                                                dict_feature_eng=None):
    if dict_feature_eng is None:
        dict_feature_eng = {}
    # Sentinel encodings (kept explicit so new encodings can be handled per-case)
    df["   RR"] = df["   RR"].replace(-9999, np.nan)
    df["   RR"] = df["   RR"].replace(-999, np.nan)
    df["   RR"] = df["   RR"].where(df["   RR"] >= 0, np.nan)

    nan_lengths, _ = get_length_consecutive_nans(df)
    list_len, list_count = np.unique(nan_lengths, return_counts=True)
    if verbose:
        print(pd.DataFrame({"consecutive_nan_length": list_len,
                            "nb_nan_pack_for_this_length": list_count}))

    serie_precip = df["   RR"].tolist()
    serie_dates = df["    DATE"].tolist()
    filled = fill_solo_to_trio_nan(serie_precip)
    df["filled_precip"] = filled
    if verbose:
        print("Filled solo to trio missing values, beginning to drop remaining nan")

    sections, date_sections = extract_sections_and_dates(serie_dates, filled)
    pos_all, neg_all = [], []
    for section, date_section in zip(sections, date_sections):
        # Apply wet day threshold
        wet_day_threshold = dict_feature_eng.get("wet_day_threshold")
        if wet_day_threshold:
            apply_wet_day_threshold(section, wet_day_threshold)
        # Drop first and last spell
        drop_first_last = dict_feature_eng.get("drop_first_last")
        pos, neg = extract_pos_exc_and_drought_lengths_with_dates_from_real_data(
            section, date_section, drop_first=drop_first_last, drop_last=drop_first_last, verbose=verbose)
        pos_all.extend(pos)
        neg_all.extend(neg)
    # Filter years
    start_year = dict_feature_eng.get("start_year")
    if start_year is not None:
        pos_all = [exc_date for exc_date in pos_all if exc_date[1][0] > (start_year * 10_000)]
        neg_all = [exc_date for exc_date in neg_all if exc_date[1][0] > (start_year * 10_000)]
    # Filter number of spells
    min_number_spells = dict_feature_eng.get("min_number_spells")
    if min_number_spells:
        nb_records_pos = np.sum([len(exc_date[0]) for exc_date in pos_all])
        nb_records_neg = np.sum([exc_date[0] for exc_date in neg_all])
        if nb_records_pos + nb_records_neg < min_number_spells:
            if verbose:
                print(f"Not enough data in concatenation_pos_exc_with_dates: {pos_all}")
            raise ValueError(f"Not enough data in concatenation_pos_exc_with_dates: {len(pos_all)} spells")
    return pos_all, neg_all


def extract_pos_exc_and_drought_lengths_with_dates_from_real_data(serie,
                                                                  date_section,
                                                                  drop_first=True,
                                                                  drop_last=True,
                                                                  verbose=False):
    memory_serie = serie[:]
    positive_excursions = []
    drought_length_list = []
    first_exc_positive = serie[0] > 0
    current_exc = "positive" if first_exc_positive else "negative"

    while len(serie) != 0:
        if current_exc == "positive":
            idx_end = 0
            while idx_end < len(serie) and serie[idx_end] > 0:
                idx_end += 1
            if idx_end > 0:
                positive_excursions.append((serie[:idx_end], date_section[:idx_end]))
            serie = serie[idx_end:]
            date_section = date_section[idx_end:]
            current_exc = "negative"
        if current_exc == "negative":
            idx_end = 0
            while idx_end < len(serie) and serie[idx_end] == 0:
                idx_end += 1
            if idx_end > 0:
                drought_length_list.append((idx_end, date_section[:idx_end]))
            serie = serie[idx_end:]
            date_section = date_section[idx_end:]
            current_exc = "positive"

    if len(positive_excursions) + len(drought_length_list) < drop_first + drop_last:
        if len(memory_serie) < 20:
            if verbose:
                print(f"we dropped an extract that was alone between two nan periods : {memory_serie}")
                print(f"original serie of len {len(memory_serie)}, here is the serie : {memory_serie}")
                print(positive_excursions)
                print(drought_length_list)
                return [], []

    if drop_last:
        # TODO: Enhance estimation : instead of getting rid : P(.. + not finished)
        if current_exc == "positive":
            drought_length_list = drought_length_list[:-1]
        if current_exc == "negative":
            positive_excursions = positive_excursions[:-1]
    if drop_first:
        if first_exc_positive:
            positive_excursions = positive_excursions[1:]
        else:
            drought_length_list = drought_length_list[1:]
    return positive_excursions, drought_length_list


def apply_wet_day_threshold(section, wet_day_threshold):
    for i, val in enumerate(section):
        if val <= wet_day_threshold:
            section[i] = 0
    return section


# NaN treatment

def fill_solo_to_trio_nan(serie_precip):
    # Edges
    if np.isnan(serie_precip[0]) and not np.isnan(serie_precip[1]):
        serie_precip[0] = serie_precip[1]
    if np.isnan(serie_precip[-1]) and not np.isnan(serie_precip[-2]):
        serie_precip[-1] = serie_precip[-2]

    # Solo nans
    idx_solo = [i for i in range(1, len(serie_precip) - 1)
                if np.isnan(serie_precip[i])
                and not np.isnan(serie_precip[i - 1])
                and not np.isnan(serie_precip[i + 1])]
    for idx in idx_solo:
        serie_precip[idx] = (serie_precip[idx - 1] + serie_precip[idx + 1]) / 2

    # Duo nans
    idx_duo = [i for i in range(1, len(serie_precip) - 2)
               if not np.isnan(serie_precip[i - 1])
               and np.isnan(serie_precip[i])
               and np.isnan(serie_precip[i + 1])
               and not np.isnan(serie_precip[i + 2])]
    for idx in idx_duo:
        before, after = serie_precip[idx - 1], serie_precip[idx + 2]
        serie_precip[idx] = before + (after - before) / 3
        serie_precip[idx + 1] = before + 2 * (after - before) / 3

    # Trio nans
    idx_trio = [i for i in range(1, len(serie_precip) - 3)
                if not np.isnan(serie_precip[i - 1])
                and np.isnan(serie_precip[i])
                and np.isnan(serie_precip[i + 1])
                and np.isnan(serie_precip[i + 2])
                and not np.isnan(serie_precip[i + 3])]
    for idx in idx_trio:
        before, after = serie_precip[idx - 1], serie_precip[idx + 3]
        serie_precip[idx] = before + (after - before) / 4
        serie_precip[idx + 1] = before + 2 * (after - before) / 4
        serie_precip[idx + 2] = before + 3 * (after - before) / 4
    return serie_precip


def extract_sections_and_dates(list_dates, lst):
    sections, date_sections = [], []
    current_section, current_dates = [], []
    for date, value in zip(list_dates, lst):
        if not np.isnan(value):
            current_section.append(value)
            current_dates.append(date)
        elif current_section:
            sections.append(current_section)
            date_sections.append(current_dates)
            current_section, current_dates = [], []
    if current_section:
        sections.append(current_section)
        date_sections.append(current_dates)
    return sections, date_sections


def get_length_consecutive_nans(df):
    """Return (nan_lengths, start_dates) for consecutive NaN runs in df['   RR']."""
    dates = pd.to_datetime(df["    DATE"], format="%Y%m%d").tolist()
    nan_lengths, dates_list = [], []
    current_length = 0
    for value, date in zip(df["   RR"].to_list(), dates):
        if np.isnan(value):
            if current_length == 0:
                dates_list.append(date)
            current_length += 1
        elif current_length > 0:
            nan_lengths.append(current_length)
            current_length = 0
    if current_length > 0:
        nan_lengths.append(current_length)
    return nan_lengths, dates_list


# Utils data load


def list_station_files(data_dir, prefix = "RR_SOUID", suffix = ".txt"):
    """Return a sorted list of Paths for station files in `data_dir`."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    return sorted(p for p in data_dir.iterdir()
                  if p.is_file() and p.name.startswith(prefix) and p.name.endswith(suffix))


def load_station_dataframe(file_path,
                           skiprows = 18,
                           sep = ",",
                           header = 0):
    """Load a station .txt/.csv-like file into a DataFrame."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path, skiprows=skiprows, sep=sep, header=header)


def from_date_to_season(date):
    d = int(str(date)[4:])
    if 301 <= d < 601:
        return "spring"
    if 601 <= d < 901:
        return "summer"
    if 901 <= d < 1201:
        return "autumn"
    return "winter"


def convert_to_datetime(date):
    return datetime.strptime(str(date), "%Y%m%d")

