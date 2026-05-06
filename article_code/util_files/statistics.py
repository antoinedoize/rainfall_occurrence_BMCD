from .spell_models import *
from . import config
from collections import defaultdict
from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np
import scipy
import re
import pandas as pd
from math import isclose
from scipy.special import beta, betainc


### Compute statistics


# Exit probability
def get_proba_leaving_by_day(list_spells):
    # Empirical estimation
    ar_spells = np.array(list_spells)
    days = list(range(1,np.max(list_spells)+1))
    probas = list()
    nb_days_state_list = list()
    for k in days:
        nb_left = np.sum(ar_spells == k)
        nb_days_state = np.sum(ar_spells >= k)
        q = nb_left / nb_days_state
        probas.append(q)
        nb_days_state_list.append(nb_days_state)
    probas = np.array(probas)
    return days,probas,nb_days_state_list


def get_proba_leaving_state_n_kozu(cdf_func,n):
    # Parametric estimation (from Kozubowski article)
    return (cdf_func(n)-cdf_func(n-1))/(1-cdf_func(n-1))


# Bivariate autocorrelation function
def pooled_bivariate_autocorr(data_by_year, L):
    """
    Pooled bivariate autocorrelation across disjoint year blocks.

    Parameters
    ----------
    data_by_year : dict[int, list]
        {year: [ [[tau0, tau1], [start_date, end_date]], ... ], ... }
        Only the durations [tau0, tau1] are used.
    L : int
        Max lag.

    Returns
    -------
    R : np.ndarray
        Array of shape (L+1, 2, 2) with autocorrelation matrices R_hat(l).
        If N_l==0 for some lag, R[l] is filled with np.nan.
    Gamma : np.ndarray
        Array of shape (L+1, 2, 2) with covariance matrices Gamma_hat(l).
    N_l : np.ndarray
        Array of shape (L+1,) with pooled pair counts N_l.
    bands : np.ndarray
        Array of shape (L+1,) with classical bands 2/sqrt(N_l) (nan when N_l==0).
    Vbar : np.ndarray
        Vector of shape (2,) pooled mean over all observations.
    """

    # ---- extract per-year duration arrays V_{y,k} ----
    V_year = {}
    for y, seq in data_by_year.items():
        # seq: list of entries like [[tau0, tau1], [date0, date1]]
        V = np.array([entry[0] for entry in seq], dtype=float)  # shape (N_y, 2)
        V_year[y] = V
    # ---- pooled mean over all years ----
    allV = np.vstack(list(V_year.values()))  # shape (N_tot, 2)
    Vbar = allV.mean(axis=0)
    # ---- N_l = sum_y max(N_y - l, 0) ----
    N_l = np.zeros(L + 1, dtype=int)
    for l in range(L + 1):
        N_l[l] = sum(max(V.shape[0] - l, 0) for V in V_year.values())
    # ---- pooled covariance estimators Gamma_hat(l) ----
    Gamma = np.full((L + 1, 2, 2), np.nan, dtype=float)
    for l in range(L + 1):
        if N_l[l] == 0:
            continue
        acc = np.zeros((2, 2), dtype=float)
        for y, V in V_year.items():
            Ny = V.shape[0]
            if Ny <= l:
                continue
            A = V[:Ny - l] - Vbar            # shape (Ny-l, 2)
            B = V[l:Ny] - Vbar               # shape (Ny-l, 2)
            acc += A.T @ B                   # sum_k (V_{y,k}-Vbar)(V_{y,k+l}-Vbar)^T

        Gamma[l] = acc / N_l[l]
    D = np.diag(np.diag(Gamma[0]))
    d = np.diag(D)
    # protect against zeros (if a component has zero variance)
    if np.any(d <= 0):
        raise ValueError(f"Non-positive variance components in Gamma(0): {d}")
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))

    R = np.full((L + 1, 2, 2), np.nan, dtype=float)
    for l in range(L + 1):
        if N_l[l] == 0:
            continue
        R[l] = D_inv_sqrt @ Gamma[l] @ D_inv_sqrt
    # ---- classic confidence band 2/sqrt(N_l) (componentwise heuristic) ----
    bands = np.full(L + 1, np.nan, dtype=float)
    for l in range(L + 1):
        if N_l[l] > 0:
            bands[l] = 2.0 / np.sqrt(N_l[l])
    return R, Gamma, N_l, bands, Vbar


### Goodness of fit

def goodness_of_fit_true_all_cities_seasons(
    data, df_fit, seasons=config.SEASONS,
    nb_days_min_for_D=40, force_D=None, spell_type="dry",
):
    dict_chi2, dict_p_value, dict_D = {}, {}, {}
    list_rejected = []
    list_city = sorted(list(data.keys()))
    for city in tqdm(list_city):
        sp = data[city]["dry_spell"] if spell_type == "dry" else data[city]["wet_spell"]
        all_durations, all_dates = sp["duration_spell"], sp["start_date_spell"]
        for season in seasons:
            season_durations = [dur for dur, date in zip(all_durations, all_dates)
                                if from_date_to_season(date) == season]
            if len(season_durations) == 0:
                continue
            extract_df = df_fit[df_fit["data_source"] == f"{city} {season}"]
            if len(extract_df) != 1:
                list_rejected.append(f"{city} - {season} (missing/duplicate fit row)")
                continue
            days, probas_emp, nb_days_state_list = get_proba_leaving_by_day(season_durations)
            D = int(force_D) if force_D is not None else adaptive_D(nb_days_state_list, nb_days_min_for_D)
            if D < 2:
                continue
            xi_gpd = extract_df["xi"].item()
            sigma_gpd = extract_df["sigma"].item()
            kappa_gpd = extract_df["kappa"].item()
            proba1 = sum(s == 1 for s in season_durations) / len(season_durations)
            cdf_fitted = make_cdf_fitted_hdeGPD_from_params(proba1, xi_gpd, sigma_gpd, kappa_gpd)
            fitted_probas = np.array(
                [get_proba_leaving_state_n_kozu(cdf_fitted, d) for d in range(1, D)],
                dtype=float)
            probas_emp = np.asarray(probas_emp, dtype=float)[:D-1]
            T = build_T_matrix(D, cdf_fitted)
            Sigma = build_Sigma_matrix_new(D, cdf_fitted)
            try:
                inv_matrix_mul = np.linalg.inv(T @ Sigma @ T.T)
            except np.linalg.LinAlgError:
                list_rejected.append(f"{city} - {season} (singular matrix)")
                continue
            diff = probas_emp - fitted_probas
            N_cycles = len(season_durations)
            Q_n = float(N_cycles * diff.T @ inv_matrix_mul @ diff)
            key = f"{city} - {season}"
            dict_chi2[key] = Q_n
            dict_D[key] = D
            dict_p_value[key] = 1.0 - scipy.stats.chi2.cdf(Q_n, df=D - 1)
    return dict_chi2, dict_p_value, dict_D, list_rejected

def goodness_of_fit_true_all_cities_seasons_geometric(
    data, seasons=config.SEASONS, nb_days_min_for_D=40, force_D=None, spell_type="dry",
):
    dict_chi2, dict_p_value, dict_D, dict_p_hat = {}, {}, {}, {}
    list_rejected = []
    for city in tqdm(sorted(list(data.keys()))):
        sp = data[city]["dry_spell"] if spell_type == "dry" else data[city]["wet_spell"]
        all_durations, all_dates = sp["duration_spell"], sp["start_date_spell"]
        for season in seasons:
            season_durations = [dur for dur, date in zip(all_durations, all_dates)
                                if from_date_to_season(date) == season]
            if len(season_durations) == 0:
                continue
            days, probas_emp, nb_days_state_list = get_proba_leaving_by_day(season_durations)
            D = int(force_D) if force_D is not None else adaptive_D(nb_days_state_list, nb_days_min_for_D)
            if D < 2:
                continue
            try:
                p_hat = fit_geometric_p_mle(season_durations)
                cdf_fitted = make_cdf_fitted_geometric_from_p(p_hat)
            except Exception as e:
                list_rejected.append(f"{city} - {season} (fit failed: {e})")
                continue
            fitted_probas = np.array(
                [get_proba_leaving_state_n_kozu(cdf_fitted, d) for d in range(1, D)],
                dtype=float)
            probas_emp_ = np.asarray(probas_emp, dtype=float)[:D-1]
            T = build_T_matrix(D, cdf_fitted)
            Sigma = build_Sigma_matrix_new(D, cdf_fitted)
            try:
                inv_matrix_mul = np.linalg.inv(T @ Sigma @ T.T)
            except np.linalg.LinAlgError:
                list_rejected.append(f"{city} - {season} (singular matrix)")
                continue
            diff = probas_emp_ - fitted_probas
            N_cycles = len(season_durations)
            Q_n = float(N_cycles * diff.T @ inv_matrix_mul @ diff)
            key = f"{city} - {season}"
            dict_chi2[key] = Q_n
            dict_D[key] = D
            dict_p_value[key] = 1.0 - scipy.stats.chi2.cdf(Q_n, df=D - 1)
            dict_p_hat[key] = p_hat
    return dict_chi2, dict_p_value, dict_D, dict_p_hat, list_rejected

def build_Sigma_matrix_new(D, cdf_fitted):
    barF = lambda k: 1.0 - float(cdf_fitted(k))
    Sigma = np.zeros((D - 1, D - 1), dtype=float)
    for k in range(1, D):
        for l in range(1, D):
            m = max(k, l)
            Sigma[k - 1, l - 1] = barF(m) - barF(k) * barF(l)
    return Sigma

def build_T_matrix(D, cdf_fitted):
    barF = lambda k: 1.0 - float(cdf_fitted(k))
    T = np.zeros((D - 1, D - 1), dtype=float)
    T[0, 0] = -1.0
    for i in range(2, D):
        denom = barF(i - 1)
        if denom <= 0:
            return None
        T[i - 1, i - 2] = barF(i) / (denom ** 2)
        T[i - 1, i - 1] = -1.0 / denom
    return T


def adaptive_D(nb_days_state_list, nb_days_min_for_D):
    D = 1
    while D < len(nb_days_state_list) and nb_days_state_list[D] > nb_days_min_for_D:
        D += 1
    return D

def _normalize_city(s):
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).strip().upper())

def build_gof_results_df(dict_chi2, dict_p_value, dict_D, df_info=None):
    rows = []
    for key, Q in dict_chi2.items():
        if " - " not in key:
            continue
        city, season = key.rsplit(" - ", 1)
        season = season.strip().lower()
        if season not in config.SEASONS:
            continue
        rows.append({
            "city": city.strip(),
            "season": season,
            "Q_n": float(Q),
            "p_value": float(dict_p_value.get(key, np.nan)),
            "D": int(dict_D[key]) if key in dict_D else np.nan,
        })
    df_res = pd.DataFrame(rows)
    if df_res.empty or df_info is None:
        return df_res

    df_info_merge = df_info.copy()
    df_info_merge["city_norm"] = df_info_merge["city"].map(_normalize_city)
    df_res["city_norm"] = df_res["city"].map(_normalize_city)

    df_out = df_res.merge(
        df_info_merge[["city_norm", "country", "lat", "lon"]],
        on="city_norm", how="left", validate="many_to_one",
    ).drop(columns=["city_norm"])
    return df_out.sort_values(["season", "city"]).reset_index(drop=True)


### Mean excess duration in dry spell

def expected_tau1(pi, p1, p2):
    if not (0.0 <= pi <= 1.0 and 0.0 < p1 <= 1.0 and 0.0 < p2 <= 1.0):
        raise ValueError(f"parameters must satisfy pi in [0,1], p1,p2 in (0,1]: {(pi, p1, p2)}")
    return pi / p1 + (1.0 - pi) / p2


def lower_incomplete_beta(x, a, b):
    """b_x(a,b) = int_0^x t^{a-1}(1-t)^{b-1} dt
    WARNING scipy.special.betainc = b_x(a,b) / beta(a,b)
    where beta(a,b) is the beta function
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return beta(a, b)
    return beta(a, b) * betainc(a, b, x)

def tail_integral_from_u(sigma, kappa, xi, u):
    if isclose(xi, 0.0, abs_tol=1e-14) or xi < 0:
        raise ValueError("This formula requires xi > 0.")
    base = 1.0 + xi * u / sigma
    a_u = 1.0 - base ** (-1.0 / xi)
    B = beta(kappa, 1.0 - xi)
    b_lower = lower_incomplete_beta(a_u, kappa, 1.0 - xi)
    tail = (sigma * kappa / xi) * (B - b_lower) \
           - (sigma / xi + u) * (1.0 - a_u ** kappa)
    return tail

def compute_LU_u(u, sigma, kappa, xi, survival_x):
    finite_sum = sum(survival_x(m) for m in range(u))
    tail_int = tail_integral_from_u(sigma, kappa, xi, u)
    L_p = finite_sum + tail_int
    U_p = L_p + survival_x(u)
    return L_p, U_p

def compute_bounds_mean_excess(d_thresh, u,
    sigma, kappa, xi, f1):
    """Bounds for E[tau0 - d | tau0 > d] using eq. (bounds_value_long_dry_spells_refined)."""
    if not (0.0 <= f1 <= 1.0):
        raise ValueError("f1 must be in [0,1].")
    if xi < 0:
        raise ValueError("xi must be positive (if not make exact evaluation)")

    def cdf_X(x):
        if x <= 0:
            return 0.0
        return get_ext_gpd_type_1_cdf(x, xi, sigma, kappa)
    def survival_X(x):
        return 1.0 - cdf_X(x)
    def cdf_tau0(z):
        if z <= 0:
            return 0.0
        elif z <= 1:
            return f1
        else:
            return f1 + (1.0 - f1) * cdf_X(z - 1)
    def survival_tau0(z):
        return 1.0 - cdf_tau0(z)

    L_u, U_u = compute_LU_u(u, sigma, kappa, xi, survival_X)

    # Numerator: E[tau0] - sum_{k=1}^{d} survival_tau0(k-1)
    # E[tau0] = 1 + (1-f1)*E[ceil(X)], with E[ceil(X)] in [L_u, U_u]
    tail_sum = sum(survival_tau0(k - 1) for k in range(1, d_thresh + 1))

    num_lower = 1.0 + (1.0 - f1) * L_u - tail_sum
    num_upper = 1.0 + (1.0 - f1) * U_u - tail_sum

    # Denominator: survival_tau0(d) — no bounds needed, it's exact
    denom = survival_tau0(d_thresh)

    if denom <= 0:
        return {"lower_bound": float('inf'), "upper_bound": float('inf'),
                "denom": denom}

    lower_bound = num_lower / denom
    upper_bound = num_upper / denom

    return {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "L_u": L_u,
        "U_u": U_u,
        "tail_sum_tau0": tail_sum,
        "numerator_lower": num_lower,
        "numerator_upper": num_upper,
        "denominator": denom,
    }


def make_approx_mean_excess(
    d_thresh,
    sigma, kappa, xi, f1,
    target_precision=1e-15):
    """Approximate E[tau0 - d | tau0 > d]."""

    def cdf_tau0(z):
        if z <= 0:
            return 0.0
        elif z <= 1:
            return f1
        else:
            return f1 + (1.0 - f1) * get_ext_gpd_type_1_cdf(z - 1, xi, sigma, kappa)
    def survival_tau0(z):
        return 1.0 - cdf_tau0(z)

    if xi < 0:
        d_lim = -sigma / xi
        if d_thresh > d_lim:
            return 0.0
        E_tau0 = np.sum([survival_tau0(z) for z in range(int(d_lim) + 1)])
        tail_sum = sum(survival_tau0(k - 1) for k in range(1, d_thresh + 1))
        denom = survival_tau0(d_thresh)
        if denom <= 0:
            return 0.0
        return (E_tau0 - tail_sum) / denom
    else:
        u = 5000
        precision = 10
        while precision > target_precision:
            u = u * 10
            bounds = compute_bounds_mean_excess(d_thresh, u,
                    sigma, kappa, xi, f1)
            precision = bounds["upper_bound"] - bounds["lower_bound"]
        return bounds["lower_bound"]


def mean_excess_markov_order_1(p_geom_dry, d):
    """E[tau0 - d | tau0 > d] for geometric dry spells (eq. share_time_long_dry_spells_markov_case)."""
    return 1/p_geom_dry
    # Simplifies to 1/p_geom_dry (memoryless property)


### Reorganize data structure utils
def split_spells_by_season_simple(
    data,
    start_key="start_date_spell",
    dur_key="duration_spell",
    seasons=None,
):
    """
    Group per-city spells by season.

    Input  : data[city]["dry_spell"|"wet_spell"][start_key|dur_key] = list
    Output : out[season][city]["dry_spell"|"wet_spell"][start_key|dur_key] = filtered list
    """
    if seasons is None:
        seasons = {
            "winter": {12, 1, 2},
            "spring": {3, 4, 5},
            "summer": {6, 7, 8},
            "autumn": {9, 10, 11},
        }

    out = {s: {} for s in seasons}
    for city, city_data in data.items():
        for season_name, months_set in seasons.items():
            out[season_name][city] = {}
            for spell_type in ("dry_spell", "wet_spell"):
                starts = city_data[spell_type][start_key]
                durs   = city_data[spell_type][dur_key]
                f_starts, f_durs = [], []
                for s, d in zip(starts, durs):
                    month = int(str(s)[4:6])
                    if month in months_set:
                        f_starts.append(s)
                        f_durs.append(d)
                out[season_name][city][spell_type] = {
                    start_key: f_starts,
                    dur_key:   f_durs,
                }
    return out


def build_consecutive_pairs(
    dry_durations, dry_start_ints,
    wet_durations, wet_start_ints,
):
    """
    Returns:
      pairs: list of [[[dry_dur, wet_dur], [dry_start_int, wet_start_int]], ...]
      dropped: dict with counts {"dry":..., "wet":...}
    """
    # Convert to dates
    dry_starts = [int_yyyymmdd_to_date(x) for x in dry_start_ints]
    wet_starts = [int_yyyymmdd_to_date(x) for x in wet_start_ints]

    i, j = 0, 0
    pairs = []
    dropped_dry = 0
    dropped_wet = 0


    while i < len(dry_durations) and j < len(wet_durations):
        ds = dry_starts[i]
        dd = int(dry_durations[i])
        de = ds + timedelta(days=dd)  # end date = start + duration days

        ws = wet_starts[j]
        wd = int(wet_durations[j])

        # Rule 1: wet starts before the "next" dry spell -> drop wet
        if ws < ds:
            j += 1
            dropped_wet += 1
            continue

        # Rule 2: consecutive match
        if ws == de:
            pairs.append([
                [dd, wd],
                [date_to_int_yyyymmdd(ds), date_to_int_yyyymmdd(ws)]
            ])
            i += 1
            j += 1
            continue

        # Rule 3: "cut" situations (not consecutive) -> drop until consecutive
        # If wet starts after dry ends, this dry cannot match any later wet (unless you skip wet backward),
        # so drop dry and try next dry with same wet.
        if ws > de:
            i += 1
            dropped_dry += 1
            continue

        # If wet starts after dry start but before dry end => overlap/inconsistency.
        # Best “drop until consecutive” strategy: drop the wet (too early to be consecutive with dry end).
        if ds < ws < de:
            j += 1
            dropped_wet += 1
            continue

        # Fallback (should rarely hit)
        # Move the earlier-starting spell forward
        if ds <= ws:
            i += 1
            dropped_dry += 1
        else:
            j += 1
            dropped_wet += 1

    dropped = {"dry": dropped_dry, "wet": dropped_wet}
    return pairs, dropped


def build_data_per_city_per_season_per_year_couple_vector_duration_vector_date(data_by_season):
    """
    Output structure:
    data_per_city_per_season_per_year_couple_vector_duration_vector_date[city][season][year_couple]
        = [
            [[dry_duration, wet_duration], [dry_start_int, wet_start_int]],
            ...
          ]
    """
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for season, cities_dict in data_by_season.items():
        for city, city_data in cities_dict.items():
            dry = city_data["dry_spell"]
            wet = city_data["wet_spell"]

            dry_start = dry["start_date_spell"]
            dry_dur   = dry["duration_spell"]
            wet_start = wet["start_date_spell"]
            wet_dur   = wet["duration_spell"]

            pairs, dropped = build_consecutive_pairs(
                dry_durations=dry_dur,
                dry_start_ints=dry_start,
                wet_durations=wet_dur,
                wet_start_ints=wet_start,
            )

            for durations, dates in pairs:
                dry_start_int, wet_start_int = dates
                dry_start_date = int_yyyymmdd_to_date(dry_start_int)

                yc = year_couple_from_date(dry_start_date, season)
                out[city][season][yc].append([durations, dates])

    return out


def season_spell_durations(data, city, season):
    dry, wet = data[city]["dry_spell"], data[city]["wet_spell"]
    dry_durs = [dur for dur, date in zip(dry["duration_spell"], dry["start_date_spell"])
                if from_date_to_season(date) == season]
    wet_durs = [dur for dur, date in zip(wet["duration_spell"], wet["start_date_spell"])
                if from_date_to_season(date) == season]
    return dry_durs, wet_durs


### Datetime utils
def int_yyyymmdd_to_date(x: int):
    return datetime.strptime(str(x), "%Y%m%d").date()

def date_to_int_yyyymmdd(d):
    return int(d.strftime("%Y%m%d"))

def year_couple_from_date(d, season):
    """
    Return the seasonal year key.

    For spring/summer/autumn:
        1947-xx-xx -> (1947, 1947)

    For winter:
        Dec 1946, Jan 1947, Feb 1947 -> (1946, 1947)
    """
    y = d.year
    m = d.month

    if season == "winter":
        if m == 12:
            return (y, y + 1)
        else:  # Jan / Feb / maybe Mar if your splitter allows it
            return (y - 1, y)
    else:
        return (y, y)


def from_date_to_season(date):
    date_without_year = int(str(date)[4:])
    if date_without_year >= 301 and date_without_year < 601:
        return("spring")
    if date_without_year >= 601 and date_without_year < 901:
        return("summer")
    if date_without_year >= 901 and date_without_year < 1201:
        return("autumn")
    if date_without_year >= 1201 or date_without_year < 301:
        return("winter")