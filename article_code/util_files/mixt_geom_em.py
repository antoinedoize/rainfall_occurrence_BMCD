import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from . import config


def _logsumexp(a, axis=None, keepdims=False):
    a = np.asarray(a)
    amax = np.max(a, axis=axis, keepdims=True)
    out = amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def fit_geometric_mixture_em_support1(x, n_init=10, max_iter=1000, tol=1e-8, seed=0,
                                      min_p=1e-10, min_pi=1e-10, sort_params=True):
    """
    Fit a 2-component mixture of Geom(p) on support {1,2,...} via EM.
    Returns: dict with keys pi, p1, p2, loglik, n_iter
    """
    x = np.asarray(x, dtype=np.int64)
    if np.any(x < 1):
        raise ValueError("x must be positive integers for support {1,2,3,...}.")
    n = x.size
    if n == 0:
        raise ValueError("Empty data.")
    rng = np.random.default_rng(seed)
    def log_pmf_geom1(x, p):
        p = np.clip(p, min_p, 1 - min_p)
        return np.log(p) + (x - 1) * np.log1p(-p)
    def loglik(pi, p1, p2):
        pi = np.clip(pi, min_pi, 1 - min_pi)
        lp1 = np.log(pi)    + log_pmf_geom1(x, p1)
        lp2 = np.log1p(-pi) + log_pmf_geom1(x, p2)
        return float(np.sum(_logsumexp(np.vstack([lp1, lp2]), axis=0)))
    best = None
    mean_x = float(np.mean(x))
    base_p = 1.0 / (mean_x + 1e-12)
    for _ in range(n_init):
        pi = rng.uniform(0.2, 0.8)
        p1 = np.clip(base_p * rng.uniform(0.6, 1.4), min_p, 1 - min_p)
        p2 = np.clip(base_p * rng.uniform(0.6, 1.4), min_p, 1 - min_p)
        if abs(p1 - p2) < 1e-6:
            p2 = np.clip(p2 * 0.8, min_p, 1 - min_p)
        prev_ll = -np.inf
        for it in range(1, max_iter + 1):
            pi = np.clip(pi, min_pi, 1 - min_pi)
            lp1 = np.log(pi)    + log_pmf_geom1(x, p1)
            lp2 = np.log1p(-pi) + log_pmf_geom1(x, p2)
            denom = _logsumexp(np.vstack([lp1, lp2]), axis=0)
            r = np.exp(lp1 - denom)
            r_sum = float(np.sum(r))
            r2_sum = float(n - r_sum)
            pi = np.clip(r_sum / n, min_pi, 1 - min_pi)
            denom1 = float(np.sum(r * x))
            denom2 = float(np.sum((1 - r) * x))
            p1 = np.clip(r_sum / max(denom1, 1e-300), min_p, 1 - min_p)
            p2 = np.clip(r2_sum / max(denom2, 1e-300), min_p, 1 - min_p)
            ll = loglik(pi, p1, p2)
            if abs(ll - prev_ll) <= tol * (1.0 + abs(ll)):
                break
            prev_ll = ll
        if sort_params and p1 < p2:
            p1, p2 = p2, p1
            pi = 1.0 - pi
        res = {"pi": float(pi), "p1": float(p1), "p2": float(p2),
               "loglik": loglik(pi, p1, p2), "n_iter": it}
        if best is None or res["loglik"] > best["loglik"]:
            best = res
    return best


def fit_mixt_geom_wet_spell_durations(json_path,
    path_folder_save_results,
    subset_city_to_fit=False):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    list_cities = sorted(data.keys()) if subset_city_to_fit is False else sorted(subset_city_to_fit)
    results_rows = []
    for city in tqdm(list_cities, desc=f"{"wet_spell"}"):
        node = data[city]["wet_spell"]
        durations   = np.asarray(node["duration_spell"], dtype=float)
        start_dates = np.asarray(node["start_date_spell"], dtype=str)
        months = np.array([s[4:6] for s in start_dates], dtype="<U2")
        for vec_month_seasons in config.LIST_MONTH_SEASONS:
            season_label = vec_month_seasons[3]
            title = f"{city} {season_label}"
            month_set = set(vec_month_seasons)
            season_mask = np.array([m in month_set for m in months])
            spells_season = durations[season_mask].copy()
            if spells_season.size == 0:
                continue
            fit = fit_geometric_mixture_em_support1(spells_season)
            results_rows.append(dict(data_source=title,
                city=city, season=season_label, pi=fit['pi'], p1=fit['p1'], p2=fit['p2']))
    df_out = pd.DataFrame(results_rows,
                        columns=["data_source","city","season", "pi", "p1", "p2"])
    csv_path = path_folder_save_results / (
        f"wet_spell_fit_mixt_geom"
        f"result_fit_parameters.csv")
    df_out.to_csv(csv_path, index=False)