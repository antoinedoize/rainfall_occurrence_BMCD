"""
Python translation of R `mev::fit.extgp` for the specific case used in the
calling script:

    fit.extgp(data, model = 1, method = "pwm",
              init = c(kappa0, sigma0, xi0),
              censoring = c(0, Inf), rounded = 1, plot = FALSE)

Model 1 = Naveau et al. (2016) extended GP with carrier G(u) = u^kappa.

The `rounded` argument only affects the MLE branch in R, so it is not used
here. PWM with an exactly identified system (3 params, 3 moments) reduces
to solving g(theta) = 0, which maps onto scipy.optimize.least_squares.

Reference:
  Naveau, Huser, Ribereau, Hannart (2016). Modeling jointly low, moderate,
  and heavy rainfall intensities without a threshold selection.
  Water Resour. Res., 52, 2753-2769. doi:10.1002/2015WR018552
"""
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from scipy.special import beta as beta_func, comb
from scipy.stats import beta as beta_dist
from scipy.optimize import least_squares
from .config import LIST_MONTH_SEASONS


# LIST_MONTH_SEASONS = [
#     ["01", "02", "03", "all","04", "05", "06", "07", "08", "09", "10", "11", "12"],
#     ["03", "04", "05", "spring"],
#     ["06", "07", "08", "summer"],
#     ["09", "10", "11", "autumn"],
#     ["12", "01", "02", "winter"],
# ]


# Generalized Pareto distribution helpers (equivalent to mev::pgp / mev::qgp)

def pgp(q, loc=0.0, scale=1.0, shape=0.0):
    """CDF of the Generalized Pareto Distribution."""
    q = np.asarray(q, dtype=float)
    z = np.maximum((q - loc) / scale, 0.0)
    if abs(shape) < 1e-12:
        return 1.0 - np.exp(-z)
    base = np.maximum(1.0 + shape * z, 0.0)
    return 1.0 - base ** (-1.0 / shape)


def qgp(p, loc=0.0, scale=1.0, shape=0.0):
    """Quantile function of the Generalized Pareto Distribution."""
    p = np.asarray(p, dtype=float)
    if abs(shape) < 1e-12:
        return loc - scale * np.log1p(-p)
    return loc + scale * ((1.0 - p) ** (-shape) - 1.0) / shape


# Extended GP type 1 distribution: F(x) = H(x)^kappa with H = GPD(sigma, xi)

def pextgp_type1(q, kappa, sigma, xi):
    """CDF of the extended GP type 1."""
    u = np.clip(pgp(q, scale=sigma, shape=xi), 0.0, 1.0)
    return u ** kappa


def qextgp_type1(p, kappa, sigma, xi):
    """Quantile of the extended GP type 1."""
    p = np.asarray(p, dtype=float)
    u = p ** (1.0 / kappa)
    return qgp(u, scale=sigma, shape=xi)


def rextgp_type1(n, kappa, sigma, xi, censoring=(0.0, np.inf), rng=None):
    """Random sample from the extended GP type 1 (with optional censoring)."""
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random(n)
    F_L = pextgp_type1(censoring[0], kappa, sigma, xi) if censoring[0] > 0 else 0.0
    F_U = pextgp_type1(censoring[1], kappa, sigma, xi) if censoring[1] < np.inf else 1.0
    return qextgp_type1(F_L + (F_U - F_L) * u, kappa, sigma, xi)


# Theoretical Probability Weighted Moments, type 1
#   alpha_r = E[X * (1 - F(X))^r]

def extgp_pwm_type1(orders, kappa, sigma, xi, censoring=(0.0, np.inf)):
    """
    Theoretical PWMs for the type-1 extended GP (closed-form from Naveau et al.).

    Parameters
    ----------
    orders : array-like of ints
        PWM orders r = 0, 1, 2, ...
    kappa, sigma, xi : float
        Extended GP parameters.
    censoring : (lo, hi)
        Censoring interval (defaults to no censoring).

    Returns
    -------
    ndarray of length len(orders).
    """
    orders = np.asarray(orders, dtype=int)

    # Carrier-scale bounds
    H_L = pgp(censoring[0], scale=sigma, shape=xi) if censoring[0] > 0 else 0.0
    H_U = pgp(censoring[1], scale=sigma, shape=xi) if censoring[1] < np.inf else 1.0

    # Data-scale bounds
    F_L = pextgp_type1(censoring[0], kappa, sigma, xi) if censoring[0] > 0 else 0.0
    F_U = pextgp_type1(censoring[1], kappa, sigma, xi) if censoring[1] < np.inf else 1.0
    prob_LU = F_U - F_L

    # E2 term (independent of kappa)
    E2 = ((1.0 - F_L) ** (orders + 1) - (1.0 - F_U) ** (orders + 1)) / (prob_LU * (orders + 1))

    # E1 term: closed-form sum of incomplete-Beta integrals
    max_order = int(orders.max())
    k_vals = np.arange(1, max_order + 2)              # k = 1 ... max_order+1
    beta_coefs = beta_func(k_vals * kappa, 1.0 - xi)  # B(k*kappa, 1-xi)
    probs_beta = (beta_dist.cdf(H_U, k_vals * kappa, 1.0 - xi)
                  - beta_dist.cdf(H_L, k_vals * kappa, 1.0 - xi))

    E1 = np.empty(len(orders))
    for i, ord_i in enumerate(orders):
        ord_i = int(ord_i)
        j = np.arange(ord_i + 1)
        E1[i] = (kappa / prob_LU) * np.sum(
            (-1.0) ** j
            * comb(ord_i, j, exact=False)
            * beta_coefs[:ord_i + 1]
            * probs_beta[:ord_i + 1]
        )

    return (sigma / xi) * (E1 - E2)


# PWM fitting

def fit_extgp_pwm(x, init=(1.0, 20.0, 0.1), censoring=(0.0, np.inf)):
    """
    Fit the type-1 extended GP by Probability Weighted Moments.

    Python equivalent of:
        fit.extgp(data, model = 1, method = "pwm",
                  init = c(kappa0, sigma0, xi0), censoring = c(lo, hi))
    (PWM branch only; `rounded` is ignored because it is MLE-specific.)

    Parameters
    ----------
    x : array-like
        Data vector. Non-positive values are dropped, matching R behaviour.
    init : (kappa0, sigma0, xi0)
        Initial parameter values.
    censoring : (lo, hi)
        Censoring interval.

    Returns
    -------
    dict with keys 'kappa', 'sigma', 'xi', 'success', 'message', 'cost'.
    """
    x = np.asarray(x, dtype=float)
    x = x[x > 0]                           # matches R: data <- data[data > 0]
    n = x.size
    if n < 3:
        raise ValueError("Need at least 3 positive observations.")

    sorted_x = np.sort(x)

    mask = (x > censoring[0]) & (x < censoring[1])
    x_sub = x[mask]

    # ecdf(x)(x_sub): fraction of data values <= x_sub[i]
    F_emp = np.searchsorted(sorted_x, x_sub, side="right") / n

    # Empirical PWMs of orders 0, 1, 2
    mu_hat = np.array([np.mean(x_sub * (1.0 - F_emp) ** r) for r in range(3)])

    def residuals(theta):
        kappa, sigma, xi = theta
        try:
            pwm_th = extgp_pwm_type1([0, 1, 2], kappa, sigma, xi, censoring)
        except Exception:
            return np.full(3, 1e6)
        if not np.all(np.isfinite(pwm_th)):
            return np.full(3, 1e6)
        return pwm_th - mu_hat

    # Bounds that mirror the R gmm(...) call for model = 1
    lo = [1e-4, 1e-4, -0.5]
    hi = [np.inf, np.inf, 0.99]

    theta0 = np.clip(np.asarray(init, dtype=float), lo, [1e6, 1e6, 0.98])

    res = least_squares(residuals, theta0, method="trf", bounds=(lo, hi))

    return {
        "kappa":   float(res.x[0]),
        "sigma":   float(res.x[1]),
        "xi":      float(res.x[2]),
        "success": bool(res.success),
        "message": str(res.message),
        "cost":    float(res.cost),
    }


def get_dry_spells_from_json(data_json, city):
    """
    Extract durations + start months for a given city / spell_type.
    spell_type : 'dry_spell' or 'wet_spell'.
    """
    if city not in data_json:
        return np.array([]), np.array([], dtype="<U2")
    node = data_json[city]["dry_spell"]
    if node is None:
        return np.array([]), np.array([], dtype="<U2")

    durations   = np.asarray(node["duration_spell"], dtype=float)
    start_dates = np.asarray(node["start_date_spell"], dtype=str)
    # months = positions 5..6 of YYYYMMDD strings (1-indexed), keep zero-padding
    months = np.array([s[4:6] for s in start_dates], dtype="<U2")
    return durations, months


def fit_hdegpd_dry_spell_durations(
    json_path,
    path_folder_save_results,
    subset_city_to_fit=False,
    init=(1.0, 10.0, 0.1)):
    with open(json_path, "r", encoding="utf-8") as f:
        data_json = json.load(f)
    df_names = sorted(data_json.keys()) if subset_city_to_fit is False else sorted(subset_city_to_fit)
    results_rows = []
    for df_name in tqdm(df_names, desc=f"dry_spell"):
        durations, months = get_dry_spells_from_json(data_json, df_name)
        for vec_month_seasons in LIST_MONTH_SEASONS:
            season_label = vec_month_seasons[3]
            month_set = set(vec_month_seasons)
            season_mask = np.array([m in month_set for m in months])
            spells_season = durations[season_mask].copy()
            if spells_season.size == 0:
                continue
            f_1 = len(spells_season[spells_season == 1]) / len(spells_season)
            spells_season = spells_season[spells_season>=2] - 2
            ### warining : je dois filtrer quand > 0 (géré par f_1)???
            if spells_season.size < 3:
                continue  # skip if not enough data
            title = f"{df_name} {season_label}"
            fit = fit_extgp_pwm(spells_season,init=init,censoring=(0.0, np.inf),)
            results_rows.append(
                {"data_source": title, "f_1":f_1,"kappa": fit["kappa"],"sigma": fit["sigma"],"xi":fit["xi"],})
        # --- save outputs ---
        # thr_dir = path_folder_save_results #/ f"thr_{val_threshold}"
        # thr_dir.mkdir(parents=True, exist_ok=True)
        df_out = pd.DataFrame(results_rows,
                              columns=["data_source", "f_1", "kappa", "sigma", "xi"])
        csv_path = path_folder_save_results / (
            f"dry_spell_fit_egpd1_excess_over_1"
            f"result_fit_parameters.csv")
        df_out.to_csv(csv_path, index=False)
