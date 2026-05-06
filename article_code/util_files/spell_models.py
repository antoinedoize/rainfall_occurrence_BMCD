import numpy as np


# Empirical model

def S_hat(values, t):
    values = np.asarray(values)
    est = np.mean(values >= t)
    var = est*(1-est)/len(values)
    return est, np.sqrt(var)


# Geometric model
def fit_geometric_p_mle(season_durations):
    season_durations = np.asarray(season_durations, dtype=float)
    mean_ = float(np.mean(season_durations))
    p_hat = 1.0 / mean_
    p_hat = min(max(p_hat, 1e-12), 1.0)
    return p_hat

def make_cdf_fitted_geometric_from_p(p):
    p = float(p)
    if not (0.0 < p <= 1.0):
        raise ValueError(f"Geometric p must be in (0,1], got {p}")
    q = 1.0 - p
    def cdf(x):
        x = np.asarray(x)
        k = np.floor(x).astype(int)
        out = np.zeros_like(k, dtype=float)
        mask = k >= 1
        out[mask] = 1.0 - np.power(q, k[mask])
        return out if out.shape != () else float(out)
    return cdf


# Mixture geometric model
def geom1_pmf(ks, p):
    ks = np.asarray(ks, dtype=int)
    return p * (1.0 - p) ** (ks - 1)

def geom1_survival(ks, p):
    # S(k) = P(X >= k) = (1-p)^(k-1) for support {1,2,...}
    ks = np.asarray(ks, dtype=int)
    return (1.0 - p) ** (ks - 1)

def mix_geom_pmf(ks, pi, p1, p2):
    ks = np.asarray(ks, dtype=int)
    return pi * geom1_pmf(ks, p1) + (1.0 - pi) * geom1_pmf(ks, p2)

def sample_geom_mix(n, pi, p1, p2, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    z = rng.random(n) < pi
    x = np.empty(n, dtype=np.int64)
    x[z]  = rng.geometric(p1, size=z.sum())
    x[~z] = rng.geometric(p2, size=(~z).sum())
    return x


# hdeGPD model

def get_ext_gpd_type_1_cdf(z,xi_gpd, sigma_gpd, kappa_gpd):
    if xi_gpd < 0:
        right_bound = - sigma_gpd/xi_gpd
        if z > right_bound:
            return 1
    inside_factor = 1+xi_gpd*z/sigma_gpd
    temp = 1 - inside_factor**(-1/xi_gpd)
    output = temp ** kappa_gpd
    return output

def inv_gpd_distrib(xi_gpd,u):
    num = (1-u)**(-xi_gpd)-1
    return num / xi_gpd

def get_ext_gpd_type_1(n_var,xi_gpd, sigma_gpd, kappa_gpd):
    uniform = np.random.rand(n_var)
    modif_uniform = uniform**(1/kappa_gpd)
    gpd = sigma_gpd*inv_gpd_distrib(xi_gpd,modif_uniform)
    return gpd

def make_cdf_fitted_hdeGPD_from_params(f_1, xi_gpd, sigma_gpd, kappa_gpd):
    def cdf_fitted(z):
        if z <= 0:
            return 0.0
        if z <= 1:
            return float(f_1)
        tail = get_ext_gpd_type_1_cdf(int(z - 1), xi_gpd, sigma_gpd, kappa_gpd)
        return float(f_1 + (1.0 - f_1) * tail)
    return cdf_fitted

def make_cdf_fitted_extgpd_from_season(season_durations, df_fit, city, season):
    extract_df = df_fit[df_fit["data_source"] == f"{city} {season}"]
    xi_gpd = extract_df["xi"].item()
    sigma_gpd = extract_df["sigma"].item()
    kappa_gpd = extract_df["kappa"].item()
    proba1 = sum(s == 1 for s in season_durations) / len(season_durations)
    return make_cdf_fitted_hdeGPD_from_params(proba1, xi_gpd, sigma_gpd, kappa_gpd)

def get_spell_length_degenerate_mixture_order_1_extgpd1(xi_gpd, sigma_gpd, kappa_gpd,f_1):
    alea = np.random.random()
    if alea < f_1:
        return 1
    else:
        return int(2 + get_ext_gpd_type_1(1,xi_gpd, sigma_gpd, kappa_gpd)[0])