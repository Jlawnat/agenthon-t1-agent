from __future__ import annotations
import math
import numpy as np
from scipy import optimize, stats

def monitoring_times(T: float, n_monitoring: int) -> np.ndarray:
    return np.linspace(T / n_monitoring, T, n_monitoring, dtype=float)

def geometric_asian_call(*, S0: float, K: float, T: float, r: float, sigma: float, n_monitoring: int) -> float:
    times = monitoring_times(T, n_monitoring)
    covariance = np.minimum.outer(times, times)
    mu_log_g = float(math.log(S0) + (r - 0.5 * sigma * sigma) * float(np.mean(times)))
    var_log_g = float(sigma * sigma * np.sum(covariance) / (n_monitoring * n_monitoring))
    if var_log_g <= 0.0:
        forward_g = math.exp(mu_log_g)
        return float(math.exp(-r * T) * max(forward_g - K, 0.0))
    root_var = math.sqrt(var_log_g)
    d1 = (mu_log_g - math.log(K) + var_log_g) / root_var
    d2 = d1 - root_var
    expected_g = math.exp(mu_log_g + 0.5 * var_log_g)
    return float(math.exp(-r * T) * (expected_g * stats.norm.cdf(d1) - K * stats.norm.cdf(d2)))

def arithmetic_moments(*, S0: float, T: float, r: float, sigma: float, n_monitoring: int) -> tuple[float, float]:
    times = monitoring_times(T, n_monitoring)
    m1 = float(S0 * np.mean(np.exp(r * times)))
    ti = times[:, None]
    tj = times[None, :]
    m2 = float(S0 * S0 * np.mean(np.exp(r * (ti + tj) + sigma * sigma * np.minimum(ti, tj))))
    return (m1, m2)

def levy_asian_call(*, S0: float, K: float, T: float, r: float, sigma: float, n_monitoring: int) -> float:
    m1, m2 = arithmetic_moments(S0=S0, T=T, r=r, sigma=sigma, n_monitoring=n_monitoring)
    variance_log = float(math.log(max(m2 / (m1 * m1), 1.0)))
    if variance_log <= 1e-16:
        return float(math.exp(-r * T) * max(m1 - K, 0.0))
    root_var = math.sqrt(variance_log)
    d1 = (math.log(m1 / K) + 0.5 * variance_log) / root_var
    d2 = d1 - root_var
    return float(math.exp(-r * T) * (m1 * stats.norm.cdf(d1) - K * stats.norm.cdf(d2)))

def curran_asian_call(*, S0: float, K: float, T: float, r: float, sigma: float, n_monitoring: int) -> float:
    times = monitoring_times(T, n_monitoring)
    n = int(n_monitoring)
    cov_w = sigma * sigma * np.minimum.outer(times, times)
    mu_x = math.log(S0) + (r - 0.5 * sigma * sigma) * times
    mu_y = float(np.mean(mu_x))
    cov_xy = np.sum(cov_w, axis=1) / n
    var_y = float(np.sum(cov_w) / (n * n))
    var_x = sigma * sigma * times
    if var_y <= 0.0:
        return levy_asian_call(S0=S0, K=K, T=T, r=r, sigma=sigma, n_monitoring=n_monitoring)
    slopes = cov_xy / var_y
    conditional_intercepts = mu_x - slopes * mu_y + 0.5 * (var_x - cov_xy * cov_xy / var_y)

    def conditional_average(y_value: float) -> float:
        return float(np.mean(np.exp(conditional_intercepts + slopes * y_value)))
    root_var_y = math.sqrt(var_y)
    lower = mu_y - 12.0 * root_var_y
    upper = mu_y + 12.0 * root_var_y
    lower_value = conditional_average(lower) - K
    upper_value = conditional_average(upper) - K
    if lower_value >= 0.0:
        y_star = lower
    elif upper_value <= 0.0:
        y_star = upper
    else:
        y_star = float(optimize.brentq(lambda y: conditional_average(y) - K, lower, upper, xtol=1e-12, rtol=1e-12))
    asset_tail_expectations = S0 * np.exp(r * times) * stats.norm.cdf((mu_y + cov_xy - y_star) / root_var_y)
    exercise_probability = float(stats.norm.cdf((mu_y - y_star) / root_var_y))
    price = math.exp(-r * T) * (float(np.mean(asset_tail_expectations)) - K * exercise_probability)
    return float(max(price, 0.0))

def monte_carlo_asian(*, S0: float, strikes: list[float], T: float, r: float, sigma: float, n_monitoring: int, n_paths: int, rng: np.random.Generator) -> list[tuple[float, float, float]]:
    dt = T / n_monitoring
    normals = rng.standard_normal((n_paths, n_monitoring))
    log_increments = (r - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * normals
    log_paths = math.log(S0) + np.cumsum(log_increments, axis=1)
    paths = np.exp(log_paths)
    arithmetic_average = np.mean(paths, axis=1)
    geometric_average = np.exp(np.mean(log_paths, axis=1))
    discount = math.exp(-r * T)
    results = []
    for strike in strikes:
        arithmetic_payoff = np.maximum(arithmetic_average - strike, 0.0)
        geometric_payoff = np.maximum(geometric_average - strike, 0.0)
        discounted_arithmetic = discount * arithmetic_payoff
        mc_arith = float(np.mean(discounted_arithmetic))
        mc_geo = float(discount * np.mean(geometric_payoff))
        standard_error = float(np.std(discounted_arithmetic, ddof=1) / math.sqrt(float(n_paths)))
        results.append((mc_arith, mc_geo, standard_error))
    return results
__all__ = ('monitoring_times', 'geometric_asian_call', 'arithmetic_moments', 'levy_asian_call', 'curran_asian_call', 'monte_carlo_asian')
