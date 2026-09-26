from __future__ import annotations

"""
Curated quantitative-finance primitives exposed to generated candidates.

This module is intentionally self-contained and side-effect free:
- no network access,
- no filesystem access,
- no subprocesses,
- no model/orchestrator imports.

Candidate workspaces receive a read-only copy as ``qf_primitives.py``.
"""

import math
from typing import Sequence

import numpy as np


PRIMITIVE_API_CATALOG = (
    "black_scholes_price(spot, strike, rate, dividend_yield, volatility, maturity, option_type) -> float",
    "black_scholes_greeks(spot, strike, rate, dividend_yield, volatility, maturity, option_type) -> dict",
    "historical_log_return_calibration(closes, annualization=252.0) -> dict",
    "discount_cashflow(amount, rate, maturity) -> float",
    "sma_seeded_ema(values, span) -> ndarray",
    "ewma_annualized_volatility(log_returns, lookback, annualization=252.0) -> ndarray",
    "log_return_performance(log_returns, risk_free_annual, annualization=252.0) -> dict",
    "down_and_out_call_price(spot, strike, barrier, maturity, rate, volatility, dividend_yield=0.0) -> float",
    "central_price_delta(price_fn, spot, relative_step=0.01) -> float",
    "ols_with_intercept(x, y) -> dict",
    "fit_ou_euler(values, dt) -> dict",
    "fit_ou_exact_ar1(values, dt) -> dict",
    "ou_exact_step_parameters(kappa, sigma, dt) -> dict",
    "simulate_ou_exact(initial, kappa, mu, sigma, dt, n_steps, n_paths, seed) -> ndarray[n_paths,n_steps]",
    "brownian_running_max_hit_probability(mu, sigma, horizon, barrier) -> float",
    "brownian_running_min_hit_probability(mu, sigma, horizon, barrier) -> float",
    "brownian_joint_terminal_max_cdf(mu, sigma, horizon, terminal_level, max_barrier) -> float",
    "empirical_var_es_from_losses(losses, confidence, strict_tail=False) -> (var, es)",
    "normalize_nonnegative(values) -> ndarray",
    "ewma_weights(length, half_life, newest_first=True) -> ndarray",
    "largest_remainder_allocate(total, weights) -> ndarray[int]",
    "fit_garch11_zero_mean(returns, min_observations=50) -> dict",
    "garch11_forecast_variance(omega, alpha, beta, last_return, last_variance, horizon) -> dict",
    "implied_volatility_black_scholes(market_price, spot, strike, rate, dividend_yield, maturity, option_type, upper_volatility=5.0) -> float",
    "discount_factor_from_continuous_zero_rate(zero_rate, maturity) -> float",
    "continuous_zero_rate_from_discount_factor(discount_factor, maturity) -> float",
    "log_linear_discount_factor(pillars, maturity) -> float",
    "bootstrap_annual_par_discount_factors(par_yields) -> dict",
    "pseudo_observations(values) -> ndarray",
    "sample_gaussian_copula(n, rho, rng) -> tuple[ndarray, ndarray]",
    "sample_student_t_copula(n, rho, degrees_of_freedom, rng) -> tuple[ndarray, ndarray]",
    "fit_gpd_exceedances(excesses) -> dict",
    "evt_var_es_from_gpd(probability, threshold, shape, scale, n_total, n_exceedances) -> dict",
    "hill_tail_index(losses) -> float",
)

__all__ = [
    "PRIMITIVE_API_CATALOG",
    "black_scholes_price",
    "black_scholes_greeks",
    "historical_log_return_calibration",
    "discount_cashflow",
    "sma_seeded_ema",
    "ewma_annualized_volatility",
    "log_return_performance",
    "down_and_out_call_price",
    "central_price_delta",
    "ols_with_intercept",
    "fit_ou_euler",
    "fit_ou_exact_ar1",
    "ou_exact_step_parameters",
    "simulate_ou_exact",
    "brownian_running_max_hit_probability",
    "brownian_running_min_hit_probability",
    "brownian_joint_terminal_max_cdf",
    "empirical_var_es_from_losses",
    "normalize_nonnegative",
    "ewma_weights",
    "largest_remainder_allocate",
    "fit_garch11_zero_mean",
    "garch11_forecast_variance",
    "implied_volatility_black_scholes",
    "discount_factor_from_continuous_zero_rate",
    "continuous_zero_rate_from_discount_factor",
    "log_linear_discount_factor",
    "bootstrap_annual_par_discount_factors",
    "pseudo_observations",
    "sample_gaussian_copula",
    "sample_student_t_copula",
    "fit_gpd_exceedances",
    "evt_var_es_from_gpd",
    "hill_tail_index",
]


def _finite_1d(values: Sequence[float], *, minimum: int = 1) -> np.ndarray:
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < minimum:
        raise ValueError(f"Expected at least {minimum} finite observations.")
    return x


def _normal_cdf(value: float) -> float:
    return float(0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0))))


def _normal_pdf(value: float) -> float:
    x = float(value)
    return float(math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi))


def black_scholes_price(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
    option_type: str,
) -> float:
    s = float(spot)
    k = float(strike)
    r = float(rate)
    q = float(dividend_yield)
    sigma = float(volatility)
    t = float(maturity)
    if s <= 0.0 or k <= 0.0:
        raise ValueError("Black-Scholes requires positive spot and strike.")
    if sigma < 0.0 or t < 0.0:
        raise ValueError("Volatility and maturity must be non-negative.")
    kind = str(option_type).strip().lower()
    if kind not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    if t == 0.0:
        return float(max(s - k, 0.0) if kind == "call" else max(k - s, 0.0))
    discounted_spot = s * math.exp(-q * t)
    discounted_strike = k * math.exp(-r * t)
    if sigma == 0.0:
        forward_pv = discounted_spot - discounted_strike
        return float(max(forward_pv, 0.0) if kind == "call" else max(-forward_pv, 0.0))
    sqrt_t = math.sqrt(t)
    d1 = (
        math.log(s / k)
        + (r - q + 0.5 * sigma * sigma) * t
    ) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    if kind == "call":
        return float(
            discounted_spot * _normal_cdf(d1)
            - discounted_strike * _normal_cdf(d2)
        )
    return float(
        discounted_strike * _normal_cdf(-d2)
        - discounted_spot * _normal_cdf(-d1)
    )


def black_scholes_greeks(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
    option_type: str,
) -> dict[str, float]:
    s = float(spot)
    k = float(strike)
    r = float(rate)
    q = float(dividend_yield)
    sigma = float(volatility)
    t = float(maturity)
    if s <= 0.0 or k <= 0.0 or sigma <= 0.0 or t <= 0.0:
        raise ValueError(
            "Greek formulas require positive spot, strike, volatility and maturity."
        )
    kind = str(option_type).strip().lower()
    if kind not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    sqrt_t = math.sqrt(t)
    d1 = (
        math.log(s / k)
        + (r - q + 0.5 * sigma * sigma) * t
    ) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    disc_q = math.exp(-q * t)
    disc_r = math.exp(-r * t)
    pdf = _normal_pdf(d1)
    gamma = disc_q * pdf / (s * sigma * sqrt_t)
    vega = s * disc_q * pdf * sqrt_t
    if kind == "call":
        delta = disc_q * _normal_cdf(d1)
        theta = (
            -s * disc_q * pdf * sigma / (2.0 * sqrt_t)
            - r * k * disc_r * _normal_cdf(d2)
            + q * s * disc_q * _normal_cdf(d1)
        )
        rho = k * t * disc_r * _normal_cdf(d2)
    else:
        delta = disc_q * (_normal_cdf(d1) - 1.0)
        theta = (
            -s * disc_q * pdf * sigma / (2.0 * sqrt_t)
            + r * k * disc_r * _normal_cdf(-d2)
            - q * s * disc_q * _normal_cdf(-d1)
        )
        rho = -k * t * disc_r * _normal_cdf(-d2)
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta_annual": float(theta),
        "rho": float(rho),
    }


def historical_log_return_calibration(
    closes: Sequence[float],
    annualization: float = 252.0,
) -> dict[str, float | int]:
    prices = _finite_1d(closes, minimum=3)
    if np.any(prices <= 0.0):
        raise ValueError("Close prices must be positive.")
    ann = float(annualization)
    if ann <= 0.0:
        raise ValueError("annualization must be positive.")
    returns = np.log(prices[1:] / prices[:-1])
    sample_std = float(np.std(returns, ddof=1))
    return {
        "n_returns": int(returns.size),
        "return_mean": float(np.mean(returns)),
        "return_std": sample_std,
        "annualized_vol": float(sample_std * math.sqrt(ann)),
        "spot": float(prices[-1]),
    }



def discount_cashflow(
    amount: float,
    rate: float,
    maturity: float,
) -> float:
    return float(
        float(amount)
        * math.exp(
            -float(rate)
            * float(maturity)
        )
    )


def sma_seeded_ema(
    values: Sequence[float],
    span: int,
) -> np.ndarray:
    x = np.asarray(
        values,
        dtype=float,
    ).ravel()

    if isinstance(span, bool):
        raise ValueError(
            "EMA span must be a positive integer."
        )

    n = int(span)

    if (
        n <= 0
        or x.size < n
        or not np.all(np.isfinite(x))
    ):
        raise ValueError(
            "EMA requires finite values and a positive "
            "span no larger than the series."
        )

    out = np.full(
        x.shape,
        np.nan,
        dtype=float,
    )

    alpha = (
        2.0
        / (float(n) + 1.0)
    )

    out[n - 1] = float(
        np.mean(x[:n])
    )

    for index in range(
        n,
        x.size,
    ):
        out[index] = (
            alpha * x[index]
            + (1.0 - alpha)
            * out[index - 1]
        )

    return out


def ewma_annualized_volatility(
    log_returns: Sequence[float],
    lookback: int,
    annualization: float = 252.0,
) -> np.ndarray:
    returns = np.asarray(
        log_returns,
        dtype=float,
    ).ravel()

    if isinstance(lookback, bool):
        raise ValueError(
            "EWMA lookback must be a positive integer."
        )

    window = int(lookback)
    annual = float(annualization)

    if (
        window <= 0
        or annual <= 0.0
        or returns.size == 0
        or not np.all(np.isfinite(returns))
    ):
        raise ValueError(
            "EWMA requires finite non-empty returns, "
            "positive lookback and positive annualization."
        )

    alpha = (
        2.0
        / (float(window) + 1.0)
    )

    variance = np.empty_like(
        returns,
        dtype=float,
    )

    variance[0] = (
        returns[0]
        * returns[0]
    )

    for index in range(
        1,
        returns.size,
    ):
        variance[index] = (
            alpha
            * returns[index]
            * returns[index]
            + (1.0 - alpha)
            * variance[index - 1]
        )

    return np.sqrt(
        np.maximum(
            variance,
            0.0,
        )
        * annual
    )


def log_return_performance(
    log_returns: Sequence[float],
    risk_free_annual: float,
    annualization: float = 252.0,
) -> dict[str, float]:
    returns = np.asarray(
        log_returns,
        dtype=float,
    ).ravel()

    returns = returns[
        np.isfinite(returns)
    ]

    annual = float(
        annualization
    )

    if (
        returns.size == 0
        or annual <= 0.0
    ):
        raise ValueError(
            "Performance metrics require finite returns "
            "and positive annualization."
        )

    mean_period = float(
        np.mean(returns)
    )

    std_period = float(
        np.std(
            returns,
            ddof=0,
        )
    )

    annual_log_return = (
        mean_period
        * annual
    )

    annualized_return = float(
        math.exp(
            annual_log_return
        )
        - 1.0
    )

    annualized_volatility = float(
        std_period
        * math.sqrt(annual)
    )

    sharpe_ratio = (
        (
            annual_log_return
            - float(risk_free_annual)
        )
        / annualized_volatility
        if annualized_volatility > 0.0
        else 0.0
    )

    wealth = np.exp(
        np.cumsum(returns)
    )

    drawdown = (
        wealth
        / np.maximum.accumulate(
            wealth
        )
        - 1.0
    )

    max_drawdown = float(
        np.min(drawdown)
    )

    calmar_ratio = (
        annualized_return
        / abs(max_drawdown)
        if max_drawdown < 0.0
        else 0.0
    )

    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": max_drawdown,
        "calmar_ratio": float(calmar_ratio),
    }


def down_and_out_call_price(
    spot: float,
    strike: float,
    barrier: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    s = float(spot)
    k = float(strike)
    h = float(barrier)
    t = float(maturity)
    r = float(rate)
    sigma = float(volatility)
    q = float(dividend_yield)

    if (
        s <= h
        or sigma <= 0.0
        or t <= 0.0
    ):
        return 0.0

    if (
        s <= 0.0
        or k <= 0.0
        or h <= 0.0
    ):
        raise ValueError(
            "Barrier option requires positive "
            "spot, strike and barrier."
        )

    root_t = math.sqrt(t)

    d1 = (
        math.log(s / k)
        + (
            r
            - q
            + 0.5 * sigma * sigma
        )
        * t
    ) / (
        sigma
        * root_t
    )

    d2 = (
        d1
        - sigma
        * root_t
    )

    vanilla = (
        s
        * math.exp(-q * t)
        * _normal_cdf(d1)
        - k
        * math.exp(-r * t)
        * _normal_cdf(d2)
    )

    lam = (
        r
        - q
        + 0.5 * sigma * sigma
    ) / (
        sigma
        * sigma
    )

    ratio = h / s

    d1_h = (
        math.log(
            h * h
            / (s * k)
        )
        + (
            r
            - q
            + 0.5 * sigma * sigma
        )
        * t
    ) / (
        sigma
        * root_t
    )

    d2_h = (
        d1_h
        - sigma
        * root_t
    )

    down_in = (
        s
        * math.exp(-q * t)
        * ratio ** (2.0 * lam)
        * _normal_cdf(d1_h)
        - k
        * math.exp(-r * t)
        * ratio ** (
            2.0 * lam
            - 2.0
        )
        * _normal_cdf(d2_h)
    )

    return max(
        float(
            vanilla
            - down_in
        ),
        0.0,
    )


def central_price_delta(
    price_fn,
    spot: float,
    relative_step: float = 0.01,
) -> float:
    s = float(spot)
    rel = float(relative_step)

    if rel <= 0.0:
        raise ValueError(
            "relative_step must be positive."
        )

    step = max(
        abs(s) * rel,
        1e-8,
    )

    return float(
        (
            float(
                price_fn(
                    s + step
                )
            )
            - float(
                price_fn(
                    s - step
                )
            )
        )
        / (2.0 * step)
    )


def ols_with_intercept(
    x: Sequence[float],
    y: Sequence[float],
) -> dict[str, float | int | np.ndarray]:
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    if x_arr.size != y_arr.size:
        raise ValueError("OLS inputs must have equal length.")
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 2:
        raise ValueError("OLS requires at least two aligned finite observations.")
    design = np.column_stack([np.ones(x_arr.size, dtype=float), x_arr])
    coefficients, _, _, _ = np.linalg.lstsq(design, y_arr, rcond=None)
    fitted = design @ coefficients
    residuals = y_arr - fitted
    ss_res = float(residuals @ residuals)
    centered = y_arr - float(np.mean(y_arr))
    ss_tot = float(centered @ centered)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return {
        "intercept": float(coefficients[0]),
        "slope": float(coefficients[1]),
        "residuals": residuals,
        "r_squared": float(r_squared),
        "n_observations": int(x_arr.size),
    }


def fit_ou_euler(
    values: Sequence[float],
    dt: float,
) -> dict[str, float | int | np.ndarray]:
    series = _finite_1d(values, minimum=3)
    delta_t = float(dt)
    if delta_t <= 0.0:
        raise ValueError("OU dt must be positive.")
    fit = ols_with_intercept(series[:-1], series[1:] - series[:-1])
    slope = float(fit["slope"])
    intercept = float(fit["intercept"])
    if abs(slope) < 1e-15:
        raise ValueError("OU Euler slope is numerically zero.")
    kappa = -slope / delta_t
    theta = -intercept / slope
    residuals = np.asarray(fit["residuals"], dtype=float)
    residual_std = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
    sigma = residual_std / math.sqrt(delta_t)
    return {
        "intercept": intercept,
        "slope": slope,
        "kappa": float(kappa),
        "theta": float(theta),
        "sigma": float(sigma),
        "residual_std": residual_std,
        "residuals": residuals,
        "r_squared": float(fit["r_squared"]),
        "n_observations": int(residuals.size),
    }


def fit_ou_exact_ar1(
    values: Sequence[float],
    dt: float,
) -> dict[str, float | int | np.ndarray]:
    series = _finite_1d(values, minimum=3)
    delta_t = float(dt)
    if delta_t <= 0.0:
        raise ValueError("OU dt must be positive.")
    fit = ols_with_intercept(series[:-1], series[1:])
    intercept = float(fit["intercept"])
    phi = float(fit["slope"])
    if not 0.0 < phi < 1.0:
        raise ValueError("Exact OU calibration requires 0 < phi < 1.")
    kappa = -math.log(phi) / delta_t
    theta = intercept / (1.0 - phi)
    residuals = np.asarray(fit["residuals"], dtype=float)
    residual_std = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
    sigma = residual_std * math.sqrt(
        2.0 * kappa / max(1.0 - phi * phi, 1e-18)
    )
    return {
        "intercept": intercept,
        "phi": phi,
        "kappa": float(kappa),
        "theta": float(theta),
        "sigma": float(sigma),
        "residual_std": residual_std,
        "residuals": residuals,
        "r_squared": float(fit["r_squared"]),
        "n_observations": int(residuals.size),
    }


def ou_exact_step_parameters(
    kappa: float,
    sigma: float,
    dt: float,
) -> dict[str, float]:
    speed = float(kappa)
    vol = float(sigma)
    delta_t = float(dt)
    if speed <= 0.0 or vol < 0.0 or delta_t <= 0.0:
        raise ValueError("Exact OU step requires kappa>0, sigma>=0 and dt>0.")
    decay = math.exp(-speed * delta_t)
    step_std = vol * math.sqrt((1.0 - decay * decay) / (2.0 * speed))
    return {"decay": float(decay), "step_std": float(step_std)}


def simulate_ou_exact(
    initial: float,
    kappa: float,
    mu: float,
    sigma: float,
    dt: float,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    if isinstance(n_steps, bool) or isinstance(n_paths, bool):
        raise ValueError("n_steps and n_paths must be integers.")
    n_steps_i = int(n_steps)
    n_paths_i = int(n_paths)
    if n_steps_i <= 0 or n_paths_i <= 0:
        raise ValueError("n_steps and n_paths must be positive integers.")
    step = ou_exact_step_parameters(kappa, sigma, dt)
    rng = np.random.default_rng(int(seed))
    current = np.full(n_paths_i, float(initial), dtype=float)
    out = np.empty((n_paths_i, n_steps_i), dtype=float)
    decay = float(step["decay"])
    step_std = float(step["step_std"])
    long_run = float(mu)
    for index in range(n_steps_i):
        current = (
            long_run
            + (current - long_run) * decay
            + step_std * rng.standard_normal(n_paths_i)
        )
        out[:, index] = current
    return out


def _validate_brownian_hit_inputs(
    mu: float,
    sigma: float,
    horizon: float,
) -> tuple[float, float, float]:
    drift = float(mu)
    vol = float(sigma)
    time = float(horizon)
    if vol <= 0.0:
        raise ValueError("Brownian hit probability requires sigma > 0.")
    if time <= 0.0:
        raise ValueError("Brownian hit probability requires horizon > 0.")
    return drift, vol, time


def brownian_running_max_hit_probability(
    mu: float,
    sigma: float,
    horizon: float,
    barrier: float,
) -> float:
    drift, vol, time = _validate_brownian_hit_inputs(mu, sigma, horizon)
    level = float(barrier)
    if level <= 0.0:
        return 1.0
    root_time = math.sqrt(time)
    probability = (
        _normal_cdf((drift * time - level) / (vol * root_time))
        + math.exp(2.0 * drift * level / (vol * vol))
        * _normal_cdf((-drift * time - level) / (vol * root_time))
    )
    return float(min(max(probability, 0.0), 1.0))


def brownian_running_min_hit_probability(
    mu: float,
    sigma: float,
    horizon: float,
    barrier: float,
) -> float:
    drift, vol, time = _validate_brownian_hit_inputs(mu, sigma, horizon)
    level = float(barrier)
    if level >= 0.0:
        return 1.0
    root_time = math.sqrt(time)
    probability = (
        _normal_cdf((level - drift * time) / (vol * root_time))
        + math.exp(2.0 * drift * level / (vol * vol))
        * _normal_cdf((level + drift * time) / (vol * root_time))
    )
    return float(min(max(probability, 0.0), 1.0))


def brownian_joint_terminal_max_cdf(
    mu: float,
    sigma: float,
    horizon: float,
    terminal_level: float,
    max_barrier: float,
) -> float:
    drift, vol, time = _validate_brownian_hit_inputs(mu, sigma, horizon)
    barrier = float(max_barrier)
    if barrier <= 0.0:
        return 0.0
    x = min(float(terminal_level), barrier)
    root_time = math.sqrt(time)
    probability = (
        _normal_cdf((x - drift * time) / (vol * root_time))
        - math.exp(2.0 * drift * barrier / (vol * vol))
        * _normal_cdf((x - 2.0 * barrier - drift * time) / (vol * root_time))
    )
    return float(min(max(probability, 0.0), 1.0))


def empirical_var_es_from_losses(
    losses: Sequence[float],
    confidence: float,
    strict_tail: bool = False,
) -> tuple[float, float]:
    values = _finite_1d(losses, minimum=1)
    level = float(confidence)
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must lie in (0, 1).")
    ordered = np.sort(values)
    index = int(math.ceil(level * ordered.size) - 1)
    index = min(max(index, 0), ordered.size - 1)
    var = float(ordered[index])
    tail = values[values > var] if strict_tail else values[values >= var]
    es = float(np.mean(tail)) if tail.size else var
    return var, es


def normalize_nonnegative(values: Sequence[float]) -> np.ndarray:
    weights = np.asarray(values, dtype=float).ravel()
    if (
        weights.size == 0
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
    ):
        raise ValueError("Weights must be a non-empty finite non-negative vector.")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("Weights must have positive total mass.")
    return weights / total


def ewma_weights(
    length: int,
    half_life: float,
    newest_first: bool = True,
) -> np.ndarray:
    if isinstance(length, bool):
        raise ValueError("length must be a positive integer.")
    n = int(length)
    half = float(half_life)
    if n <= 0:
        raise ValueError("length must be a positive integer.")
    if half <= 0.0:
        raise ValueError("half_life must be positive.")
    lags = np.arange(n, dtype=float)
    weights = np.exp(-math.log(2.0) * lags / half)
    if not newest_first:
        weights = weights[::-1]
    return normalize_nonnegative(weights)


def largest_remainder_allocate(
    total: int,
    weights: Sequence[float],
) -> np.ndarray:
    if isinstance(total, bool):
        raise ValueError("total must be a non-negative integer.")
    amount = int(total)
    if amount < 0:
        raise ValueError("total must be a non-negative integer.")
    normalized = normalize_nonnegative(weights)
    exact = normalized * amount
    base = np.floor(exact).astype(np.int64)
    remainder = amount - int(np.sum(base))
    if remainder > 0:
        fractions = exact - base
        order = np.argsort(-fractions, kind="stable")
        base[order[:remainder]] += 1
    return base


def fit_garch11_zero_mean(
    returns: Sequence[float],
    min_observations: int = 50,
) -> dict[str, object]:
    """
    Fit a zero-mean Gaussian GARCH(1,1) model.

    Input returns are expressed in decimal units, for example 0.01 for 1%.
    Fitting is performed on percent-scaled returns for numerical stability.
    Returned omega, conditional variances, and long-run variance are converted
    back to decimal-return squared units.

    Non-finite observations are removed before fitting.
    """
    if (
        isinstance(min_observations, bool)
        or not isinstance(min_observations, int)
        or min_observations <= 0
    ):
        raise ValueError(
            "min_observations must be a positive integer."
        )

    values = np.asarray(
        returns,
        dtype=float,
    ).ravel()

    values = values[
        np.isfinite(values)
    ]

    if values.size < min_observations:
        raise ValueError(
            "GARCH fitting requires at least "
            f"{min_observations} finite observations."
        )

    if np.all(values == values[0]):
        raise ValueError(
            "GARCH fitting requires non-constant returns."
        )

    try:
        from arch import arch_model
    except ImportError as exc:
        raise RuntimeError(
            "The arch package is required for GARCH fitting."
        ) from exc

    percent_returns = (
        values
        * 100.0
    )

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )

        model = arch_model(
            percent_returns,
            mean="Zero",
            vol="GARCH",
            p=1,
            q=1,
            dist="Normal",
            rescale=False,
        )

        fitted = model.fit(
            update_freq=0,
            disp="off",
            show_warning=False,
        )

    omega = (
        float(
            fitted.params["omega"]
        )
        / 10000.0
    )

    alpha = float(
        fitted.params["alpha[1]"]
    )

    beta = float(
        fitted.params["beta[1]"]
    )

    persistence = (
        alpha
        + beta
    )

    if persistence < 1.0:
        long_run_variance = (
            omega
            / (
                1.0
                - persistence
            )
        )
    else:
        long_run_variance = float(
            np.var(
                values,
                ddof=1,
            )
        )

    conditional_variance = (
        np.asarray(
            fitted.conditional_volatility,
            dtype=float,
        )
        ** 2
        / 10000.0
    )

    convergence_flag = int(
        getattr(
            fitted,
            "convergence_flag",
            0,
        )
    )

    return {
        "omega": float(
            omega
        ),
        "alpha": float(
            alpha
        ),
        "beta": float(
            beta
        ),
        "persistence": float(
            persistence
        ),
        "long_run_variance": float(
            long_run_variance
        ),
        "conditional_variance": (
            conditional_variance
        ),
        "convergence_flag": (
            convergence_flag
        ),
        "n_observations": int(
            values.size
        ),
    }


def garch11_forecast_variance(
    *,
    omega: float,
    alpha: float,
    beta: float,
    last_return: float,
    last_variance: float,
    horizon: int,
) -> dict[str, object]:
    """
    Forecast GARCH(1,1) conditional variance for one or more future periods.

    All variance quantities use the same squared-return units.
    """
    omega_value = float(
        omega
    )
    alpha_value = float(
        alpha
    )
    beta_value = float(
        beta
    )
    last_return_value = float(
        last_return
    )
    last_variance_value = float(
        last_variance
    )

    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or horizon <= 0
    ):
        raise ValueError(
            "horizon must be a positive integer."
        )

    numeric_values = np.asarray(
        [
            omega_value,
            alpha_value,
            beta_value,
            last_return_value,
            last_variance_value,
        ],
        dtype=float,
    )

    if not np.all(
        np.isfinite(
            numeric_values
        )
    ):
        raise ValueError(
            "GARCH forecast inputs must be finite."
        )

    if omega_value < 0.0:
        raise ValueError(
            "omega must be non-negative."
        )

    if alpha_value < 0.0:
        raise ValueError(
            "alpha must be non-negative."
        )

    if beta_value < 0.0:
        raise ValueError(
            "beta must be non-negative."
        )

    if last_variance_value < 0.0:
        raise ValueError(
            "last_variance must be non-negative."
        )

    variance_path = np.empty(
        horizon,
        dtype=float,
    )

    variance_path[0] = (
        omega_value
        + alpha_value
        * last_return_value ** 2
        + beta_value
        * last_variance_value
    )

    persistence = (
        alpha_value
        + beta_value
    )

    for step in range(
        1,
        horizon,
    ):
        variance_path[step] = (
            omega_value
            + persistence
            * variance_path[
                step - 1
            ]
        )

    return {
        "variance_path": (
            variance_path
        ),
        "aggregate_variance": float(
            np.sum(
                variance_path
            )
        ),
        "persistence": float(
            persistence
        ),
    }


def implied_volatility_black_scholes(
    *,
    market_price: float,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    maturity: float,
    option_type: str,
    upper_volatility: float = 5.0,
    price_tolerance: float = 1e-12,
    volatility_tolerance: float = 1e-12,
    max_iterations: int = 300,
) -> float:
    """
    Recover Black-Scholes implied volatility by bounded root finding.

    The market price must satisfy the no-arbitrage bounds for a European
    call or put with continuous dividend yield. A price equal to the
    zero-volatility lower bound returns 0.0.

    The solver uses monotone bisection over volatility in
    [0, upper_volatility].
    """
    market = float(
        market_price
    )
    s = float(
        spot
    )
    k = float(
        strike
    )
    r = float(
        rate
    )
    q = float(
        dividend_yield
    )
    t = float(
        maturity
    )
    upper_sigma = float(
        upper_volatility
    )
    price_tol = float(
        price_tolerance
    )
    vol_tol = float(
        volatility_tolerance
    )

    values = (
        market,
        s,
        k,
        r,
        q,
        t,
        upper_sigma,
        price_tol,
        vol_tol,
    )

    if not all(
        math.isfinite(value)
        for value in values
    ):
        raise ValueError(
            "Implied-volatility inputs must be finite."
        )

    if s <= 0.0 or k <= 0.0:
        raise ValueError(
            "spot and strike must be positive."
        )

    if t <= 0.0:
        raise ValueError(
            "maturity must be positive."
        )

    if upper_sigma <= 0.0:
        raise ValueError(
            "upper_volatility must be positive."
        )

    if price_tol <= 0.0:
        raise ValueError(
            "price_tolerance must be positive."
        )

    if vol_tol <= 0.0:
        raise ValueError(
            "volatility_tolerance must be positive."
        )

    if (
        isinstance(max_iterations, bool)
        or not isinstance(
            max_iterations,
            int,
        )
        or max_iterations <= 0
    ):
        raise ValueError(
            "max_iterations must be a positive integer."
        )

    kind = str(
        option_type
    ).strip().lower()

    if kind not in {
        "call",
        "put",
    }:
        raise ValueError(
            "option_type must be 'call' or 'put'."
        )

    discounted_spot = (
        s
        * math.exp(
            -q * t
        )
    )

    discounted_strike = (
        k
        * math.exp(
            -r * t
        )
    )

    if kind == "call":
        lower_bound = max(
            discounted_spot
            - discounted_strike,
            0.0,
        )

        upper_bound = (
            discounted_spot
        )

    else:
        lower_bound = max(
            discounted_strike
            - discounted_spot,
            0.0,
        )

        upper_bound = (
            discounted_strike
        )

    arbitrage_tolerance = max(
        price_tol,
        1e-12,
    )

    if (
        market
        < lower_bound
        - arbitrage_tolerance
        or market
        > upper_bound
        + arbitrage_tolerance
    ):
        raise ValueError(
            "market_price violates Black-Scholes "
            "no-arbitrage bounds."
        )

    if abs(
        market
        - lower_bound
    ) <= price_tol:
        return 0.0

    def residual(
        volatility: float,
    ) -> float:
        return (
            black_scholes_price(
                spot=s,
                strike=k,
                rate=r,
                dividend_yield=q,
                volatility=volatility,
                maturity=t,
                option_type=kind,
            )
            - market
        )

    low = 0.0
    high = upper_sigma

    low_value = residual(
        low
    )

    high_value = residual(
        high
    )

    if abs(
        low_value
    ) <= price_tol:
        return 0.0

    if abs(
        high_value
    ) <= price_tol:
        return float(
            high
        )

    if (
        low_value > 0.0
        or high_value < 0.0
    ):
        raise ValueError(
            "Could not bracket Black-Scholes "
            "implied volatility within the "
            "requested volatility interval."
        )

    for _ in range(
        max_iterations
    ):
        midpoint = (
            0.5
            * (
                low
                + high
            )
        )

        midpoint_value = residual(
            midpoint
        )

        if (
            abs(
                midpoint_value
            )
            <= price_tol
            or (
                high
                - low
            )
            <= vol_tol
        ):
            return float(
                midpoint
            )

        if midpoint_value > 0.0:
            high = midpoint
        else:
            low = midpoint

    midpoint = (
        0.5
        * (
            low
            + high
        )
    )

    if abs(
        residual(
            midpoint
        )
    ) > max(
        price_tol,
        1e-10,
    ):
        raise RuntimeError(
            "Black-Scholes implied-volatility "
            "solver did not converge."
        )

    return float(
        midpoint
    )


def discount_factor_from_continuous_zero_rate(
    *,
    zero_rate: float,
    maturity: float,
) -> float:
    """
    Convert a continuously compounded zero rate into a discount factor.
    """
    rate = float(
        zero_rate
    )
    time = float(
        maturity
    )

    if not (
        math.isfinite(rate)
        and math.isfinite(time)
    ):
        raise ValueError(
            "zero_rate and maturity must be finite."
        )

    if time < 0.0:
        raise ValueError(
            "maturity must be non-negative."
        )

    if time == 0.0:
        return 1.0

    return float(
        math.exp(
            -rate * time
        )
    )


def continuous_zero_rate_from_discount_factor(
    *,
    discount_factor: float,
    maturity: float,
) -> float:
    """
    Convert a positive discount factor into a continuously compounded
    zero rate.
    """
    discount = float(
        discount_factor
    )
    time = float(
        maturity
    )

    if not (
        math.isfinite(discount)
        and math.isfinite(time)
    ):
        raise ValueError(
            "discount_factor and maturity must be finite."
        )

    if discount <= 0.0:
        raise ValueError(
            "discount_factor must be positive."
        )

    if time <= 0.0:
        raise ValueError(
            "maturity must be positive."
        )

    return float(
        -math.log(
            discount
        )
        / time
    )


def log_linear_discount_factor(
    *,
    pillars: Sequence[
        tuple[
            float,
            float,
        ]
    ],
    maturity: float,
) -> float:
    """
    Interpolate discount factors linearly in log-discount space.

    Inside the pillar range, log discount factors are linearly interpolated.
    Outside the pillar range, the zero rate of the nearest endpoint is held
    constant.
    """
    target = float(
        maturity
    )

    if not math.isfinite(
        target
    ):
        raise ValueError(
            "maturity must be finite."
        )

    if target <= 0.0:
        return 1.0

    if not pillars:
        raise ValueError(
            "pillars must not be empty."
        )

    ordered = sorted(
        (
            float(time),
            float(discount),
        )
        for time, discount
        in pillars
    )

    previous_time = None

    for time, discount in ordered:
        if not (
            math.isfinite(time)
            and math.isfinite(discount)
        ):
            raise ValueError(
                "curve pillars must be finite."
            )

        if time <= 0.0:
            raise ValueError(
                "pillar maturities must be positive."
            )

        if discount <= 0.0:
            raise ValueError(
                "pillar discount factors must be positive."
            )

        if (
            previous_time is not None
            and time <= previous_time
        ):
            raise ValueError(
                "pillar maturities must be unique."
            )

        previous_time = time

    first_time, first_discount = (
        ordered[0]
    )

    if target <= first_time:
        zero_rate = (
            -math.log(
                first_discount
            )
            / first_time
        )

        return float(
            math.exp(
                -zero_rate
                * target
            )
        )

    last_time, last_discount = (
        ordered[-1]
    )

    if target >= last_time:
        zero_rate = (
            -math.log(
                last_discount
            )
            / last_time
        )

        return float(
            math.exp(
                -zero_rate
                * target
            )
        )

    for index in range(
        1,
        len(ordered),
    ):
        t0, d0 = (
            ordered[
                index - 1
            ]
        )

        t1, d1 = (
            ordered[index]
        )

        if (
            t0
            <= target
            <= t1
        ):
            weight = (
                target
                - t0
            ) / (
                t1
                - t0
            )

            return float(
                math.exp(
                    (
                        1.0
                        - weight
                    )
                    * math.log(
                        d0
                    )
                    + weight
                    * math.log(
                        d1
                    )
                )
            )

    raise RuntimeError(
        "Curve interpolation failed."
    )


def bootstrap_annual_par_discount_factors(
    par_yields: Sequence[
        float
    ],
) -> dict[str, object]:
    """
    Bootstrap discount factors from consecutive annual par yields.

    The input sequence represents annual-coupon par yields for maturities
    1, 2, ..., N years. Coupon frequency is annual and par value is 1.

    Returns discount factors, continuously compounded zero rates, and
    one-year continuously compounded forward rates.
    """
    yields = np.asarray(
        par_yields,
        dtype=float,
    ).ravel()

    if yields.size == 0:
        raise ValueError(
            "par_yields must not be empty."
        )

    if not np.all(
        np.isfinite(
            yields
        )
    ):
        raise ValueError(
            "par_yields must contain only finite values."
        )

    discounts = np.empty(
        yields.size,
        dtype=float,
    )

    previous_sum = 0.0

    for index, par_yield in enumerate(
        yields
    ):
        discount = (
            1.0
            - float(
                par_yield
            )
            * previous_sum
        ) / (
            1.0
            + float(
                par_yield
            )
        )

        if not (
            math.isfinite(
                discount
            )
            and 0.0
            < discount
            < 1.0
        ):
            raise ValueError(
                "par yields imply an invalid discount factor."
            )

        discounts[
            index
        ] = discount

        previous_sum += discount

    maturities = np.arange(
        1,
        yields.size + 1,
        dtype=float,
    )

    zero_rates = (
        -np.log(
            discounts
        )
        / maturities
    )

    forward_rates = np.empty(
        yields.size,
        dtype=float,
    )

    forward_rates[0] = (
        -math.log(
            float(
                discounts[0]
            )
        )
    )

    if yields.size > 1:
        forward_rates[1:] = (
            -np.log(
                discounts[1:]
                / discounts[:-1]
            )
        )

    return {
        "maturities": (
            maturities
        ),
        "discount_factors": (
            discounts
        ),
        "zero_rates": (
            zero_rates
        ),
        "forward_rates": (
            forward_rates
        ),
    }


def pseudo_observations(
    values: Sequence[float],
) -> np.ndarray:
    """
    Convert a one-dimensional sample to rank-based pseudo-observations.

    Average ranks are used for ties and divided by n + 1 so every
    pseudo-observation lies strictly inside the unit interval.
    """
    from scipy import stats

    sample = np.asarray(
        values,
        dtype=float,
    ).ravel()

    if sample.size == 0:
        raise ValueError(
            "values must not be empty."
        )

    if not np.all(
        np.isfinite(
            sample
        )
    ):
        raise ValueError(
            "values must contain only finite observations."
        )

    return np.asarray(
        stats.rankdata(
            sample
        )
        / (
            sample.size
            + 1.0
        ),
        dtype=float,
    )


def sample_gaussian_copula(
    *,
    n: int,
    rho: float,
    rng,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Draw bivariate Gaussian-copula pseudo-uniform samples.

    The supplied random generator controls reproducibility.
    """
    from scipy import stats

    if (
        isinstance(n, bool)
        or not isinstance(n, int)
        or n <= 0
    ):
        raise ValueError(
            "n must be a positive integer."
        )

    correlation = float(
        rho
    )

    if not math.isfinite(
        correlation
    ):
        raise ValueError(
            "rho must be finite."
        )

    if not (
        -1.0
        <= correlation
        <= 1.0
    ):
        raise ValueError(
            "rho must lie between -1 and 1."
        )

    z = rng.multivariate_normal(
        [0.0, 0.0],
        [
            [1.0, correlation],
            [correlation, 1.0],
        ],
        n,
    )

    uniforms = stats.norm.cdf(
        z
    )

    return (
        np.asarray(
            uniforms[:, 0],
            dtype=float,
        ),
        np.asarray(
            uniforms[:, 1],
            dtype=float,
        ),
    )


def sample_student_t_copula(
    *,
    n: int,
    rho: float,
    degrees_of_freedom: int,
    rng,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Draw bivariate Student-t copula pseudo-uniform samples.

    Correlated standard-normal draws are scaled by one common chi-square
    variate per observation before applying the Student-t CDF.
    """
    from scipy import stats

    if (
        isinstance(n, bool)
        or not isinstance(n, int)
        or n <= 0
    ):
        raise ValueError(
            "n must be a positive integer."
        )

    correlation = float(
        rho
    )

    if not math.isfinite(
        correlation
    ):
        raise ValueError(
            "rho must be finite."
        )

    if not (
        -1.0
        <= correlation
        <= 1.0
    ):
        raise ValueError(
            "rho must lie between -1 and 1."
        )

    if (
        isinstance(
            degrees_of_freedom,
            bool,
        )
        or not isinstance(
            degrees_of_freedom,
            int,
        )
        or degrees_of_freedom <= 0
    ):
        raise ValueError(
            "degrees_of_freedom must be a positive integer."
        )

    z = rng.multivariate_normal(
        [0.0, 0.0],
        [
            [1.0, correlation],
            [correlation, 1.0],
        ],
        n,
    )

    chi2 = rng.chisquare(
        degrees_of_freedom,
        n,
    )

    t_values = (
        z
        * np.sqrt(
            degrees_of_freedom
            / chi2
        )[:, None]
    )

    uniforms = stats.t.cdf(
        t_values,
        degrees_of_freedom,
    )

    return (
        np.asarray(
            uniforms[:, 0],
            dtype=float,
        ),
        np.asarray(
            uniforms[:, 1],
            dtype=float,
        ),
    )


def fit_gpd_exceedances(
    excesses: Sequence[float],
) -> dict[str, object]:
    """
    Fit a Generalized Pareto distribution to non-negative threshold
    exceedances with location fixed at zero.
    """
    from scipy import stats

    values = np.asarray(
        excesses,
        dtype=float,
    ).ravel()

    if values.size < 3:
        raise ValueError(
            "GPD fitting requires at least 3 exceedances."
        )

    if not np.all(
        np.isfinite(
            values
        )
    ):
        raise ValueError(
            "excesses must contain only finite values."
        )

    if np.any(
        values < 0.0
    ):
        raise ValueError(
            "excesses must be non-negative."
        )

    shape, _location, scale = (
        stats.genpareto.fit(
            values,
            floc=0.0,
        )
    )

    shape_value = float(
        shape
    )

    scale_value = float(
        scale
    )

    if not (
        math.isfinite(
            shape_value
        )
        and math.isfinite(
            scale_value
        )
        and scale_value > 0.0
    ):
        raise RuntimeError(
            "GPD fit produced invalid parameters."
        )

    return {
        "shape": (
            shape_value
        ),
        "scale": (
            scale_value
        ),
        "n_exceedances": int(
            values.size
        ),
        "mean_excess": float(
            np.mean(
                values
            )
        ),
    }


def evt_var_es_from_gpd(
    *,
    probability: float,
    threshold: float,
    shape: float,
    scale: float,
    n_total: int,
    n_exceedances: int,
) -> dict[str, float]:
    """
    Compute POT/EVT VaR and expected shortfall from a fitted GPD tail.

    The exceedance frequency is n_exceedances / n_total.
    """
    p = float(
        probability
    )

    u = float(
        threshold
    )

    xi = float(
        shape
    )

    sigma = float(
        scale
    )

    numeric_values = (
        p,
        u,
        xi,
        sigma,
    )

    if not all(
        math.isfinite(value)
        for value in numeric_values
    ):
        raise ValueError(
            "EVT inputs must be finite."
        )

    if not (
        0.0
        < p
        < 1.0
    ):
        raise ValueError(
            "probability must lie strictly between 0 and 1."
        )

    if sigma <= 0.0:
        raise ValueError(
            "scale must be positive."
        )

    if (
        isinstance(n_total, bool)
        or not isinstance(n_total, int)
        or n_total <= 0
    ):
        raise ValueError(
            "n_total must be a positive integer."
        )

    if (
        isinstance(n_exceedances, bool)
        or not isinstance(n_exceedances, int)
        or n_exceedances <= 0
        or n_exceedances > n_total
    ):
        raise ValueError(
            "n_exceedances must be a positive integer "
            "not exceeding n_total."
        )

    tail_probability = (
        float(
            n_total
        )
        / float(
            n_exceedances
        )
        * (
            1.0
            - p
        )
    )

    if tail_probability <= 0.0:
        raise ValueError(
            "Invalid POT tail probability."
        )

    if abs(
        xi
    ) < 1e-10:
        var = (
            u
            - sigma
            * math.log(
                tail_probability
            )
        )
    else:
        var = (
            u
            + sigma
            / xi
            * (
                tail_probability
                ** (
                    -xi
                )
                - 1.0
            )
        )

    if xi >= 1.0:
        raise ValueError(
            "GPD expected shortfall is infinite for shape >= 1."
        )

    expected_shortfall = (
        var
        / (
            1.0
            - xi
        )
        + (
            sigma
            - xi
            * u
        )
        / (
            1.0
            - xi
        )
    )

    return {
        "var": float(
            var
        ),
        "expected_shortfall": float(
            expected_shortfall
        ),
    }


def hill_tail_index(
    losses: Sequence[float],
) -> float:
    """
    Estimate an upper-tail index using the Hill estimator.

    The number of upper order statistics is floor(sqrt(n)), matching the
    existing Phase 8R convention.
    """
    values = np.asarray(
        losses,
        dtype=float,
    ).ravel()

    if values.size == 0:
        raise ValueError(
            "losses must not be empty."
        )

    if not np.all(
        np.isfinite(
            values
        )
    ):
        raise ValueError(
            "losses must contain only finite values."
        )

    ordered = np.sort(
        values
    )[::-1]

    n = int(
        ordered.size
    )

    k = int(
        math.floor(
            math.sqrt(
                n
            )
        )
    )

    if (
        k < 1
        or k >= n
        or ordered[k] <= 0.0
    ):
        raise ValueError(
            "Hill estimator requires positive upper-tail observations."
        )

    top = ordered[
        :k
    ]

    if np.any(
        top <= 0.0
    ):
        raise ValueError(
            "Hill estimator upper-tail observations must be positive."
        )

    return float(
        np.mean(
            np.log(
                top
            )
        )
        - math.log(
            float(
                ordered[k]
            )
        )
    )
