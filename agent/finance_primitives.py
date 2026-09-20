from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

TRADING_DAYS = 252.0


@dataclass(frozen=True)
class Garch11Fit:
    omega: float
    alpha: float
    beta: float
    persistence: float
    long_run_variance: float
    conditional_variance: np.ndarray


def fit_garch11_zero_mean(returns_decimal: Sequence[float]) -> Garch11Fit:
    """Fit normal zero-mean GARCH(1,1) on percentage returns, return decimal variance."""
    from arch import arch_model  # lazy: benchmark image supplies arch

    values = np.asarray(returns_decimal, dtype=float).ravel()
    if values.size < 20 or not np.all(np.isfinite(values)):
        raise RuntimeError("GARCH requires at least 20 finite returns.")

    fitted = arch_model(
        values * 100.0,
        mean="Zero",
        vol="Garch",
        p=1,
        q=1,
        dist="normal",
    ).fit(disp="off")

    omega = float(fitted.params["omega"] / 10000.0)
    alpha = float(fitted.params["alpha[1]"])
    beta = float(fitted.params["beta[1]"])
    persistence = alpha + beta
    sample_var = float(np.var(values, ddof=1))
    long_run = omega / (1.0 - persistence) if persistence < 1.0 else sample_var
    cond_var = np.asarray((fitted.conditional_volatility ** 2) / 10000.0, dtype=float)
    return Garch11Fit(
        omega=omega,
        alpha=alpha,
        beta=beta,
        persistence=float(persistence),
        long_run_variance=float(long_run),
        conditional_variance=cond_var,
    )


def sma_seeded_ema(values: Sequence[float], span: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if span <= 0 or x.size < span:
        raise RuntimeError("EMA span must be positive and no larger than the series.")
    out = np.full(x.shape, np.nan, dtype=float)
    alpha = 2.0 / (float(span) + 1.0)
    out[span - 1] = float(np.mean(x[:span]))
    for i in range(span, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def ewma_annualized_volatility(
    log_returns: Sequence[float],
    lookback: int,
    *,
    annualization: float = TRADING_DAYS,
) -> np.ndarray:
    r = np.asarray(log_returns, dtype=float)
    if lookback <= 0 or r.size == 0:
        raise RuntimeError("EWMA requires positive lookback and non-empty returns.")
    alpha = 2.0 / (float(lookback) + 1.0)
    var = np.empty_like(r, dtype=float)
    var[0] = r[0] * r[0]
    for i in range(1, len(r)):
        var[i] = alpha * r[i] * r[i] + (1.0 - alpha) * var[i - 1]
    return np.sqrt(np.maximum(var, 0.0) * float(annualization))


@dataclass(frozen=True)
class PerformanceMetrics:
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float


def log_return_performance(
    log_returns: Sequence[float],
    *,
    risk_free_annual: float,
    annualization: float = TRADING_DAYS,
) -> PerformanceMetrics:
    x = np.asarray(log_returns, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise RuntimeError("Performance metrics require at least one finite return.")
    mean_daily = float(np.mean(x))
    std_daily = float(np.std(x, ddof=0))
    ann_log_return = mean_daily * annualization
    ann_return = float(math.exp(ann_log_return) - 1.0)
    ann_vol = std_daily * math.sqrt(annualization)
    sharpe = (ann_log_return - risk_free_annual) / ann_vol if ann_vol > 0.0 else 0.0
    wealth = np.exp(np.cumsum(x))
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    max_drawdown = float(np.min(drawdown))
    calmar = ann_return / abs(max_drawdown) if max_drawdown < 0.0 else 0.0
    return PerformanceMetrics(
        annualized_return=ann_return,
        annualized_volatility=float(ann_vol),
        sharpe_ratio=float(sharpe),
        max_drawdown=max_drawdown,
        calmar_ratio=float(calmar),
    )


def discount_cashflow(amount: float, rate: float, maturity: float) -> float:
    return float(amount * math.exp(-rate * maturity))


def down_and_out_call_price(
    *,
    spot: float,
    strike: float,
    barrier: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    if spot <= barrier or volatility <= 0.0 or maturity <= 0.0:
        return 0.0
    root_t = math.sqrt(maturity)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility) * maturity
    ) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    vanilla = (
        spot * math.exp(-dividend_yield * maturity) * norm.cdf(d1)
        - strike * math.exp(-rate * maturity) * norm.cdf(d2)
    )
    lam = (rate - dividend_yield + 0.5 * volatility * volatility) / (
        volatility * volatility
    )
    ratio = barrier / spot
    d1_h = (
        math.log(barrier * barrier / (spot * strike))
        + (rate - dividend_yield + 0.5 * volatility * volatility) * maturity
    ) / (volatility * root_t)
    d2_h = d1_h - volatility * root_t
    down_in = (
        spot
        * math.exp(-dividend_yield * maturity)
        * ratio ** (2.0 * lam)
        * norm.cdf(d1_h)
        - strike
        * math.exp(-rate * maturity)
        * ratio ** (2.0 * lam - 2.0)
        * norm.cdf(d2_h)
    )
    return max(float(vanilla - down_in), 0.0)


def central_price_delta(
    *,
    price_fn,
    spot: float,
    relative_step: float = 0.01,
) -> float:
    step = max(abs(spot) * relative_step, 1e-8)
    return float((price_fn(spot + step) - price_fn(spot - step)) / (2.0 * step))



def normal_delta_var(
    *,
    exposure: float,
    annualized_volatility: float,
    confidence: float,
    annualization: float = TRADING_DAYS,
) -> float:
    if not 0.5 < confidence < 1.0:
        raise RuntimeError("VaR confidence must lie in (0.5, 1).")
    daily_sigma = float(annualized_volatility) / math.sqrt(float(annualization))
    z = float(norm.ppf(confidence))
    return float(abs(exposure) * daily_sigma * z)


def normal_tail_multiplier(confidence: float) -> float:
    if not 0.5 < confidence < 1.0:
        raise RuntimeError("Tail confidence must lie in (0.5, 1).")
    z = float(norm.ppf(confidence))
    return float(norm.pdf(z) / (1.0 - confidence))


def normal_expected_shortfall(
    *,
    exposure: float,
    annualized_volatility: float,
    confidence: float,
    annualization: float = TRADING_DAYS,
) -> float:
    daily_sigma = float(annualized_volatility) / math.sqrt(float(annualization))
    return float(abs(exposure) * daily_sigma * normal_tail_multiplier(confidence))

def empirical_loss_var_es(
    returns: Sequence[float],
    *,
    exposure: float,
    confidence: float,
) -> tuple[float, float]:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or not 0.5 < confidence < 1.0:
        raise RuntimeError("VaR/ES requires finite returns and confidence in (0.5, 1).")
    q = float(np.quantile(values, 1.0 - confidence))
    tail = values[values <= q]
    var = max(-q * exposure, 0.0)
    es = max(-float(np.mean(tail)) * exposure, var) if tail.size else var
    return float(var), float(es)


def solve_breakeven_volatility(
    *,
    target_value: float,
    fixed_value: float,
    units: float,
    option_price_at_vol,
    lower: float = 1e-6,
    upper: float = 5.0,
) -> float:
    def objective(vol: float) -> float:
        return fixed_value + units * float(option_price_at_vol(vol)) - target_value

    lo = objective(lower)
    hi = objective(upper)
    if lo == 0.0:
        return lower
    if hi == 0.0:
        return upper
    if lo * hi > 0.0:
        raise RuntimeError("Could not bracket breakeven volatility.")
    return float(brentq(objective, lower, upper, xtol=1e-12, rtol=1e-12, maxiter=200))
