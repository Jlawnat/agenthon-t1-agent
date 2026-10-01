from __future__ import annotations
import math
from scipy.stats import norm

def floating_lookback_call(*, spot: float, running_min: float, maturity: float, risk_free_rate: float, dividend_yield: float, volatility: float) -> float:
    b = risk_free_rate - dividend_yield
    if abs(b) < 1e-12:
        raise RuntimeError('Zero cost-of-carry floating lookback limit is not implemented.')
    sqrt_t = math.sqrt(maturity)
    a1 = (math.log(spot / running_min) + (b + 0.5 * volatility * volatility) * maturity) / (volatility * sqrt_t)
    a2 = a1 - volatility * sqrt_t
    a3 = a1 - 2.0 * b * sqrt_t / volatility
    disc_r = math.exp(-risk_free_rate * maturity)
    disc_q = math.exp(-dividend_yield * maturity)
    ratio_term = (running_min / spot) ** (2.0 * b / (volatility * volatility))
    correction = spot * disc_r * volatility * volatility / (2.0 * b) * (ratio_term * norm.cdf(-a1 + 2.0 * b * sqrt_t / volatility) - math.exp(b * maturity) * norm.cdf(-a1))
    return float(spot * disc_q * norm.cdf(a1) - running_min * disc_r * norm.cdf(a2) + correction)

def floating_lookback_put(*, spot: float, running_max: float, maturity: float, risk_free_rate: float, dividend_yield: float, volatility: float) -> float:
    b = risk_free_rate - dividend_yield
    if abs(b) < 1e-12:
        raise RuntimeError('Zero cost-of-carry floating lookback limit is not implemented.')
    sqrt_t = math.sqrt(maturity)
    b1 = (math.log(spot / running_max) + (b + 0.5 * volatility * volatility) * maturity) / (volatility * sqrt_t)
    b2 = b1 - volatility * sqrt_t
    disc_r = math.exp(-risk_free_rate * maturity)
    disc_q = math.exp(-dividend_yield * maturity)
    ratio_term = (spot / running_max) ** (-2.0 * b / (volatility * volatility))
    correction = spot * disc_r * volatility * volatility / (2.0 * b) * (-ratio_term * norm.cdf(b1 - 2.0 * b * sqrt_t / volatility) + math.exp(b * maturity) * norm.cdf(b1))
    return float(-spot * disc_q * norm.cdf(-b1) + running_max * disc_r * norm.cdf(-b2) + correction)

def fixed_strike_lookback_call(*, spot: float, running_max: float, strike: float, maturity: float, risk_free_rate: float, dividend_yield: float, volatility: float) -> float:
    b = risk_free_rate - dividend_yield
    if abs(b) < 1e-12:
        raise RuntimeError('Zero cost-of-carry fixed lookback limit is not implemented.')
    sqrt_t = math.sqrt(maturity)
    disc_r = math.exp(-risk_free_rate * maturity)
    disc_q = math.exp(-dividend_yield * maturity)
    if strike > running_max:
        d1 = (math.log(spot / strike) + (b + 0.5 * volatility * volatility) * maturity) / (volatility * sqrt_t)
        d2 = d1 - volatility * sqrt_t
        correction = spot * disc_r * volatility * volatility / (2.0 * b) * (-(spot / strike) ** (-2.0 * b / (volatility * volatility)) * norm.cdf(d1 - 2.0 * b * sqrt_t / volatility) + math.exp(b * maturity) * norm.cdf(d1))
        return float(spot * disc_q * norm.cdf(d1) - strike * disc_r * norm.cdf(d2) + correction)
    e1 = (math.log(spot / running_max) + (b + 0.5 * volatility * volatility) * maturity) / (volatility * sqrt_t)
    e2 = e1 - volatility * sqrt_t
    correction = spot * disc_r * volatility * volatility / (2.0 * b) * (-(spot / running_max) ** (-2.0 * b / (volatility * volatility)) * norm.cdf(e1 - 2.0 * b * sqrt_t / volatility) + math.exp(b * maturity) * norm.cdf(e1))
    return float(disc_r * (running_max - strike) + spot * disc_q * norm.cdf(e1) - running_max * disc_r * norm.cdf(e2) + correction)
__all__ = ('floating_lookback_call', 'floating_lookback_put', 'fixed_strike_lookback_call')
