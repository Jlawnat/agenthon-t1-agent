from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, norm

try:
    from agent.offline_common.vanilla_options import black_scholes_greeks
except ModuleNotFoundError:  # Candidate workspace exposes offline_common directly.
    from offline_common.vanilla_options import black_scholes_greeks


def barone_adesi_whaley(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
    option_type: str,
) -> tuple[float, float]:
    """Price an American call or put with the BAW quadratic approximation."""
    kind = option_type.lower()
    if kind not in {"call", "put"} or min(spot, strike, volatility, maturity) <= 0.0:
        raise ValueError("invalid American option inputs")
    european = black_scholes_greeks(spot, strike, rate, dividend_yield, volatility, maturity, kind).price
    if kind == "call" and dividend_yield <= 0.0:
        return european, math.inf
    sigma2 = volatility**2
    m = 2.0 * rate / sigma2
    n = 2.0 * (rate - dividend_yield) / sigma2
    capital_k = 1.0 - math.exp(-rate * maturity)
    if capital_k <= 0.0:
        raise ValueError("BAW requires a positive rate")
    root_term = math.sqrt((n - 1.0) ** 2 + 4.0 * m / capital_k)
    q = (-(n - 1.0) + root_term) / 2.0 if kind == "call" else (-(n - 1.0) - root_term) / 2.0
    discount_q = math.exp(-dividend_yield * maturity)

    def equation(boundary: float) -> float:
        d1 = (math.log(boundary / strike) + (rate - dividend_yield + 0.5 * sigma2) * maturity) / (volatility * math.sqrt(maturity))
        euro = black_scholes_greeks(boundary, strike, rate, dividend_yield, volatility, maturity, kind).price
        if kind == "call":
            return boundary - strike - euro - (1.0 - discount_q * norm.cdf(d1)) * boundary / q
        return strike - boundary - euro + (1.0 - discount_q * norm.cdf(-d1)) * boundary / q

    boundary = float(brentq(equation, strike * (1.0 + 1e-12), strike * 100.0, maxiter=200)) if kind == "call" else float(brentq(equation, strike * 1e-12, strike * (1.0 - 1e-12), maxiter=200))
    d1_boundary = (math.log(boundary / strike) + (rate - dividend_yield + 0.5 * sigma2) * maturity) / (volatility * math.sqrt(maturity))
    if kind == "call":
        if spot >= boundary:
            return float(spot - strike), boundary
        coefficient = (1.0 - discount_q * norm.cdf(d1_boundary)) * boundary / q
    else:
        if spot <= boundary:
            return float(strike - spot), boundary
        coefficient = -(1.0 - discount_q * norm.cdf(-d1_boundary)) * boundary / q
    return float(european + coefficient * (spot / boundary) ** q), boundary


def crr_american_option(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
    option_type: str,
    *,
    steps: int = 1000,
) -> float:
    """Price an American option with a CRR tree and early exercise."""
    if steps < 1:
        raise ValueError("steps must be positive")
    dt = maturity / steps
    up = math.exp(volatility * math.sqrt(dt))
    down = 1.0 / up
    probability = (math.exp((rate - dividend_yield) * dt) - down) / (up - down)
    if not 0.0 < probability < 1.0:
        raise ValueError("risk-neutral probability is outside (0,1)")
    indices = np.arange(steps + 1, dtype=float)
    stock = spot * up**indices * down ** (steps - indices)
    call = option_type.lower() == "call"
    values = np.maximum(stock - strike, 0.0) if call else np.maximum(strike - stock, 0.0)
    discount = math.exp(-rate * dt)
    for step in range(steps - 1, -1, -1):
        values = discount * (probability * values[1:] + (1.0 - probability) * values[:-1])
        indices = np.arange(step + 1, dtype=float)
        stock = spot * up**indices * down ** (step - indices)
        exercise = np.maximum(stock - strike, 0.0) if call else np.maximum(strike - stock, 0.0)
        values = np.maximum(values, exercise)
    return float(values[0])


def geske_compound_option(
    spot: float,
    outer_strike: float,
    inner_strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    outer_maturity: float,
    inner_maturity: float,
    option_type: str,
) -> tuple[float, float]:
    """Geske call-on-call or put-on-put price and critical inner spot."""
    kind = option_type.lower()
    if kind not in {"call_on_call", "put_on_put"} or not 0.0 < outer_maturity < inner_maturity:
        raise ValueError("invalid compound option type or maturities")
    inner_kind = "call" if kind == "call_on_call" else "put"
    remaining = inner_maturity - outer_maturity
    boundary = float(brentq(lambda value: black_scholes_greeks(value, inner_strike, rate, dividend_yield, volatility, remaining, inner_kind).price - outer_strike, 1e-8, 5.0 * max(spot, inner_strike), maxiter=200))

    def d_values(level: float, strike: float, maturity: float) -> tuple[float, float]:
        first = (math.log(level / strike) + (rate - dividend_yield + 0.5 * volatility**2) * maturity) / (volatility * math.sqrt(maturity))
        return first, first - volatility * math.sqrt(maturity)

    a1, a2 = d_values(spot, boundary, outer_maturity)
    b1, b2 = d_values(spot, inner_strike, inner_maturity)
    rho = math.sqrt(outer_maturity / inner_maturity)
    bivar = lambda x, y, correlation: float(multivariate_normal.cdf([x, y], mean=[0.0, 0.0], cov=[[1.0, correlation], [correlation, 1.0]]))
    if kind == "call_on_call":
        price = spot * math.exp(-dividend_yield * inner_maturity) * bivar(a1, b1, rho) - inner_strike * math.exp(-rate * inner_maturity) * bivar(a2, b2, rho) - outer_strike * math.exp(-rate * outer_maturity) * norm.cdf(a2)
    else:
        price = spot * math.exp(-dividend_yield * inner_maturity) * bivar(a1, -b1, -rho) - inner_strike * math.exp(-rate * inner_maturity) * bivar(a2, -b2, -rho) + outer_strike * math.exp(-rate * outer_maturity) * norm.cdf(a2)
    return float(price), boundary


def simple_chooser_option(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    choose_time: float,
    maturity: float,
) -> float:
    """Price a simple chooser using its call-plus-adjusted-put decomposition."""
    call = black_scholes_greeks(spot, strike, rate, dividend_yield, volatility, maturity, "call").price
    adjusted_strike = strike * math.exp(-(rate - dividend_yield) * (maturity - choose_time))
    put = black_scholes_greeks(spot, adjusted_strike, rate, dividend_yield, volatility, choose_time, "put").price
    return float(call + put)
