from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class BlackScholesGreeks:
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


@dataclass(frozen=True)
class ParityForward:
    synthetic_bid: float
    synthetic_ask: float
    synthetic_mid: float
    theoretical_forward: float
    implied_borrow_rate: float
    violation_amount: float


def black_scholes_greeks(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
    option_type: str,
) -> BlackScholesGreeks:
    """Black-Scholes price and annualized analytical Greeks."""
    if spot <= 0.0 or strike <= 0.0 or volatility <= 0.0 or maturity <= 0.0:
        raise ValueError("spot, strike, volatility, and maturity must be positive")
    kind = option_type.lower()
    if kind not in {"call", "put"}:
        raise ValueError("option_type must be call or put")
    root = math.sqrt(maturity)
    discount_r = math.exp(-rate * maturity)
    discount_q = math.exp(-dividend_yield * maturity)
    d1 = (math.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * maturity) / (volatility * root)
    d2 = d1 - volatility * root
    gamma = discount_q * norm.pdf(d1) / (spot * volatility * root)
    vega = spot * discount_q * norm.pdf(d1) * root
    common_theta = -spot * discount_q * norm.pdf(d1) * volatility / (2.0 * root)
    if kind == "call":
        price = spot * discount_q * norm.cdf(d1) - strike * discount_r * norm.cdf(d2)
        delta = discount_q * norm.cdf(d1)
        theta = common_theta + dividend_yield * spot * discount_q * norm.cdf(d1) - rate * strike * discount_r * norm.cdf(d2)
        rho = strike * maturity * discount_r * norm.cdf(d2)
    else:
        price = strike * discount_r * norm.cdf(-d2) - spot * discount_q * norm.cdf(-d1)
        delta = -discount_q * norm.cdf(-d1)
        theta = common_theta - dividend_yield * spot * discount_q * norm.cdf(-d1) + rate * strike * discount_r * norm.cdf(-d2)
        rho = -strike * maturity * discount_r * norm.cdf(-d2)
    return BlackScholesGreeks(float(price), float(delta), float(gamma), float(theta), float(vega), float(rho))


def black_scholes_pde_residual(
    price: float,
    delta: float,
    gamma: float,
    theta: float,
    *,
    spot: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    """Evaluate the Black-Scholes PDE residual from price and Greeks."""
    return float(theta + 0.5 * volatility**2 * spot**2 * gamma + (rate - dividend_yield) * spot * delta - rate * price)


def implied_volatility_newton(
    market_price: float,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    maturity: float,
    *,
    option_type: str = "call",
    initial: float = 0.3,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> tuple[float, bool]:
    """Invert Black-Scholes with guarded Newton-Raphson iterations."""
    sigma = float(initial)
    for _ in range(int(max_iterations)):
        result = black_scholes_greeks(spot, strike, rate, dividend_yield, sigma, maturity, option_type)
        error = result.price - float(market_price)
        if abs(error) < tolerance:
            return sigma, True
        if result.vega <= 1e-14:
            return sigma, False
        updated = sigma - error / result.vega
        if not np.isfinite(updated) or updated <= 0.0:
            updated = sigma / 2.0
        if abs(updated - sigma) < tolerance:
            return float(updated), True
        sigma = float(updated)
    return sigma, False


def brenner_subrahmanyam_atm(call_price: float, discounted_spot: float, maturity: float) -> float:
    """Brenner-Subrahmanyam ATM implied-volatility approximation."""
    return float(call_price / discounted_spot * math.sqrt(2.0 * math.pi / maturity))


def li_atm(call_price: float, discounted_spot: float, maturity: float) -> float:
    """Li trigonometric ATM implied-volatility approximation."""
    c = -3.0 * math.sqrt(math.pi) * call_price / (4.0 * discounted_spot)
    if not -1.0 <= c <= 1.0:
        return float("nan")
    angle = 2.0 * math.pi / 3.0 - math.acos(c) / 3.0
    return float(4.0 * math.sqrt(2.0 / maturity) * math.cos(angle))


def li_non_atm(call_price: float, discounted_spot: float, discounted_strike: float, maturity: float) -> float:
    """Li non-ATM implied-volatility approximation."""
    term = call_price - (discounted_spot - discounted_strike) / 2.0
    discriminant = term**2 - (discounted_spot - discounted_strike) ** 2 * (1.0 + discounted_spot / discounted_strike) / (2.0 * math.pi)
    if discriminant < -1e-10:
        return float("nan")
    return float(math.sqrt(2.0 * math.pi / maturity) * (term + math.sqrt(max(discriminant, 0.0))) / (discounted_spot + discounted_strike))


def corrado_miller_hallerbach(call_price: float, discounted_spot: float, discounted_strike: float, maturity: float) -> float:
    """Corrado-Miller-Hallerbach implied-volatility approximation."""
    moneyness = discounted_spot / discounted_strike
    alpha = math.sqrt(2.0 * math.pi) / (moneyness + 1.0) * (2.0 * call_price / discounted_strike - (moneyness - 1.0))
    beta = 0.5 * ((moneyness - 1.0) / (moneyness + 1.0)) ** 2
    discriminant = alpha**2 - 8.0 * beta
    if discriminant < -1e-10:
        return float("nan")
    return float((alpha + math.sqrt(max(discriminant, 0.0))) / math.sqrt(maturity))


def put_call_parity_forward(
    strike: float,
    call_bid: float,
    call_ask: float,
    put_bid: float,
    put_ask: float,
    *,
    spot: float,
    rate: float,
    dividend_yield: float,
    borrow_rate: float,
    maturity: float,
) -> ParityForward:
    """Build a no-arbitrage synthetic-forward interval from option bid/asks."""
    if maturity <= 0.0 or spot <= 0.0 or min(call_bid, call_ask, put_bid, put_ask) < 0.0:
        raise ValueError("maturity and spot must be positive and quotes non-negative")
    if call_bid > call_ask or put_bid > put_ask:
        raise ValueError("crossed quote")
    discount = math.exp(-rate * maturity)
    bid = strike + (call_bid - put_ask) / discount
    ask = strike + (call_ask - put_bid) / discount
    mid = (bid + ask) / 2.0
    theoretical = spot * math.exp((rate - dividend_yield - borrow_rate) * maturity)
    implied_borrow = rate - dividend_yield - math.log(mid / spot) / maturity
    violation = max(bid - theoretical, theoretical - ask, 0.0)
    return ParityForward(float(bid), float(ask), float(mid), float(theoretical), float(implied_borrow), float(violation))
