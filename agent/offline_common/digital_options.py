from __future__ import annotations

import math

from scipy.stats import norm


def cash_or_nothing(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
    option_type: str,
    *,
    cash_payoff: float = 1.0,
) -> float:
    """Price a European cash-or-nothing binary option."""
    d2 = (math.log(spot / strike) + (rate - dividend_yield - 0.5 * volatility**2) * maturity) / (volatility * math.sqrt(maturity))
    probability = norm.cdf(d2) if option_type.lower() == "call" else norm.cdf(-d2)
    return float(cash_payoff * math.exp(-rate * maturity) * probability)


def asset_or_nothing(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
    option_type: str,
) -> float:
    """Price a European asset-or-nothing binary option."""
    d1 = (math.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * maturity) / (volatility * math.sqrt(maturity))
    probability = norm.cdf(d1) if option_type.lower() == "call" else norm.cdf(-d1)
    return float(spot * math.exp(-dividend_yield * maturity) * probability)


def gap_option(
    spot: float,
    payoff_strike: float,
    trigger_strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
    option_type: str,
) -> float:
    """Price a European gap call or put with distinct trigger and payoff strikes."""
    kind = option_type.lower()
    asset = asset_or_nothing(spot, trigger_strike, rate, dividend_yield, volatility, maturity, kind)
    cash = cash_or_nothing(spot, trigger_strike, rate, dividend_yield, volatility, maturity, kind)
    return float(asset - payoff_strike * cash) if kind == "call" else float(payoff_strike * cash - asset)


def reflected_barrier_binary(
    spot: float,
    strike: float,
    barrier: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
    barrier_type: str,
) -> tuple[float, float]:
    """Price supported continuous knock-out/in cash binaries by reflection."""
    kind = barrier_type.lower()
    if kind not in {"down_out_call", "up_out_put"}:
        raise ValueError("barrier_type must be down_out_call or up_out_put")
    mu = rate - dividend_yield - 0.5 * volatility**2
    root = volatility * math.sqrt(maturity)
    d_strike = (math.log(spot / strike) + mu * maturity) / root
    d_image = (math.log(barrier**2 / (spot * strike)) + mu * maturity) / root
    reflection = (barrier / spot) ** (2.0 * mu / volatility**2)
    if kind == "down_out_call":
        vanilla = cash_or_nothing(spot, strike, rate, dividend_yield, volatility, maturity, "call")
        knockout = math.exp(-rate * maturity) * (norm.cdf(d_strike) - reflection * norm.cdf(d_image))
    else:
        vanilla = cash_or_nothing(spot, strike, rate, dividend_yield, volatility, maturity, "put")
        knockout = math.exp(-rate * maturity) * (norm.cdf(-d_strike) - reflection * norm.cdf(-d_image))
    return float(knockout), float(vanilla - knockout)
