from __future__ import annotations
import math
import numpy as np
from scipy.stats import norm

def forward_start_atm_call_price(*, spot: float, rate: float, dividend_yield: float, volatility: float, start: float, end: float) -> float:
    tau = float(end - start)
    if tau <= 0.0:
        return 0.0
    if volatility <= 0.0:
        unit = max(math.exp(-dividend_yield * tau) - math.exp(-rate * tau), 0.0)
        return float(spot * math.exp(-dividend_yield * start) * unit)
    root_tau = math.sqrt(tau)
    d1 = (rate - dividend_yield + 0.5 * volatility * volatility) * tau / (volatility * root_tau)
    d2 = d1 - volatility * root_tau
    unit = math.exp(-dividend_yield * tau) * norm.cdf(d1) - math.exp(-rate * tau) * norm.cdf(d2)
    return float(spot * math.exp(-dividend_yield * start) * unit)

def cliquet_forward_start_prices(*, spot: float, rate: float, dividend_yield: float, volatility: float, maturity: float, resets: int) -> np.ndarray:
    if maturity <= 0.0 or resets <= 0:
        raise RuntimeError('Cliquet maturity and reset count must be positive.')
    dt = maturity / float(resets)
    return np.asarray([forward_start_atm_call_price(spot=spot, rate=rate, dividend_yield=dividend_yield, volatility=volatility, start=i * dt, end=(i + 1) * dt) for i in range(resets)], dtype=float)
__all__ = ('forward_start_atm_call_price', 'cliquet_forward_start_prices')
