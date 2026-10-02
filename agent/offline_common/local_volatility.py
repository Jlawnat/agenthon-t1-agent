from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class SviCalibration:
    parameters: np.ndarray
    volatility_rmse: float
    converged: bool


@dataclass(frozen=True)
class BarrierMonteCarlo:
    vanilla_price: float
    vanilla_standard_error: float
    knockout_price: float
    knockout_standard_error: float
    hit_probability: float


def svi_total_variance(log_moneyness: np.ndarray, parameters: Sequence[float]) -> np.ndarray:
    """Raw SVI total variance ``a+b(rho(k-m)+sqrt((k-m)^2+s^2))``."""
    a, b, rho, center, width = (float(value) for value in parameters)
    k = np.asarray(log_moneyness, dtype=float)
    return a + b * (rho * (k - center) + np.sqrt((k - center) ** 2 + width**2))


def calibrate_svi(
    log_moneyness: np.ndarray,
    implied_volatility: np.ndarray,
    maturity: float,
) -> SviCalibration:
    """Calibrate bounded raw SVI parameters by volatility RMSE."""
    k = np.asarray(log_moneyness, dtype=float).ravel()
    iv = np.asarray(implied_volatility, dtype=float).ravel()
    if k.shape != iv.shape or k.size < 5 or maturity <= 0.0 or np.any(iv <= 0.0):
        raise ValueError("SVI calibration requires at least five positive-vol observations")
    observed = iv**2 * maturity

    def objective(parameters: np.ndarray) -> float:
        total = svi_total_variance(k, parameters)
        if np.any(total <= 0.0):
            return 1e10
        return float(np.mean((np.sqrt(total / maturity) - iv) ** 2))

    result = minimize(
        objective,
        np.array([max(float(observed.min()) * 0.5, 0.001), 0.1, -0.3, 0.0, 0.1]),
        method="L-BFGS-B",
        bounds=[(1e-8, 2.0), (1e-8, 3.0), (-0.999, 0.999), (-2.0, 2.0), (1e-5, 3.0)],
    )
    return SviCalibration(np.asarray(result.x, dtype=float), float(math.sqrt(result.fun)), bool(result.success))


def dupire_local_volatility(
    strike: float,
    call_price: float,
    call_time_derivative: float,
    call_strike_derivative: float,
    call_strike_second_derivative: float,
    *,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> float:
    """Evaluate Dupire local volatility from call-price derivatives."""
    numerator = call_time_derivative + (rate - dividend_yield) * strike * call_strike_derivative + dividend_yield * call_price
    denominator = 0.5 * strike**2 * call_strike_second_derivative
    if denominator <= 0.0 or numerator < 0.0:
        return float("nan")
    return float(math.sqrt(numerator / denominator))


def dupire_local_vol_grid(
    call_prices: np.ndarray,
    maturities: Sequence[float],
    strikes: Sequence[float],
    *,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    variance_floor: float = 1e-8,
    volatility_cap: float = 5.0,
) -> np.ndarray:
    """Differentiate a maturity-by-strike call surface into local volatility."""
    calls = np.asarray(call_prices, dtype=float)
    times = np.asarray(maturities, dtype=float).ravel()
    strike_grid = np.asarray(strikes, dtype=float).ravel()
    if calls.shape != (times.size, strike_grid.size) or min(times.size, strike_grid.size) < 3:
        raise ValueError("surface shape must match grids with at least three points per axis")
    dcdt = np.gradient(calls, times, axis=0, edge_order=2)
    dcdk = np.gradient(calls, strike_grid, axis=1, edge_order=2)
    d2cdk2 = np.gradient(dcdk, strike_grid, axis=1, edge_order=2)
    numerator = dcdt + (rate - dividend_yield) * strike_grid[None, :] * dcdk + dividend_yield * calls
    denominator = 0.5 * strike_grid[None, :] ** 2 * d2cdk2
    variance = np.divide(numerator, denominator, out=np.full_like(calls, variance_floor), where=denominator > 0.0)
    variance = np.clip(variance, variance_floor, volatility_cap**2)
    return np.sqrt(variance)


def bilinear_surface_value(
    maturity: float,
    spot: float,
    maturities: Sequence[float],
    spots: Sequence[float],
    values: np.ndarray,
) -> float:
    """Bilinearly interpolate a rectangular time/spot surface with edge clipping."""
    times = np.asarray(maturities, dtype=float)
    levels = np.asarray(spots, dtype=float)
    surface = np.asarray(values, dtype=float)
    if surface.shape != (times.size, levels.size):
        raise ValueError("surface shape does not match grids")
    t = float(np.clip(maturity, times[0], times[-1]))
    s = float(np.clip(spot, levels[0], levels[-1]))
    upper_t = min(int(np.searchsorted(times, t, side="right")), times.size - 1)
    upper_s = min(int(np.searchsorted(levels, s, side="right")), levels.size - 1)
    lower_t, lower_s = max(upper_t - 1, 0), max(upper_s - 1, 0)
    wt = 0.0 if times[upper_t] == times[lower_t] else (t - times[lower_t]) / (times[upper_t] - times[lower_t])
    ws = 0.0 if levels[upper_s] == levels[lower_s] else (s - levels[lower_s]) / (levels[upper_s] - levels[lower_s])
    return float((1 - wt) * ((1 - ws) * surface[lower_t, lower_s] + ws * surface[lower_t, upper_s]) + wt * ((1 - ws) * surface[upper_t, lower_s] + ws * surface[upper_t, upper_s]))


def local_vol_barrier_monte_carlo(
    spot: float,
    strike: float,
    barrier: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    surface_maturities: Sequence[float],
    surface_spots: Sequence[float],
    local_volatilities: np.ndarray,
    *,
    paths: int,
    steps: int,
    monitoring_interval: int = 1,
    seed: int = 0,
) -> BarrierMonteCarlo:
    """Antithetic local-vol Monte Carlo for a discretely monitored down-and-out call."""
    if paths < 2 or steps < 1 or monitoring_interval < 1 or barrier >= spot:
        raise ValueError("invalid Monte Carlo or down-barrier inputs")
    half = (paths + 1) // 2
    rng = np.random.default_rng(seed)
    normals = rng.standard_normal((half, steps))
    normals = np.vstack([normals, -normals])[:paths]
    levels = np.full(paths, float(spot))
    alive = np.ones(paths, dtype=bool)
    dt = maturity / steps
    for index in range(steps):
        time = index * dt
        vols = np.array([bilinear_surface_value(time, value, surface_maturities, surface_spots, local_volatilities) for value in levels])
        levels *= np.exp((rate - dividend_yield - 0.5 * vols**2) * dt + vols * math.sqrt(dt) * normals[:, index])
        if (index + 1) % monitoring_interval == 0 or index + 1 == steps:
            alive &= levels > barrier
    discount = math.exp(-rate * maturity)
    vanilla = discount * np.maximum(levels - strike, 0.0)
    knockout = vanilla * alive
    standard_error = lambda values: float(values.std(ddof=1) / math.sqrt(values.size))
    return BarrierMonteCarlo(float(vanilla.mean()), standard_error(vanilla), float(knockout.mean()), standard_error(knockout), float(1.0 - alive.mean()))
