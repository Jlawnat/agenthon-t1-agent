from __future__ import annotations
from dataclasses import dataclass
import math
import re
from typing import Sequence
import numpy as np

@dataclass(frozen=True)
class FiniteDifferenceOptionSpec:
    spot: float
    strike: float
    rate: float
    dividend_yield: float
    volatility: float
    maturity: float
    s_max: float
    stock_steps: int
    time_steps: int
    dividends: tuple[tuple[float, float], ...] = ()
    psor_omega: float = 1.2
    psor_tolerance: float = 1e-08
    psor_max_iterations: int = 10000

@dataclass(frozen=True)
class FiniteDifferenceResult:
    value: float
    delta: float
    stock_grid: np.ndarray
    time_grid: np.ndarray
    value_grid: np.ndarray | None
    exercise_boundary: np.ndarray | None
    psor_iterations_max: int

def _payoff(stock_grid: np.ndarray, strike: float, option_type: str) -> np.ndarray:
    if option_type == 'put':
        return np.maximum(strike - stock_grid, 0.0)
    if option_type == 'call':
        return np.maximum(stock_grid - strike, 0.0)
    raise RuntimeError("option_type must be 'put' or 'call'.")

def crank_nicolson_option(spec: FiniteDifferenceOptionSpec, *, option_type: str, exercise_type: str, dividends: Sequence[tuple[float, float]] | None=None, return_grid: bool=False, return_boundary: bool=False) -> FiniteDifferenceResult:
    """Price a European or American option using CN, PSOR and cash-dividend jumps."""
    if exercise_type not in {'american', 'european'}:
        raise RuntimeError("exercise_type must be 'american' or 'european'.")
    if spec.stock_steps < 3 or spec.time_steps < 2:
        raise RuntimeError('Finite-difference grid is too small.')
    n_s = int(spec.stock_steps)
    n_t = int(spec.time_steps)
    d_s = spec.s_max / n_s
    dt = spec.maturity / n_t
    stock = np.linspace(0.0, spec.s_max, n_s + 1)
    times = np.arange(n_t + 1, dtype=float) * dt
    payoff = _payoff(stock, spec.strike, option_type)
    values = payoff.copy()
    boundary = None
    if return_boundary:
        boundary = np.zeros(n_t + 1, dtype=float)
        if option_type == 'put' and exercise_type == 'american':
            boundary[n_t] = spec.strike
    full_grid = None
    if return_grid:
        full_grid = np.empty((n_s + 1, n_t + 1), dtype=float)
        full_grid[:, n_t] = values
    i = np.arange(1, n_s, dtype=float)
    sigma2 = spec.volatility * spec.volatility
    alpha = 0.25 * dt * (sigma2 * i * i - spec.rate * i)
    beta = -0.5 * dt * (sigma2 * i * i + spec.rate)
    gamma = 0.25 * dt * (sigma2 * i * i + spec.rate * i)
    a_lower = -alpha
    a_diag = 1.0 - beta
    a_upper = -gamma
    b_lower = alpha
    b_diag = 1.0 + beta
    b_upper = gamma
    active_dividends = tuple(spec.dividends if dividends is None else dividends)
    div_steps = {int(round(float(time) / dt)): float(amount) for time, amount in active_dividends}
    max_iterations_seen = 0
    for n in range(n_t - 1, -1, -1):
        tau_new = spec.maturity - n * dt
        tau_old = spec.maturity - (n + 1) * dt
        dividend_amount = div_steps.get(n + 1)
        if dividend_amount is not None:
            adjusted = np.zeros(n_s + 1, dtype=float)
            for node in range(n_s + 1):
                shifted = stock[node] - dividend_amount
                if shifted <= 0.0:
                    adjusted[node] = spec.strike * math.exp(-spec.rate * tau_old) if option_type == 'put' else 0.0
                elif shifted >= spec.s_max:
                    adjusted[node] = 0.0 if option_type == 'put' else spec.s_max - spec.strike * math.exp(-spec.rate * tau_old)
                else:
                    left = min(int(shifted / d_s), n_s - 1)
                    fraction = (shifted - stock[left]) / d_s
                    adjusted[node] = values[left] + fraction * (values[left + 1] - values[left])
            values = adjusted
        if option_type == 'put':
            old_left = spec.strike * math.exp(-spec.rate * tau_old)
            old_right = 0.0
            new_left = spec.strike * math.exp(-spec.rate * tau_new)
            new_right = 0.0
        else:
            old_left = 0.0
            old_right = spec.s_max - spec.strike * math.exp(-spec.rate * tau_old)
            new_left = 0.0
            new_right = spec.s_max - spec.strike * math.exp(-spec.rate * tau_new)
        values[0] = old_left
        values[n_s] = old_right
        rhs = b_lower * values[0:n_s - 1] + b_diag * values[1:n_s] + b_upper * values[2:n_s + 1]
        rhs[0] += alpha[0] * new_left
        rhs[-1] += gamma[-1] * new_right
        interior = values[1:n_s].copy()
        obstacle = payoff[1:n_s] if exercise_type == 'american' else None
        iteration_count = 0
        for iteration in range(spec.psor_max_iterations):
            max_change = 0.0
            for row in range(n_s - 1):
                rhs_value = rhs[row]
                if row > 0:
                    rhs_value -= a_lower[row] * interior[row - 1]
                if row < n_s - 2:
                    rhs_value -= a_upper[row] * interior[row + 1]
                candidate = rhs_value / a_diag[row]
                updated = interior[row] + spec.psor_omega * (candidate - interior[row])
                if obstacle is not None:
                    updated = max(updated, obstacle[row])
                max_change = max(max_change, abs(updated - interior[row]))
                interior[row] = updated
            iteration_count = iteration + 1
            if max_change < spec.psor_tolerance:
                break
        else:
            raise RuntimeError('PSOR failed to converge.')
        max_iterations_seen = max(max_iterations_seen, iteration_count)
        values[1:n_s] = interior
        values[0] = new_left
        values[n_s] = new_right
        if boundary is not None and option_type == 'put' and (exercise_type == 'american'):
            s_star = 0.0
            for node in range(n_s, 0, -1):
                if payoff[node] > 0.0 and abs(values[node] - payoff[node]) < 1e-06:
                    s_star = stock[node]
                    break
            boundary[n] = s_star
        if full_grid is not None:
            full_grid[:, n] = values
    i0 = int(round(spec.spot / d_s))
    if not 1 <= i0 <= n_s - 1:
        raise RuntimeError('Spot must lie strictly inside the stock grid.')
    value = float(values[i0])
    delta = float((values[i0 + 1] - values[i0 - 1]) / (2.0 * d_s))
    return FiniteDifferenceResult(value=value, delta=delta, stock_grid=stock, time_grid=times, value_grid=full_grid, exercise_boundary=boundary, psor_iterations_max=max_iterations_seen)

def with_grid_size(spec: FiniteDifferenceOptionSpec, *, stock_steps: int, time_steps: int) -> FiniteDifferenceOptionSpec:
    return FiniteDifferenceOptionSpec(spot=spec.spot, strike=spec.strike, rate=spec.rate, dividend_yield=spec.dividend_yield, volatility=spec.volatility, maturity=spec.maturity, s_max=spec.s_max, stock_steps=int(stock_steps), time_steps=int(time_steps), dividends=spec.dividends, psor_omega=spec.psor_omega, psor_tolerance=spec.psor_tolerance, psor_max_iterations=spec.psor_max_iterations)

def richardson_second_order(fine: float, coarse: float) -> float:
    return float((4.0 * float(fine) - float(coarse)) / 3.0)

__all__ = (
    "FiniteDifferenceOptionSpec",
    "FiniteDifferenceResult",
    "crank_nicolson_option",
    "with_grid_size",
    "richardson_second_order",
)
