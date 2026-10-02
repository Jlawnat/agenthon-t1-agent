from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BrinsonEffects:
    allocation: np.ndarray
    selection: np.ndarray
    interaction: np.ndarray


def brinson_fachler_effects(
    portfolio_weights: Sequence[float],
    benchmark_weights: Sequence[float],
    portfolio_returns: Sequence[float],
    benchmark_returns: Sequence[float],
) -> BrinsonEffects:
    """Compute per-segment Brinson-Fachler attribution effects."""
    wp, wb, rp, rb = (np.asarray(values, dtype=float).ravel() for values in (portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns))
    if not (wp.shape == wb.shape == rp.shape == rb.shape) or wp.size == 0:
        raise ValueError("weights and returns must be equally sized non-empty vectors")
    if not all(np.all(np.isfinite(values)) for values in (wp, wb, rp, rb)):
        raise ValueError("weights and returns must be finite")
    total_benchmark_return = float(np.dot(wb, rb))
    active_weight = wp - wb
    active_return = rp - rb
    return BrinsonEffects(
        allocation=active_weight * (rb - total_benchmark_return),
        selection=wb * active_return,
        interaction=active_weight * active_return,
    )


def drift_weights(weights: Sequence[float], returns: Sequence[float]) -> np.ndarray:
    """Drift beginning weights through one return period and renormalize."""
    values = np.asarray(weights, dtype=float).ravel() * (1.0 + np.asarray(returns, dtype=float).ravel())
    if values.size == 0 or not np.all(np.isfinite(values)) or float(values.sum()) <= 0.0:
        raise ValueError("post-return portfolio value must be finite and positive")
    return values / values.sum()


def compound_returns(returns: Sequence[float]) -> float:
    """Compound simple returns over time."""
    values = np.asarray(returns, dtype=float).ravel()
    if not np.all(np.isfinite(values)) or np.any(values <= -1.0):
        raise ValueError("simple returns must be finite and greater than -1")
    return float(np.prod(1.0 + values) - 1.0)
