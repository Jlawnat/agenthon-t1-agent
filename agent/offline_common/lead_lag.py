from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LeadLagResult:
    first: str
    second: str
    correlations: dict[int, float]
    positive_peak: float
    negative_peak: float
    asymmetry: float
    leader: str
    lagger: str
    fisher_delta_z: float
    significant: bool


def lagged_correlation(first: np.ndarray, second: np.ndarray, lag: int) -> float:
    """Correlation of first[t] with second[t+lag]."""
    one = np.asarray(first, dtype=float).ravel()
    two = np.asarray(second, dtype=float).ravel()
    if one.size != two.size or one.size <= abs(int(lag)) + 1:
        raise ValueError("series must have equal length and overlap at the requested lag")
    if lag > 0:
        one, two = one[:-lag], two[lag:]
    elif lag < 0:
        one, two = one[-lag:], two[:lag]
    if not np.all(np.isfinite(one)) or not np.all(np.isfinite(two)):
        raise ValueError("series must be finite")
    return float(np.corrcoef(one, two)[0, 1])


def lead_lag_result(
    first: np.ndarray,
    second: np.ndarray,
    *,
    first_name: str,
    second_name: str,
    max_lag: int,
    z_critical: float = 1.959963984540054,
) -> LeadLagResult:
    """Measure peak positive/negative lag asymmetry with a Fisher-z gate."""
    if isinstance(max_lag, bool) or int(max_lag) < 1:
        raise ValueError("max_lag must be a positive integer")
    correlations = {
        lag: round(lagged_correlation(first, second, lag), 6)
        for lag in range(-int(max_lag), int(max_lag) + 1)
    }
    positive = max(correlations[lag] for lag in range(1, int(max_lag) + 1))
    negative = max(correlations[lag] for lag in range(-int(max_lag), 0))
    asymmetry = round(positive - negative, 6)
    clipped = np.clip([positive, negative], -0.999999, 0.999999)
    delta_z = float(abs(np.arctanh(clipped[0]) - np.arctanh(clipped[1])))
    n_obs = len(np.asarray(first).ravel())
    threshold = float(z_critical * np.sqrt(2.0 / (n_obs - 3))) if n_obs > 3 else float("inf")
    leader, lagger = (first_name, second_name) if asymmetry >= 0.0 else (second_name, first_name)
    return LeadLagResult(first_name, second_name, correlations, positive, negative, asymmetry, leader, lagger, delta_z, bool(delta_z > threshold))


def pairwise_lead_lag(returns: pd.DataFrame, *, max_lag: int) -> list[LeadLagResult]:
    """Compute lead-lag results for every column pair in input order."""
    rows = [
        lead_lag_result(
            returns[first].to_numpy(),
            returns[second].to_numpy(),
            first_name=str(first),
            second_name=str(second),
            max_lag=max_lag,
        )
        for first, second in combinations(returns.columns, 2)
    ]
    return sorted(rows, key=lambda row: abs(row.asymmetry), reverse=True)


def fit_market_models(returns: pd.DataFrame, benchmark: str) -> dict[str, dict[str, float]]:
    """Fit intercept market models for all non-benchmark columns."""
    if benchmark not in returns:
        raise ValueError("benchmark column is missing")
    market = returns[benchmark].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(market)), market])
    result: dict[str, dict[str, float]] = {}
    for name in returns.columns:
        if name == benchmark:
            continue
        values = returns[name].to_numpy(dtype=float)
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        fitted = design @ coefficients
        ss_res = float(np.sum((values - fitted) ** 2))
        ss_tot = float(np.sum((values - values.mean()) ** 2))
        result[str(name)] = {
            "alpha": float(coefficients[0]),
            "beta": float(coefficients[1]),
            "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 0.0,
        }
    return result


def market_model_residuals(returns: pd.DataFrame, models: Mapping[str, Mapping[str, float]], benchmark: str) -> pd.DataFrame:
    """Apply previously fitted market models without refitting."""
    market = returns[benchmark].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            name: returns[name].to_numpy(dtype=float) - (float(model["alpha"]) + float(model["beta"]) * market)
            for name, model in models.items()
        },
        index=returns.index,
    )
