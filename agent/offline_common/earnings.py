from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class EarningsSurprise:
    surprise_dollars: float
    surprise_pct: float
    sue: float
    yoy_growth: float
    classification: str


def earnings_surprise(
    actual_eps: float,
    consensus_estimate: float,
    std_estimate: float,
    prior_year_eps: float,
    *,
    threshold_pct: float = 2.0,
) -> EarningsSurprise:
    """Compute common earnings-surprise measures for one observation."""
    values = np.asarray(
        [actual_eps, consensus_estimate, std_estimate, prior_year_eps, threshold_pct],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("earnings inputs must be finite")
    if threshold_pct < 0.0:
        raise ValueError("threshold_pct must be non-negative")

    surprise_dollars = float(actual_eps - consensus_estimate)
    surprise_pct = (
        float(surprise_dollars / abs(consensus_estimate) * 100.0)
        if consensus_estimate != 0.0
        else 0.0
    )
    sue = float(surprise_dollars / std_estimate) if std_estimate != 0.0 else 0.0
    yoy_growth = (
        float((actual_eps - prior_year_eps) / abs(prior_year_eps) * 100.0)
        if prior_year_eps != 0.0
        else 0.0
    )
    classification = (
        "beat"
        if surprise_pct > threshold_pct
        else "miss"
        if surprise_pct < -threshold_pct
        else "meet"
    )
    return EarningsSurprise(
        surprise_dollars=surprise_dollars,
        surprise_pct=surprise_pct,
        sue=sue,
        yoy_growth=yoy_growth,
        classification=classification,
    )


def aggregate_earnings_surprises(
    observations: Sequence[EarningsSurprise],
) -> dict[str, float]:
    """Aggregate a non-empty earnings-surprise history."""
    rows = tuple(observations)
    if not rows:
        raise ValueError("observations must not be empty")
    return {
        "avg_surprise_pct": float(np.mean([row.surprise_pct for row in rows])),
        "avg_sue": float(np.mean([row.sue for row in rows])),
        "beat_rate": float(np.mean([row.classification == "beat" for row in rows])),
        "avg_yoy_growth": float(np.mean([row.yoy_growth for row in rows])),
    }
