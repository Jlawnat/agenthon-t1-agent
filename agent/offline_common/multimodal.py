from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def normalize_cross_section(values: Sequence[float]) -> np.ndarray:
    """Median-fill, population-z-score, and clip a cross section."""
    series = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan)
    if not series.notna().any():
        return np.zeros(len(series), dtype=float)
    filled = series.fillna(series.median())
    scale = float(filled.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0.0:
        return np.zeros(len(filled), dtype=float)
    return np.clip(((filled - filled.mean()) / scale).to_numpy(dtype=float), -3.0, 3.0)


def signed_component_score(feature_columns: np.ndarray, signs: Sequence[float]) -> np.ndarray:
    """Normalize signed feature columns cross-sectionally and average them."""
    matrix = np.asarray(feature_columns, dtype=float)
    direction = np.asarray(signs, dtype=float).ravel()
    if matrix.ndim != 2 or direction.shape != (matrix.shape[1],):
        raise ValueError("signs must contain one direction per feature column")
    components = np.column_stack([normalize_cross_section(matrix[:, index] * direction[index]) for index in range(matrix.shape[1])])
    return components.mean(axis=1)


def fuse_committee_scores(
    agent_scores: np.ndarray,
    weights: Sequence[float],
    *,
    disagreement_penalty: float,
) -> dict[str, np.ndarray]:
    """Fuse agent scores with a population-dispersion agreement penalty."""
    scores = np.asarray(agent_scores, dtype=float)
    weight = np.asarray(weights, dtype=float).ravel()
    if scores.ndim != 2 or weight.shape != (scores.shape[1],):
        raise ValueError("weights must contain one value per agent")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(weight)) or disagreement_penalty < 0.0:
        raise ValueError("scores and weights must be finite and penalty non-negative")
    committee = scores @ weight
    dispersion = scores.std(axis=1, ddof=0)
    agreement = np.maximum(0.0, 1.0 - float(disagreement_penalty) * dispersion)
    return {
        "committee_score": committee,
        "committee_dispersion": dispersion,
        "agreement_multiplier": agreement,
        "committee_alpha": committee * agreement,
    }


def reference_vintage_disagreement(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    asof_date: str | pd.Timestamp,
    stale_after_days: int,
) -> dict[str, float | int]:
    """Compare latest field vintages available at an as-of date."""
    required = {"date", "field", "value"}
    if not required.issubset(first.columns) or not required.issubset(second.columns):
        raise ValueError("vintages require date, field, and value columns")
    asof = pd.Timestamp(asof_date)

    def latest(frame: pd.DataFrame) -> pd.DataFrame:
        work = frame.copy()
        work["date"] = pd.to_datetime(work["date"], errors="raise")
        return work.loc[work["date"] <= asof].sort_values("date").groupby("field", as_index=False).tail(1).set_index("field")

    one, two = latest(first), latest(second)
    fields = sorted(set(one.index) | set(two.index))
    mismatches = stale = 0
    absolute = 0.0
    for field in fields:
        available = [frame.loc[field] for frame in (one, two) if field in frame.index]
        if not available or (asof - max(row["date"] for row in available)).days > stale_after_days:
            stale += 1
        if len(available) != 2:
            continue
        left, right = available[0]["value"], available[1]["value"]
        try:
            difference = abs(float(left) - float(right))
        except (TypeError, ValueError):
            difference = 0.0 if str(left) == str(right) else 1.0
        if difference > 1e-12:
            mismatches += 1
            absolute += difference
    return {"mismatch_count": mismatches, "mismatch_abs_sum": float(absolute), "stale_field_count": stale}
