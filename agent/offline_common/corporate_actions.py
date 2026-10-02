from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AdjustmentResult:
    prices: pd.DataFrame
    actions_applied: int
    cumulative_price_factor: float
    cumulative_volume_factor: float
    total_cash_distributions: float


def parse_split_ratio(value: object) -> float:
    """Parse ratios such as ``4:1`` or numeric split factors."""
    if isinstance(value, str) and ":" in value:
        numerator, denominator = value.split(":", 1)
        ratio = float(numerator) / float(denominator)
    else:
        ratio = float(value)
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("split ratio must be finite and positive")
    return ratio


def backward_adjust_prices(
    prices: pd.DataFrame,
    actions: Sequence[Mapping[str, object]],
    *,
    date_column: str = "date",
    close_column: str = "close",
    volume_column: str = "volume",
) -> AdjustmentResult:
    """Apply chronologically ordered splits and cash distributions backward."""
    required = {date_column, close_column, volume_column}
    if not required.issubset(prices.columns):
        raise ValueError("prices lacks date, close, or volume")
    work = prices.copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="raise")
    work = work.sort_values(date_column, kind="stable").reset_index(drop=True)
    close = pd.to_numeric(work[close_column], errors="raise").astype(float)
    volume = pd.to_numeric(work[volume_column], errors="raise").astype(float)
    adjusted_close = close.copy()
    adjusted_volume = volume.copy()
    price_factor = volume_factor = 1.0
    distributions = 0.0

    ordered = sorted(actions, key=lambda action: pd.Timestamp(action["date"]))
    for action in ordered:
        action_date = pd.Timestamp(action["date"])
        references = work.loc[work[date_column] <= action_date]
        if references.empty:
            raise ValueError("action precedes available price history")
        reference_close = float(close.loc[references.index[-1]])
        prior = work[date_column] < action_date
        action_type = str(action["type"]).strip().lower()
        if action_type == "split":
            ratio = parse_split_ratio(action["ratio"])
            adjusted_close.loc[prior] /= ratio
            adjusted_volume.loc[prior] *= ratio
            price_factor /= ratio
            volume_factor *= ratio
        elif action_type in {"dividend", "cash_dividend", "distribution"}:
            amount = float(action["amount"])
            factor = (reference_close - amount) / reference_close
            if not np.isfinite(factor) or factor <= 0.0:
                raise ValueError("cash distribution implies a non-positive factor")
            adjusted_close.loc[prior] *= factor
            price_factor *= factor
            distributions += amount
        else:
            raise ValueError(f"unsupported corporate action: {action_type!r}")

    result = pd.DataFrame(
        {
            date_column: work[date_column],
            "close_unadjusted": close,
            "close_adjusted": adjusted_close,
            "volume_unadjusted": volume,
            "volume_adjusted": adjusted_volume,
        }
    )
    return AdjustmentResult(result, len(ordered), float(price_factor), float(volume_factor), float(distributions))
