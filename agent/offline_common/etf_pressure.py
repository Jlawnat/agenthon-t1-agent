from __future__ import annotations

import re

import numpy as np
import pandas as pd


def parse_financial_amount(value: object) -> float:
    """Parse common workbook money strings with K/M/B suffixes."""
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        text = str(value).strip().replace("$", "").replace(",", "")
        match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s*([KMB]?)", text, re.I)
        if match is None:
            raise ValueError(f"invalid financial amount: {value!r}")
        result = float(match.group(1)) * {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9}[match.group(2).upper()]
    if not np.isfinite(result):
        raise ValueError("financial amount must be finite")
    return result


def parse_percentage(value: object) -> float:
    """Parse a displayed percentage to a decimal ratio."""
    if isinstance(value, str):
        return float(value.strip().removesuffix("%").replace(",", "")) / 100.0
    return float(value)


def select_top_holdings(
    holdings: pd.DataFrame,
    *,
    count: int,
    ticker_column: str = "ticker",
    weight_column: str = "weight",
) -> pd.DataFrame:
    """Select top holdings by descending weight and ticker tie-break."""
    if count < 1 or not {ticker_column, weight_column}.issubset(holdings.columns):
        raise ValueError("invalid top-holdings request")
    result = holdings.copy()
    result[weight_column] = pd.to_numeric(result[weight_column], errors="raise")
    return result.sort_values([weight_column, ticker_column], ascending=[False, True], kind="stable").head(count).reset_index(drop=True)


def reconcile_constituent_price(
    shares: np.ndarray,
    weights: np.ndarray,
    fund_assets: np.ndarray,
) -> float:
    """Reconcile constituent price from combined fund-implied holding dollars."""
    share_values = np.asarray(shares, dtype=float)
    holding_dollars = np.asarray(weights, dtype=float) * np.asarray(fund_assets, dtype=float)
    if share_values.shape != holding_dollars.shape or np.any(share_values < 0.0):
        raise ValueError("shares, weights, and assets must align and shares be non-negative")
    total_shares = float(share_values.sum())
    if total_shares <= 0.0:
        raise ValueError("combined shares must be positive")
    return float(holding_dollars.sum() / total_shares)


def redemption_pressure(
    holding_shares: np.ndarray,
    holding_dollars: np.ndarray,
    redemption_rates: np.ndarray,
) -> dict[str, np.ndarray]:
    """Apply per-fund redemption rates and aggregate constituent sale pressure."""
    shares = np.asarray(holding_shares, dtype=float)
    dollars = np.asarray(holding_dollars, dtype=float)
    rates = np.asarray(redemption_rates, dtype=float).ravel()
    if shares.ndim != 2 or dollars.shape != shares.shape or rates.shape != (shares.shape[1],):
        raise ValueError("holdings must be names-by-funds and rates one per fund")
    if np.any(shares < 0.0) or np.any(dollars < 0.0) or np.any(rates < 0.0):
        raise ValueError("holdings and redemption rates must be non-negative")
    sell_shares = shares * rates
    sell_dollars = dollars * rates
    return {
        "sell_shares_by_fund": sell_shares,
        "sell_dollars_by_fund": sell_dollars,
        "combined_sell_shares": sell_shares.sum(axis=1),
        "combined_sell_dollars": sell_dollars.sum(axis=1),
    }


def concentration_hhi(values: np.ndarray) -> float:
    """HHI of non-negative amounts after sum normalization."""
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0 or np.any(array < 0.0) or not np.all(np.isfinite(array)):
        raise ValueError("values must be finite and non-negative")
    total = float(array.sum())
    return float(np.dot(array / total, array / total)) if total > 0.0 else 0.0
