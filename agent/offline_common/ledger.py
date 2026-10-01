from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _finite_float(
    value: float,
    *,
    name: str,
) -> float:
    result = float(value)

    if not np.isfinite(result):
        raise ValueError(
            f"{name} must be finite."
        )

    return result


@dataclass(frozen=True)
class TradeUpdate:
    quantity: float
    cash: float
    trade_value: float
    transaction_cost: float


def proportional_trade_cost(
    quantity_change: float,
    price: float,
    *,
    buy_rate: float = 0.0,
    sell_rate: float | None = None,
) -> float:
    """Cost for a signed trade under proportional buy/sell fee rates."""
    change = _finite_float(
        quantity_change,
        name="quantity_change",
    )
    spot = _finite_float(
        price,
        name="price",
    )
    buy = _finite_float(
        buy_rate,
        name="buy_rate",
    )

    if sell_rate is None:
        sell = buy
    else:
        sell = _finite_float(
            sell_rate,
            name="sell_rate",
        )

    if spot < 0.0:
        raise ValueError(
            "price must be non-negative."
        )

    if buy < 0.0 or sell < 0.0:
        raise ValueError(
            "fee rates must be non-negative."
        )

    if change == 0.0:
        return 0.0

    rate = (
        buy
        if change > 0.0
        else sell
    )

    return float(
        abs(change * spot)
        * rate
    )


def apply_trade(
    *,
    cash: float,
    quantity: float,
    quantity_change: float,
    price: float,
    buy_rate: float = 0.0,
    sell_rate: float | None = None,
) -> TradeUpdate:
    """Apply a signed trade to cash and position state."""
    cash_value = _finite_float(
        cash,
        name="cash",
    )
    old_quantity = _finite_float(
        quantity,
        name="quantity",
    )
    change = _finite_float(
        quantity_change,
        name="quantity_change",
    )
    spot = _finite_float(
        price,
        name="price",
    )

    cost = proportional_trade_cost(
        change,
        spot,
        buy_rate=buy_rate,
        sell_rate=sell_rate,
    )

    trade_value = float(
        change
        * spot
    )

    return TradeUpdate(
        quantity=float(
            old_quantity
            + change
        ),
        cash=float(
            cash_value
            - trade_value
            - cost
        ),
        trade_value=trade_value,
        transaction_cost=float(
            cost
        ),
    )


def mark_to_market(
    cash: float,
    quantities: np.ndarray,
    prices: np.ndarray,
) -> float:
    """Portfolio NAV from cash plus signed position market values."""
    cash_value = _finite_float(
        cash,
        name="cash",
    )

    quantity_array = np.asarray(
        quantities,
        dtype=float,
    )
    price_array = np.asarray(
        prices,
        dtype=float,
    )

    if (
        quantity_array.ndim != 1
        or price_array.ndim != 1
    ):
        raise ValueError(
            "quantities and prices must be one-dimensional."
        )

    if (
        quantity_array.shape
        != price_array.shape
    ):
        raise ValueError(
            "quantities and prices must have identical shapes."
        )

    if not np.all(
        np.isfinite(quantity_array)
    ):
        raise ValueError(
            "quantities must be finite."
        )

    if not np.all(
        np.isfinite(price_array)
    ):
        raise ValueError(
            "prices must be finite."
        )

    return float(
        cash_value
        + np.dot(
            quantity_array,
            price_array,
        )
    )


def position_pnl(
    quantity: float,
    entry_price: float,
    exit_price: float,
    *,
    total_cost: float = 0.0,
) -> float:
    """Signed position P&L between entry and exit prices."""
    qty = _finite_float(
        quantity,
        name="quantity",
    )
    entry = _finite_float(
        entry_price,
        name="entry_price",
    )
    exit_value = _finite_float(
        exit_price,
        name="exit_price",
    )
    costs = _finite_float(
        total_cost,
        name="total_cost",
    )

    if costs < 0.0:
        raise ValueError(
            "total_cost must be non-negative."
        )

    return float(
        qty
        * (
            exit_value
            - entry
        )
        - costs
    )
