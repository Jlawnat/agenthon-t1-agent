from __future__ import annotations

from dataclasses import dataclass
import bisect
import csv
import json
from pathlib import Path


RESULT_KEYS = (
    "schema_version",
    "task_slug",
    "symbol",
    "side",
    "start_time_ms",
    "end_time_ms",
    "bucket_ms",
    "target_qty",
    "executed_qty",
    "unfilled_qty",
    "fill_rate",
    "arrival_mid",
    "arrival_quote_update_id",
    "arrival_quote_transaction_time_ms",
    "arrival_quote_event_time_ms",
    "future_reference_time_ms",
    "future_mid",
    "future_quote_update_id",
    "future_quote_transaction_time_ms",
    "future_quote_event_time_ms",
    "avg_fill_price",
    "market_vwap",
    "implementation_shortfall_bps",
    "effective_spread_bps",
    "realized_spread_bps",
    "temporary_impact_bps",
    "market_trade_count",
    "market_volume",
    "bucket_count_total",
    "bucket_count_filled",
    "bucket_count_empty",
)

FILL_COLUMNS = (
    "bucket_index",
    "bucket_start_ms",
    "bucket_end_ms",
    "eligible_trade_window_start_ms",
    "eligible_trade_window_end_ms",
    "eligible_trade_count",
    "eligible_trade_qty",
    "eligible_trade_notional",
    "eligible_first_trade_id",
    "eligible_last_trade_id",
    "eligible_first_trade_time_ms",
    "eligible_last_trade_time_ms",
    "participation_cap_qty",
    "max_child_qty",
    "child_order_qty",
    "fill_qty",
    "fill_price",
    "reference_bid",
    "reference_ask",
    "fill_quote_update_id",
    "fill_quote_transaction_time_ms",
    "fill_quote_event_time_ms",
    "remaining_qty",
    "is_empty_bucket",
)


def _find(task_dir: Path, name: str) -> Path:
    matches = [
        p
        for p in task_dir.rglob(name)
        if p.is_file() and "checks" not in p.parts
    ]
    if not matches:
        raise RuntimeError(
            f"Offline TCA skill could not find {name}."
        )
    return sorted(matches)[0]


def _r6(value: float) -> float:
    return round(float(value), 6)


def _f6(value: float) -> str:
    return f"{float(value):.6f}"


def _load_quotes(path: Path) -> list[dict[str, int | float]]:
    rows: list[dict[str, int | float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "update_id": int(row["update_id"]),
                    "best_bid_price": float(row["best_bid_price"]),
                    "best_ask_price": float(row["best_ask_price"]),
                    "transaction_time": int(row["transaction_time"]),
                    "event_time": int(row["event_time"]),
                }
            )
    rows.sort(key=lambda row: int(row["transaction_time"]))
    return rows


def _load_trades(path: Path) -> list[dict[str, int | float]]:
    rows: list[dict[str, int | float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "agg_trade_id": int(row["agg_trade_id"]),
                    "price": float(row["price"]),
                    "quantity": float(row["quantity"]),
                    "transact_time": int(row["transact_time"]),
                }
            )
    rows.sort(key=lambda row: int(row["transact_time"]))
    return rows


def _quote_at_or_after(
    quotes: list[dict[str, int | float]],
    quote_times: list[int],
    timestamp: int,
) -> dict[str, int | float]:
    index = bisect.bisect_left(
        quote_times,
        timestamp,
    )
    if index >= len(quotes):
        raise RuntimeError(
            f"No quote at or after {timestamp}."
        )
    return quotes[index]


@dataclass(frozen=True)
class BinanceParticipationTcaSkill:
    name: str = "binance-btc-participation-tca"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        del task_dir
        lowered = instruction.lower()
        required = (
            "participation-capped",
            "implementation_shortfall_bps",
            "realized_spread_bps",
            "fills.csv",
            "solution.json",
            "btc",
        )
        return all(token in lowered for token in required)

    def solve(
        self,
        *,
        instruction: str,
        task_dir: Path,
        out_dir: Path,
        seed: int,
    ) -> None:
        del instruction, seed

        order = json.loads(
            _find(
                task_dir,
                "order_spec.json",
            ).read_text(encoding="utf-8")
        )

        # Read the normative rules as part of the local contract.
        json.loads(
            _find(
                task_dir,
                "tca_rules.json",
            ).read_text(encoding="utf-8")
        )

        quotes = _load_quotes(
            _find(task_dir, "quotes.csv")
        )
        trades = _load_trades(
            _find(task_dir, "trades.csv")
        )

        quote_times = [
            int(row["transaction_time"])
            for row in quotes
        ]

        start = int(order["start_time_ms"])
        end = int(order["end_time_ms"])
        bucket_ms = int(order["bucket_ms"])
        target = float(order["target_quantity"])
        participation_cap = float(order["participation_cap"])
        max_child = float(order["max_child_quantity"])
        realized_horizon = int(order["realized_horizon_ms"])

        arrival_quote = _quote_at_or_after(
            quotes,
            quote_times,
            start,
        )
        arrival_mid = (
            float(arrival_quote["best_bid_price"])
            + float(arrival_quote["best_ask_price"])
        ) / 2.0

        future_reference_time = (
            end + realized_horizon
        )
        future_quote = _quote_at_or_after(
            quotes,
            quote_times,
            future_reference_time,
        )
        future_mid = (
            float(future_quote["best_bid_price"])
            + float(future_quote["best_ask_price"])
        ) / 2.0

        trade_times = [
            int(row["transact_time"])
            for row in trades
        ]

        remaining = target
        fill_rows: list[dict[str, str]] = []
        fill_notional = 0.0
        executed_qty = 0.0
        eligible_count_total = 0
        eligible_volume_total = 0.0
        eligible_notional_total = 0.0
        first_fill_quote_time: int | None = None
        last_fill_quote_time: int | None = None

        for bucket_index, bucket_start in enumerate(
            range(start, end, bucket_ms)
        ):
            bucket_end = min(
                bucket_start + bucket_ms,
                end,
            )

            left = bisect.bisect_left(
                trade_times,
                bucket_start,
            )
            right = bisect.bisect_left(
                trade_times,
                bucket_end,
            )
            eligible = trades[left:right]

            eligible_count = len(eligible)
            eligible_qty = sum(
                float(row["quantity"])
                for row in eligible
            )
            eligible_notional = sum(
                float(row["price"])
                * float(row["quantity"])
                for row in eligible
            )

            eligible_count_total += eligible_count
            eligible_volume_total += eligible_qty
            eligible_notional_total += eligible_notional

            participation_cap_qty = (
                participation_cap
                * eligible_qty
            )

            child_qty = min(
                remaining,
                participation_cap_qty,
                max_child,
            )
            child_qty = max(
                0.0,
                child_qty,
            )

            fill_quote = _quote_at_or_after(
                quotes,
                quote_times,
                bucket_start,
            )

            if child_qty > 0.0:
                fill_price = float(
                    fill_quote["best_ask_price"]
                )
                executed_qty += child_qty
                fill_notional += (
                    child_qty
                    * fill_price
                )
                remaining -= child_qty
                if abs(remaining) < 5e-12:
                    remaining = 0.0

                quote_tx = int(
                    fill_quote["transaction_time"]
                )
                if first_fill_quote_time is None:
                    first_fill_quote_time = quote_tx
                last_fill_quote_time = quote_tx

                fill_price_text = _f6(fill_price)
                bid_text = _f6(
                    float(
                        fill_quote["best_bid_price"]
                    )
                )
                ask_text = _f6(
                    float(
                        fill_quote["best_ask_price"]
                    )
                )
                quote_id_text = str(
                    int(fill_quote["update_id"])
                )
                quote_tx_text = str(quote_tx)
                quote_event_text = str(
                    int(fill_quote["event_time"])
                )
            else:
                fill_price_text = ""
                bid_text = ""
                ask_text = ""
                quote_id_text = ""
                quote_tx_text = ""
                quote_event_text = ""

            is_empty = (
                eligible_count == 0
            )

            fill_rows.append(
                {
                    "bucket_index": str(bucket_index),
                    "bucket_start_ms": str(bucket_start),
                    "bucket_end_ms": str(bucket_end),
                    "eligible_trade_window_start_ms": str(bucket_start),
                    "eligible_trade_window_end_ms": str(bucket_end),
                    "eligible_trade_count": str(eligible_count),
                    "eligible_trade_qty": _f6(eligible_qty),
                    "eligible_trade_notional": _f6(eligible_notional),
                    "eligible_first_trade_id": (
                        str(
                            int(
                                eligible[0]["agg_trade_id"]
                            )
                        )
                        if eligible
                        else ""
                    ),
                    "eligible_last_trade_id": (
                        str(
                            int(
                                eligible[-1]["agg_trade_id"]
                            )
                        )
                        if eligible
                        else ""
                    ),
                    "eligible_first_trade_time_ms": (
                        str(
                            int(
                                eligible[0]["transact_time"]
                            )
                        )
                        if eligible
                        else ""
                    ),
                    "eligible_last_trade_time_ms": (
                        str(
                            int(
                                eligible[-1]["transact_time"]
                            )
                        )
                        if eligible
                        else ""
                    ),
                    "participation_cap_qty": _f6(
                        participation_cap_qty
                    ),
                    "max_child_qty": _f6(max_child),
                    "child_order_qty": _f6(child_qty),
                    "fill_qty": _f6(child_qty),
                    "fill_price": fill_price_text,
                    "reference_bid": bid_text,
                    "reference_ask": ask_text,
                    "fill_quote_update_id": quote_id_text,
                    "fill_quote_transaction_time_ms": quote_tx_text,
                    "fill_quote_event_time_ms": quote_event_text,
                    "remaining_qty": _f6(remaining),
                    "is_empty_bucket": (
                        "true"
                        if is_empty
                        else "false"
                    ),
                }
            )

        if executed_qty <= 0.0:
            raise RuntimeError(
                "Participation TCA produced no fills."
            )

        avg_fill_price = (
            fill_notional
            / executed_qty
        )
        market_vwap = (
            eligible_notional_total
            / eligible_volume_total
        )

        fill_rate = (
            executed_qty
            / target
        )
        unfilled_qty = (
            target
            - executed_qty
        )

        implementation_shortfall_bps = (
            10000.0
            * (
                avg_fill_price
                - arrival_mid
            )
            / arrival_mid
        )
        effective_spread_bps = (
            20000.0
            * (
                avg_fill_price
                - arrival_mid
            )
            / arrival_mid
        )
        realized_spread_bps = (
            20000.0
            * (
                avg_fill_price
                - future_mid
            )
            / arrival_mid
        )
        temporary_impact_bps = (
            10000.0
            * (
                future_mid
                - arrival_mid
            )
            / arrival_mid
        )

        bucket_count_total = len(fill_rows)
        bucket_count_filled = sum(
            1
            for row in fill_rows
            if float(row["fill_qty"]) > 0.0
        )
        bucket_count_empty = sum(
            1
            for row in fill_rows
            if row["is_empty_bucket"] == "true"
        )

        result_values = {
            "schema_version": "2.0.0",
            "task_slug": "binance-btc-participation-tca",
            "symbol": str(order["symbol"]),
            "side": str(order["side"]),
            "start_time_ms": start,
            "end_time_ms": end,
            "bucket_ms": bucket_ms,
            "target_qty": _r6(target),
            "executed_qty": _r6(executed_qty),
            "unfilled_qty": _r6(unfilled_qty),
            "fill_rate": _r6(fill_rate),
            "arrival_mid": _r6(arrival_mid),
            "arrival_quote_update_id": int(
                arrival_quote["update_id"]
            ),
            "arrival_quote_transaction_time_ms": int(
                arrival_quote["transaction_time"]
            ),
            "arrival_quote_event_time_ms": int(
                arrival_quote["event_time"]
            ),
            "future_reference_time_ms": future_reference_time,
            "future_mid": _r6(future_mid),
            "future_quote_update_id": int(
                future_quote["update_id"]
            ),
            "future_quote_transaction_time_ms": int(
                future_quote["transaction_time"]
            ),
            "future_quote_event_time_ms": int(
                future_quote["event_time"]
            ),
            "avg_fill_price": _r6(avg_fill_price),
            "market_vwap": _r6(market_vwap),
            "implementation_shortfall_bps": _r6(
                implementation_shortfall_bps
            ),
            "effective_spread_bps": _r6(
                effective_spread_bps
            ),
            "realized_spread_bps": _r6(
                realized_spread_bps
            ),
            "temporary_impact_bps": _r6(
                temporary_impact_bps
            ),
            "market_trade_count": eligible_count_total,
            "market_volume": _r6(
                eligible_volume_total
            ),
            "bucket_count_total": bucket_count_total,
            "bucket_count_filled": bucket_count_filled,
            "bucket_count_empty": bucket_count_empty,
        }

        # Rebuild in normative order explicitly.
        results = {
            key: result_values[key]
            for key in RESULT_KEYS
        }

        if first_fill_quote_time is None or last_fill_quote_time is None:
            raise RuntimeError(
                "Missing fill quote timestamps."
            )

        def value_box(value):
            return {"value": value}

        checkpoints = {}
        for index in (0, 4, 9):
            if index >= len(fill_rows):
                continue

            row = fill_rows[index]
            checkpoints[
                f"bucket_{index}"
            ] = {
                "bucket_start_ms": value_box(
                    int(row["bucket_start_ms"])
                ),
                "eligible_trade_count": value_box(
                    int(row["eligible_trade_count"])
                ),
                "eligible_trade_qty": value_box(
                    _r6(
                        float(
                            row["eligible_trade_qty"]
                        )
                    )
                ),
                "child_order_qty": value_box(
                    _r6(
                        float(
                            row["child_order_qty"]
                        )
                    )
                ),
                "remaining_qty": value_box(
                    _r6(
                        float(
                            row["remaining_qty"]
                        )
                    )
                ),
                "is_empty_bucket": value_box(
                    row["is_empty_bucket"]
                    == "true"
                ),
            }

        solution = {
            "intermediates": {
                "bucket_count_total": value_box(
                    bucket_count_total
                ),
                "bucket_count_filled": value_box(
                    bucket_count_filled
                ),
                "bucket_count_empty": value_box(
                    bucket_count_empty
                ),
                "eligible_trade_count": value_box(
                    eligible_count_total
                ),
                "eligible_trade_volume": value_box(
                    _r6(
                        eligible_volume_total
                    )
                ),
                "eligible_trade_notional": value_box(
                    _r6(
                        eligible_notional_total
                    )
                ),
                "first_fill_quote_transaction_time_ms": value_box(
                    first_fill_quote_time
                ),
                "last_fill_quote_transaction_time_ms": value_box(
                    last_fill_quote_time
                ),
                "arrival_quote_transaction_time_ms": value_box(
                    int(
                        arrival_quote[
                            "transaction_time"
                        ]
                    )
                ),
                "arrival_quote_event_time_ms": value_box(
                    int(
                        arrival_quote[
                            "event_time"
                        ]
                    )
                ),
                "future_quote_transaction_time_ms": value_box(
                    int(
                        future_quote[
                            "transaction_time"
                        ]
                    )
                ),
                "future_quote_event_time_ms": value_box(
                    int(
                        future_quote[
                            "event_time"
                        ]
                    )
                ),
            },
            "checkpoints": checkpoints,
        }

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            out_dir
            / "results.json"
        ).write_text(
            json.dumps(
                results,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        with (
            out_dir
            / "fills.csv"
        ).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=FILL_COLUMNS,
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(fill_rows)

        (
            out_dir
            / "solution.json"
        ).write_text(
            json.dumps(
                solution,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
