from __future__ import annotations

import csv
import json
from pathlib import Path

from agent.offline_tca import (
    BinanceParticipationTcaSkill,
    RESULT_KEYS,
)


def _write_task(root: Path) -> Path:
    task = root / "task"
    data = task / "environment" / "data"
    data.mkdir(parents=True)

    (
        task
        / "instruction.md"
    ).write_text(
        """
# BTC participation-capped execution
Compute implementation_shortfall_bps and realized_spread_bps.
Write fills.csv and solution.json for BTC.
""".strip(),
        encoding="utf-8",
    )

    order = {
        "symbol": "BTCUSDT",
        "side": "buy",
        "start_time_ms": 1000,
        "end_time_ms": 3000,
        "bucket_ms": 1000,
        "target_quantity": 2.0,
        "participation_cap": 0.5,
        "max_child_quantity": 1.0,
        "realized_horizon_ms": 500,
    }
    (
        data
        / "order_spec.json"
    ).write_text(
        json.dumps(order),
        encoding="utf-8",
    )

    (
        data
        / "tca_rules.json"
    ).write_text(
        json.dumps(
            {"normative": True}
        ),
        encoding="utf-8",
    )

    with (
        data
        / "quotes.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "update_id",
                "best_bid_price",
                "best_ask_price",
                "transaction_time",
                "event_time",
            ]
        )
        writer.writerow(
            [
                1,
                99.0,
                101.0,
                1000,
                1001,
            ]
        )
        writer.writerow(
            [
                2,
                100.0,
                102.0,
                2000,
                2001,
            ]
        )
        writer.writerow(
            [
                3,
                101.0,
                103.0,
                3500,
                3501,
            ]
        )

    with (
        data
        / "trades.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "agg_trade_id",
                "price",
                "quantity",
                "transact_time",
            ]
        )
        writer.writerow(
            [10, 100.0, 1.0, 1100]
        )
        writer.writerow(
            [11, 101.0, 3.0, 2100]
        )

    return task


def test_tca_skill_contract(tmp_path: Path) -> None:
    task = _write_task(tmp_path)
    out = tmp_path / "out"
    instruction = (
        task
        / "instruction.md"
    ).read_text(encoding="utf-8")

    skill = BinanceParticipationTcaSkill()

    assert skill.matches(
        instruction=instruction,
        task_dir=task,
    )

    skill.solve(
        instruction=instruction,
        task_dir=task,
        out_dir=out,
        seed=42,
    )

    assert {
        path.name
        for path in out.iterdir()
    } == {
        "results.json",
        "fills.csv",
        "solution.json",
    }

    results = json.loads(
        (
            out
            / "results.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    assert list(results) == list(RESULT_KEYS)
    assert results["executed_qty"] == 1.5
    assert results["unfilled_qty"] == 0.5
    assert results["bucket_count_total"] == 2
    assert results["bucket_count_filled"] == 2
    assert results["bucket_count_empty"] == 0

    with (
        out
        / "fills.csv"
    ).open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(
            csv.DictReader(handle)
        )

    assert len(rows) == 2
    assert rows[0]["fill_qty"] == "0.500000"
    assert rows[1]["fill_qty"] == "1.000000"
    assert rows[1]["remaining_qty"] == "0.500000"

    solution = json.loads(
        (
            out
            / "solution.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert set(solution) == {
        "intermediates",
        "checkpoints",
    }
