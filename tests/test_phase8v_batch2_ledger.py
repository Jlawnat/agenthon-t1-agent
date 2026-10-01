from __future__ import annotations

import numpy as np
import pytest

from agent.offline_common import ledger as extracted
from agent.offline_delta_hedging import (
    _transaction_cost as mature_transaction_cost,
)


def test_proportional_cost_matches_delta_hedging_buy() -> None:
    fee_rates = {
        "buy": 0.0012,
        "sell": 0.0018,
    }

    expected = mature_transaction_cost(
        share_change=5.0,
        spot=101.5,
        fee_rates=fee_rates,
    )

    actual = extracted.proportional_trade_cost(
        5.0,
        101.5,
        buy_rate=fee_rates["buy"],
        sell_rate=fee_rates["sell"],
    )

    assert np.isclose(
        actual,
        expected,
        rtol=0.0,
        atol=1e-15,
    )


def test_proportional_cost_matches_delta_hedging_sell() -> None:
    fee_rates = {
        "buy": 0.0012,
        "sell": 0.0018,
    }

    expected = mature_transaction_cost(
        share_change=-3.5,
        spot=98.0,
        fee_rates=fee_rates,
    )

    actual = extracted.proportional_trade_cost(
        -3.5,
        98.0,
        buy_rate=fee_rates["buy"],
        sell_rate=fee_rates["sell"],
    )

    assert np.isclose(
        actual,
        expected,
        rtol=0.0,
        atol=1e-15,
    )


def test_apply_trade_matches_delta_cash_identity() -> None:
    cash = 1000.0
    quantity = 2.0
    change = 3.0
    price = 50.0
    rate = 0.001

    expected_cost = (
        abs(change * price)
        * rate
    )

    expected_cash = (
        cash
        - change * price
        - expected_cost
    )

    result = extracted.apply_trade(
        cash=cash,
        quantity=quantity,
        quantity_change=change,
        price=price,
        buy_rate=rate,
    )

    assert np.isclose(
        result.quantity,
        5.0,
    )
    assert np.isclose(
        result.cash,
        expected_cash,
    )
    assert np.isclose(
        result.transaction_cost,
        expected_cost,
    )


def test_sell_trade_adds_proceeds_less_cost() -> None:
    result = extracted.apply_trade(
        cash=100.0,
        quantity=10.0,
        quantity_change=-4.0,
        price=20.0,
        sell_rate=0.01,
    )

    assert np.isclose(
        result.quantity,
        6.0,
    )

    assert np.isclose(
        result.cash,
        100.0
        + 80.0
        - 0.8,
    )


def test_mark_to_market_matches_cash_plus_positions() -> None:
    cash = 125.0

    quantities = np.array(
        [
            10.0,
            -5.0,
        ]
    )

    prices = np.array(
        [
            20.0,
            8.0,
        ]
    )

    expected = (
        cash
        + 10.0 * 20.0
        - 5.0 * 8.0
    )

    assert np.isclose(
        extracted.mark_to_market(
            cash,
            quantities,
            prices,
        ),
        expected,
    )


def test_mark_to_market_matches_time_series_identity() -> None:
    cash = 400.0
    shares = 7.0
    close = 35.0

    expected = (
        cash
        + shares * close
    )

    actual = extracted.mark_to_market(
        cash,
        np.array([shares]),
        np.array([close]),
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_position_pnl_long() -> None:
    assert np.isclose(
        extracted.position_pnl(
            10.0,
            100.0,
            112.0,
            total_cost=4.0,
        ),
        116.0,
    )


def test_position_pnl_short() -> None:
    assert np.isclose(
        extracted.position_pnl(
            -10.0,
            100.0,
            90.0,
            total_cost=3.0,
        ),
        97.0,
    )


def test_round_trip_cash_reconciles_with_pnl() -> None:
    initial_cash = 1000.0
    quantity = 5.0
    entry = 100.0
    exit_price = 110.0
    rate = 0.001

    buy = extracted.apply_trade(
        cash=initial_cash,
        quantity=0.0,
        quantity_change=quantity,
        price=entry,
        buy_rate=rate,
        sell_rate=rate,
    )

    sell = extracted.apply_trade(
        cash=buy.cash,
        quantity=buy.quantity,
        quantity_change=-quantity,
        price=exit_price,
        buy_rate=rate,
        sell_rate=rate,
    )

    total_cost = (
        buy.transaction_cost
        + sell.transaction_cost
    )

    expected_pnl = (
        extracted.position_pnl(
            quantity,
            entry,
            exit_price,
            total_cost=total_cost,
        )
    )

    assert np.isclose(
        sell.quantity,
        0.0,
    )

    assert np.isclose(
        sell.cash
        - initial_cash,
        expected_pnl,
    )


def test_nonfinite_inputs_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        extracted.apply_trade(
            cash=np.nan,
            quantity=0.0,
            quantity_change=1.0,
            price=10.0,
        )


def test_negative_fee_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="fee rates",
    ):
        extracted.proportional_trade_cost(
            1.0,
            10.0,
            buy_rate=-0.01,
        )


from pathlib import Path
import json

from agent.candidate_runner import run_candidate
from agent.candidate_workspace import CandidateWorkspace
from agent.capability_bridge import rank_capabilities


def _task(
    tmp_path: Path,
    instruction: str,
) -> Path:
    task = tmp_path / "task"

    (
        task
        / "environment"
        / "data"
    ).mkdir(
        parents=True,
    )

    (
        task
        / "instruction.md"
    ).write_text(
        instruction,
        encoding="utf-8",
    )

    return task


def test_cash_position_ledger_routing(
    tmp_path: Path,
) -> None:
    cases = (
        (
            "momentum",
            (
                "Track daily portfolio value. "
                "When flat, portfolio value = cash. "
                "When invested, use cash + shares times close."
            ),
        ),
        (
            "delta",
            (
                "Liquidate the share position and report "
                "the terminal cash balance."
            ),
        ),
        (
            "event",
            (
                "Build the Cash, Equity, and Daily P&L "
                "ledger through time."
            ),
        ),
    )

    for name, instruction in cases:
        task = _task(
            tmp_path / name,
            instruction,
        )

        selected = rank_capabilities(
            instruction=instruction,
            task_dir=task,
        )

        assert (
            "cash-position-ledger"
            in {
                item.descriptor.capability_id
                for item in selected
            }
        )


def test_mark_to_market_phrase_alone_does_not_select_ledger(
    tmp_path: Path,
) -> None:
    instruction = (
        "Price and risk-manage a mark-to-market "
        "cross-currency swap and value its future cashflows."
    )

    task = _task(
        tmp_path,
        instruction,
    )

    selected = rank_capabilities(
        instruction=instruction,
        task_dir=task,
    )

    assert (
        "cash-position-ledger"
        not in {
            item.descriptor.capability_id
            for item in selected
        }
    )


def test_candidate_executes_ledger_helpers(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path,
        "Maintain a cash and position ledger.",
    )

    workspace = CandidateWorkspace.create(
        base_dir=tmp_path / "workspaces",
        candidate_id=1,
        task_dir=task,
    )

    try:
        script_path = (
            workspace.source_dir
            / "solver.py"
        )

        script_path.write_text(
            '''
import json
import numpy as np

from offline_common.ledger import (
    apply_trade,
    mark_to_market,
    position_pnl,
)

buy = apply_trade(
    cash=1000.0,
    quantity=0.0,
    quantity_change=5.0,
    price=100.0,
    buy_rate=0.001,
    sell_rate=0.001,
)

nav = mark_to_market(
    buy.cash,
    np.array([buy.quantity]),
    np.array([110.0]),
)

pnl = position_pnl(
    5.0,
    100.0,
    110.0,
    total_cost=0.5,
)

payload = {
    "quantity": buy.quantity,
    "cash": buy.cash,
    "nav": nav,
    "pnl": pnl,
}

with open(
    "output/ledger_probe.json",
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        payload,
        handle,
    )
'''.strip()
            + "\n",
            encoding="utf-8",
        )

        completed = run_candidate(
            workspace,
            script_path,
            timeout_seconds=30,
        )

        assert completed.return_code == 0

        payload = json.loads(
            (
                workspace.output_dir
                / "ledger_probe.json"
            ).read_text(
                encoding="utf-8",
            )
        )

        assert np.isclose(
            payload["quantity"],
            5.0,
        )

        assert np.isclose(
            payload["cash"],
            499.5,
        )

        assert np.isclose(
            payload["nav"],
            1049.5,
        )

        assert np.isclose(
            payload["pnl"],
            49.5,
        )

    finally:
        workspace.cleanup()
