from __future__ import annotations

import numpy as np
import pytest

from agent.offline_common import execution as extracted
from agent.offline_stable_residual import (
    _turnover_one_way as mature_one_way_turnover,
)


def test_one_way_turnover_matches_mature_implementation() -> None:
    previous = {
        101: 0.25,
        102: -0.25,
        103: 0.50,
    }
    current = {
        101: 0.10,
        103: 0.35,
        104: -0.20,
    }

    expected = mature_one_way_turnover(
        previous,
        current,
    )

    actual = extracted.one_way_turnover(
        previous,
        current,
    )

    assert np.isclose(
        actual,
        expected,
        rtol=0.0,
        atol=1e-15,
    )


def test_two_way_turnover_is_twice_one_way() -> None:
    previous = {
        "A": 0.5,
        "B": 0.5,
    }
    current = {
        "A": 0.2,
        "C": 0.8,
    }

    two_way = extracted.two_way_turnover(
        previous,
        current,
    )
    one_way = extracted.one_way_turnover(
        previous,
        current,
    )

    assert np.isclose(
        two_way,
        2.0 * one_way,
    )


def test_two_way_turnover_handles_entries_and_exits() -> None:
    previous = {
        "A": 0.6,
        "B": 0.4,
    }
    current = {
        "B": 0.25,
        "C": 0.75,
    }

    expected = (
        abs(0.0 - 0.6)
        + abs(0.25 - 0.4)
        + abs(0.75 - 0.0)
    )

    assert np.isclose(
        extracted.two_way_turnover(
            previous,
            current,
        ),
        expected,
    )


def test_transaction_cost_matches_double_sort_convention() -> None:
    turnover = 0.8
    gross_return = 0.035
    rate = 0.0015

    expected = (
        gross_return
        - rate * turnover
    )

    actual = (
        extracted.net_return_after_cost(
            gross_return,
            turnover,
            rate,
        )
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_transaction_cost_rejects_negative_inputs() -> None:
    with pytest.raises(
        ValueError,
        match="turnover",
    ):
        extracted.transaction_cost(
            -0.1,
            0.001,
        )

    with pytest.raises(
        ValueError,
        match="rate",
    ):
        extracted.transaction_cost(
            0.1,
            -0.001,
        )


def test_lag_positions_one_period() -> None:
    weights = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=float,
    )

    expected = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )

    actual = extracted.lag_positions(
        weights,
        periods=1,
    )

    assert np.array_equal(
        actual,
        expected,
    )


def test_lag_positions_multiple_periods() -> None:
    values = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
        ]
    )

    actual = extracted.lag_positions(
        values,
        periods=2,
        fill_value=-1.0,
    )

    assert np.array_equal(
        actual,
        np.array(
            [
                -1.0,
                -1.0,
                1.0,
                2.0,
            ]
        ),
    )


def test_lag_positions_zero_returns_copy() -> None:
    weights = np.array(
        [
            [0.5, 0.5],
            [0.3, 0.7],
        ]
    )

    actual = extracted.lag_positions(
        weights,
        periods=0,
    )

    assert np.array_equal(
        actual,
        weights,
    )
    assert actual is not weights


def test_rowwise_portfolio_returns() -> None:
    weights = np.array(
        [
            [0.5, 0.5],
            [1.0, -1.0],
        ]
    )

    returns = np.array(
        [
            [0.10, -0.02],
            [0.03, -0.01],
        ]
    )

    actual = (
        extracted.rowwise_portfolio_returns(
            weights,
            returns,
        )
    )

    expected = np.array(
        [
            0.04,
            0.04,
        ]
    )

    assert np.allclose(
        actual,
        expected,
    )


def test_execution_lag_prevents_same_period_signal_use() -> None:
    target_weights = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )

    asset_returns = np.array(
        [
            [0.50, 0.00],
            [0.10, 0.20],
            [-0.10, 0.30],
        ]
    )

    held_weights = (
        extracted.lag_positions(
            target_weights,
            periods=1,
        )
    )

    portfolio_returns = (
        extracted.rowwise_portfolio_returns(
            held_weights,
            asset_returns,
        )
    )

    assert np.allclose(
        portfolio_returns,
        np.array(
            [
                0.0,
                0.10,
                0.30,
            ]
        ),
    )


def test_nonfinite_weight_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        extracted.two_way_turnover(
            {"A": 1.0},
            {"A": np.nan},
        )


from pathlib import Path

from agent.candidate_runner import run_candidate
from agent.candidate_workspace import CandidateWorkspace
from agent.capability_bridge import (
    CANDIDATE_LIBRARY_EXPORTS,
    rank_capabilities,
)


def _task(
    tmp_path: Path,
    instruction: str,
) -> Path:
    task = tmp_path / "task"
    data = task / "environment" / "data"
    data.mkdir(
        parents=True,
    )

    (
        task / "instruction.md"
    ).write_text(
        instruction,
        encoding="utf-8",
    )

    return task


def _load_curated_execution(
    tmp_path: Path,
):
    task = _task(
        tmp_path,
        "Evaluate portfolio execution.",
    )

    workspace = CandidateWorkspace.create(
        base_dir=tmp_path / "workspaces",
        candidate_id=1,
        task_dir=task,
    )

    module_path = (
        workspace.root_dir
        / "lib"
        / "offline_common"
        / "execution.py"
    )

    namespace = {}

    exec(
        compile(
            module_path.read_text(
                encoding="utf-8",
            ),
            str(module_path),
            "exec",
        ),
        namespace,
    )

    return (
        workspace,
        namespace,
    )


def test_curated_execution_has_exact_surface(
    tmp_path: Path,
) -> None:
    workspace, namespace = (
        _load_curated_execution(
            tmp_path
        )
    )

    try:
        assert tuple(
            namespace["__all__"]
        ) == (
            CANDIDATE_LIBRARY_EXPORTS[
                "execution.py"
            ]
        )

        assert callable(
            namespace[
                "two_way_turnover"
            ]
        )
        assert callable(
            namespace[
                "lag_positions"
            ]
        )

    finally:
        workspace.cleanup()


def test_execution_routing_is_specific(
    tmp_path: Path,
) -> None:
    positive = _task(
        tmp_path / "positive",
        (
            "Apply a one-day execution lag and "
            "deduct transaction costs from "
            "portfolio returns."
        ),
    )

    selected = rank_capabilities(
        instruction=(
            positive
            / "instruction.md"
        ).read_text(),
        task_dir=positive,
    )

    assert (
        "causal-portfolio-execution"
        in {
            item.descriptor.capability_id
            for item in selected
        }
    )

    negative = _task(
        tmp_path / "negative",
        "Analyse the supplied observations.",
    )

    selected_negative = (
        rank_capabilities(
            instruction=(
                negative
                / "instruction.md"
            ).read_text(),
            task_dir=negative,
        )
    )

    assert (
        "causal-portfolio-execution"
        not in {
            item.descriptor.capability_id
            for item
            in selected_negative
        }
    )


def test_candidate_executes_execution_helpers(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path,
        "Evaluate portfolio execution.",
    )

    workspace = CandidateWorkspace.create(
        base_dir=tmp_path / "workspaces",
        candidate_id=1,
        task_dir=task,
    )

    try:
        script = '''
import json
import numpy as np

from offline_common.execution import (
    lag_positions,
    one_way_turnover,
    rowwise_portfolio_returns,
    transaction_cost,
)

previous = {
    "A": 0.5,
    "B": 0.5,
}

current = {
    "A": 0.2,
    "C": 0.8,
}

weights = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
])

returns = np.array([
    [0.5, 0.0],
    [0.1, 0.2],
    [-0.1, 0.3],
])

held = lag_positions(
    weights,
    periods=1,
)

portfolio = rowwise_portfolio_returns(
    held,
    returns,
)

payload = {
    "turnover": one_way_turnover(
        previous,
        current,
    ),
    "cost": transaction_cost(
        0.4,
        0.0015,
    ),
    "returns": portfolio.tolist(),
}

with open(
    "output/execution_probe.json",
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        payload,
        handle,
    )
'''

        script_path = (
            workspace.source_dir
            / "solver.py"
        )

        script_path.write_text(
            script.strip()
            + "\n",
            encoding="utf-8",
        )

        completed = run_candidate(
            workspace,
            script_path,
            timeout_seconds=30,
        )

        assert completed.return_code == 0

        import json

        payload = json.loads(
            (
                workspace.output_dir
                / "execution_probe.json"
            ).read_text(
                encoding="utf-8",
            )
        )

        assert np.isclose(
            payload["turnover"],
            0.8,
        )

        assert np.isclose(
            payload["cost"],
            0.0006,
        )

        assert np.allclose(
            payload["returns"],
            [
                0.0,
                0.1,
                0.3,
            ],
        )

    finally:
        workspace.cleanup()
