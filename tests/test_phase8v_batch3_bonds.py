from __future__ import annotations

import numpy as np
import pytest

from agent.offline_common import bonds as extracted
from agent.offline_curve_immunization import (
    _discount_cash_flows as mature_discount_cash_flows,
    _z_spread as mature_z_spread,
)
from agent.offline_dated_curve_immunization import (
    _parallel_duration_convexity as mature_parallel_duration_convexity,
    _yield_and_durations as mature_yield_and_durations,
)


def test_discounted_cashflow_price_matches_mature() -> None:
    times = np.array(
        [1, 2, 3],
        dtype=int,
    )

    cash = np.array(
        [5.0, 5.0, 105.0]
    )

    full_curve = np.array(
        [0.03, 0.035, 0.04]
    )

    expected = mature_discount_cash_flows(
        times,
        cash,
        full_curve,
    )

    actual = extracted.discounted_cashflow_price(
        times.astype(float),
        cash,
        full_curve,
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_yield_and_durations_matches_mature() -> None:
    times = np.array(
        [0.5, 1.0, 1.5, 2.0]
    )

    cash = np.array(
        [2.5, 2.5, 2.5, 102.5]
    )

    price = 99.25

    expected = mature_yield_and_durations(
        price=price,
        times=times,
        cash_flows=cash,
        frequency=2,
    )

    actual = extracted.yield_and_durations(
        price=price,
        times=times,
        cash_flows=cash,
        frequency=2,
    )

    assert np.allclose(
        [
            actual.yield_to_maturity,
            actual.macaulay_duration,
            actual.modified_duration,
        ],
        expected,
    )


def test_z_spread_matches_mature() -> None:
    times = np.array(
        [1, 2, 3],
        dtype=int,
    )

    cash = np.array(
        [5.0, 5.0, 105.0]
    )

    full_curve = np.array(
        [0.03, 0.035, 0.04]
    )

    target_spread = 0.0125

    price = float(
        np.sum(
            cash
            * np.exp(
                -(
                    full_curve
                    + target_spread
                )
                * times
            )
        )
    )

    expected = mature_z_spread(
        price,
        times,
        cash,
        full_curve,
    )

    actual = (
        extracted.z_spread_from_continuous_curve(
            price=price,
            times=times.astype(float),
            cash_flows=cash,
            zero_rates=full_curve,
        )
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_parallel_duration_convexity_matches_mature() -> None:
    expected = mature_parallel_duration_convexity(
        base_price=100.0,
        price_up=99.95,
        price_down=100.05,
        bump=1e-4,
    )

    actual = extracted.parallel_duration_convexity(
        base_price=100.0,
        price_up=99.95,
        price_down=100.05,
        bump=1e-4,
    )

    assert np.allclose(
        actual,
        expected,
    )


def test_symmetric_key_rate_durations() -> None:
    actual = (
        extracted.symmetric_key_rate_durations(
            base_value=100.0,
            values_up=np.array(
                [99.99, 99.98, 99.97]
            ),
            values_down=np.array(
                [100.01, 100.02, 100.03]
            ),
            bump=1e-4,
        )
    )

    assert np.allclose(
        actual,
        np.array(
            [1.0, 2.0, 3.0]
        ),
    )


def test_dv01_from_duration() -> None:
    assert np.isclose(
        extracted.dv01_from_duration(
            1_000_000.0,
            4.5,
        ),
        450.0,
    )


def test_shape_mismatch_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="identical shapes",
    ):
        extracted.discounted_cashflow_price(
            [1.0, 2.0],
            [5.0],
            [0.03, 0.04],
        )


def test_invalid_frequency_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        extracted.yield_and_durations(
            price=100.0,
            times=[1.0],
            cash_flows=[105.0],
            frequency=0,
        )


def test_invalid_bump_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="bump",
    ):
        extracted.parallel_duration_convexity(
            base_price=100.0,
            price_up=99.0,
            price_down=101.0,
            bump=0.0,
        )


from pathlib import Path
import json

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


def test_bond_risk_routing_is_specific(
    tmp_path: Path,
) -> None:
    positive_cases = (
        (
            "immunization",
            (
                "Construct a bond immunization hedge and "
                "measure key-rate duration exposures."
            ),
        ),
        (
            "analytics",
            (
                "Report Macaulay duration, modified duration, "
                "and z-spread for the bond."
            ),
        ),
    )

    for name, instruction in positive_cases:
        task = _task(
            tmp_path / name,
            instruction,
        )

        selected = rank_capabilities(
            instruction=instruction,
            task_dir=task,
        )

        assert (
            "bond-risk-analytics"
            in {
                item.descriptor.capability_id
                for item in selected
            }
        )

    negative_instruction = (
        "Price a Hull-White swaption and report its DV01."
    )

    negative_task = _task(
        tmp_path / "negative",
        negative_instruction,
    )

    selected = rank_capabilities(
        instruction=negative_instruction,
        task_dir=negative_task,
    )

    assert (
        "bond-risk-analytics"
        not in {
            item.descriptor.capability_id
            for item in selected
        }
    )


def test_curated_bonds_has_exact_surface(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path,
        "Evaluate bond risk analytics.",
    )

    workspace = CandidateWorkspace.create(
        base_dir=tmp_path / "workspaces",
        candidate_id=1,
        task_dir=task,
    )

    try:
        module_path = (
            workspace.root_dir
            / "lib"
            / "offline_common"
            / "bonds.py"
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

        assert tuple(
            namespace["__all__"]
        ) == (
            CANDIDATE_LIBRARY_EXPORTS[
                "bonds.py"
            ]
        )

    finally:
        workspace.cleanup()


def test_candidate_executes_bond_helpers(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path,
        "Evaluate bond risk analytics.",
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

from offline_common.bonds import (
    discounted_cashflow_price,
    dv01_from_duration,
    yield_and_durations,
)

times = np.array(
    [0.5, 1.0, 1.5, 2.0]
)

cash = np.array(
    [2.5, 2.5, 2.5, 102.5]
)

rates = np.array(
    [0.03, 0.032, 0.034, 0.036]
)

price = discounted_cashflow_price(
    times,
    cash,
    rates,
)

analytics = yield_and_durations(
    price=price,
    times=times,
    cash_flows=cash,
    frequency=2,
)

payload = {
    "price_positive": price > 0.0,
    "ytm_finite": bool(
        np.isfinite(
            analytics.yield_to_maturity
        )
    ),
    "dv01_positive": (
        dv01_from_duration(
            price,
            analytics.modified_duration,
        )
        > 0.0
    ),
}

with open(
    "output/bond_probe.json",
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
                / "bond_probe.json"
            ).read_text(
                encoding="utf-8",
            )
        )

        assert payload == {
            "price_positive": True,
            "ytm_finite": True,
            "dv01_positive": True,
        }

    finally:
        workspace.cleanup()
