from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from agent.candidate_runner import run_candidate
from agent.candidate_workspace import CandidateWorkspace
from agent.capability_bridge import (
    CANDIDATE_LIBRARY_EXPORTS,
    rank_capabilities,
)
from agent.offline_common import invariants as extracted
from agent.quant_invariants import (
    QuantInvariantConfig,
    evaluate_quant_invariants,
)


def _task(tmp_path: Path, instruction: str) -> Path:
    task = tmp_path / "task"
    (task / "environment" / "data").mkdir(parents=True)
    (task / "instruction.md").write_text(
        instruction,
        encoding="utf-8",
    )
    return task


def _by_name(evidence):
    return {
        item.name: item
        for item in evidence
    }


def _evaluate(
    tmp_path: Path,
    frame: pd.DataFrame,
    *,
    config: QuantInvariantConfig,
):
    path = tmp_path / "result.csv"
    frame.to_csv(
        path,
        index=False,
    )
    return evaluate_quant_invariants(
        output_path=path,
        config=config,
    )


def _load_curated_invariants(
    tmp_path: Path,
):
    task = _task(
        tmp_path,
        (
            "Validate probability bounds, causal timestamps "
            "and accounting reconciliation."
        ),
    )

    workspace = CandidateWorkspace.create(
        base_dir=tmp_path / "work",
        candidate_id=1,
        task_dir=task,
    )

    path = (
        workspace.root_dir
        / "lib"
        / "offline_common"
        / "invariants.py"
    )

    spec = importlib.util.spec_from_file_location(
        "_phase8v_curated_invariants",
        path,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return workspace, module


def test_finite_mask_matches_mature_violation_count(
    tmp_path: Path,
) -> None:
    values = np.asarray(
        [
            1.0,
            np.inf,
            -2.0,
            np.nan,
        ]
    )

    mask = extracted.finite_numeric_mask(
        values
    )

    np.testing.assert_array_equal(
        mask,
        [
            True,
            False,
            True,
            False,
        ],
    )

    evidence = _evaluate(
        tmp_path,
        pd.DataFrame(
            {
                "price": values,
            }
        ),
        config=QuantInvariantConfig(
            category="fixed-income"
        ),
    )

    mature = _by_name(
        evidence
    )["finite_numeric_values"]

    assert mature.status == "fail"

    assert (
        mature.details[
            "violations"
        ]["price"]["violation_count"]
        == int((~mask).sum())
    )


def test_probability_bounds_match_mature_behavior(
    tmp_path: Path,
) -> None:
    values = np.asarray(
        [
            0.0,
            0.25,
            1.0,
            1.20,
            -0.10,
        ]
    )

    mask = extracted.probability_bounds_mask(
        values,
        upper=1.0,
        tolerance=0.0,
    )

    np.testing.assert_array_equal(
        mask,
        [
            True,
            True,
            True,
            False,
            False,
        ],
    )

    evidence = _evaluate(
        tmp_path,
        pd.DataFrame(
            {
                "default_probability": values,
            }
        ),
        config=QuantInvariantConfig(
            category="credit"
        ),
    )

    mature = _by_name(
        evidence
    )["probability_bounds"]

    assert mature.status == "fail"

    assert (
        mature.details["violations"][
            "default_probability"
        ]["violation_count"]
        == int((~mask).sum())
    )


def test_probability_percentage_upper_bound() -> None:
    values = np.asarray(
        [
            0.0,
            25.0,
            100.0,
            101.0,
        ]
    )

    np.testing.assert_array_equal(
        extracted.probability_bounds_mask(
            values,
            upper=100.0,
        ),
        [
            True,
            True,
            True,
            False,
        ],
    )


def test_psd_validation_accepts_valid_covariance() -> None:
    covariance = np.asarray(
        [
            [0.04, 0.01],
            [0.01, 0.09],
        ],
        dtype=float,
    )

    assert extracted.is_positive_semidefinite(
        covariance
    )


def test_psd_validation_rejects_indefinite_matrix() -> None:
    indefinite = np.asarray(
        [
            [1.0, 2.0],
            [2.0, 1.0],
        ],
        dtype=float,
    )

    assert not extracted.is_positive_semidefinite(
        indefinite
    )


def test_psd_validation_rejects_nonsymmetric_matrix() -> None:
    matrix = np.asarray(
        [
            [1.0, 0.2],
            [0.1, 1.0],
        ],
        dtype=float,
    )

    assert not extracted.is_positive_semidefinite(
        matrix
    )


def test_weight_sum_error_properties() -> None:
    assert np.isclose(
        extracted.weight_sum_error(
            [0.2, 0.3, 0.5]
        ),
        0.0,
    )

    assert np.isclose(
        extracted.weight_sum_error(
            [0.2, 0.3, 0.4]
        ),
        -0.1,
    )

    assert np.isclose(
        extracted.weight_sum_error(
            [0.25, -0.25],
            target=0.0,
        ),
        0.0,
    )


def test_causal_order_matches_mature_invariant(
    tmp_path: Path,
) -> None:
    information = [
        "2026-01-01T09:00:00Z",
        "2026-01-01T10:00:00Z",
    ]

    signal = [
        "2026-01-01T09:01:00Z",
        "2026-01-01T10:01:00Z",
    ]

    execution = [
        "2026-01-01T09:02:00Z",
        "2026-01-01T10:01:00Z",
    ]

    info_signal = extracted.causal_order_mask(
        information,
        signal,
        strict=False,
    )

    signal_execution = (
        extracted.causal_order_mask(
            signal,
            execution,
            strict=True,
        )
    )

    np.testing.assert_array_equal(
        info_signal,
        [True, True],
    )

    np.testing.assert_array_equal(
        signal_execution,
        [True, False],
    )

    evidence = _evaluate(
        tmp_path,
        pd.DataFrame(
            {
                "information_time": information,
                "signal_time": signal,
                "execution_time": execution,
            }
        ),
        config=QuantInvariantConfig(
            category="backtesting",
            review_packs=(
                "data-causality",
            ),
        ),
    )

    mature = _by_name(
        evidence
    )["time_causality"]

    assert mature.status == "fail"

    assert (
        mature.details[
            "ordering_failure_count"
        ]
        == int(
            (~signal_execution).sum()
        )
    )


def test_causal_order_rejects_unparseable_timestamps() -> None:
    mask = extracted.causal_order_mask(
        [
            "2026-01-01",
            "not-a-date",
        ],
        [
            "2026-01-02",
            "2026-01-03",
        ],
    )

    np.testing.assert_array_equal(
        mask,
        [
            True,
            False,
        ],
    )


def test_nav_reconciliation_matches_mature_engine(
    tmp_path: Path,
) -> None:
    nav = np.asarray(
        [
            150.0,
            205.0,
        ]
    )

    components = np.asarray(
        [
            [50.0, 100.0],
            [80.0, 120.0],
        ]
    )

    mask = extracted.reconciliation_close(
        nav,
        components,
    )

    np.testing.assert_array_equal(
        mask,
        [
            True,
            False,
        ],
    )

    evidence = _evaluate(
        tmp_path,
        pd.DataFrame(
            {
                "nav": nav,
                "cash": components[:, 0],
                "market_value": components[:, 1],
            }
        ),
        config=QuantInvariantConfig(
            category="backtesting",
            review_packs=(
                "accounting",
            ),
        ),
    )

    mature = _by_name(
        evidence
    )["nav_reconciliation"]

    assert mature.status == "fail"

    assert (
        mature.details[
            "violation_count"
        ]
        == int((~mask).sum())
    )


def test_pnl_reconciliation_matches_mature_engine(
    tmp_path: Path,
) -> None:
    total = np.asarray(
        [
            12.0,
            9.0,
        ]
    )

    components = np.asarray(
        [
            [5.0, 7.0],
            [4.0, 4.0],
        ]
    )

    mask = extracted.reconciliation_close(
        total,
        components,
    )

    np.testing.assert_array_equal(
        mask,
        [
            True,
            False,
        ],
    )

    evidence = _evaluate(
        tmp_path,
        pd.DataFrame(
            {
                "total_pnl": total,
                "realized_pnl": components[:, 0],
                "unrealized_pnl": components[:, 1],
            }
        ),
        config=QuantInvariantConfig(
            category="backtesting",
            review_packs=(
                "accounting",
            ),
        ),
    )

    mature = _by_name(
        evidence
    )["pnl_reconciliation"]

    assert mature.status == "fail"


def test_curated_invariants_have_exact_surface(
    tmp_path: Path,
) -> None:
    workspace, curated = (
        _load_curated_invariants(
            tmp_path
        )
    )

    try:
        assert tuple(curated.__all__) == (
            CANDIDATE_LIBRARY_EXPORTS[
                "invariants.py"
            ]
        )

        assert curated.is_positive_semidefinite(
            np.eye(3)
        )

        mask = curated.probability_bounds_mask(
            [0.1, 0.5, 1.2]
        )

        np.testing.assert_array_equal(
            mask,
            [
                True,
                True,
                False,
            ],
        )

    finally:
        workspace.cleanup()


def test_invariant_routing_is_specific(
    tmp_path: Path,
) -> None:
    positive = _task(
        tmp_path / "positive",
        (
            "Check whether the covariance matrix is positive "
            "semidefinite and verify that portfolio weights "
            "sum to one."
        ),
    )

    selected = rank_capabilities(
        instruction=(
            positive / "instruction.md"
        ).read_text(),
        task_dir=positive,
    )

    ids = [
        item.descriptor.capability_id
        for item in selected
    ]

    assert (
        "quantitative-validation-invariants"
        in ids
    )

    generic = _task(
        tmp_path / "generic",
        "Check the supplied results.",
    )

    generic_selected = rank_capabilities(
        instruction=(
            generic / "instruction.md"
        ).read_text(),
        task_dir=generic,
    )

    assert all(
        item.descriptor.capability_id
        != "quantitative-validation-invariants"
        for item in generic_selected
    )


def test_candidate_executes_invariant_helpers(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path,
        (
            "Validate covariance PSD, probability bounds, "
            "weight totals and causal ordering."
        ),
    )

    workspace = CandidateWorkspace.create(
        base_dir=tmp_path / "work",
        candidate_id=1,
        task_dir=task,
    )

    try:
        script = (
            workspace.source_dir
            / "solver.py"
        )

        script.write_text(
            """
from pathlib import Path
import json
import os
import numpy as np

from offline_common.invariants import (
    causal_order_mask,
    is_positive_semidefinite,
    probability_bounds_mask,
    weight_sum_error,
)

covariance = np.array([
    [0.04, 0.01],
    [0.01, 0.09],
])

probabilities = probability_bounds_mask(
    [0.1, 0.4, 1.2]
)

causal = causal_order_mask(
    ["2026-01-01", "2026-01-02"],
    ["2026-01-02", "2026-01-03"],
    strict=True,
)

payload = {
    "psd": bool(
        is_positive_semidefinite(
            covariance
        )
    ),
    "probability_valid": (
        probabilities.tolist()
    ),
    "weight_error": float(
        weight_sum_error(
            [0.2, 0.3, 0.5]
        )
    ),
    "causal": causal.tolist(),
}

Path(os.environ["OUTPUT_DIR"]).joinpath(
    "invariant_probe.json"
).write_text(
    json.dumps(
        payload,
        sort_keys=True,
    ),
    encoding="utf-8",
)
""".strip()
            + "\n",
            encoding="utf-8",
        )

        execution = run_candidate(
            workspace,
            script,
            timeout_seconds=30,
        )

        assert execution.return_code == 0, (
            execution.stderr
        )

        payload = json.loads(
            (
                workspace.output_dir
                / "invariant_probe.json"
            ).read_text()
        )

        assert payload["psd"] is True

        assert payload[
            "probability_valid"
        ] == [
            True,
            True,
            False,
        ]

        assert abs(
            payload["weight_error"]
        ) < 1e-12

        assert payload["causal"] == [
            True,
            True,
        ]

    finally:
        workspace.cleanup()
