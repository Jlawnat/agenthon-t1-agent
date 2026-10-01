from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from agent.candidate_runner import run_candidate
from agent.candidate_workspace import CandidateWorkspace
from agent.capability_bridge import (
    CANDIDATE_LIBRARY_EXPORTS,
    rank_capabilities,
)
from agent.offline_common import panel as extracted_panel
from agent.offline_double_sort import _quintile_labels
from agent.offline_residual_momentum import _spearman_average_ranks
from agent.offline_stable_residual import (
    _beta_neutral_project,
    _ols_pinv,
    _zscore_ddof0,
)


def _task(tmp_path: Path, instruction: str) -> Path:
    task = tmp_path / "task"
    (task / "environment" / "data").mkdir(parents=True)
    (task / "instruction.md").write_text(
        instruction,
        encoding="utf-8",
    )
    return task


def _load_curated_panel(tmp_path: Path):
    task = _task(
        tmp_path,
        "Residualize a cross-sectional signal and impose beta-neutral weights.",
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
        / "panel.py"
    )

    spec = importlib.util.spec_from_file_location(
        "_phase8v_curated_panel",
        path,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return workspace, module


def test_zscore_matches_mature_implementation() -> None:
    values = np.asarray(
        [3.0, 7.0, -2.0, 4.5, 9.0, 1.0],
        dtype=float,
    )

    expected = _zscore_ddof0(values)
    actual = extracted_panel.zscore_ddof0(values)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )

    assert abs(float(actual.mean())) < 1e-12
    assert abs(float(actual.std(ddof=0)) - 1.0) < 1e-12


def test_zscore_constant_cross_section_is_zero() -> None:
    actual = extracted_panel.zscore_ddof0(
        [5.0, 5.0, 5.0, 5.0]
    )

    np.testing.assert_array_equal(
        actual,
        np.zeros(4),
    )


def test_equal_count_quintiles_match_mature_implementation() -> None:
    for n in (
        1,
        4,
        5,
        6,
        9,
        10,
        17,
        25,
        101,
    ):
        expected = _quintile_labels(n)

        actual = extracted_panel.equal_count_bucket_labels(
            n,
            buckets=5,
        )

        np.testing.assert_array_equal(
            actual,
            expected,
        )

        assert actual.min() >= 1
        assert actual.max() <= 5


def test_equal_count_bucket_generalization_is_deterministic() -> None:
    first = extracted_panel.equal_count_bucket_labels(
        20,
        buckets=4,
    )
    second = extracted_panel.equal_count_bucket_labels(
        20,
        buckets=4,
    )

    np.testing.assert_array_equal(
        first,
        second,
    )

    np.testing.assert_array_equal(
        first,
        np.repeat(
            np.arange(1, 5),
            5,
        ),
    )


def test_spearman_rank_matches_mature_implementation() -> None:
    first = np.asarray(
        [10.0, 20.0, 20.0, 5.0, 30.0, 15.0],
        dtype=float,
    )
    second = np.asarray(
        [2.0, 4.0, 3.0, 1.0, 6.0, 5.0],
        dtype=float,
    )

    expected = _spearman_average_ranks(
        first,
        second,
    )
    actual = extracted_panel.spearman_rank_correlation(
        first,
        second,
    )

    assert np.isclose(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_spearman_constant_rank_returns_nan() -> None:
    actual = extracted_panel.spearman_rank_correlation(
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 3.0],
    )

    assert np.isnan(actual)


def test_residualization_matches_mature_pinv_projection() -> None:
    n = 80
    index = np.arange(n, dtype=float)

    control_one = np.sin(index / 8.0)
    control_two = np.cos(index / 11.0)

    controls = np.column_stack(
        [
            control_one,
            control_two,
        ]
    )

    values = (
        1.25
        + 0.8 * control_one
        - 0.35 * control_two
        + 0.05 * np.sin(index / 2.5)
    )

    design = np.column_stack(
        [
            np.ones(n),
            controls,
        ]
    )

    expected_coefficients = _ols_pinv(
        design,
        values,
    )

    expected_residuals = (
        values
        - design @ expected_coefficients
    )

    residuals, coefficients = (
        extracted_panel.residualize_against_controls(
            values,
            controls,
            add_intercept=True,
        )
    )

    np.testing.assert_allclose(
        coefficients,
        expected_coefficients,
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        residuals,
        expected_residuals,
        rtol=1e-12,
        atol=1e-12,
    )

    assert abs(float(residuals.mean())) < 1e-10

    np.testing.assert_allclose(
        design.T @ residuals,
        np.zeros(design.shape[1]),
        atol=1e-10,
    )


def test_residualization_without_intercept() -> None:
    x = np.linspace(
        -1.0,
        1.0,
        50,
    ).reshape(-1, 1)

    y = 2.0 * x[:, 0] + 0.01 * np.sin(
        np.arange(50)
    )

    residuals, coefficients = (
        extracted_panel.residualize_against_controls(
            y,
            x,
            add_intercept=False,
        )
    )

    expected_coefficients = _ols_pinv(
        x,
        y,
    )

    np.testing.assert_allclose(
        coefficients,
        expected_coefficients,
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        x.T @ residuals,
        np.zeros(1),
        atol=1e-10,
    )


def test_beta_neutral_projection_matches_mature_implementation() -> None:
    weights = np.asarray(
        [0.30, 0.20, -0.15, -0.10, -0.25],
        dtype=float,
    )

    beta = np.asarray(
        [1.2, 0.8, 1.5, 0.6, 1.0],
        dtype=float,
    )

    expected = _beta_neutral_project(
        weights,
        beta,
    )

    actual = extracted_panel.beta_neutral_projection(
        weights,
        beta,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )

    assert abs(float(actual.sum())) < 1e-10
    assert abs(float(actual @ beta)) < 1e-10


def test_curated_panel_has_exact_approved_surface(
    tmp_path: Path,
) -> None:
    workspace, curated = _load_curated_panel(
        tmp_path
    )

    try:
        assert tuple(curated.__all__) == (
            CANDIDATE_LIBRARY_EXPORTS[
                "panel.py"
            ]
        )

        values = np.asarray(
            [1.0, 2.0, 4.0, 8.0],
            dtype=float,
        )

        np.testing.assert_allclose(
            curated.zscore_ddof0(values),
            extracted_panel.zscore_ddof0(
                values
            ),
        )

        np.testing.assert_array_equal(
            curated.equal_count_bucket_labels(
                10,
                buckets=5,
            ),
            extracted_panel.equal_count_bucket_labels(
                10,
                buckets=5,
            ),
        )

    finally:
        workspace.cleanup()


def test_panel_routing_is_specific(
    tmp_path: Path,
) -> None:
    positive = _task(
        tmp_path / "positive",
        (
            "Residualize the cross-sectional signal against controls, "
            "then construct beta-neutral portfolio weights."
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
        "cross-sectional-panel-operations"
        in ids
    )

    generic = _task(
        tmp_path / "generic",
        "Sort the supplied observations and report the result.",
    )

    generic_selected = rank_capabilities(
        instruction=(
            generic / "instruction.md"
        ).read_text(),
        task_dir=generic,
    )

    assert all(
        item.descriptor.capability_id
        != "cross-sectional-panel-operations"
        for item in generic_selected
    )


def test_candidate_executes_panel_helpers(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path,
        (
            "Residualize a factor signal and create "
            "beta-neutral portfolio weights."
        ),
    )

    workspace = CandidateWorkspace.create(
        base_dir=tmp_path / "work",
        candidate_id=1,
        task_dir=task,
    )

    try:
        script = workspace.source_dir / "solver.py"

        script.write_text(
            """
from pathlib import Path
import json
import os
import numpy as np

from offline_common.panel import (
    beta_neutral_projection,
    residualize_against_controls,
)

controls = np.column_stack([
    np.linspace(-1.0, 1.0, 20),
    np.sin(np.arange(20) / 3.0),
])

signal = (
    0.4
    + 0.7 * controls[:, 0]
    - 0.2 * controls[:, 1]
    + 0.01 * np.cos(np.arange(20))
)

residuals, coefficients = residualize_against_controls(
    signal,
    controls,
)

weights = np.linspace(
    -0.2,
    0.2,
    20,
)

beta = np.linspace(
    0.5,
    1.5,
    20,
)

neutral = beta_neutral_projection(
    weights,
    beta,
)

payload = {
    "residual_mean": float(residuals.mean()),
    "coefficient_count": int(len(coefficients)),
    "weight_sum": float(neutral.sum()),
    "beta_exposure": float(neutral @ beta),
}

Path(os.environ["OUTPUT_DIR"]).joinpath(
    "panel_probe.json"
).write_text(
    json.dumps(payload, sort_keys=True),
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
                / "panel_probe.json"
            ).read_text()
        )

        assert abs(
            payload["residual_mean"]
        ) < 1e-10

        assert (
            payload["coefficient_count"]
            == 3
        )

        assert abs(
            payload["weight_sum"]
        ) < 1e-10

        assert abs(
            payload["beta_exposure"]
        ) < 1e-10

    finally:
        workspace.cleanup()


def test_spearman_and_quintile_language_selects_panel_helpers(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path,
        (
            "Form deterministic Q1 and Q5 quintile sorts "
            "and compute the Spearman correlation between "
            "predictions and next-period returns."
        ),
    )

    selected = rank_capabilities(
        instruction=(
            task / "instruction.md"
        ).read_text(),
        task_dir=task,
    )

    assert (
        "cross-sectional-panel-operations"
        in {
            item.descriptor.capability_id
            for item in selected
        }
    )
