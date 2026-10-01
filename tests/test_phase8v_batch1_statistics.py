from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from agent import finance_factor_stats as mature_stats
from agent.finance_multimodal import (
    normalize_cross_section as mature_normalize_cross_section,
)
from agent.candidate_runner import run_candidate
from agent.candidate_workspace import CandidateWorkspace
from agent.capability_bridge import (
    CANDIDATE_LIBRARY_EXPORTS,
    rank_capabilities,
)
from agent.offline_common import statistics as extracted_stats


def _task(tmp_path: Path, instruction: str) -> Path:
    task = tmp_path / "task"
    (task / "environment" / "data").mkdir(parents=True)
    (task / "instruction.md").write_text(
        instruction,
        encoding="utf-8",
    )
    return task


def _load_curated_statistics(
    tmp_path: Path,
):
    task = _task(
        tmp_path,
        "Estimate a factor regression using Newey-West covariance.",
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
        / "statistics.py"
    )

    spec = importlib.util.spec_from_file_location(
        "_phase8v_curated_statistics",
        path,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return workspace, module


def _regression_inputs():
    n = 120
    index = np.arange(n, dtype=float)

    x1 = np.sin(index / 8.0)
    x2 = np.cos(index / 13.0) + 0.15 * x1

    design = np.column_stack(
        [
            np.ones(n),
            x1,
            x2,
        ]
    )

    response = (
        0.7
        + 1.4 * x1
        - 0.6 * x2
        + 0.08 * np.sin(index / 3.0)
    )

    return design, response


def _assert_ols_equal(left, right) -> None:
    np.testing.assert_allclose(
        left.coefficients,
        right.coefficients,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        left.fitted,
        right.fitted,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        left.residuals,
        right.residuals,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        left.standard_errors,
        right.standard_errors,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        left.t_statistics,
        right.t_statistics,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        left.p_values,
        right.p_values,
        rtol=1e-12,
        atol=1e-12,
    )

    assert left.r_squared == right.r_squared
    assert left.adjusted_r_squared == right.adjusted_r_squared
    assert left.degrees_of_freedom == right.degrees_of_freedom


def test_extracted_statistics_match_mature_implementation() -> None:
    design, response = _regression_inputs()

    mature_fit = mature_stats.ols_with_inference(
        design,
        response,
    )
    extracted_fit = extracted_stats.ols_with_inference(
        design,
        response,
    )

    _assert_ols_equal(mature_fit, extracted_fit)

    lag = mature_stats.optimal_newey_west_lag(
        len(response)
    )
    assert (
        extracted_stats.optimal_newey_west_lag(
            len(response)
        )
        == lag
    )

    mature_hac = mature_stats.newey_west_covariance(
        design,
        mature_fit.residuals,
        lag=lag,
    )
    extracted_hac = extracted_stats.newey_west_covariance(
        design,
        extracted_fit.residuals,
        lag=lag,
    )

    np.testing.assert_allclose(
        extracted_hac,
        mature_hac,
        rtol=1e-12,
        atol=1e-12,
    )

    assert np.allclose(
        extracted_hac,
        extracted_hac.T,
        rtol=1e-12,
        atol=1e-12,
    )

    assert np.isfinite(
        extracted_fit.standard_errors
    ).all()

    assert (
        extracted_stats.durbin_watson(
            extracted_fit.residuals
        )
        == mature_stats.durbin_watson(
            mature_fit.residuals
        )
    )

    factors = design[:, 1:]
    np.testing.assert_allclose(
        extracted_stats.variance_inflation_factors(
            factors
        ),
        mature_stats.variance_inflation_factors(
            factors
        ),
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        extracted_stats.rolling_ols_coefficient(
            design,
            response,
            window=40,
            coefficient_index=1,
        ),
        mature_stats.rolling_ols_coefficient(
            design,
            response,
            window=40,
            coefficient_index=1,
        ),
        rtol=1e-12,
        atol=1e-12,
    )


def test_grs_matches_mature_implementation_and_is_finite() -> None:
    n = 140
    index = np.arange(n, dtype=float)

    factors = np.column_stack(
        [
            np.sin(index / 7.0),
            np.cos(index / 11.0),
        ]
    )

    residuals = np.column_stack(
        [
            0.03 * np.sin(index / 3.0),
            0.04 * np.cos(index / 5.0),
            0.02 * np.sin(index / 9.0 + 0.4),
        ]
    )

    alphas = np.array(
        [0.001, -0.0005, 0.0008],
        dtype=float,
    )

    mature = mature_stats.grs_joint_alpha_test(
        alphas,
        residuals,
        factors,
    )
    extracted = extracted_stats.grs_joint_alpha_test(
        alphas,
        residuals,
        factors,
    )

    assert asdict(extracted) == asdict(mature)
    assert np.isfinite(extracted.statistic)
    assert np.isfinite(extracted.p_value)
    assert 0.0 <= extracted.p_value <= 1.0
    assert extracted.df1 == 3
    assert extracted.df2 == n - 3 - 2


def test_vif_detects_collinearity() -> None:
    x = np.linspace(-1.0, 1.0, 100)

    factors = np.column_stack(
        [
            x,
            2.0 * x + 0.001 * np.sin(
                np.arange(100)
            ),
            np.cos(np.arange(100) / 9.0),
        ]
    )

    vif = extracted_stats.variance_inflation_factors(
        factors
    )

    assert np.isfinite(vif).all()
    assert vif.shape == (3,)
    assert vif[0] > 100.0
    assert vif[1] > 100.0


def test_rolling_ols_is_causal_with_respect_to_future_rows() -> None:
    design, response = _regression_inputs()

    original = extracted_stats.rolling_ols_coefficient(
        design,
        response,
        window=30,
        coefficient_index=1,
    )

    changed_response = response.copy()
    changed_response[-1] += 1000.0

    changed = extracted_stats.rolling_ols_coefficient(
        design,
        changed_response,
        window=30,
        coefficient_index=1,
    )

    np.testing.assert_allclose(
        original[:-1],
        changed[:-1],
        rtol=1e-12,
        atol=1e-12,
    )

    assert not np.isclose(
        original[-1],
        changed[-1],
    )


def test_cross_section_normalization_matches_mature_source() -> None:
    series = pd.Series(
        [
            1.0,
            2.0,
            np.nan,
            np.inf,
            -2.0,
            100.0,
        ],
        index=list("abcdef"),
    )

    expected = mature_normalize_cross_section(
        series
    )
    actual = extracted_stats.normalize_cross_section(
        series
    )

    pd.testing.assert_series_equal(
        actual,
        expected,
    )

    assert np.isfinite(actual).all()
    assert actual.min() >= -3.0
    assert actual.max() <= 3.0


def test_curated_statistics_module_has_exact_approved_surface(
    tmp_path: Path,
) -> None:
    workspace, curated = _load_curated_statistics(
        tmp_path
    )

    try:
        assert tuple(curated.__all__) == (
            CANDIDATE_LIBRARY_EXPORTS[
                "statistics.py"
            ]
        )

        design, response = _regression_inputs()

        mature = mature_stats.ols_with_inference(
            design,
            response,
        )
        curated_fit = curated.ols_with_inference(
            design,
            response,
        )

        _assert_ols_equal(
            mature,
            curated_fit,
        )

        lag = curated.optimal_newey_west_lag(
            len(response)
        )

        covariance = curated.newey_west_covariance(
            design,
            curated_fit.residuals,
            lag=lag,
        )

        assert covariance.shape == (3, 3)
        assert np.isfinite(covariance).all()

    finally:
        workspace.cleanup()


def test_statistics_routing_is_specific_not_generic(
    tmp_path: Path,
) -> None:
    positive = _task(
        tmp_path / "positive",
        (
            "Estimate a Fama-French factor regression "
            "with Newey-West covariance and report "
            "Durbin-Watson diagnostics."
        ),
    )

    selected = rank_capabilities(
        instruction=(
            positive / "instruction.md"
        ).read_text(),
        task_dir=positive,
    )

    assert (
        selected[0].descriptor.capability_id
        == "regression-factor-statistics"
    )

    generic = _task(
        tmp_path / "generic",
        "Run a regression on the supplied data.",
    )

    generic_selected = rank_capabilities(
        instruction=(
            generic / "instruction.md"
        ).read_text(),
        task_dir=generic,
    )

    assert all(
        item.descriptor.capability_id
        != "regression-factor-statistics"
        for item in generic_selected
    )


def test_candidate_can_execute_statistics_from_workspace(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path,
        "Estimate a factor regression with HAC inference.",
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

from offline_common.statistics import (
    ols_with_inference,
    newey_west_covariance,
)

x = np.arange(60, dtype=float)
design = np.column_stack([
    np.ones(60),
    x / 60.0,
])
response = 1.5 + 0.8 * (x / 60.0) + 0.02 * np.sin(x)

fit = ols_with_inference(design, response)
hac = newey_west_covariance(
    design,
    fit.residuals,
    lag=3,
)

payload = {
    "beta": fit.coefficients.tolist(),
    "r_squared": fit.r_squared,
    "hac_shape": list(hac.shape),
    "hac_finite": bool(np.isfinite(hac).all()),
}

Path(os.environ["OUTPUT_DIR"]).joinpath(
    "statistics_probe.json"
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
                / "statistics_probe.json"
            ).read_text()
        )

        assert payload["hac_shape"] == [2, 2]
        assert payload["hac_finite"] is True
        assert len(payload["beta"]) == 2

    finally:
        workspace.cleanup()


def test_newey_west_mean_tstat_matches_mature_implementation() -> None:
    from agent.offline_residual_momentum import _nw_tstat

    values = (
        0.01
        + 0.02
        * np.sin(
            np.arange(
                180,
                dtype=float,
            )
            / 7.0
        )
    )

    for lag in (
        0,
        1,
        3,
        6,
        24,
    ):
        expected = _nw_tstat(
            values,
            lag,
        )

        actual = (
            extracted_stats
            .newey_west_mean_tstat(
                values,
                lag=lag,
            )
        )

        assert np.isclose(
            actual,
            expected,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )


def test_curated_newey_west_mean_tstat(
    tmp_path: Path,
) -> None:
    workspace, curated = (
        _load_curated_statistics(
            tmp_path
        )
    )

    try:
        values = (
            0.015
            + 0.01
            * np.cos(
                np.arange(
                    100,
                    dtype=float,
                )
                / 5.0
            )
        )

        expected = (
            extracted_stats
            .newey_west_mean_tstat(
                values,
                lag=6,
            )
        )

        actual = (
            curated
            .newey_west_mean_tstat(
                values,
                lag=6,
            )
        )

        assert np.isclose(
            actual,
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

    finally:
        workspace.cleanup()
