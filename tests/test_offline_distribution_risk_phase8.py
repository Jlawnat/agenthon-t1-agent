
from __future__ import annotations

from pathlib import Path

import numpy as np

from agent.offline_distribution_risk import (
    DistributionRiskSkill,
    analytical_true_values,
    generate_loss_samples,
    historical_estimate,
)


def test_distribution_risk_primitives() -> None:
    truth = analytical_true_values()

    assert np.isclose(
        truth["normal"]["0.95"]["var"],
        1.6448536270,
        rtol=1e-8,
    )

    assert np.isclose(
        truth["student_t"]["0.99"]["var"],
        3.3649299989,
        rtol=1e-8,
    )

    samples = generate_loss_samples(
        n_samples=1000,
        seed=42,
    )

    assert set(samples) == {
        "normal",
        "student_t",
        "contaminated_normal",
    }

    var, es = historical_estimate(
        samples["normal"],
        alpha=0.95,
    )

    assert es >= var


def test_distribution_risk_skill_outputs(
    tmp_path: Path,
) -> None:
    task = tmp_path / "task"
    task.mkdir()

    instruction = """
    Compare Value-at-Risk and Expected Shortfall on simulated loss data.
    Generate N = 200 samples with seed = 42 for a Normal, Student-t,
    and contaminated normal distribution. Compare historical,
    parametric, and Gaussian kernel density estimation methods.
    """.strip()

    skill = DistributionRiskSkill()

    assert skill.matches(
        instruction=instruction,
        task_dir=task,
    )

    out = tmp_path / "out"

    skill.solve(
        instruction=instruction,
        task_dir=task,
        out_dir=out,
        seed=7,
    )

    assert {
        path.name
        for path
        in out.iterdir()
    } == {
        "true_values.json",
        "estimates.csv",
        "comparison.json",
        "summary.json",
    }
