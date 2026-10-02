from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from agent import offline_copula_fitting as mature
from agent.offline_common import copula_fitting as extracted


def _pseudo_pair() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123456)

    x = rng.normal(size=300)
    y = (
        0.55 * x
        + np.sqrt(1.0 - 0.55**2)
        * rng.normal(size=300)
    )

    return (
        stats.rankdata(x) / (len(x) + 1.0),
        stats.rankdata(y) / (len(y) + 1.0),
    )


def test_gaussian_loglik_matches_mature() -> None:
    u1, u2 = _pseudo_pair()
    rho = 0.45

    assert np.isclose(
        extracted.gaussian_copula_loglik(
            u1,
            u2,
            rho,
        ),
        mature._gaussian_ll(
            u1,
            u2,
            rho,
        ),
    )


def test_student_t_loglik_matches_mature() -> None:
    u1, u2 = _pseudo_pair()
    rho = 0.45
    nu = 7.5

    assert np.isclose(
        extracted.student_t_copula_loglik(
            u1,
            u2,
            rho,
            nu,
        ),
        mature._t_ll(
            u1,
            u2,
            rho,
            nu,
        ),
    )


def test_clayton_loglik_matches_mature() -> None:
    u1, u2 = _pseudo_pair()
    theta = 1.3

    assert np.isclose(
        extracted.clayton_copula_loglik(
            u1,
            u2,
            theta,
        ),
        mature._clayton_ll(
            u1,
            u2,
            theta,
        ),
    )


def test_gumbel_loglik_matches_mature() -> None:
    u1, u2 = _pseudo_pair()
    theta = 1.6

    assert np.isclose(
        extracted.gumbel_copula_loglik(
            u1,
            u2,
            theta,
        ),
        mature._gumbel_ll(
            u1,
            u2,
            theta,
        ),
    )


def test_kendall_tau_conversion() -> None:
    tau = 0.4

    expected = np.sin(
        np.pi
        * tau
        / 2.0
    )

    assert np.isclose(
        extracted.kendall_tau_to_gaussian_rho(
            tau
        ),
        expected,
    )


def test_student_t_df_fit_matches_mature_optimization() -> None:
    u1, u2 = _pseudo_pair()

    tau = float(
        stats.kendalltau(
            u1,
            u2,
        )[0]
    )

    rho = float(
        np.sin(
            np.pi
            * tau
            / 2.0
        )
    )

    expected = mature.minimize_scalar(
        lambda nu: -mature._t_ll(
            u1,
            u2,
            rho,
            nu,
        ),
        bounds=(
            2.1,
            100.0,
        ),
        method="bounded",
    )

    actual = (
        extracted.fit_student_t_degrees_of_freedom(
            u1,
            u2,
            rho=rho,
        )
    )

    assert expected.success
    assert np.isclose(
        actual,
        expected.x,
    )


def test_empirical_tail_dependence() -> None:
    u1 = np.array(
        [
            0.01,
            0.03,
            0.20,
            0.70,
            0.96,
            0.98,
        ]
    )

    u2 = np.array(
        [
            0.02,
            0.40,
            0.10,
            0.80,
            0.97,
            0.60,
        ]
    )

    lower, upper = (
        extracted.empirical_tail_dependence(
            u1,
            u2,
            quantile=0.95,
        )
    )

    assert np.isclose(
        lower,
        0.5,
    )

    assert np.isclose(
        upper,
        0.5,
    )


@pytest.mark.parametrize(
    "fn,args",
    (
        (
            extracted.gaussian_copula_loglik,
            (
                [0.2, 0.4],
                [0.3],
                0.2,
            ),
        ),
        (
            extracted.clayton_copula_loglik,
            (
                [0.2, 0.4],
                [0.3, 0.5],
                0.0,
            ),
        ),
        (
            extracted.gumbel_copula_loglik,
            (
                [0.2, 0.4],
                [0.3, 0.5],
                0.8,
            ),
        ),
    ),
)
def test_invalid_inputs_rejected(
    fn,
    args,
) -> None:
    with pytest.raises(ValueError):
        fn(*args)


from pathlib import Path

from agent.capability_bridge import rank_capabilities


def _task(
    tmp_path: Path,
    instruction: str,
) -> Path:
    task = tmp_path / "task"
    (task / "environment" / "data").mkdir(
        parents=True,
    )
    (task / "instruction.md").write_text(
        instruction,
        encoding="utf-8",
    )
    return task


@pytest.mark.parametrize(
    "instruction",
    (
        (
            "Perform copula fitting across the equity pairs "
            "and compare the candidate families by likelihood."
        ),
        (
            "Construct pseudo-observations and fit dependence models."
        ),
    ),
)
def test_specific_copula_fitting_language_selects_capability(
    tmp_path: Path,
    instruction: str,
) -> None:
    task = _task(tmp_path, instruction)

    selected = rank_capabilities(
        instruction=instruction,
        task_dir=task,
    )

    assert (
        "copula-dependence-fitting"
        in {
            item.descriptor.capability_id
            for item in selected
        }
    )


def test_generic_copula_sampling_does_not_select_fitting(
    tmp_path: Path,
) -> None:
    instruction = (
        "Sample observations from a Gaussian copula "
        "and calculate rank correlations."
    )

    task = _task(tmp_path, instruction)

    selected = rank_capabilities(
        instruction=instruction,
        task_dir=task,
    )

    assert (
        "copula-dependence-fitting"
        not in {
            item.descriptor.capability_id
            for item in selected
        }
    )
