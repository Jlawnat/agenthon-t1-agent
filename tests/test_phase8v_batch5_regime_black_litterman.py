from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from agent import offline_bl_regime_hmm as mature
from agent.offline_common import regime_black_litterman as extracted


def _synthetic_inputs():
    rng = np.random.default_rng(20261002)

    n = 240
    n_assets = 3

    returns = np.empty(
        (n, n_assets),
        dtype=float,
    )

    returns[:120] = rng.normal(
        loc=[0.0008, 0.0006, 0.0005],
        scale=[0.008, 0.009, 0.007],
        size=(120, n_assets),
    )

    returns[120:] = rng.normal(
        loc=[-0.0005, -0.0007, -0.0004],
        scale=[0.014, 0.016, 0.012],
        size=(120, n_assets),
    )

    params = {
        "market_cap_weights": [
            0.45,
            0.35,
            0.20,
        ],
        "init_pi": [
            0.5,
            0.5,
        ],
        "init_trans_matrix": [
            [0.95, 0.05],
            [0.05, 0.95],
        ],
        "hmm_max_iter": 100,
        "hmm_tol": 1e-9,
        "risk_aversion": 2.5,
        "tau": 0.05,
        "annualization_factor": 252.0,
        "views": [
            {
                "asset": 0,
                "return": 0.08,
                "confidence": 0.12,
            },
            {
                "asset": 2,
                "return": 0.04,
                "confidence": 0.15,
            },
        ],
    }

    return returns, params


def test_full_math_matches_mature_workflow(
    tmp_path,
) -> None:
    returns, params = _synthetic_inputs()

    task = tmp_path / "task"
    data = task / "environment" / "data"
    output = tmp_path / "mature-output"

    data.mkdir(parents=True)

    frame = pd.DataFrame(
        returns,
        columns=[
            "ASSET_1",
            "ASSET_2",
            "ASSET_3",
        ],
    )

    frame.insert(
        0,
        "date",
        pd.date_range(
            "2020-01-01",
            periods=len(frame),
            freq="B",
        ),
    )

    frame.to_csv(
        data / "returns.csv",
        index=False,
    )

    (
        data / "params.json"
    ).write_text(
        json.dumps(params),
        encoding="utf-8",
    )

    mature.BlackLittermanRegimeHmmSkill().solve(
        instruction="",
        task_dir=task,
        out_dir=output,
        seed=0,
    )

    mature_solution = json.loads(
        (
            output / "solution.json"
        ).read_text()
    )

    inter = mature_solution[
        "intermediates"
    ]

    market_weights = np.asarray(
        params["market_cap_weights"],
        dtype=float,
    )

    portfolio_returns = (
        returns @ market_weights
    )

    hmm = (
        extracted.fit_two_state_gaussian_hmm(
            portfolio_returns,
            initial_probabilities=params[
                "init_pi"
            ],
            transition_matrix=np.asarray(
                params[
                    "init_trans_matrix"
                ],
                dtype=float,
            ),
            max_iterations=params[
                "hmm_max_iter"
            ],
            tolerance=params[
                "hmm_tol"
            ],
        )
    )

    bull_state = int(
        np.argmax(
            hmm.means
        )
    )

    bear_state = (
        1 - bull_state
    )

    assert (
        hmm.iterations
        == inter["hmm_iterations"]["value"]
    )

    assert np.isclose(
        hmm.means[bull_state],
        inter["hmm_mu_bull"]["value"],
    )

    assert np.isclose(
        hmm.standard_deviations[
            bear_state
        ],
        inter[
            "hmm_sigma_bear"
        ]["value"],
    )

    assert np.isclose(
        hmm.posterior_probabilities[
            -1,
            bull_state,
        ],
        inter[
            "regime_prob_bull"
        ]["value"],
    )

    regime = (
        extracted.hard_regime_covariances(
            returns,
            hmm.posterior_probabilities,
            bull_state=bull_state,
            threshold=0.5,
            minimum_observations=11,
        )
    )

    assert (
        int(
            regime.bull_mask.sum()
        )
        == inter[
            "n_bull_days"
        ]["value"]
    )

    assert np.isclose(
        regime.bull_covariance[
            0,
            0,
        ],
        inter[
            "cov_bull_diag_asset1"
        ]["value"],
    )

    current_is_bull = (
        hmm.posterior_probabilities[
            -1,
            bull_state,
        ]
        > 0.5
    )

    covariance = (
        regime.bull_covariance
        if current_is_bull
        else regime.bear_covariance
    )

    views = params["views"]

    p_matrix = np.zeros(
        (
            len(views),
            returns.shape[1],
        ),
        dtype=float,
    )

    q_vector = np.zeros(
        len(views),
        dtype=float,
    )

    omega_diag = np.zeros(
        len(views),
        dtype=float,
    )

    for index, view in enumerate(
        views
    ):
        p_matrix[
            index,
            int(view["asset"]),
        ] = 1.0

        q_vector[index] = (
            float(view["return"])
            / 252.0
        )

        omega_diag[index] = (
            float(
                view["confidence"]
            )
            ** 2
            / 252.0
        )

    bl = (
        extracted.black_litterman_posterior(
            covariance,
            market_weights,
            risk_aversion=params[
                "risk_aversion"
            ],
            tau=params["tau"],
            view_matrix=p_matrix,
            view_returns=q_vector,
            view_covariance=np.diag(
                omega_diag
            ),
            normalize_gross=True,
        )
    )

    assert np.isclose(
        bl.implied_returns[0]
        * 252.0,
        inter[
            "implied_return_asset1"
        ]["value"],
    )

    assert np.isclose(
        bl.weights[0],
        inter[
            "bl_weight_asset1"
        ]["value"],
    )


def test_hmm_posteriors_and_transition_are_probabilities() -> None:
    returns, params = _synthetic_inputs()

    series = (
        returns
        @ np.asarray(
            params[
                "market_cap_weights"
            ]
        )
    )

    result = (
        extracted.fit_two_state_gaussian_hmm(
            series,
            initial_probabilities=[
                0.5,
                0.5,
            ],
            transition_matrix=np.array(
                [
                    [0.95, 0.05],
                    [0.05, 0.95],
                ]
            ),
        )
    )

    assert np.allclose(
        result.posterior_probabilities.sum(
            axis=1
        ),
        1.0,
    )

    assert np.allclose(
        result.transition_matrix.sum(
            axis=1
        ),
        1.0,
    )


def test_regime_masks_partition_observations() -> None:
    returns, params = _synthetic_inputs()

    series = (
        returns
        @ np.asarray(
            params[
                "market_cap_weights"
            ]
        )
    )

    hmm = (
        extracted.fit_two_state_gaussian_hmm(
            series,
            initial_probabilities=[
                0.5,
                0.5,
            ],
            transition_matrix=np.array(
                [
                    [0.95, 0.05],
                    [0.05, 0.95],
                ]
            ),
        )
    )

    bull = int(
        np.argmax(hmm.means)
    )

    regime = (
        extracted.hard_regime_covariances(
            returns,
            hmm.posterior_probabilities,
            bull_state=bull,
        )
    )

    assert np.all(
        regime.bull_mask
        ^ regime.bear_mask
    )

    assert (
        regime.bull_mask.sum()
        + regime.bear_mask.sum()
        == len(returns)
    )


def test_small_regime_uses_unconditional_covariance() -> None:
    rng = np.random.default_rng(7)

    returns = rng.normal(
        size=(20, 3)
    )

    posterior = np.zeros(
        (20, 2)
    )

    posterior[:, 1] = 1.0
    posterior[0, 0] = 0.9
    posterior[0, 1] = 0.1

    result = (
        extracted.hard_regime_covariances(
            returns,
            posterior,
            bull_state=0,
            minimum_observations=11,
        )
    )

    expected = np.cov(
        returns.T,
        ddof=1,
    )

    assert np.allclose(
        result.bull_covariance,
        expected,
    )


def test_black_litterman_gross_normalization() -> None:
    covariance = np.array(
        [
            [0.04, 0.01],
            [0.01, 0.05],
        ]
    )

    result = (
        extracted.black_litterman_posterior(
            covariance,
            [0.6, 0.4],
            risk_aversion=2.5,
            tau=0.05,
            view_matrix=np.array(
                [
                    [1.0, 0.0],
                ]
            ),
            view_returns=[
                0.0003,
            ],
            view_covariance=np.array(
                [
                    [0.0001],
                ]
            ),
        )
    )

    assert np.isclose(
        np.sum(
            np.abs(
                result.weights
            )
        ),
        1.0,
    )


def test_invalid_hmm_transition_rejected() -> None:
    with pytest.raises(ValueError):
        extracted.fit_two_state_gaussian_hmm(
            [
                -0.01,
                0.01,
                -0.02,
                0.02,
            ],
            initial_probabilities=[
                0.5,
                0.5,
            ],
            transition_matrix=np.array(
                [
                    [0.9, 0.2],
                    [0.1, 0.9],
                ]
            ),
        )


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
            "Build a regime-aware Black-Litterman portfolio "
            "using a two-state Hidden Markov Model."
        ),
        (
            "Fit the regimes using Baum-Welch and then run "
            "Black-Litterman portfolio construction."
        ),
    ),
)
def test_regime_black_litterman_language_selects_capability(
    tmp_path: Path,
    instruction: str,
) -> None:
    task = _task(tmp_path, instruction)

    selected = rank_capabilities(
        instruction=instruction,
        task_dir=task,
    )

    assert (
        "regime-black-litterman"
        in {
            item.descriptor.capability_id
            for item in selected
        }
    )


def test_generic_hmm_language_does_not_select_bl_capability(
    tmp_path: Path,
) -> None:
    instruction = (
        "Fit a generic hidden Markov model "
        "to a time series."
    )

    task = _task(tmp_path, instruction)

    selected = rank_capabilities(
        instruction=instruction,
        task_dir=task,
    )

    assert (
        "regime-black-litterman"
        not in {
            item.descriptor.capability_id
            for item in selected
        }
    )
