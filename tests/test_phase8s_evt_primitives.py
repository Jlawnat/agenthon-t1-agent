from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from agent.offline_evt_pot import (
    _evt_var_es as old_evt_var_es,
    _hill_xi as old_hill_xi,
)

from agent.qf_primitives import (
    evt_var_es_from_gpd,
    fit_gpd_exceedances,
    hill_tail_index,
)


@pytest.mark.parametrize(
    ("shape", "scale", "seed"),
    [
        (0.10, 0.75, 101),
        (0.25, 1.20, 202),
        (-0.15, 0.80, 303),
        (0.00, 1.00, 404),
    ],
)
def test_fit_gpd_exceedances_recovers_scipy_contract(
    shape: float,
    scale: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(
        seed
    )

    excesses = stats.genpareto.rvs(
        c=shape,
        loc=0.0,
        scale=scale,
        size=3000,
        random_state=rng,
    )

    old_shape, _old_loc, old_scale = (
        stats.genpareto.fit(
            excesses,
            floc=0.0,
        )
    )

    new = fit_gpd_exceedances(
        excesses
    )

    assert new["shape"] == pytest.approx(
        float(old_shape),
        rel=0.0,
        abs=1e-12,
    )

    assert new["scale"] == pytest.approx(
        float(old_scale),
        rel=0.0,
        abs=1e-12,
    )

    assert new["n_exceedances"] == (
        len(excesses)
    )

    assert new["mean_excess"] == pytest.approx(
        float(
            np.mean(
                excesses
            )
        ),
        rel=0.0,
        abs=1e-14,
    )


def test_fit_gpd_exceedances_requires_nonnegative_finite_values() -> None:
    with pytest.raises(ValueError):
        fit_gpd_exceedances(
            []
        )

    with pytest.raises(ValueError):
        fit_gpd_exceedances(
            [
                0.1,
                np.nan,
                0.2,
            ]
        )

    with pytest.raises(ValueError):
        fit_gpd_exceedances(
            [
                0.1,
                -0.01,
                0.2,
            ]
        )


@pytest.mark.parametrize(
    (
        "probability",
        "threshold",
        "shape",
        "scale",
        "n_total",
        "n_exceedances",
    ),
    [
        (
            0.95,
            2.0,
            0.20,
            0.75,
            1000,
            100,
        ),
        (
            0.99,
            1.5,
            0.05,
            0.50,
            2500,
            200,
        ),
        (
            0.975,
            3.0,
            -0.10,
            1.25,
            1500,
            150,
        ),
        (
            0.99,
            2.5,
            0.0,
            0.90,
            2000,
            250,
        ),
    ],
)
def test_evt_var_es_matches_phase8r(
    probability: float,
    threshold: float,
    shape: float,
    scale: float,
    n_total: int,
    n_exceedances: int,
) -> None:
    old_var, old_es = old_evt_var_es(
        probability=probability,
        threshold=threshold,
        shape=shape,
        scale=scale,
        n_total=n_total,
        n_exceedances=n_exceedances,
    )

    new = evt_var_es_from_gpd(
        probability=probability,
        threshold=threshold,
        shape=shape,
        scale=scale,
        n_total=n_total,
        n_exceedances=n_exceedances,
    )

    assert new["var"] == pytest.approx(
        old_var,
        rel=0.0,
        abs=1e-14,
    )

    assert new["expected_shortfall"] == pytest.approx(
        old_es,
        rel=0.0,
        abs=1e-14,
    )


def test_evt_var_es_rejects_infinite_expected_shortfall() -> None:
    with pytest.raises(
        ValueError,
        match="infinite",
    ):
        evt_var_es_from_gpd(
            probability=0.99,
            threshold=2.0,
            shape=1.0,
            scale=0.8,
            n_total=1000,
            n_exceedances=100,
        )


@pytest.mark.parametrize(
    "seed",
    [
        11,
        22,
        33,
    ],
)
def test_hill_tail_index_matches_phase8r(
    seed: int,
) -> None:
    rng = np.random.default_rng(
        seed
    )

    losses = (
        rng.pareto(
            3.5,
            size=1000,
        )
        + 1.0
    )

    old = old_hill_xi(
        losses
    )

    new = hill_tail_index(
        losses
    )

    assert new == pytest.approx(
        old,
        rel=0.0,
        abs=1e-15,
    )


def test_hill_tail_index_requires_positive_upper_tail() -> None:
    with pytest.raises(ValueError):
        hill_tail_index(
            [
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )


def test_hill_tail_index_validates_input() -> None:
    with pytest.raises(ValueError):
        hill_tail_index(
            []
        )

    with pytest.raises(ValueError):
        hill_tail_index(
            [
                1.0,
                np.nan,
                2.0,
            ]
        )
