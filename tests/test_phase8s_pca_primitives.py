from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA

from agent.qf_primitives import (
    pca_from_covariance,
    pca_from_observations,
)


@pytest.mark.parametrize(
    ("n_obs", "n_features", "n_components", "seed"),
    [
        (100, 5, 3, 11),
        (250, 8, 4, 22),
        (80, 3, 2, 33),
    ],
)
def test_pca_from_observations_matches_sklearn(
    n_obs: int,
    n_features: int,
    n_components: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)

    x = rng.normal(
        size=(n_obs, n_features)
    )

    # Add non-zero means and different scales so centering matters.
    x = (
        x
        * np.linspace(
            0.5,
            2.0,
            n_features,
        )
        + np.linspace(
            -1.0,
            1.5,
            n_features,
        )
    )

    reference = PCA(
        n_components=n_components
    )
    reference_scores = reference.fit_transform(
        x
    )

    result = pca_from_observations(
        x,
        n_components=n_components,
    )

    np.testing.assert_allclose(
        result["mean"],
        reference.mean_,
        rtol=0.0,
        atol=1e-14,
    )

    np.testing.assert_allclose(
        result["components"],
        reference.components_,
        rtol=0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        result["explained_variance"],
        reference.explained_variance_,
        rtol=0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        result["explained_variance_ratio"],
        reference.explained_variance_ratio_,
        rtol=0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        result["scores"],
        reference_scores,
        rtol=0.0,
        atol=1e-11,
    )


def test_pca_from_observations_matches_phase8r_factor_contract() -> None:
    rng = np.random.default_rng(
        12345
    )

    returns = rng.normal(
        loc=0.01,
        scale=0.04,
        size=(120, 7),
    )

    old = PCA(
        n_components=3
    )
    old.fit(
        returns
    )

    new = pca_from_observations(
        returns,
        n_components=3,
    )

    np.testing.assert_allclose(
        new["components"],
        old.components_,
        rtol=0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        new["explained_variance_ratio"],
        old.explained_variance_ratio_,
        rtol=0.0,
        atol=1e-12,
    )


def test_pca_from_observations_validates_input() -> None:
    with pytest.raises(ValueError):
        pca_from_observations(
            [],
            n_components=1,
        )

    with pytest.raises(ValueError):
        pca_from_observations(
            [[1.0, np.nan], [2.0, 3.0]],
            n_components=1,
        )

    with pytest.raises(ValueError):
        pca_from_observations(
            [[1.0, 2.0], [3.0, 4.0]],
            n_components=3,
        )


@pytest.mark.parametrize(
    ("n_features", "n_components", "seed"),
    [
        (4, 2, 101),
        (7, 3, 202),
        (10, 5, 303),
    ],
)
def test_pca_from_covariance_matches_numpy_eigh(
    n_features: int,
    n_components: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(
        seed
    )

    a = rng.normal(
        size=(n_features, n_features)
    )

    covariance = (
        a.T @ a
        / n_features
    )

    eigenvalues, eigenvectors = (
        np.linalg.eigh(
            covariance
        )
    )

    order = np.argsort(
        eigenvalues
    )[::-1]

    eigenvalues = eigenvalues[
        order
    ]

    eigenvectors = eigenvectors[
        :,
        order
    ]

    expected_ratio = (
        eigenvalues
        / eigenvalues.sum()
    )

    result = pca_from_covariance(
        covariance,
        n_components=n_components,
    )

    np.testing.assert_allclose(
        result["eigenvalues"],
        eigenvalues[:n_components],
        rtol=0.0,
        atol=1e-13,
    )

    np.testing.assert_allclose(
        result["components"],
        eigenvectors[:, :n_components],
        rtol=0.0,
        atol=1e-13,
    )

    np.testing.assert_allclose(
        result["explained_variance_ratio"],
        expected_ratio[:n_components],
        rtol=0.0,
        atol=1e-13,
    )

    assert result[
        "cumulative_explained_variance_ratio"
    ] == pytest.approx(
        float(
            expected_ratio[
                :n_components
            ].sum()
        ),
        rel=0.0,
        abs=1e-14,
    )


def test_pca_from_covariance_matches_phase8r_yield_curve_contract() -> None:
    rng = np.random.default_rng(
        456
    )

    changes = rng.normal(
        size=(150, 10)
    )

    covariance = np.cov(
        changes,
        rowvar=False,
    )

    old_values, old_vectors = (
        np.linalg.eigh(
            covariance
        )
    )

    order = np.argsort(
        old_values
    )[::-1]

    old_values = old_values[
        order
    ]

    old_vectors = old_vectors[
        :,
        order
    ]

    old_ratio = (
        old_values
        / old_values.sum()
    )

    new = pca_from_covariance(
        covariance,
        n_components=3,
    )

    np.testing.assert_allclose(
        new["eigenvalues"],
        old_values[:3],
        rtol=0.0,
        atol=1e-14,
    )

    np.testing.assert_allclose(
        new["components"],
        old_vectors[:, :3],
        rtol=0.0,
        atol=1e-14,
    )

    np.testing.assert_allclose(
        new["explained_variance_ratio"],
        old_ratio[:3],
        rtol=0.0,
        atol=1e-14,
    )


def test_pca_from_covariance_can_return_all_components() -> None:
    covariance = np.asarray(
        [
            [2.0, 0.5, 0.2],
            [0.5, 1.5, 0.1],
            [0.2, 0.1, 0.8],
        ],
        dtype=float,
    )

    result = pca_from_covariance(
        covariance
    )

    assert result[
        "components"
    ].shape == (
        3,
        3,
    )

    assert result[
        "eigenvalues"
    ].shape == (
        3,
    )

    assert result[
        "cumulative_explained_variance_ratio"
    ] == pytest.approx(
        1.0,
        rel=0.0,
        abs=1e-14,
    )


def test_pca_from_covariance_validates_input() -> None:
    with pytest.raises(ValueError):
        pca_from_covariance(
            [[1.0, 0.2]]
        )

    with pytest.raises(ValueError):
        pca_from_covariance(
            [
                [1.0, 0.5],
                [0.1, 1.0],
            ]
        )

    with pytest.raises(ValueError):
        pca_from_covariance(
            [
                [1.0, np.nan],
                [np.nan, 1.0],
            ]
        )

    with pytest.raises(ValueError):
        pca_from_covariance(
            np.eye(3),
            n_components=4,
        )
