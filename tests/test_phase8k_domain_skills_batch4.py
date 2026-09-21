from __future__ import annotations

import numpy as np

from agent.offline_cta_basel import (
    _sma_seeded_ema,
)
from agent.offline_intraday_volume import (
    _largest_remainder,
    _profiles,
)


def test_sma_seeded_ema_definition() -> None:
    x = np.arange(1.0, 8.0)
    got = _sma_seeded_ema(x, 3)
    assert np.isnan(got[:2]).all()
    assert np.isclose(got[2], 2.0)
    # alpha = 2 / (3 + 1) = 0.5
    assert np.isclose(got[3], 0.5 * 4.0 + 0.5 * 2.0)


def test_volume_profiles_are_nonnegative_and_normalized() -> None:
    rng = np.random.default_rng(7)
    x = rng.uniform(1.0, 10.0, size=(20, 78))
    shares = x / x.sum(axis=1, keepdims=True)
    profiles = _profiles(shares)

    assert set(profiles) == {
        "historical_mean_profile",
        "historical_median_profile",
        "ewma_profile",
        "winsorized_mean_profile",
    }

    for profile in profiles.values():
        assert np.all(profile >= 0.0)
        assert np.isclose(profile.sum(), 1.0)


def test_largest_remainder_sums_exactly_and_breaks_tie_early() -> None:
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    got = _largest_remainder(3, weights)
    assert got.tolist() == [1, 1, 1, 0]
    assert int(got.sum()) == 3
