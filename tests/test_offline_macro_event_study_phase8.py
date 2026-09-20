from __future__ import annotations

from pathlib import Path

import numpy as np

from agent.offline_macro_event_study import (
    MacroTextEventStudySkill,
    normalize_tokens,
    _newey_west_covariance,
)


def test_macro_event_study_text_normalization() -> None:
    tokens = normalize_tokens(
        "Inflation, STRONG! 2% -- but easing?"
    )

    assert tokens == [
        "inflation",
        "strong",
        "but",
        "easing",
    ]


def test_macro_event_study_hac_and_match() -> None:
    X = np.column_stack(
        [
            np.ones(
                6
            ),
            np.arange(
                6,
                dtype=float,
            ),
        ]
    )

    residuals = np.asarray(
        [
            0.2,
            -0.1,
            0.3,
            -0.2,
            0.1,
            -0.05,
        ],
        dtype=float,
    )

    covariance = _newey_west_covariance(
        X,
        residuals,
        lag=2,
    )

    assert covariance.shape == (
        2,
        2,
    )

    assert np.isfinite(
        covariance
    ).all()

    skill = MacroTextEventStudySkill()

    assert skill.matches(
        instruction=(
            "Run an FOMC tone event study on Treasury yields using "
            "a lexicon, TF-IDF novelty and Newey-West regression."
        ),
        task_dir=Path("."),
    )
