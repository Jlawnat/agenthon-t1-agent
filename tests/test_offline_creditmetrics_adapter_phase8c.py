from __future__ import annotations

import numpy as np

from agent.offline_credit_portfolio import (
    _parse_creditmetrics_spec,
)


def test_parse_parameter_only_creditmetrics_spec() -> None:
    instruction = """
Initial rating distribution: 2 AAA, 3 AA
Recovery Rate
- Upon default, each bond recovers 40% of face value
- Each bond has face value 1 million
- asset correlation rho = 0.20
- Number of simulations: 200,000
- Random seed: 42

| From\\To | AAA | AA | Default |
|---------|-----|----|---------|
| AAA | 90.0 | 9.0 | 1.0 |
| AA  | 5.0  | 90.0 | 5.0 |
"""

    spec = _parse_creditmetrics_spec(
        instruction
    )

    assert spec[
        "initial_counts"
    ] == {
        "AAA": 2,
        "AA": 3,
    }

    assert spec[
        "n_simulations"
    ] == 200000

    assert np.isclose(
        spec[
            "recovery_rate"
        ],
        0.40,
    )

    assert np.isclose(
        spec[
            "rho"
        ],
        0.20,
    )

    assert np.isclose(
        spec[
            "face_value_millions"
        ],
        1.0,
    )
