from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from agent.offline_first_passage import FirstPassageTimeSkill
from agent.offline_lob_pc_signal import _weighted_ofi
from agent.offline_ou_jump import _fit_ar1


def test_first_passage_matcher() -> None:
    text = """
    First Passage Time with running maximum and running minimum
    using the reflection principle. Write running_max_probs.csv,
    running_min_probs.csv and first_passage_time.csv.
    """
    assert FirstPassageTimeSkill().matches(
        instruction=text,
        task_dir=Path("."),
    )


def test_weighted_ofi_formula() -> None:
    data = {}
    for depth in range(15):
        data[f"bids_limit_notional_{depth}"] = [100.0 + depth]
        data[f"bids_cancel_notional_{depth}"] = [10.0]
        data[f"bids_market_notional_{depth}"] = [5.0]
        data[f"asks_limit_notional_{depth}"] = [80.0 + depth]
        data[f"asks_cancel_notional_{depth}"] = [4.0]
        data[f"asks_market_notional_{depth}"] = [1.0]
        data[f"bids_distance_{depth}"] = [-0.10]
        data[f"asks_distance_{depth}"] = [0.20]

    frame = pd.DataFrame(data)
    result = _weighted_ofi(frame)
    expected = (100.0 - 10.0 - 5.0) / 1.10 - (80.0 - 4.0 - 1.0) / 1.20
    assert np.isclose(
        result.loc[0, "weighted_ofi_0"],
        expected,
    )


def test_ou_ar1_exact_inversion() -> None:
    dt = 1.0 / 252.0
    kappa = 3.0
    theta = 4.5
    b = math.exp(-kappa * dt)
    a = theta * (1.0 - b)

    values = [3.0]
    for _ in range(500):
        values.append(a + b * values[-1] + 0.001 * math.sin(len(values)))

    fit = _fit_ar1(
        np.asarray(values, dtype=float),
        dt,
    )

    assert float(fit["kappa"]) > 0.0
    assert float(fit["theta"]) > 0.0
