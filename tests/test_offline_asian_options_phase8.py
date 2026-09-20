from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from agent.offline_asian_options import (
    AsianOptionSkill,
    calibrate_gbm,
    geometric_asian_call,
    levy_asian_call,
    curran_asian_call,
)


def test_asian_pricing_primitives() -> None:
    kwargs = {
        "S0": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.20,
        "n_monitoring": 12,
    }

    geo = geometric_asian_call(
        **kwargs
    )
    levy = levy_asian_call(
        **kwargs
    )
    curran = curran_asian_call(
        **kwargs
    )

    assert geo > 0.0
    assert levy > 0.0
    assert curran > 0.0
    assert geo <= levy + 1.0


def test_asian_skill_calibration_and_match(
    tmp_path: Path,
) -> None:
    dates = pd.bdate_range(
        "2024-01-02",
        periods=30,
    )

    close = 100.0 * np.exp(
        0.001
        * np.arange(
            30
        )
        + 0.01
        * np.sin(
            np.arange(
                30
            )
        )
    )

    frame = pd.DataFrame(
        {
            "date": dates,
            "close": close,
        }
    )

    calibration = calibrate_gbm(
        frame
    )

    assert calibration[
        "n_prices"
    ] == 30

    assert calibration[
        "sigma"
    ] > 0.0

    skill = AsianOptionSkill()

    assert skill.matches(
        instruction=(
            "Price arithmetic Asian options with exact geometric Asian, "
            "Levy moment matching, Curran conditioning and Monte Carlo."
        ),
        task_dir=tmp_path,
    )
