from __future__ import annotations

import numpy as np
import pytest

from agent.offline_common import (
    volume_scheduling as extracted,
)
from agent.offline_intraday_volume import (
    _largest_remainder as mature_largest_remainder,
    _normalize_profile as mature_normalize_profile,
    _profiles as mature_profiles,
    _r2 as mature_r2,
)


def _history() -> np.ndarray:
    return np.array(
        [
            [0.10, 0.20, 0.30, 0.40],
            [0.20, 0.20, 0.20, 0.40],
            [0.15, 0.25, 0.25, 0.35],
            [0.30, 0.10, 0.20, 0.40],
            [0.12, 0.18, 0.40, 0.30],
        ],
        dtype=float,
    )


def test_normalize_profile_matches_mature() -> None:
    values = np.array(
        [
            -1.0,
            2.0,
            3.0,
            5.0,
        ]
    )

    expected = mature_normalize_profile(
        values
    )

    actual = extracted.normalize_profile(
        values
    )

    assert np.allclose(
        actual,
        expected,
    )


def test_mean_profile_matches_mature() -> None:
    history = _history()

    expected = mature_profiles(
        history
    )[
        "historical_mean_profile"
    ]

    actual = extracted.mean_volume_profile(
        history
    )

    assert np.allclose(
        actual,
        expected,
    )


def test_median_profile_matches_mature() -> None:
    history = _history()

    expected = mature_profiles(
        history
    )[
        "historical_median_profile"
    ]

    actual = extracted.median_volume_profile(
        history
    )

    assert np.allclose(
        actual,
        expected,
    )


def test_ewma_profile_matches_mature_half_life_20() -> None:
    history = _history()

    expected = mature_profiles(
        history
    )[
        "ewma_profile"
    ]

    actual = (
        extracted.exponentially_weighted_volume_profile(
            history,
            half_life=20.0,
        )
    )

    assert np.allclose(
        actual,
        expected,
    )


def test_winsorized_profile_matches_mature() -> None:
    history = _history()

    expected = mature_profiles(
        history
    )[
        "winsorized_mean_profile"
    ]

    actual = (
        extracted.winsorized_mean_volume_profile(
            history,
            lower_percentile=5.0,
            upper_percentile=95.0,
        )
    )

    assert np.allclose(
        actual,
        expected,
    )


def test_profile_r_squared_matches_mature() -> None:
    realized = np.array(
        [
            0.1,
            0.2,
            0.3,
            0.4,
        ]
    )

    predicted = np.array(
        [
            0.12,
            0.18,
            0.29,
            0.41,
        ]
    )

    expected = mature_r2(
        realized,
        predicted,
    )

    actual = extracted.profile_r_squared(
        realized,
        predicted,
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_largest_remainder_matches_mature() -> None:
    weights = np.array(
        [
            0.37,
            0.27,
            0.22,
            0.14,
        ]
    )

    expected = mature_largest_remainder(
        101,
        weights,
    )

    actual = (
        extracted.largest_remainder_allocation(
            101,
            weights,
        )
    )

    assert np.array_equal(
        actual,
        expected,
    )


def test_largest_remainder_preserves_total_quantity() -> None:
    allocation = (
        extracted.largest_remainder_allocation(
            137,
            np.array(
                [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                ]
            ),
        )
    )

    assert allocation.dtype == np.int64
    assert int(
        allocation.sum()
    ) == 137


def test_largest_remainder_ties_favour_earlier_index() -> None:
    actual = (
        extracted.largest_remainder_allocation(
            2,
            np.array(
                [
                    1.0,
                    1.0,
                    1.0,
                ]
            ),
        )
    )

    assert np.array_equal(
        actual,
        np.array(
            [
                1,
                1,
                0,
            ],
            dtype=np.int64,
        ),
    )


def test_zero_profile_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="positive total",
    ):
        extracted.normalize_profile(
            np.zeros(4)
        )


def test_invalid_half_life_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="half_life",
    ):
        extracted.exponentially_weighted_volume_profile(
            _history(),
            half_life=0.0,
        )


def test_invalid_percentiles_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="percentiles",
    ):
        extracted.winsorized_mean_volume_profile(
            _history(),
            lower_percentile=90.0,
            upper_percentile=10.0,
        )


from pathlib import Path
import json

from agent.candidate_runner import run_candidate
from agent.candidate_workspace import CandidateWorkspace
from agent.capability_bridge import (
    CANDIDATE_LIBRARY_EXPORTS,
    rank_capabilities,
)


def _task(
    tmp_path: Path,
    instruction: str,
) -> Path:
    task = tmp_path / "task"

    (
        task
        / "environment"
        / "data"
    ).mkdir(
        parents=True,
    )

    (
        task
        / "instruction.md"
    ).write_text(
        instruction,
        encoding="utf-8",
    )

    return task


def test_volume_scheduling_routing_is_specific(
    tmp_path: Path,
) -> None:
    instruction = (
        "Fit intraday volume-share models, estimate a "
        "volume-share profile, and produce the final execution schedule."
    )

    task = _task(
        tmp_path,
        instruction,
    )

    selected = rank_capabilities(
        instruction=instruction,
        task_dir=task,
    )

    assert (
        "intraday-volume-scheduling"
        in {
            item.descriptor.capability_id
            for item in selected
        }
    )

    negative_instruction = (
        "Estimate daily portfolio volatility "
        "from historical closing prices."
    )

    negative_task = _task(
        tmp_path / "negative",
        negative_instruction,
    )

    selected = rank_capabilities(
        instruction=negative_instruction,
        task_dir=negative_task,
    )

    assert (
        "intraday-volume-scheduling"
        not in {
            item.descriptor.capability_id
            for item in selected
        }
    )


def test_curated_volume_scheduling_has_exact_surface(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path,
        "Evaluate intraday volume scheduling.",
    )

    workspace = CandidateWorkspace.create(
        base_dir=tmp_path / "workspaces",
        candidate_id=1,
        task_dir=task,
    )

    try:
        module_path = (
            workspace.root_dir
            / "lib"
            / "offline_common"
            / "volume_scheduling.py"
        )

        namespace = {}

        exec(
            compile(
                module_path.read_text(
                    encoding="utf-8",
                ),
                str(module_path),
                "exec",
            ),
            namespace,
        )

        assert tuple(
            namespace["__all__"]
        ) == (
            CANDIDATE_LIBRARY_EXPORTS[
                "volume_scheduling.py"
            ]
        )

    finally:
        workspace.cleanup()


def test_candidate_executes_volume_scheduling_helpers(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path,
        "Evaluate intraday volume scheduling.",
    )

    workspace = CandidateWorkspace.create(
        base_dir=tmp_path / "workspaces",
        candidate_id=1,
        task_dir=task,
    )

    try:
        script_path = (
            workspace.source_dir
            / "solver.py"
        )

        script_path.write_text(
            '''
import json
import numpy as np

from offline_common.volume_scheduling import (
    largest_remainder_allocation,
    mean_volume_profile,
    profile_r_squared,
)

history = np.array(
    [
        [0.10, 0.20, 0.30, 0.40],
        [0.20, 0.20, 0.20, 0.40],
        [0.15, 0.25, 0.25, 0.35],
    ]
)

profile = mean_volume_profile(
    history
)

allocation = largest_remainder_allocation(
    101,
    profile,
)

score = profile_r_squared(
    profile,
    profile,
)

payload = {
    "profile_sum": float(profile.sum()),
    "allocation_sum": int(allocation.sum()),
    "score": score,
}

with open(
    "output/volume_probe.json",
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        payload,
        handle,
    )
'''.strip()
            + "\n",
            encoding="utf-8",
        )

        completed = run_candidate(
            workspace,
            script_path,
            timeout_seconds=30,
        )

        assert completed.return_code == 0

        payload = json.loads(
            (
                workspace.output_dir
                / "volume_probe.json"
            ).read_text(
                encoding="utf-8",
            )
        )

        assert np.isclose(
            payload["profile_sum"],
            1.0,
        )

        assert (
            payload["allocation_sum"]
            == 101
        )

        assert np.isclose(
            payload["score"],
            1.0,
        )

    finally:
        workspace.cleanup()
