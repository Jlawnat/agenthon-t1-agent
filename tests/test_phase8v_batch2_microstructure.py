from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent.offline_common import microstructure as extracted
from agent.offline_tca import (
    _quote_at_or_after as mature_quote_at_or_after,
)


def test_first_index_matches_mature_quote_lookup() -> None:
    quotes = [
        {
            "transaction_time": 100,
            "best_bid_price": 99.0,
        },
        {
            "transaction_time": 150,
            "best_bid_price": 100.0,
        },
        {
            "transaction_time": 220,
            "best_bid_price": 101.0,
        },
    ]

    times = [
        int(row["transaction_time"])
        for row in quotes
    ]

    expected = mature_quote_at_or_after(
        quotes,
        times,
        120,
    )

    index = extracted.first_index_at_or_after(
        times,
        120,
    )

    assert quotes[index] == expected


def test_first_index_accepts_exact_timestamp() -> None:
    assert (
        extracted.first_index_at_or_after(
            [100, 150, 220],
            150,
        )
        == 1
    )


def test_first_index_rejects_target_after_end() -> None:
    with pytest.raises(
        ValueError,
        match="at or after",
    ):
        extracted.first_index_at_or_after(
            [100, 150],
            200,
        )


def test_half_open_window_indices() -> None:
    timestamps = [
        100,
        150,
        200,
        250,
        300,
    ]

    left, right = (
        extracted.half_open_window_indices(
            timestamps,
            start=150,
            end=250,
        )
    )

    assert (
        left,
        right,
    ) == (
        1,
        3,
    )

    assert timestamps[left:right] == [
        150,
        200,
    ]


def test_unsorted_timestamps_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="sorted",
    ):
        extracted.first_index_at_or_after(
            [100, 90, 110],
            95,
        )


def test_participation_capped_quantity() -> None:
    actual = (
        extracted.participation_capped_quantity(
            remaining_quantity=10.0,
            market_volume=50.0,
            participation_cap=0.10,
            max_child_quantity=4.0,
        )
    )

    expected = min(
        10.0,
        0.10 * 50.0,
        4.0,
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_zero_market_volume_yields_zero_child() -> None:
    assert (
        extracted.participation_capped_quantity(
            remaining_quantity=10.0,
            market_volume=0.0,
            participation_cap=0.25,
            max_child_quantity=5.0,
        )
        == 0.0
    )


def test_exact_synchronized_venue_rows() -> None:
    frame = pd.DataFrame(
        [
            ["A", 100, "X", 0.40],
            ["A", 100, "Y", 0.60],

            ["B", 100, "X", 0.45],
            ["B", 100, "X", 0.46],
            ["B", 100, "Y", 0.55],

            ["C", 100, "X", 0.50],

            ["D", 100, "X", 0.30],
            ["D", 100, "Y", 0.70],
            ["D", 100, "Z", 0.50],

            ["A", 200, "X", 0.42],
            ["A", 200, "Y", 0.58],
        ],
        columns=[
            "pair_id",
            "timestamp",
            "venue",
            "price",
        ],
    )

    result = (
        extracted.exact_synchronized_venue_rows(
            frame,
            group_columns=[
                "pair_id",
                "timestamp",
            ],
            venue_column="venue",
            required_venues=[
                "X",
                "Y",
            ],
        )
    )

    assert (
        result[
            [
                "pair_id",
                "timestamp",
            ]
        ]
        .drop_duplicates()
        .to_records(
            index=False
        )
        .tolist()
        == [
            ("A", 100),
            ("A", 200),
        ]
    )

    assert len(result) == 4


def test_exact_sync_requires_unique_required_venues() -> None:
    frame = pd.DataFrame(
        {
            "id": ["A"],
            "timestamp": [1],
            "venue": ["X"],
        }
    )

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        extracted.exact_synchronized_venue_rows(
            frame,
            group_columns=[
                "id",
                "timestamp",
            ],
            venue_column="venue",
            required_venues=[
                "X",
                "X",
            ],
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


def test_microstructure_routing_is_specific(
    tmp_path: Path,
) -> None:
    positive_cases = (
        (
            "tca",
            (
                "Reconstruct participation-capped fills "
                "bucket-by-bucket and use the first quote "
                "at or after each bucket start."
            ),
        ),
        (
            "sync",
            (
                "Build synchronized snapshots across venues "
                "using exact timestamps."
            ),
        ),
    )

    for name, instruction in positive_cases:
        task = _task(
            tmp_path / name,
            instruction,
        )

        selected = rank_capabilities(
            instruction=instruction,
            task_dir=task,
        )

        assert (
            "microstructure-temporal-alignment"
            in {
                item.descriptor.capability_id
                for item in selected
            }
        )

    negative_instruction = (
        "Estimate a daily return forecast from historical prices."
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
        "microstructure-temporal-alignment"
        not in {
            item.descriptor.capability_id
            for item in selected
        }
    )


def test_curated_microstructure_has_exact_surface(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path,
        "Evaluate temporal market alignment.",
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
            / "microstructure.py"
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
                "microstructure.py"
            ]
        )

    finally:
        workspace.cleanup()


def test_candidate_executes_microstructure_helpers(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path,
        "Evaluate temporal market alignment.",
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
import pandas as pd

from offline_common.microstructure import (
    exact_synchronized_venue_rows,
    first_index_at_or_after,
    half_open_window_indices,
    participation_capped_quantity,
)

times = [100, 150, 220]

frame = pd.DataFrame(
    [
        ["A", 100, "X"],
        ["A", 100, "Y"],
        ["B", 100, "X"],
    ],
    columns=[
        "id",
        "timestamp",
        "venue",
    ],
)

sync = exact_synchronized_venue_rows(
    frame,
    group_columns=[
        "id",
        "timestamp",
    ],
    venue_column="venue",
    required_venues=[
        "X",
        "Y",
    ],
)

payload = {
    "first_index": first_index_at_or_after(
        times,
        120,
    ),
    "window": list(
        half_open_window_indices(
            times,
            start=100,
            end=220,
        )
    ),
    "child": participation_capped_quantity(
        10.0,
        50.0,
        0.1,
        4.0,
    ),
    "sync_rows": len(sync),
}

with open(
    "output/microstructure_probe.json",
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
                / "microstructure_probe.json"
            ).read_text(
                encoding="utf-8",
            )
        )

        assert payload == {
            "first_index": 1,
            "window": [0, 2],
            "child": 4.0,
            "sync_rows": 2,
        }

    finally:
        workspace.cleanup()
