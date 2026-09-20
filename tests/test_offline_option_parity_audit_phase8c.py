from __future__ import annotations

import numpy as np
import pandas as pd

from agent.offline_option_parity_audit import (
    OptionParityAuditSkill,
    _act365_years,
    _clean_quotes,
    _compute_pair_row,
)


def test_option_parity_skill_matches() -> None:
    skill = OptionParityAuditSkill()

    assert skill.matches(
        instruction=(
            "Audit option bid/ask quotes using put-call parity, "
            "synthetic forward bounds, and violation thresholds."
        ),
        task_dir=None,
    )


def test_act365_timestamp_fraction() -> None:
    t = _act365_years(
        "2025-03-03T15:30:00Z",
        "2025-03-21T16:00:00Z",
    )

    assert np.isclose(
        t,
        0.0493721461,
        atol=1e-10,
    )


def test_cleaning_and_forward_math() -> None:
    quotes = pd.DataFrame(
        [
            {
                "quote_id": "C",
                "quote_ts": "2025-03-03T15:30:00Z",
                "expiry_ts": "2025-03-21T16:00:00Z",
                "option_type": "call",
                "strike": 100.0,
                "bid": 2.4,
                "ask": 2.62,
                "quote_age_sec": 12,
            },
            {
                "quote_id": "P",
                "quote_ts": "2025-03-03T15:30:00Z",
                "expiry_ts": "2025-03-21T16:00:00Z",
                "option_type": "put",
                "strike": 100.0,
                "bid": 2.081198,
                "ask": 2.100666,
                "quote_age_sec": 12,
            },
            {
                "quote_id": "STALE",
                "quote_ts": "2025-03-03T15:30:00Z",
                "expiry_ts": "2025-03-21T16:00:00Z",
                "option_type": "call",
                "strike": 95.0,
                "bid": 5.5,
                "ask": 5.9,
                "quote_age_sec": 480,
            },
        ]
    )

    clean, counts = _clean_quotes(
        quotes,
        max_quote_age_sec=120,
    )

    assert len(clean) == 2
    assert counts["stale"] == 1

    call = clean[
        clean[
            "option_type"
        ]
        == "call"
    ].iloc[0]

    put = clean[
        clean[
            "option_type"
        ]
        == "put"
    ].iloc[0]

    row = _compute_pair_row(
        call=call,
        put=put,
        quote_ts="2025-03-03T15:30:00Z",
        spot_price=100.25,
        risk_free_rate=0.045,
        dividend_yield=0.012,
        reference_borrow_rate=0.0,
        violation_threshold=0.15,
    )

    assert np.isclose(
        row[
            "synthetic_forward_bid"
        ],
        100.3,
        atol=5e-7,
    )

    assert np.isclose(
        row[
            "synthetic_forward_ask"
        ],
        100.54,
        atol=5e-7,
    )

    assert row[
        "violation_type"
    ] == "none"
