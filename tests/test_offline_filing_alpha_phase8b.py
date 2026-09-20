from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from agent.offline_filing_alpha import (
    FilingEventAlphaSkill,
    _extract_debt_amount_musd,
    _extract_event_date,
    _extract_guidance_midpoint_musd,
    _extract_item_numbers,
)


def test_filing_event_skill_matches() -> None:
    skill = FilingEventAlphaSkill()

    assert skill.matches(
        instruction=(
            "Read SEC 8-K filings, classify guidance, executive "
            "departure and debt financing events, calculate an "
            "event alpha score, and write filing_signals.csv."
        ),
        task_dir=Path("."),
    )


def test_extract_guidance_and_cover_fields() -> None:
    text = (
        "Date of Report (Date of earliest event reported): May 22, 2024 "
        "Item 2.02 Results. Item 7.01 Regulation FD. Item 9.01 Exhibits. "
        "Business Outlook: Revenue is expected to be $28.0 billion, "
        "plus or minus 2%."
    )

    assert (
        _extract_event_date(
            text
        )
        == "2024-05-22"
    )

    assert (
        _extract_item_numbers(
            text
        )
        == "2.02|7.01|9.01"
    )

    assert np.isclose(
        _extract_guidance_midpoint_musd(
            text
        ),
        28000.0,
    )


def test_extract_guidance_range() -> None:
    text = (
        "Forecasting fourth-quarter revenue of "
        "$13.3 billion to $14.3 billion."
    )

    assert np.isclose(
        _extract_guidance_midpoint_musd(
            text
        ),
        13800.0,
    )


def test_debt_amount_removes_exercised_overallotment() -> None:
    text = (
        "The Company issued $1,725,000,000 principal amount of Notes. "
        "The Company granted the purchasers an option to purchase an "
        "additional $225,000,000 principal amount of Notes. "
        "The Notes issued include $225,000,000 principal amount "
        "pursuant to the full exercise of such option."
    )

    assert np.isclose(
        _extract_debt_amount_musd(
            text
        ),
        1500.0,
    )
