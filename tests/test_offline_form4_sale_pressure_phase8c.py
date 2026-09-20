from __future__ import annotations

import numpy as np

from agent.offline_form4_sale_pressure import (
    Form4SalePressureSkill,
    _price_band,
)


def test_form4_sale_pressure_skill_matches() -> None:
    skill = Form4SalePressureSkill()

    assert skill.matches(
        instruction=(
            "Reconstruct Form 4 non-derivative insider sale activity, "
            "ownership buckets, cross-sectional sale-pressure rankings."
        ),
        task_dir=None,
    )


def test_price_band_parser() -> None:
    low, high, width = _price_band(
        "The reported price is a weighted average. "
        "Shares were sold in multiple transactions at prices "
        "ranging from $106.73 to $107.73 per share, inclusive."
    )

    assert np.isclose(
        low,
        106.73,
    )

    assert np.isclose(
        high,
        107.73,
    )

    expected = (
        10000.0
        * (
            107.73
            - 106.73
        )
        / (
            (
                107.73
                + 106.73
            )
            / 2.0
        )
    )

    assert np.isclose(
        width,
        expected,
    )


def test_iot_trust_roman_numerals_do_not_collapse() -> None:
    from agent.offline_form4_sale_pressure import _normalize_entity

    footnotes = {
        "F7": (
            "Consists of shares held by Jordan Park Trust Company, LLC, "
            "Trustee of The Bicket-Dobson Trust I u/a/d 11/10/2021."
        ),
        "F9": (
            "Consists of shares held by Jordan Park Trust Company, LLC, "
            "Trustee of The Bicket-Dobson Trust II u/a/d 10/8/2021."
        ),
    }

    first = _normalize_entity(
        ownership_form="I",
        nature="See footnote",
        ownership_footnotes=["F7"],
        footnotes=footnotes,
    )

    second = _normalize_entity(
        ownership_form="I",
        nature="See footnote",
        ownership_footnotes=["F9"],
        footnotes=footnotes,
    )

    assert first == "Bicket-Dobson Trust I"
    assert second == "Bicket-Dobson Trust II"
    assert first != second
