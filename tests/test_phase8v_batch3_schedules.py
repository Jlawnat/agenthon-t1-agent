from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from agent.offline_common import schedules as extracted
from agent.offline_dual_curve import (
    act_360 as mature_act_360,
    act_365_25 as mature_act_365_25,
    add_months as mature_add_months,
    generate_schedule as mature_generate_schedule,
    thirty_360 as mature_thirty_360,
)
from agent.offline_fx_carry_hedge import (
    _add_common_business_days as mature_add_business_days,
    _add_months_following as mature_add_months_following,
    _following_adjust as mature_following_adjust,
    _is_business_day as mature_is_business_day,
)


def test_parse_iso_date() -> None:
    expected = date(
        2026,
        10,
        1,
    )

    assert (
        extracted.parse_iso_date(
            "2026-10-01"
        )
        == expected
    )

    assert (
        extracted.parse_iso_date(
            expected
        )
        == expected
    )


@pytest.mark.parametrize(
    ("start", "months"),
    [
        (
            date(2024, 1, 31),
            1,
        ),
        (
            date(2024, 3, 31),
            -1,
        ),
        (
            date(2025, 8, 31),
            6,
        ),
        (
            date(2025, 12, 15),
            14,
        ),
    ],
)
def test_add_months_matches_mature(
    start: date,
    months: int,
) -> None:
    assert (
        extracted.add_months(
            start,
            months,
        )
        == mature_add_months(
            start,
            months,
        )
    )


def test_act_360_matches_mature() -> None:
    start = date(
        2025,
        1,
        15,
    )

    end = date(
        2025,
        7,
        15,
    )

    assert (
        extracted.year_fraction(
            start,
            end,
            convention="ACT/360",
        )
        == mature_act_360(
            start,
            end,
        )
    )


def test_act_365_25_matches_mature() -> None:
    start = date(
        2024,
        2,
        29,
    )

    end = date(
        2025,
        2,
        28,
    )

    assert (
        extracted.year_fraction(
            start,
            end,
            convention="ACT/365.25",
        )
        == mature_act_365_25(
            start,
            end,
        )
    )


def test_act_365() -> None:
    start = date(
        2025,
        1,
        1,
    )

    end = date(
        2026,
        1,
        1,
    )

    assert (
        extracted.year_fraction(
            start,
            end,
            convention="ACT/365",
        )
        == 1.0
    )

    assert (
        extracted.year_fraction(
            start,
            end,
            convention="ACT/365F",
        )
        == 1.0
    )


def test_thirty_360_matches_mature() -> None:
    start = date(
        2025,
        1,
        31,
    )

    end = date(
        2025,
        7,
        31,
    )

    assert (
        extracted.year_fraction(
            start,
            end,
            convention="30/360",
        )
        == mature_thirty_360(
            start,
            end,
        )
    )


def test_generate_schedule_matches_mature() -> None:
    effective = date(
        2025,
        1,
        15,
    )

    maturity = date(
        2027,
        1,
        15,
    )

    assert (
        extracted.generate_schedule(
            effective,
            maturity,
            6,
        )
        == mature_generate_schedule(
            effective,
            maturity,
            6,
        )
    )


def test_business_day_matches_mature() -> None:
    holidays = {
        "USD": {
            date(
                2026,
                1,
                1,
            ),
        },
        "EUR": {
            date(
                2026,
                1,
                2,
            ),
        },
    }

    generic_holidays = (
        holidays["USD"]
        | holidays["EUR"]
    )

    for value in (
        date(2026, 1, 1),
        date(2026, 1, 2),
        date(2026, 1, 3),
        date(2026, 1, 5),
    ):
        assert (
            extracted.is_business_day(
                value,
                holidays=generic_holidays,
            )
            == mature_is_business_day(
                value,
                ["USD", "EUR"],
                holidays,
            )
        )


def test_add_business_days_matches_mature() -> None:
    holidays = {
        "USD": {
            date(
                2026,
                1,
                1,
            ),
        },
        "EUR": {
            date(
                2026,
                1,
                2,
            ),
        },
    }

    generic_holidays = (
        holidays["USD"]
        | holidays["EUR"]
    )

    start = date(
        2025,
        12,
        31,
    )

    expected = (
        mature_add_business_days(
            start,
            2,
            ["USD", "EUR"],
            holidays,
        )
        .date()
    )

    actual = (
        extracted.add_business_days(
            start,
            2,
            holidays=generic_holidays,
        )
    )

    assert actual == expected


def test_following_adjust_matches_mature() -> None:
    holidays = {
        "USD": set(),
        "EUR": set(),
    }

    value = date(
        2026,
        1,
        3,
    )

    expected = (
        mature_following_adjust(
            value,
            ["USD", "EUR"],
            holidays,
        )
        .date()
    )

    actual = (
        extracted.following_business_day(
            value,
        )
    )

    assert actual == expected


def test_add_months_following_matches_mature() -> None:
    holidays = {
        "USD": set(),
        "EUR": set(),
    }

    start = date(
        2026,
        1,
        31,
    )

    expected = (
        mature_add_months_following(
            start,
            1,
            ["USD", "EUR"],
            holidays,
        )
        .date()
    )

    actual = (
        extracted.add_months_following(
            start,
            1,
        )
    )

    assert actual == expected


def test_invalid_convention_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported day-count",
    ):
        extracted.year_fraction(
            date(2025, 1, 1),
            date(2025, 2, 1),
            convention="UNKNOWN",
        )


def test_invalid_frequency_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        extracted.generate_schedule(
            date(2025, 1, 1),
            date(2026, 1, 1),
            0,
        )
