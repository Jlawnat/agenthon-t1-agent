from __future__ import annotations

from collections.abc import Collection
from datetime import date, datetime, timedelta
import calendar


def parse_iso_date(
    value: str | date,
) -> date:
    if isinstance(value, date):
        return value

    return datetime.strptime(
        str(value),
        "%Y-%m-%d",
    ).date()


def add_months(
    value: date,
    months: int,
) -> date:
    total = (
        value.month
        - 1
        + int(months)
    )

    year = (
        value.year
        + total // 12
    )

    month = (
        total % 12
        + 1
    )

    last_day = calendar.monthrange(
        year,
        month,
    )[1]

    return date(
        year,
        month,
        min(
            value.day,
            last_day,
        ),
    )


def year_fraction(
    start: date,
    end: date,
    *,
    convention: str,
) -> float:
    normalized = (
        convention
        .upper()
        .replace(" ", "")
    )

    days = (
        end
        - start
    ).days

    if normalized == "ACT/360":
        return float(
            days / 360.0
        )

    if normalized in {
        "ACT/365",
        "ACT/365F",
    }:
        return float(
            days / 365.0
        )

    if normalized in {
        "ACT/365.25",
        "ACT/36525",
    }:
        return float(
            days / 365.25
        )

    if normalized in {
        "30/360",
        "30/360US",
    }:
        d1 = min(
            start.day,
            30,
        )

        d2 = (
            30
            if (
                d1 == 30
                and end.day == 31
            )
            else min(
                end.day,
                30,
            )
        )

        return float(
            (
                360
                * (
                    end.year
                    - start.year
                )
                + 30
                * (
                    end.month
                    - start.month
                )
                + (
                    d2
                    - d1
                )
            )
            / 360.0
        )

    raise ValueError(
        f"Unsupported day-count convention: {convention}"
    )


def generate_schedule(
    effective: date,
    maturity: date,
    frequency_months: int,
) -> list[date]:
    if frequency_months <= 0:
        raise ValueError(
            "frequency_months must be positive."
        )

    if maturity <= effective:
        return []

    dates = [
        maturity
    ]

    while True:
        previous = add_months(
            dates[-1],
            -frequency_months,
        )

        if previous <= effective:
            break

        dates.append(
            previous
        )

    dates.reverse()

    return dates


def is_business_day(
    value: date,
    *,
    holidays: Collection[date] = (),
) -> bool:
    if value.weekday() >= 5:
        return False

    return value not in holidays


def add_business_days(
    start: date,
    count: int,
    *,
    holidays: Collection[date] = (),
) -> date:
    if count < 0:
        raise ValueError(
            "count must be non-negative."
        )

    current = start
    added = 0

    while added < count:
        current += timedelta(
            days=1
        )

        if is_business_day(
            current,
            holidays=holidays,
        ):
            added += 1

    return current


def following_business_day(
    value: date,
    *,
    holidays: Collection[date] = (),
) -> date:
    current = value

    while not is_business_day(
        current,
        holidays=holidays,
    ):
        current += timedelta(
            days=1
        )

    return current


def add_months_following(
    value: date,
    months: int,
    *,
    holidays: Collection[date] = (),
) -> date:
    return following_business_day(
        add_months(
            value,
            months,
        ),
        holidays=holidays,
    )
