from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from agent.offline_router import (
    rank_fallback_candidates,
    select_fallback_skill,
)


@dataclass(frozen=True)
class _Skill:
    name: str


_SKILLS = (
    _Skill(
        "portfolio-strategy-domain"
    ),
    _Skill(
        "market-risk-domain"
    ),
    _Skill(
        "fixed-income-curve-domain"
    ),
    _Skill(
        "two-asset-derivatives-domain"
    ),
    _Skill(
        "event-study-domain"
    ),
    _Skill(
        "ohlc-volatility-domain"
    ),
)


def _task(
    root: Path,
) -> Path:
    task = root / "task"
    (
        task
        / "environment"
        / "data"
    ).mkdir(
        parents=True
    )
    return task


def test_routes_related_domains_without_exact_contract_words(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path
    )

    pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-02",
            ],
            "AAA": [
                100,
                101,
            ],
            "BBB": [
                90,
                91,
            ],
            "CCC": [
                80,
                82,
            ],
        }
    ).to_csv(
        task
        / "environment"
        / "data"
        / "prices.csv",
        index=False,
    )

    skill = select_fallback_skill(
        instruction=(
            "Backtest a momentum trading strategy "
            "using the supplied price history."
        ),
        task_dir=task,
        skills=_SKILLS,
    )

    assert skill is not None
    assert (
        skill.name
        == "portfolio-strategy-domain"
    )


def test_routes_risk_language(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path
    )

    pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-01",
            ],
            "symbol": [
                "AAA",
                "BBB",
            ],
            "close": [
                100,
                90,
            ],
        }
    ).to_csv(
        task
        / "environment"
        / "data"
        / "market.csv",
        index=False,
    )

    skill = select_fallback_skill(
        instruction=(
            "Estimate portfolio VaR and expected shortfall "
            "using historical returns."
        ),
        task_dir=task,
        skills=_SKILLS,
    )

    assert skill is not None
    assert (
        skill.name
        == "market-risk-domain"
    )


def test_routes_curve_bootstrap_language(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path
    )

    (
        task
        / "environment"
        / "data"
        / "curve.json"
    ).write_text(
        json.dumps(
            {
                "maturities": [
                    1,
                    2,
                ],
                "rates": [
                    0.04,
                    0.05,
                ],
            }
        ),
        encoding="utf-8",
    )

    skill = select_fallback_skill(
        instruction=(
            "Bootstrap an OIS swap curve and report "
            "discount factors."
        ),
        task_dir=task,
        skills=_SKILLS,
    )

    assert skill is not None
    assert (
        skill.name
        == "fixed-income-curve-domain"
    )


def test_routes_realized_volatility_from_semantics_and_ohlc(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path
    )

    pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-02",
            ],
            "open": [
                100,
                101,
            ],
            "high": [
                102,
                103,
            ],
            "low": [
                99,
                100,
            ],
            "close": [
                101,
                102,
            ],
        }
    ).to_csv(
        task
        / "environment"
        / "data"
        / "ohlc.csv",
        index=False,
    )

    skill = select_fallback_skill(
        instruction=(
            "Compare realized volatility estimators "
            "from daily price ranges."
        ),
        task_dir=task,
        skills=_SKILLS,
    )

    assert skill is not None
    assert (
        skill.name
        == "ohlc-volatility-domain"
    )


def test_routes_event_study_language(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path
    )

    pd.DataFrame(
        {
            "ticker": [
                "AAA"
            ],
            "event_date": [
                "2024-01-10"
            ],
        }
    ).to_csv(
        task
        / "environment"
        / "data"
        / "events.csv",
        index=False,
    )

    skill = select_fallback_skill(
        instruction=(
            "Run an FOMC event study and calculate "
            "abnormal returns around announcements."
        ),
        task_dir=task,
        skills=_SKILLS,
    )

    assert skill is not None
    assert (
        skill.name
        == "event-study-domain"
    )


def test_does_not_misroute_unsupported_asian_option(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path
    )

    pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-02",
            ],
            "open": [
                100,
                101,
            ],
            "high": [
                102,
                103,
            ],
            "low": [
                99,
                100,
            ],
            "close": [
                101,
                102,
            ],
        }
    ).to_csv(
        task
        / "environment"
        / "data"
        / "prices.csv",
        index=False,
    )

    skill = select_fallback_skill(
        instruction=(
            "Price an Asian option using Levy and Curran "
            "approximations."
        ),
        task_dir=task,
        skills=_SKILLS,
    )

    assert skill is None


def test_candidate_ranking_is_explainable(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path
    )

    candidates = rank_fallback_candidates(
        instruction=(
            "Backtest a momentum portfolio strategy."
        ),
        task_dir=task,
        skills=_SKILLS,
    )

    assert candidates[
        0
    ].skill_name == (
        "portfolio-strategy-domain"
    )

    assert (
        candidates[
            0
        ].semantic_score
        >= 3
    )
