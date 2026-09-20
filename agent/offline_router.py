from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Iterable, Protocol

import pandas as pd


class RoutableSkill(Protocol):
    name: str


@dataclass(frozen=True)
class TaskFingerprint:
    filenames: tuple[str, ...]
    csv_columns: tuple[frozenset[str], ...]
    json_keys: tuple[frozenset[str], ...]


@dataclass(frozen=True)
class RouteCandidate:
    skill_name: str
    semantic_score: int
    data_score: int

    @property
    def total_score(self) -> int:
        return (
            self.semantic_score
            + self.data_score
        )


def _normalise(
    text: str,
) -> str:
    return re.sub(
        r"\s+",
        " ",
        text.lower()
        .replace("–", "-")
        .replace("—", "-"),
    ).strip()


def inspect_task_fingerprint(
    task_dir: Path,
) -> TaskFingerprint:
    filenames: list[str] = []
    csv_columns: list[
        frozenset[str]
    ] = []
    json_keys: list[
        frozenset[str]
    ] = []

    for path in sorted(
        task_dir.rglob("*")
    ):
        if (
            not path.is_file()
            or "checks" in path.parts
        ):
            continue

        filenames.append(
            path.name.lower()
        )

        if path.suffix.lower() == ".csv":
            try:
                frame = pd.read_csv(
                    path,
                    nrows=3,
                )
            except Exception:
                continue

            csv_columns.append(
                frozenset(
                    str(column).lower()
                    for column
                    in frame.columns
                )
            )

        elif path.suffix.lower() == ".json":
            try:
                value = json.loads(
                    path.read_text(
                        encoding="utf-8",
                        errors="replace",
                    )
                )
            except Exception:
                continue

            if isinstance(
                value,
                dict,
            ):
                json_keys.append(
                    frozenset(
                        str(key).lower()
                        for key
                        in value.keys()
                    )
                )

    return TaskFingerprint(
        filenames=tuple(
            filenames
        ),
        csv_columns=tuple(
            csv_columns
        ),
        json_keys=tuple(
            json_keys
        ),
    )


def _contains(
    text: str,
    phrase: str,
) -> bool:
    if phrase == "var":
        return bool(
            re.search(
                r"\bvar\b",
                text,
            )
        )

    if phrase == "car":
        return bool(
            re.search(
                r"\bcar\b",
                text,
            )
        )

    return (
        phrase in text
    )


def _score_terms(
    text: str,
    weighted_terms: Iterable[
        tuple[
            int,
            tuple[str, ...],
        ]
    ],
) -> int:
    score = 0

    for weight, alternatives in weighted_terms:
        if any(
            _contains(
                text,
                phrase,
            )
            for phrase
            in alternatives
        ):
            score += weight

    return score


def _has_csv_schema(
    fingerprint: TaskFingerprint,
    required: set[str],
) -> bool:
    return any(
        required.issubset(
            set(
                columns
            )
        )
        for columns
        in fingerprint.csv_columns
    )


def _has_json_keys(
    fingerprint: TaskFingerprint,
    required: set[str],
) -> bool:
    return any(
        required.issubset(
            set(
                keys
            )
        )
        for keys
        in fingerprint.json_keys
    )


def _portfolio_scores(
    text: str,
    fingerprint: TaskFingerprint,
) -> tuple[int, int]:
    semantic = _score_terms(
        text,
        (
            (
                4,
                (
                    "cross-sectional momentum",
                    "cross sectional momentum",
                ),
            ),
            (
                3,
                (
                    "momentum",
                    "relative strength",
                ),
            ),
            (
                2,
                (
                    "backtest",
                    "portfolio strategy",
                    "trading strategy",
                ),
            ),
            (
                2,
                (
                    "long-short",
                    "long short",
                    "long/short",
                ),
            ),
        ),
    )

    wide_price_panel = any(
        (
            "date" in columns
            and len(
                columns
            ) >= 4
            and not {
                "open",
                "high",
                "low",
                "close",
            }.issubset(
                columns
            )
        )
        for columns
        in fingerprint.csv_columns
    )

    data = (
        2
        if wide_price_panel
        else 0
    )

    return (
        semantic,
        data,
    )


def _risk_scores(
    text: str,
    fingerprint: TaskFingerprint,
) -> tuple[int, int]:
    semantic = _score_terms(
        text,
        (
            (
                4,
                (
                    "value-at-risk",
                    "value at risk",
                ),
            ),
            (
                4,
                (
                    "expected shortfall",
                    "conditional value at risk",
                    "cvar",
                ),
            ),
            (
                2,
                (
                    "var",
                ),
            ),
            (
                2,
                (
                    "historical simulation",
                    "historical var",
                ),
            ),
            (
                1,
                (
                    "ewma",
                    "student-t",
                    "student t",
                    "risk measure",
                ),
            ),
        ),
    )

    long_market = _has_csv_schema(
        fingerprint,
        {
            "date",
            "symbol",
            "close",
        },
    )

    data = (
        2
        if long_market
        else 0
    )

    return (
        semantic,
        data,
    )


def _fixed_income_scores(
    text: str,
    fingerprint: TaskFingerprint,
) -> tuple[int, int]:
    semantic = _score_terms(
        text,
        (
            (
                3,
                (
                    "bootstrap",
                    "bootstrapping",
                ),
            ),
            (
                3,
                (
                    "ois",
                    "swap curve",
                    "yield curve",
                    "zero-coupon curve",
                    "zero coupon curve",
                ),
            ),
            (
                2,
                (
                    "discount factor",
                    "zero rate",
                    "forward rate",
                    "par rate",
                ),
            ),
        ),
    )

    curve_json = (
        _has_json_keys(
            fingerprint,
            {
                "maturities",
                "par_rates",
            },
        )
        or _has_json_keys(
            fingerprint,
            {
                "maturities",
                "rates",
            },
        )
    )

    data = (
        2
        if curve_json
        else 0
    )

    return (
        semantic,
        data,
    )


def _derivatives_scores(
    text: str,
    fingerprint: TaskFingerprint,
) -> tuple[int, int]:
    del fingerprint

    # Only route to the existing two-asset engine when the task actually
    # describes the class of derivatives that engine supports.  Generic
    # "option" wording is intentionally insufficient; e.g. Asian options
    # require a different engine.
    semantic = _score_terms(
        text,
        (
            (
                5,
                (
                    "margrabe",
                    "kirk",
                ),
            ),
            (
                4,
                (
                    "spread option",
                    "exchange option",
                ),
            ),
            (
                2,
                (
                    "two-asset",
                    "two asset",
                    "correlated assets",
                ),
            ),
        ),
    )

    return (
        semantic,
        0,
    )


def _event_scores(
    text: str,
    fingerprint: TaskFingerprint,
) -> tuple[int, int]:
    semantic = _score_terms(
        text,
        (
            (
                5,
                (
                    "event study",
                    "event-study",
                ),
            ),
            (
                3,
                (
                    "abnormal return",
                    "abnormal returns",
                ),
            ),
            (
                2,
                (
                    "caar",
                    "cumulative abnormal return",
                ),
            ),
            (
                1,
                (
                    "announcement",
                    "earnings event",
                    "fomc",
                    "event date",
                ),
            ),
        ),
    )

    event_csv = _has_csv_schema(
        fingerprint,
        {
            "ticker",
            "event_date",
        },
    )

    data = (
        3
        if event_csv
        else 0
    )

    return (
        semantic,
        data,
    )


def _volatility_scores(
    text: str,
    fingerprint: TaskFingerprint,
) -> tuple[int, int]:
    semantic = _score_terms(
        text,
        (
            (
                5,
                (
                    "realized volatility",
                    "realised volatility",
                    "realized vol",
                    "realised vol",
                ),
            ),
            (
                2,
                (
                    "parkinson",
                    "garman-klass",
                    "garman klass",
                    "rogers-satchell",
                    "rogers satchell",
                    "yang-zhang",
                    "yang zhang",
                ),
            ),
            (
                2,
                (
                    "ohlc",
                    "range-based volatility",
                    "range based volatility",
                ),
            ),
        ),
    )

    ohlc = _has_csv_schema(
        fingerprint,
        {
            "date",
            "open",
            "high",
            "low",
            "close",
        },
    )

    data = (
        3
        if ohlc
        else 0
    )

    return (
        semantic,
        data,
    )


_SCORERS = {
    "portfolio-strategy-domain": (
        _portfolio_scores
    ),
    "market-risk-domain": (
        _risk_scores
    ),
    "fixed-income-curve-domain": (
        _fixed_income_scores
    ),
    "two-asset-derivatives-domain": (
        _derivatives_scores
    ),
    "event-study-domain": (
        _event_scores
    ),
    "ohlc-volatility-domain": (
        _volatility_scores
    ),
}


def rank_fallback_candidates(
    *,
    instruction: str,
    task_dir: Path,
    skills: Iterable[
        RoutableSkill
    ],
) -> list[
    RouteCandidate
]:
    text = _normalise(
        instruction
    )

    fingerprint = (
        inspect_task_fingerprint(
            task_dir
        )
    )

    skill_names = {
        skill.name
        for skill
        in skills
    }

    candidates = []

    for skill_name, scorer in _SCORERS.items():
        if skill_name not in skill_names:
            continue

        semantic_score, data_score = scorer(
            text,
            fingerprint,
        )

        candidates.append(
            RouteCandidate(
                skill_name=skill_name,
                semantic_score=(
                    semantic_score
                ),
                data_score=(
                    data_score
                ),
            )
        )

    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.total_score,
            candidate.semantic_score,
            candidate.data_score,
            candidate.skill_name,
        ),
        reverse=True,
    )


def select_fallback_skill(
    *,
    instruction: str,
    task_dir: Path,
    skills: Iterable[
        RoutableSkill
    ],
    minimum_semantic_score: int = 3,
    minimum_total_score: int = 5,
) -> RoutableSkill | None:
    skill_list = list(
        skills
    )

    by_name = {
        skill.name: skill
        for skill
        in skill_list
    }

    candidates = (
        rank_fallback_candidates(
            instruction=instruction,
            task_dir=task_dir,
            skills=skill_list,
        )
    )

    if not candidates:
        return None

    best = candidates[
        0
    ]

    if (
        best.semantic_score
        < minimum_semantic_score
        or best.total_score
        < minimum_total_score
    ):
        return None

    # Avoid arbitrary routing when two domains are genuinely tied.
    if len(
        candidates
    ) >= 2:
        second = candidates[
            1
        ]

        if (
            second.semantic_score
            >= minimum_semantic_score
            and second.total_score
            == best.total_score
        ):
            return None

    return by_name[
        best.skill_name
    ]
