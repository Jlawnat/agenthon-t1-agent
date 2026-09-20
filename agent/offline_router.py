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
    json_key_paths: tuple[str, ...]


@dataclass(frozen=True)
class RouteCandidate:
    skill_name: str
    semantic_score: int
    data_score: int

    @property
    def total_score(self) -> int:
        return self.semantic_score + self.data_score


def _normalise(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text.lower().replace("–", "-").replace("—", "-").replace("_", " "),
    ).strip()


def _collect_json_paths(value, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            part = str(key).lower()
            path = f"{prefix}.{part}" if prefix else part
            paths.append(path)
            paths.extend(_collect_json_paths(child, path))
    elif isinstance(value, list):
        for child in value[:5]:
            paths.extend(_collect_json_paths(child, prefix + "[]"))
    return paths


def inspect_task_fingerprint(task_dir: Path) -> TaskFingerprint:
    filenames: list[str] = []
    csv_columns: list[frozenset[str]] = []
    json_keys: list[frozenset[str]] = []
    json_key_paths: list[str] = []

    for path in sorted(task_dir.rglob("*")):
        if not path.is_file() or "checks" in path.parts:
            continue

        filenames.append(path.name.lower())

        if path.suffix.lower() == ".csv":
            try:
                frame = pd.read_csv(path, nrows=3)
            except Exception:
                continue
            csv_columns.append(
                frozenset(str(column).lower() for column in frame.columns)
            )

        elif path.suffix.lower() == ".json":
            try:
                value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                continue
            if isinstance(value, dict):
                json_keys.append(frozenset(str(key).lower() for key in value.keys()))
                json_key_paths.extend(_collect_json_paths(value))

    return TaskFingerprint(
        filenames=tuple(filenames),
        csv_columns=tuple(csv_columns),
        json_keys=tuple(json_keys),
        json_key_paths=tuple(sorted(set(json_key_paths))),
    )


def _contains(text: str, phrase: str) -> bool:
    if phrase in {"var", "car", "pca", "fx"}:
        return bool(re.search(rf"\b{re.escape(phrase)}\b", text))
    return phrase in text


def _score_terms(text: str, weighted_terms) -> int:
    score = 0
    for weight, alternatives in weighted_terms:
        if any(_contains(text, phrase) for phrase in alternatives):
            score += weight
    return score


def _has_csv_schema(fp: TaskFingerprint, required: set[str]) -> bool:
    return any(required.issubset(set(columns)) for columns in fp.csv_columns)


def _has_json_keys(fp: TaskFingerprint, required: set[str]) -> bool:
    return any(required.issubset(set(keys)) for keys in fp.json_keys)


def _has_files(fp: TaskFingerprint, required: set[str]) -> bool:
    return required.issubset(set(fp.filenames))


def _profile_score(text: str, fp: TaskFingerprint, profile: dict) -> tuple[int, int]:
    semantic = _score_terms(text, profile.get("terms", ()))
    data = 0
    for required, weight in profile.get("files", ()):
        if _has_files(fp, set(required)):
            data += weight
    for required, weight in profile.get("csv", ()):
        if _has_csv_schema(fp, set(required)):
            data += weight
    for required, weight in profile.get("json", ()):
        if _has_json_keys(fp, set(required)):
            data += weight
    for fragment, weight in profile.get("json_paths", ()):
        if any(fragment in path for path in fp.json_key_paths):
            data += weight
    return semantic, data


_PROFILES = {
    "portfolio-strategy-domain": {
        "terms": (
            (4, ("cross-sectional momentum", "cross sectional momentum")),
            (3, ("momentum", "relative strength")),
            (2, ("backtest", "portfolio strategy", "trading strategy")),
            (2, ("long-short", "long short", "long/short")),
        ),
        "csv": ((("date",), 1),),
    },
    "market-risk-domain": {
        "terms": (
            (4, ("value-at-risk", "value at risk")),
            (4, ("expected shortfall", "conditional value at risk", "cvar")),
            (2, ("var",)),
            (1, ("ewma", "student-t", "student t", "risk measure")),
        ),
        "csv": ((("date", "symbol", "close"), 2),),
    },
    "fixed-income-curve-domain": {
        "terms": (
            (3, ("bootstrap", "bootstrapping")),
            (3, ("ois", "swap curve", "yield curve", "zero-coupon curve", "zero coupon curve")),
            (2, ("discount factor", "zero rate", "forward rate", "par rate")),
        ),
        "json": (
            (("maturities", "par_rates"), 2),
            (("maturities", "rates"), 2),
        ),
    },
    "two-asset-derivatives-domain": {
        "terms": (
            (5, ("margrabe", "kirk")),
            (4, ("spread option", "exchange option")),
            (2, ("two-asset", "two asset", "correlated assets")),
        ),
    },
    "event-study-domain": {
        "terms": (
            (5, ("event study", "event-study")),
            (3, ("abnormal return", "abnormal returns")),
            (2, ("caar", "cumulative abnormal return")),
        ),
        "csv": ((("ticker", "event_date"), 3),),
    },
    "ohlc-volatility-domain": {
        "terms": (
            (5, ("realized volatility", "realised volatility", "realized vol", "realised vol")),
            (2, ("parkinson", "garman-klass", "garman klass", "rogers-satchell", "yang-zhang")),
            (2, ("ohlc", "range-based volatility", "range based volatility")),
        ),
        "csv": ((("date", "open", "high", "low", "close"), 3),),
    },
    "credit-migration-matrix-domain": {
        "terms": (
            (5, ("credit migration", "rating migration")),
            (4, ("transition matrix",)),
            (2, ("generator matrix", "markov")),
        ),
        "files": (((("config.json", "historical_transition_matrix.csv")), 2),),
    },
    "yield-curve-pca-dynamics-domain": {
        "terms": ((5, ("yield curve",)), (4, ("pca", "principal component")), (2, ("nelson-siegel", "nelson siegel"))),
        "files": (((("daily-treasury-rates.csv",)), 3),),
    },
    "fx-forward-cross-rate-domain": {
        "terms": ((5, ("fx forward",)), (3, ("cross rate", "triangulat")), (2, ("covered interest parity", "cip"))),
        "files": (((("spot_rates.csv", "deposit_rates.csv", "portfolio.csv")), 3),),
    },
    "tail-index-estimation-domain": {
        "terms": ((5, ("tail index",)), (4, ("hill", "smith")), (3, ("gpd", "generalized pareto"))),
    },
    "earnings-surprise-domain": {
        "terms": ((5, ("earnings surprise",)), (4, ("standardized unexpected earnings", "sue"))),
        "json_paths": (("companies[].quarters", 3),),
    },
    "implied-vol-approximation-domain": {
        "terms": ((5, ("implied volatility",)), (3, ("approximation", "brenner", "corrado", "newton"))),
    },
    "credit-spread-decomposition-domain": {
        "terms": ((5, ("credit spread",)), (4, ("newey-west", "hac")), (3, ("variance decomposition",))),
        "files": (((("baa.csv", "aaa.csv", "dgs10.csv", "tedrate.csv", "t10yie.csv")), 5),),
    },
    "fx-carry-forward-hedge-domain": {
        "terms": ((5, ("fx carry",)), (4, ("forward hedge",)), (3, ("garman-kohlhagen", "smile", "25-delta", "25d"))),
        "files": (((("spot_bidoffer.json", "deposit_bidoffer.json", "option_params.json", "market_forwards.json")), 5),),
    },
    "pca-factor-portfolio-domain": {
        "terms": ((5, ("pca", "principal component")), (4, ("factor neutral", "factor-neutral")), (3, ("hedge weights",))),
        "json": ((("tickers", "monthly_returns", "n_factors", "target_weights"), 5),),
    },
    "barrier-garch-var-domain": {
        "terms": ((5, ("barrier", "down-and-out")), (4, ("garch",)), (3, ("var", "value at risk"))),
        "csv": ((("date", "log_return"), 3),),
        "json": ((("s0", "k", "b", "confidence_level", "n_contracts"), 4),),
    },
    "copula-sampling-rank-correlation-domain": {
        "terms": ((5, ("copula",)), (4, ("kendall",)), (3, ("tail dependence", "spearman"))),
    },
    "interest-rate-cap-floor-domain": {
        "terms": ((5, ("interest rate cap", "caplet")), (5, ("floorlet", "interest rate floor")), (3, ("black's model", "black model"))),
        "json": ((("forward_rates", "vol", "strike", "discount_factors"), 5),),
    },
}


_STOP_TOKENS = {
    "domain", "offline", "generic", "engine", "skill", "strategy", "risk",
    "pricing", "analysis", "model", "calculator", "estimation",
}


def _name_overlap_score(skill_name: str, text: str) -> int:
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", skill_name.lower())
        if len(token) >= 3 and token not in _STOP_TOKENS
    ]
    matches = sum(1 for token in tokens if _contains(text, token))
    return min(matches * 2, 8)


def rank_fallback_candidates(
    *,
    instruction: str,
    task_dir: Path,
    skills: Iterable[RoutableSkill],
) -> list[RouteCandidate]:
    text = _normalise(instruction)
    fp = inspect_task_fingerprint(task_dir)
    candidates: list[RouteCandidate] = []

    for skill in skills:
        profile = _PROFILES.get(skill.name, {})
        semantic, data = _profile_score(text, fp, profile)
        semantic = max(semantic, _name_overlap_score(skill.name, text))
        candidates.append(
            RouteCandidate(
                skill_name=skill.name,
                semantic_score=semantic,
                data_score=data,
            )
        )

    return sorted(
        candidates,
        key=lambda c: (c.total_score, c.semantic_score, c.data_score, c.skill_name),
        reverse=True,
    )


def select_fallback_skill(
    *,
    instruction: str,
    task_dir: Path,
    skills: Iterable[RoutableSkill],
    minimum_semantic_score: int = 3,
    minimum_total_score: int = 5,
) -> RoutableSkill | None:
    skill_list = list(skills)
    by_name = {skill.name: skill for skill in skill_list}
    candidates = rank_fallback_candidates(
        instruction=instruction,
        task_dir=task_dir,
        skills=skill_list,
    )
    eligible = [
        candidate for candidate in candidates
        if candidate.semantic_score >= minimum_semantic_score
        and candidate.total_score >= minimum_total_score
    ]
    if not eligible:
        return None
    best = eligible[0]
    if len(eligible) >= 2 and eligible[1].total_score == best.total_score:
        return None
    return by_name[best.skill_name]
