from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import unescape
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


_MONEY = (
    r"(?:"
    r"\$?[\d,]+(?:\.\d+)?\s*(?:billion|million)"
    r"|"
    r"\$[\d,]+(?:\.\d+)?"
    r")"
)


def _find_filing_data_dir(
    task_dir: Path,
) -> Path:
    required_files = {
        "params.json",
        "universe.csv",
    }

    directories = sorted(
        {
            path.parent
            for path in task_dir.rglob("*")
            if path.is_file()
            and "checks" not in path.parts
        },
        key=lambda path: str(path),
    )

    for directory in directories:
        names = {
            path.name
            for path in directory.iterdir()
            if path.is_file()
        }

        filings_dir = (
            directory
            / "filings"
        )

        if (
            required_files.issubset(
                names
            )
            and filings_dir.is_dir()
            and any(
                filings_dir.glob(
                    "*.html"
                )
            )
        ):
            return directory

    raise RuntimeError(
        "Could not discover SEC filing-event input files."
    )


def _html_to_text(
    source: str,
) -> str:
    cleaned = re.sub(
        r"(?is)<script\b.*?</script>",
        " ",
        source,
    )

    cleaned = re.sub(
        r"(?is)<style\b.*?</style>",
        " ",
        cleaned,
    )

    cleaned = re.sub(
        r"(?is)<!--.*?-->",
        " ",
        cleaned,
    )

    cleaned = re.sub(
        r"(?s)<[^>]+>",
        " ",
        cleaned,
    )

    cleaned = unescape(
        cleaned
    )

    cleaned = (
        cleaned
        .replace(
            "\xa0",
            " ",
        )
        .replace(
            "\u0092",
            "'",
        )
    )

    return re.sub(
        r"\s+",
        " ",
        cleaned,
    ).strip()


def _normalize_name(
    value: str,
) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        value.lower(),
    ).strip()


def _extract_ticker(
    text: str,
    universe: pd.DataFrame,
) -> tuple[
    str,
    str,
]:
    normalized_text = (
        _normalize_name(
            text
        )
    )

    candidates: list[
        tuple[
            int,
            str,
            str,
        ]
    ] = []

    for row in universe.itertuples(
        index=False
    ):
        company_name = str(
            row.company_name
        )

        normalized_company = (
            _normalize_name(
                company_name
            )
        )

        if (
            normalized_company
            and normalized_company
            in normalized_text
        ):
            candidates.append(
                (
                    len(
                        normalized_company
                    ),
                    str(
                        row.ticker
                    ),
                    str(
                        row.liquidity_bucket
                    ),
                )
            )

    if not candidates:
        # Fallback: trading symbol from the cover page.
        for row in universe.itertuples(
            index=False
        ):
            ticker = str(
                row.ticker
            )

            if re.search(
                rf"\b{re.escape(ticker)}\b",
                text,
                flags=re.IGNORECASE,
            ):
                candidates.append(
                    (
                        len(
                            ticker
                        ),
                        ticker,
                        str(
                            row.liquidity_bucket
                        ),
                    )
                )

    if not candidates:
        raise RuntimeError(
            "Could not map filing to ticker universe."
        )

    candidates.sort(
        reverse=True
    )

    _length, ticker, liquidity = (
        candidates[0]
    )

    return (
        ticker,
        liquidity,
    )


def _extract_event_date(
    text: str,
) -> str:
    match = re.search(
        r"Date\s+of\s+Report"
        r"\s*\([^)]*\)"
        r"\s*:\s*"
        r"([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        raise RuntimeError(
            "Could not extract filing event date."
        )

    parsed = datetime.strptime(
        match.group(
            1
        ),
        "%B %d, %Y",
    )

    return parsed.strftime(
        "%Y-%m-%d"
    )


def _extract_item_numbers(
    text: str,
) -> str:
    numbers = {
        match.group(
            1
        )
        for match in re.finditer(
            r"\bItem\s+(\d+\.\d+)",
            text,
            flags=re.IGNORECASE,
        )
    }

    if not numbers:
        raise RuntimeError(
            "Could not extract SEC item numbers."
        )

    ordered = sorted(
        numbers,
        key=lambda value: tuple(
            int(
                part
            )
            for part in value.split(
                "."
            )
        ),
    )

    return "|".join(
        ordered
    )


def _money_to_millions(
    value: str,
) -> float:
    cleaned = (
        value.strip()
        .lower()
        .replace(
            "$",
            "",
        )
        .replace(
            ",",
            "",
        )
    )

    multiplier = 1.0

    if "billion" in cleaned:
        multiplier = 1000.0
        cleaned = cleaned.replace(
            "billion",
            "",
        )
    elif "million" in cleaned:
        cleaned = cleaned.replace(
            "million",
            "",
        )
    else:
        # Plain dollar amount such as $1,725,000,000.
        numeric = float(
            cleaned.strip()
        )

        return (
            numeric
            / 1_000_000.0
        )

    return (
        float(
            cleaned.strip()
        )
        * multiplier
    )


def _extract_guidance_midpoint_musd(
    text: str,
) -> float:
    # Explicit range:
    # "... revenue of $13.3 billion to $14.3 billion ..."
    range_pattern = re.compile(
        rf"revenue"
        rf".{{0,160}}?"
        rf"({_MONEY})"
        rf"\s*(?:to|through|[-–—])\s*"
        rf"({_MONEY})",
        flags=re.IGNORECASE,
    )

    for match in range_pattern.finditer(
        text
    ):
        context_start = max(
            0,
            match.start()
            - 250,
        )

        context = text[
            context_start:
            match.end()
            + 120
        ].lower()

        if any(
            token in context
            for token in (
                "guidance",
                "outlook",
                "forecast",
                "expect",
            )
        ):
            low = _money_to_millions(
                match.group(
                    1
                )
            )

            high = _money_to_millions(
                match.group(
                    2
                )
            )

            return (
                low
                + high
            ) / 2.0

    # Compact range:
    # "Revenue $13.3-14.3 billion"
    compact_pattern = re.compile(
        r"revenue"
        r".{0,80}?"
        r"\$?(\d+(?:\.\d+)?)"
        r"\s*[-–—]\s*"
        r"\$?(\d+(?:\.\d+)?)"
        r"\s*(billion|million)",
        flags=re.IGNORECASE,
    )

    for match in compact_pattern.finditer(
        text
    ):
        context_start = max(
            0,
            match.start()
            - 250,
        )

        context = text[
            context_start:
            match.end()
            + 120
        ].lower()

        if any(
            token in context
            for token in (
                "guidance",
                "outlook",
                "forecast",
                "expect",
            )
        ):
            unit = match.group(
                3
            )

            low = _money_to_millions(
                f"{match.group(1)} {unit}"
            )

            high = _money_to_millions(
                f"{match.group(2)} {unit}"
            )

            return (
                low
                + high
            ) / 2.0

    # Midpoint / plus-or-minus style:
    # "Revenue is expected to be $28.0 billion, plus or minus 2%."
    single_pattern = re.compile(
        rf"revenue"
        rf".{{0,100}}?"
        rf"(?:expected|forecast(?:ed|ing)?|guidance)"
        rf".{{0,80}}?"
        rf"({_MONEY})",
        flags=re.IGNORECASE,
    )

    match = single_pattern.search(
        text
    )

    if match:
        return _money_to_millions(
            match.group(
                1
            )
        )

    # Common wording puts "expected" before "revenue".
    reverse_pattern = re.compile(
        rf"(?:forecast(?:ed|ing)?|guidance|outlook)"
        rf".{{0,220}}?"
        rf"revenue"
        rf".{{0,100}}?"
        rf"({_MONEY})",
        flags=re.IGNORECASE,
    )

    match = reverse_pattern.search(
        text
    )

    if match:
        return _money_to_millions(
            match.group(
                1
            )
        )

    raise RuntimeError(
        "Could not extract revenue-guidance midpoint."
    )


def _extract_executive_departure(
    text: str,
) -> tuple[
    bool,
    str,
]:
    lowered = (
        text.lower()
    )

    departure_terms = (
        "resigned",
        "resignation",
        "stepped down",
        "step down",
    )

    if not any(
        term in lowered
        for term in departure_terms
    ):
        raise RuntimeError(
            "No executive-departure language found."
        )

    ceo_patterns = (
        r"resignation\s+of\s+chief\s+executive\s+officer",
        r"chief\s+executive\s+officer.{0,180}?(?:resigned|stepped\s+down)",
        r"(?:resigned|stepped\s+down).{0,180}?chief\s+executive\s+officer",
    )

    if any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        for pattern in ceo_patterns
    ):
        return (
            True,
            "Chief Executive Officer",
        )

    cfo_patterns = (
        r"resignation\s+of\s+chief\s+financial\s+officer",
        r"chief\s+financial\s+officer.{0,180}?(?:resigned|stepped\s+down)",
        r"(?:resigned|stepped\s+down).{0,180}?chief\s+financial\s+officer",
    )

    if any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        for pattern in cfo_patterns
    ):
        return (
            False,
            "Chief Financial Officer",
        )

    raise RuntimeError(
        "Could not determine whether departing officer is CEO or CFO."
    )


def _extract_debt_amount_musd(
    text: str,
) -> float:
    issue_pattern = re.compile(
        rf"(?:issued|offering|pricing)"
        rf".{{0,180}}?"
        rf"({_MONEY})"
        rf"\s+(?:aggregate\s+)?principal\s+amount",
        flags=re.IGNORECASE,
    )

    issue_matches = list(
        issue_pattern.finditer(
            text
        )
    )

    if not issue_matches:
        raise RuntimeError(
            "Could not extract debt principal amount."
        )

    primary = _money_to_millions(
        issue_matches[
            0
        ].group(
            1
        )
    )

    additional_pattern = re.compile(
        rf"(?:up\s+to\s+an?\s+)?additional\s+"
        rf"({_MONEY})"
        rf"\s+(?:aggregate\s+)?principal\s+amount",
        flags=re.IGNORECASE,
    )

    additional_match = (
        additional_pattern.search(
            text
        )
    )

    if additional_match:
        additional = (
            _money_to_millions(
                additional_match.group(
                    1
                )
            )
        )

        inclusion_pattern = re.compile(
            rf"(?:include|included)"
            rf".{{0,100}}?"
            rf"{re.escape(additional_match.group(1))}"
            rf".{{0,180}}?"
            rf"(?:full\s+exercise|exercise)",
            flags=re.IGNORECASE,
        )

        if (
            additional > 0.0
            and primary
            > additional
            and (
                inclusion_pattern.search(
                    text
                )
                or re.search(
                    r"Notes\s+issued"
                    r".{0,300}?"
                    r"include"
                    r".{0,250}?"
                    r"full\s+exercise",
                    text,
                    flags=re.IGNORECASE,
                )
            )
        ):
            primary -= additional

    return float(
        primary
    )


def _signal_from_score(
    score: float,
    *,
    long_threshold: float,
    short_threshold: float,
) -> str:
    if score >= long_threshold:
        return "long"

    if score <= short_threshold:
        return "short"

    return "hold"


@dataclass(frozen=True)
class FilingEventAlphaSkill:
    """Deterministic SEC filing event extraction and alpha scoring."""

    name: str = "filing-event-alpha-domain"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        del task_dir

        normalized = (
            instruction.lower()
            .replace(
                "-",
                " ",
            )
            .replace(
                "_",
                " ",
            )
        )

        return (
            (
                "8 k" in normalized
                or "sec filing" in normalized
            )
            and "alpha" in normalized
            and (
                "guidance" in normalized
                or "executive departure" in normalized
                or "debt financing" in normalized
            )
            and (
                "filing signals" in normalized
                or "event alpha score" in normalized
            )
        )

    def solve(
        self,
        *,
        instruction: str,
        task_dir: Path,
        out_dir: Path,
        seed: int,
    ) -> None:
        del instruction, seed

        data_dir = (
            _find_filing_data_dir(
                task_dir
            )
        )

        params = json.loads(
            (
                data_dir
                / "params.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        universe = pd.read_csv(
            data_dir
            / "universe.csv"
        )

        consensus = {
            str(
                key
            ): float(
                value
            )
            for key, value
            in params.get(
                "analyst_consensus_revenue_musd",
                {},
            ).items()
        }

        base_scores = params[
            "base_scores"
        ]

        liquidity_multipliers = params[
            "liquidity_multipliers"
        ]

        guidance_cap_pct = float(
            params[
                "guidance_cap_pct"
            ]
        )

        guidance_denominator = float(
            params[
                "guidance_denominator"
            ]
        )

        departure_multipliers = params[
            "executive_departure_multipliers"
        ]

        debt_threshold = float(
            params[
                "debt_large_issue_threshold_musd"
            ]
        )

        debt_large_multiplier = float(
            params[
                "debt_large_issue_multiplier"
            ]
        )

        debt_small_multiplier = float(
            params[
                "debt_small_issue_multiplier"
            ]
        )

        long_threshold = float(
            params[
                "long_threshold"
            ]
        )

        short_threshold = float(
            params[
                "short_threshold"
            ]
        )

        rows: list[
            dict[str, object]
        ] = []

        extracted: dict[
            str,
            dict[str, float]
        ] = {}

        for filing_path in sorted(
            (
                data_dir
                / "filings"
            ).glob(
                "*.html"
            )
        ):
            filing_id = (
                filing_path.stem
            )

            text = _html_to_text(
                filing_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            )

            ticker, liquidity_bucket = (
                _extract_ticker(
                    text,
                    universe,
                )
            )

            event_date = (
                _extract_event_date(
                    text
                )
            )

            item_numbers = (
                _extract_item_numbers(
                    text
                )
            )

            event_type: str
            severity_multiplier: float

            filing_values: dict[
                str,
                float
            ] = {}

            if filing_id in consensus:
                midpoint = (
                    _extract_guidance_midpoint_musd(
                        text
                    )
                )

                guidance_change_pct = (
                    (
                        midpoint
                        - consensus[
                            filing_id
                        ]
                    )
                    / consensus[
                        filing_id
                    ]
                    * 100.0
                )

                event_type = (
                    "guidance_raise"
                    if guidance_change_pct
                    > 0.0
                    else "guidance_cut"
                )

                severity_multiplier = (
                    1.0
                    + min(
                        abs(
                            guidance_change_pct
                        ),
                        guidance_cap_pct,
                    )
                    / guidance_denominator
                )

                filing_values[
                    "guidance_change_pct"
                ] = float(
                    guidance_change_pct
                )

            elif (
                re.search(
                    r"(?:resigned|resignation|stepped\s+down|step\s+down)",
                    text,
                    flags=re.IGNORECASE,
                )
                and re.search(
                    r"Chief\s+(?:Executive|Financial)\s+Officer",
                    text,
                    flags=re.IGNORECASE,
                )
            ):
                (
                    ceo_departure,
                    role,
                ) = (
                    _extract_executive_departure(
                        text
                    )
                )

                event_type = (
                    "executive_departure"
                )

                severity_multiplier = float(
                    departure_multipliers[
                        role
                    ]
                )

                filing_values[
                    "ceo_departure_flag"
                ] = float(
                    int(
                        ceo_departure
                    )
                )

            elif (
                "principal amount" in text.lower()
                and (
                    "notes" in text.lower()
                    or "debt" in text.lower()
                )
            ):
                debt_amount_musd = (
                    _extract_debt_amount_musd(
                        text
                    )
                )

                event_type = (
                    "debt_financing"
                )

                severity_multiplier = (
                    debt_large_multiplier
                    if debt_amount_musd
                    >= debt_threshold
                    else debt_small_multiplier
                )

                filing_values[
                    "debt_amount_musd"
                ] = float(
                    debt_amount_musd
                )

            else:
                raise RuntimeError(
                    f"Could not classify event type for {filing_id}."
                )

            base_score = float(
                base_scores[
                    event_type
                ]
            )

            liquidity_multiplier = float(
                liquidity_multipliers[
                    liquidity_bucket
                ]
            )

            event_alpha_score = float(
                base_score
                * severity_multiplier
                * liquidity_multiplier
            )

            signal = _signal_from_score(
                event_alpha_score,
                long_threshold=(
                    long_threshold
                ),
                short_threshold=(
                    short_threshold
                ),
            )

            rows.append(
                {
                    "filing_id": (
                        filing_id
                    ),
                    "ticker": ticker,
                    "event_date": (
                        event_date
                    ),
                    "event_type": (
                        event_type
                    ),
                    "item_numbers": (
                        item_numbers
                    ),
                    "liquidity_bucket": (
                        liquidity_bucket
                    ),
                    "base_score": (
                        base_score
                    ),
                    "severity_multiplier": (
                        float(
                            severity_multiplier
                        )
                    ),
                    "liquidity_multiplier": (
                        liquidity_multiplier
                    ),
                    "event_alpha_score": (
                        event_alpha_score
                    ),
                    "signal": signal,
                }
            )

            extracted[
                filing_id
            ] = filing_values

        if not rows:
            raise RuntimeError(
                "No SEC filings were processed."
            )

        signals = pd.DataFrame(
            rows
        ).sort_values(
            "filing_id",
            kind="stable",
        ).reset_index(
            drop=True
        )

        scores = signals[
            "event_alpha_score"
        ].to_numpy(
            dtype=float
        )

        mean_alpha = float(
            np.mean(
                scores
            )
        )

        net_score = float(
            np.sum(
                scores
            )
        )

        gross_score = float(
            np.sum(
                np.abs(
                    scores
                )
            )
        )

        top_long_index = int(
            np.argmax(
                scores
            )
        )

        top_short_index = int(
            np.argmin(
                scores
            )
        )

        results = {
            "num_filings": int(
                len(
                    signals
                )
            ),
            "num_long_signals": int(
                (
                    signals[
                        "signal"
                    ]
                    == "long"
                ).sum()
            ),
            "num_short_signals": int(
                (
                    signals[
                        "signal"
                    ]
                    == "short"
                ).sum()
            ),
            "mean_alpha_score": (
                mean_alpha
            ),
            "top_long_ticker": str(
                signals.iloc[
                    top_long_index
                ][
                    "ticker"
                ]
            ),
            "top_short_ticker": str(
                signals.iloc[
                    top_short_index
                ][
                    "ticker"
                ]
            ),
            "portfolio_net_score": (
                net_score
            ),
            "portfolio_gross_score": (
                gross_score
            ),
        }

        intermediates: dict[
            str,
            dict[
                str,
                float,
            ],
        ] = {
            "num_guidance_raise": {
                "value": float(
                    (
                        signals[
                            "event_type"
                        ]
                        == "guidance_raise"
                    ).sum()
                )
            },
            "num_guidance_cut": {
                "value": float(
                    (
                        signals[
                            "event_type"
                        ]
                        == "guidance_cut"
                    ).sum()
                )
            },
            "num_executive_departure": {
                "value": float(
                    (
                        signals[
                            "event_type"
                        ]
                        == "executive_departure"
                    ).sum()
                )
            },
            "num_debt_financing": {
                "value": float(
                    (
                        signals[
                            "event_type"
                        ]
                        == "debt_financing"
                    ).sum()
                )
            },
        }

        for filing_id, values in (
            extracted.items()
        ):
            if (
                "guidance_change_pct"
                in values
            ):
                intermediates[
                    f"guidance_change_pct_{filing_id}"
                ] = {
                    "value": float(
                        values[
                            "guidance_change_pct"
                        ]
                    )
                }

            if (
                "ceo_departure_flag"
                in values
            ):
                intermediates[
                    f"ceo_departure_flag_{filing_id}"
                ] = {
                    "value": float(
                        values[
                            "ceo_departure_flag"
                        ]
                    )
                }

            if (
                "debt_amount_musd"
                in values
            ):
                intermediates[
                    f"debt_amount_musd_{filing_id}"
                ] = {
                    "value": float(
                        values[
                            "debt_amount_musd"
                        ]
                    )
                }

        intermediates[
            "mean_severity_multiplier"
        ] = {
            "value": float(
                signals[
                    "severity_multiplier"
                ].mean()
            )
        }

        intermediates[
            "mean_liquidity_multiplier"
        ] = {
            "value": float(
                signals[
                    "liquidity_multiplier"
                ].mean()
            )
        }

        intermediates[
            "mean_alpha_score"
        ] = {
            "value": mean_alpha
        }

        intermediates[
            "portfolio_net_score"
        ] = {
            "value": net_score
        }

        intermediates[
            "portfolio_gross_score"
        ] = {
            "value": gross_score
        }

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        signals.to_csv(
            out_dir
            / "filing_signals.csv",
            index=False,
        )

        (
            out_dir
            / "results.json"
        ).write_text(
            json.dumps(
                results,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        (
            out_dir
            / "solution.json"
        ).write_text(
            json.dumps(
                {
                    "intermediates": (
                        intermediates
                    )
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
