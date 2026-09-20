from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd


def _find_parity_audit_data_dir(
    task_dir: Path,
) -> Path:
    required = {
        "option_quotes.csv",
        "spot.json",
        "carry_inputs.json",
        "audit_rules.json",
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

        if required.issubset(names):
            return directory

    raise RuntimeError(
        "Could not discover option parity-audit input files."
    )


def _parse_utc(
    value: str,
) -> datetime:
    text = str(value).strip()

    if text.endswith("Z"):
        text = (
            text[:-1]
            + "+00:00"
        )

    parsed = datetime.fromisoformat(
        text
    )

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=timezone.utc,
        )

    return parsed.astimezone(
        timezone.utc
    )


def _iso_z(
    value: str,
) -> str:
    parsed = _parse_utc(
        value
    )

    return (
        parsed
        .replace(
            microsecond=0
        )
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        )
    )


def _act365_years(
    start: str,
    end: str,
) -> float:
    delta = (
        _parse_utc(
            end
        )
        - _parse_utc(
            start
        )
    )

    return float(
        delta.total_seconds()
        / (
            365.0
            * 24.0
            * 3600.0
        )
    )


def _round6(
    value: float,
) -> float:
    return round(
        float(
            value
        ),
        6,
    )


def _fixed6(
    value: float,
) -> str:
    return f"{float(value):.6f}"


def _extract_task_slug(
    instruction: str,
) -> str:
    match = re.search(
        r'"task_slug"\s*:\s*"([^"]+)"',
        instruction,
    )

    if match:
        return match.group(
            1
        )

    return (
        "option-put-call-parity-forward-audit"
    )


def _clean_quotes(
    quotes: pd.DataFrame,
    *,
    max_quote_age_sec: float,
) -> tuple[
    pd.DataFrame,
    dict[
        str,
        int,
    ],
]:
    required = {
        "quote_id",
        "quote_ts",
        "expiry_ts",
        "option_type",
        "strike",
        "bid",
        "ask",
        "quote_age_sec",
    }

    missing = (
        required
        - set(
            quotes.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Option quote file is missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    frame = (
        quotes.copy()
        .reset_index(
            drop=True
        )
    )

    for column in (
        "strike",
        "bid",
        "ask",
        "quote_age_sec",
    ):
        frame[
            column
        ] = pd.to_numeric(
            frame[
                column
            ],
            errors="raise",
        )

    counts = {
        "stale": 0,
        "crossed": 0,
        "one_sided": 0,
        "missing_pair": 0,
    }

    clean_mask = []

    for row in frame.itertuples(
        index=False
    ):
        bid = float(
            row.bid
        )

        ask = float(
            row.ask
        )

        age = float(
            row.quote_age_sec
        )

        reason = None

        if (
            bid <= 0.0
            or ask <= 0.0
        ):
            reason = (
                "one_sided"
            )
        elif bid >= ask:
            reason = (
                "crossed"
            )
        elif age > float(
            max_quote_age_sec
        ):
            reason = (
                "stale"
            )

        if reason is None:
            clean_mask.append(
                True
            )
        else:
            clean_mask.append(
                False
            )

            counts[
                reason
            ] += 1

    clean = (
        frame.loc[
            clean_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    clean[
        "option_type"
    ] = (
        clean[
            "option_type"
        ]
        .astype(str)
        .str.lower()
    )

    clean[
        "expiry_ts"
    ] = clean[
        "expiry_ts"
    ].map(
        _iso_z
    )

    return (
        clean,
        counts,
    )


def _pair_clean_quotes(
    clean: pd.DataFrame,
    counts: dict[
        str,
        int,
    ],
) -> list[
    tuple[
        pd.Series,
        pd.Series,
    ]
]:
    pairs = []

    grouped = clean.groupby(
        [
            "expiry_ts",
            "strike",
        ],
        sort=True,
    )

    for _key, group in grouped:
        calls = group[
            group[
                "option_type"
            ]
            == "call"
        ]

        puts = group[
            group[
                "option_type"
            ]
            == "put"
        ]

        if (
            len(
                calls
            )
            == 1
            and len(
                puts
            )
            == 1
        ):
            pairs.append(
                (
                    calls.iloc[
                        0
                    ],
                    puts.iloc[
                        0
                    ],
                )
            )
            continue

        counts[
            "missing_pair"
        ] += int(
            len(
                calls
            )
            + len(
                puts
            )
        )

    pairs.sort(
        key=lambda pair: (
            str(
                pair[
                    0
                ][
                    "expiry_ts"
                ]
            ),
            float(
                pair[
                    0
                ][
                    "strike"
                ]
            ),
        )
    )

    return pairs


def _compute_pair_row(
    *,
    call: pd.Series,
    put: pd.Series,
    quote_ts: str,
    spot_price: float,
    risk_free_rate: float,
    dividend_yield: float,
    reference_borrow_rate: float,
    violation_threshold: float,
) -> dict[
    str,
    object,
]:
    expiry_ts = str(
        call[
            "expiry_ts"
        ]
    )

    strike = float(
        call[
            "strike"
        ]
    )

    if not np.isclose(
        strike,
        float(
            put[
                "strike"
            ]
        ),
        atol=1e-12,
    ):
        raise RuntimeError(
            "Matched call/put pair has inconsistent strikes."
        )

    t = _act365_years(
        quote_ts,
        expiry_ts,
    )

    if t <= 0.0:
        raise RuntimeError(
            "Option expiry must be after quote timestamp."
        )

    discount_factor = math.exp(
        -risk_free_rate
        * t
    )

    call_bid = float(
        call[
            "bid"
        ]
    )

    call_ask = float(
        call[
            "ask"
        ]
    )

    put_bid = float(
        put[
            "bid"
        ]
    )

    put_ask = float(
        put[
            "ask"
        ]
    )

    synthetic_bid = (
        strike
        + (
            call_bid
            - put_ask
        )
        / discount_factor
    )

    synthetic_ask = (
        strike
        + (
            call_ask
            - put_bid
        )
        / discount_factor
    )

    synthetic_mid = (
        synthetic_bid
        + synthetic_ask
    ) / 2.0

    theoretical_forward = (
        spot_price
        * math.exp(
            (
                risk_free_rate
                - dividend_yield
                - reference_borrow_rate
            )
            * t
        )
    )

    implied_borrow_rate = (
        risk_free_rate
        - dividend_yield
        - math.log(
            synthetic_mid
            / spot_price
        )
        / t
    )

    violation_amount = max(
        synthetic_bid
        - theoretical_forward,
        theoretical_forward
        - synthetic_ask,
        0.0,
    )

    violation_type = (
        "none"
    )

    if (
        violation_amount
        > violation_threshold
    ):
        if (
            theoretical_forward
            < synthetic_bid
        ):
            violation_type = (
                "forward_too_high"
            )
        elif (
            theoretical_forward
            > synthetic_ask
        ):
            violation_type = (
                "forward_too_low"
            )

    return {
        "expiry_ts": (
            expiry_ts
        ),
        "strike": strike,
        "call_quote_id": str(
            call[
                "quote_id"
            ]
        ),
        "put_quote_id": str(
            put[
                "quote_id"
            ]
        ),
        "time_to_expiry_years": t,
        "call_bid": call_bid,
        "call_ask": call_ask,
        "put_bid": put_bid,
        "put_ask": put_ask,
        "synthetic_forward_bid": (
            synthetic_bid
        ),
        "synthetic_forward_ask": (
            synthetic_ask
        ),
        "synthetic_forward_mid": (
            synthetic_mid
        ),
        "theoretical_forward": (
            theoretical_forward
        ),
        "implied_borrow_rate": (
            implied_borrow_rate
        ),
        "violation_amount": (
            violation_amount
        ),
        "violation_type": (
            violation_type
        ),
    }


def _write_fixed6_csv(
    *,
    path: Path,
    rows: list[
        dict[
            str,
            object,
        ]
    ],
    columns: list[
        str
    ],
    numeric_columns: set[
        str
    ],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )

        writer.writeheader()

        for row in rows:
            output = {}

            for column in columns:
                value = row[
                    column
                ]

                if column in numeric_columns:
                    output[
                        column
                    ] = _fixed6(
                        float(
                            value
                        )
                    )
                else:
                    output[
                        column
                    ] = value

            writer.writerow(
                output
            )


@dataclass(frozen=True)
class OptionParityAuditSkill:
    """Executable bid/ask put-call parity and synthetic-forward audit."""

    name: str = "option-parity-audit-domain"

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
                "put call parity"
                in normalized
                or (
                    "synthetic forward"
                    in normalized
                    and "option" in normalized
                )
            )
            and (
                "bid ask"
                in normalized
                or "bid/ask"
                in instruction.lower()
            )
            and (
                "violation"
                in normalized
                or "audit"
                in normalized
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
        del seed

        data_dir = (
            _find_parity_audit_data_dir(
                task_dir
            )
        )

        quotes = pd.read_csv(
            data_dir
            / "option_quotes.csv"
        )

        spot = json.loads(
            (
                data_dir
                / "spot.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        carry = json.loads(
            (
                data_dir
                / "carry_inputs.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        rules = json.loads(
            (
                data_dir
                / "audit_rules.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        quote_ts = _iso_z(
            str(
                spot[
                    "quote_ts"
                ]
            )
        )

        spot_price = float(
            spot[
                "spot_price"
            ]
        )

        risk_free_rate = float(
            carry[
                "risk_free_rate"
            ]
        )

        dividend_yield = float(
            carry[
                "dividend_yield"
            ]
        )

        reference_borrow_rate = float(
            carry.get(
                "reference_borrow_rate",
                0.0,
            )
        )

        max_quote_age = float(
            rules[
                "max_quote_age_sec"
            ]
        )

        violation_threshold = float(
            rules[
                "violation_threshold"
            ]
        )

        clean, counts = (
            _clean_quotes(
                quotes,
                max_quote_age_sec=(
                    max_quote_age
                ),
            )
        )

        pairs = _pair_clean_quotes(
            clean,
            counts,
        )

        audit_rows = [
            _compute_pair_row(
                call=call,
                put=put,
                quote_ts=quote_ts,
                spot_price=(
                    spot_price
                ),
                risk_free_rate=(
                    risk_free_rate
                ),
                dividend_yield=(
                    dividend_yield
                ),
                reference_borrow_rate=(
                    reference_borrow_rate
                ),
                violation_threshold=(
                    violation_threshold
                ),
            )
            for call, put
            in pairs
        ]

        if not audit_rows:
            raise RuntimeError(
                "No clean matched call-put pairs were found."
            )

        violation_rows = [
            row
            for row
            in audit_rows
            if float(
                row[
                    "violation_amount"
                ]
            )
            > violation_threshold
        ]

        forwards = [
            float(
                row[
                    "synthetic_forward_mid"
                ]
            )
            for row
            in audit_rows
        ]

        borrow_rates = [
            float(
                row[
                    "implied_borrow_rate"
                ]
            )
            for row
            in audit_rows
        ]

        worst_row = max(
            audit_rows,
            key=lambda row: float(
                row[
                    "violation_amount"
                ]
            ),
        )

        results = {
            "schema_version": (
                "1.0.0"
            ),
            "task_slug": (
                _extract_task_slug(
                    instruction
                )
            ),
            "quote_ts": quote_ts,
            "spot_price": _round6(
                spot_price
            ),
            "matched_pair_count": int(
                len(
                    audit_rows
                )
            ),
            "parity_break_count": int(
                len(
                    violation_rows
                )
            ),
            "worst_abs_violation": (
                _round6(
                    max(
                        float(
                            row[
                                "violation_amount"
                            ]
                        )
                        for row
                        in audit_rows
                    )
                )
            ),
            "median_synthetic_forward": (
                _round6(
                    float(
                        np.median(
                            forwards
                        )
                    )
                )
            ),
            "median_implied_borrow_rate": (
                _round6(
                    float(
                        np.median(
                            borrow_rates
                        )
                    )
                )
            ),
            "excluded_stale_count": int(
                counts[
                    "stale"
                ]
            ),
            "excluded_crossed_count": int(
                counts[
                    "crossed"
                ]
            ),
            "excluded_one_sided_count": int(
                counts[
                    "one_sided"
                ]
            ),
            "excluded_missing_pair_count": int(
                counts[
                    "missing_pair"
                ]
            ),
        }

        matched_pair_ids = [
            (
                f"{row['expiry_ts']}|"
                f"{float(row['strike']):.6f}"
            )
            for row
            in audit_rows
        ]

        checkpoint_forwards = {
            pair_id: _round6(
                float(
                    row[
                        "synthetic_forward_mid"
                    ]
                )
            )
            for pair_id, row
            in zip(
                matched_pair_ids,
                audit_rows,
            )
        }

        worst_pair = (
            f"{worst_row['expiry_ts']}|"
            f"{float(worst_row['strike']):.6f}"
        )

        solution = {
            "intermediates": {
                "exclusion_counts": {
                    "value": {
                        "stale": int(
                            counts[
                                "stale"
                            ]
                        ),
                        "crossed": int(
                            counts[
                                "crossed"
                            ]
                        ),
                        "one_sided": int(
                            counts[
                                "one_sided"
                            ]
                        ),
                        "missing_pair": int(
                            counts[
                                "missing_pair"
                            ]
                        ),
                    }
                },
                "matched_pair_ids": {
                    "value": (
                        matched_pair_ids
                    )
                },
                "checkpoint_forwards": {
                    "value": (
                        checkpoint_forwards
                    )
                },
                "worst_violation_pair": {
                    "value": (
                        worst_pair
                    )
                },
            }
        }

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
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

        audit_columns = [
            "expiry_ts",
            "strike",
            "call_quote_id",
            "put_quote_id",
            "time_to_expiry_years",
            "call_bid",
            "call_ask",
            "put_bid",
            "put_ask",
            "synthetic_forward_bid",
            "synthetic_forward_ask",
            "synthetic_forward_mid",
            "theoretical_forward",
            "implied_borrow_rate",
            "violation_amount",
            "violation_type",
        ]

        audit_numeric = {
            "strike",
            "time_to_expiry_years",
            "call_bid",
            "call_ask",
            "put_bid",
            "put_ask",
            "synthetic_forward_bid",
            "synthetic_forward_ask",
            "synthetic_forward_mid",
            "theoretical_forward",
            "implied_borrow_rate",
            "violation_amount",
        }

        _write_fixed6_csv(
            path=(
                out_dir
                / "parity_audit.csv"
            ),
            rows=audit_rows,
            columns=audit_columns,
            numeric_columns=(
                audit_numeric
            ),
        )

        violation_columns = [
            "expiry_ts",
            "strike",
            "call_quote_id",
            "put_quote_id",
            "violation_type",
            "violation_amount",
            "synthetic_forward_bid",
            "synthetic_forward_ask",
            "theoretical_forward",
        ]

        violation_numeric = {
            "strike",
            "violation_amount",
            "synthetic_forward_bid",
            "synthetic_forward_ask",
            "theoretical_forward",
        }

        _write_fixed6_csv(
            path=(
                out_dir
                / "violations.csv"
            ),
            rows=(
                violation_rows
            ),
            columns=(
                violation_columns
            ),
            numeric_columns=(
                violation_numeric
            ),
        )

        (
            out_dir
            / "solution.json"
        ).write_text(
            json.dumps(
                solution,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
