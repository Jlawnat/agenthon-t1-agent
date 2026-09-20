from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


TRANSACTION_COLUMNS = [
    "filing_id",
    "issuer_name",
    "ticker",
    "reporting_owner_name",
    "transaction_index",
    "transaction_date",
    "security_title",
    "transaction_code",
    "shares",
    "price_per_share",
    "gross_value",
    "acquired_disposed_code",
    "shares_following_transaction",
    "ownership_form",
    "ownership_entity",
    "price_footnote_number",
    "price_band_low",
    "price_band_high",
    "price_band_width_bps",
    "ownership_footnotes",
    "rule_10b5_1_flag",
]

ROLL_COLUMNS = [
    "filing_id",
    "ticker",
    "report_date",
    "ownership_entity",
    "ownership_form",
    "ownership_footnotes",
    "starting_shares",
    "sold_on_report_date",
    "ending_shares",
    "pct_reported_holdings_sold",
    "is_seller",
    "is_static_holding_only",
]

ENTITY_COLUMNS = [
    "filing_id",
    "ticker",
    "issuer_name",
    "transaction_date",
    "ownership_entity",
    "inventory_before_trading",
    "sale_transactions",
    "shares_sold",
    "gross_proceeds",
    "weighted_average_sale_price",
    "ending_shares_after_last_trade",
    "pct_inventory_sold",
    "shares_sold_over_adv20",
    "dollars_sold_over_adv20_dollar_volume",
    "liquidation_score",
]

ISSUER_COLUMNS = [
    "filing_id",
    "ticker",
    "issuer_name",
    "transaction_date",
    "shares_sold",
    "gross_proceeds",
    "weighted_average_sale_price",
    "seller_entities",
    "seller_concentration_hhi",
    "max_entity_pct_of_day_sales",
    "sale_shares_over_day_volume",
    "sale_shares_over_adv20",
    "sale_dollars_over_adv20_dollar_volume",
    "sale_shares_over_float",
    "sale_shares_over_outstanding",
    "sale_shares_over_reported_inventory",
    "blended_pressure_score",
]

RANK_COLUMNS = [
    "filing_id",
    "ticker",
    "transaction_date",
    "rank_shares_sold",
    "rank_gross_proceeds",
    "rank_sale_shares_over_day_volume",
    "rank_sale_shares_over_adv20",
    "rank_sale_shares_over_float",
    "rank_sale_shares_over_reported_inventory",
    "rank_blended_pressure_score",
    "best_rank",
    "worst_rank",
    "rank_spread",
    "top_metric_count",
]

DECOMP_COLUMNS = [
    "filing_id",
    "ticker",
    "transaction_date",
    "direct_sale_shares",
    "indirect_sale_shares",
    "direct_sale_pct",
    "indirect_sale_pct",
    "top_entity_name",
    "top_entity_sale_shares",
    "top_entity_sale_pct",
    "tail_entity_sale_shares",
    "tail_entity_sale_pct",
    "price_range_covered_shares",
    "price_range_coverage_pct",
    "reported_inventory_before_trading",
    "seller_ending_shares",
    "static_only_ending_shares",
    "direct_seller_flag",
    "static_holding_only_entities",
]


def _find_form4_data_dir(task_dir: Path) -> Path:
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
        xmls = list(
            directory.glob(
                "form4_*.xml"
            )
        )

        if (
            xmls
            and (
                directory
                / "market_context.csv"
            ).is_file()
        ):
            return directory

    raise RuntimeError(
        "Could not discover Form 4 sale-pressure input files."
    )


def _value(
    element: ET.Element,
    path: str,
    default: str = "",
) -> str:
    node = element.find(
        path
    )

    if node is None:
        return default

    value = node.find(
        "value"
    )

    if (
        value is not None
        and value.text is not None
    ):
        return value.text.strip()

    if node.text is not None:
        return node.text.strip()

    return default


def _footnote_ids(
    element: ET.Element | None,
) -> list[str]:
    if element is None:
        return []

    result = []

    for node in element.iter(
        "footnoteId"
    ):
        value = str(
            node.attrib.get(
                "id",
                "",
            )
        ).strip()

        if value:
            result.append(
                value
            )

    return result


def _footnote_number(
    footnote_id: str,
) -> int | None:
    match = re.fullmatch(
        r"[Ff]?(\d+)",
        str(
            footnote_id
        ).strip(),
    )

    if not match:
        return None

    return int(
        match.group(
            1
        )
    )


def _price_band(
    text: str,
) -> tuple[
    float | None,
    float | None,
    float | None,
]:
    match = re.search(
        r"(?:ranging|range)"
        r".{0,50}?"
        r"\$([0-9]+(?:\.[0-9]+)?)"
        r"\s+(?:to|through|-)\s+"
        r"\$([0-9]+(?:\.[0-9]+)?)",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return (
            None,
            None,
            None,
        )

    low = float(
        match.group(
            1
        )
    )

    high = float(
        match.group(
            2
        )
    )

    midpoint = (
        low
        + high
    ) / 2.0

    width_bps = (
        10000.0
        * (
            high
            - low
        )
        / midpoint
    )

    return (
        low,
        high,
        float(
            width_bps
        ),
    )


def _normalize_entity(
    *,
    ownership_form: str,
    nature: str,
    ownership_footnotes: list[str],
    footnotes: dict[str, str],
) -> str:
    if ownership_form == "D":
        return "Direct holdings"

    cleaned = re.sub(
        r"\s+",
        " ",
        str(
            nature
        ).strip(),
    )

    if (
        cleaned
        and cleaned.lower()
        not in {
            "see footnote",
            "see footnotes",
        }
    ):
        return cleaned

    texts = [
        footnotes.get(
            footnote_id,
            "",
        )
        for footnote_id
        in ownership_footnotes
    ]

    joined = " ".join(
        texts
    )

    # Resolve named ownership buckets using whole-name matching. A plain
    # substring check is unsafe for Roman-numeral trust names because
    # "Bicket-Dobson Trust I" is a prefix of "Bicket-Dobson Trust II".
    special = (
        "Bicket-Dobson Trust II",
        "Bicket-Dobson Trust I",
        "Bicket Revocable Trust",
    )

    for name in special:
        if re.search(
            rf"(?<!\w){re.escape(name)}(?!\w)",
            joined,
            flags=re.IGNORECASE,
        ):
            return name

    trustee = re.search(
        r"Trustee\s+of\s+The\s+(.+?)\s+u/a/d",
        joined,
        flags=re.IGNORECASE,
    )

    if trustee:
        return re.sub(
            r"\s+",
            " ",
            trustee.group(
                1
            ),
        ).strip()

    held = re.search(
        r"shares\s+held\s+by\s+(?:the\s+)?(.+?)(?:\.|,\s+over|\s+u/a/d)",
        joined,
        flags=re.IGNORECASE,
    )

    if held:
        return re.sub(
            r"\s+",
            " ",
            held.group(
                1
            ),
        ).strip()

    if cleaned:
        return cleaned

    return "Indirect holdings"


def _parse_filing(
    path: Path,
) -> tuple[
    list[
        dict[
            str,
            object,
        ]
    ],
    list[
        dict[
            str,
            object,
        ]
    ],
    dict[
        str,
        str,
    ],
]:
    root = ET.parse(
        path
    ).getroot()

    filing_id = (
        path.stem
        .replace(
            "form4_",
            "",
        )
        .lower()
    )

    issuer_name = _value(
        root,
        "issuer/issuerName",
    )

    ticker = _value(
        root,
        "issuer/issuerTradingSymbol",
    )

    owner_name = _value(
        root,
        "reportingOwner/reportingOwnerId/rptOwnerName",
    )

    report_date = _value(
        root,
        "periodOfReport",
    )

    footnotes = {}

    for node in root.findall(
        ".//footnotes/footnote"
    ):
        footnote_id = str(
            node.attrib.get(
                "id",
                "",
            )
        ).strip()

        if footnote_id:
            footnotes[
                footnote_id
            ] = re.sub(
                r"\s+",
                " ",
                "".join(
                    node.itertext()
                ),
            ).strip()

    filing_has_10b5 = any(
        (
            "10b5-1" in text.lower()
            or "10b5 1" in text.lower()
        )
        for text
        in footnotes.values()
    )

    transactions = []

    retained_index = 0

    for transaction in root.findall(
        ".//nonDerivativeTransaction"
    ):
        code = _value(
            transaction,
            "transactionCoding/transactionCode",
        )

        if code != "S":
            continue

        retained_index += 1

        shares = float(
            _value(
                transaction,
                "transactionAmounts/transactionShares",
                "0",
            )
        )

        price = float(
            _value(
                transaction,
                "transactionAmounts/transactionPricePerShare",
                "0",
            )
        )

        post = float(
            _value(
                transaction,
                "postTransactionAmounts/sharesOwnedFollowingTransaction",
                "0",
            )
        )

        ownership_form = _value(
            transaction,
            "ownershipNature/directOrIndirectOwnership",
        )

        nature = _value(
            transaction,
            "ownershipNature/natureOfOwnership",
        )

        ownership_ids = []

        post_node = transaction.find(
            "postTransactionAmounts"
        )

        ownership_node = transaction.find(
            "ownershipNature"
        )

        for footnote_id in (
            _footnote_ids(
                post_node
            )
            + _footnote_ids(
                ownership_node
            )
        ):
            if (
                footnote_id
                not in ownership_ids
            ):
                ownership_ids.append(
                    footnote_id
                )

        entity = _normalize_entity(
            ownership_form=(
                ownership_form
            ),
            nature=nature,
            ownership_footnotes=(
                ownership_ids
            ),
            footnotes=footnotes,
        )

        price_node = transaction.find(
            "transactionAmounts/transactionPricePerShare"
        )

        price_ids = _footnote_ids(
            price_node
        )

        price_id = (
            price_ids[
                0
            ]
            if price_ids
            else ""
        )

        price_number = (
            _footnote_number(
                price_id
            )
        )

        (
            price_low,
            price_high,
            price_width,
        ) = _price_band(
            footnotes.get(
                price_id,
                "",
            )
        )

        coding_ids = _footnote_ids(
            transaction.find(
                "transactionCoding"
            )
        )

        transaction_10b5 = bool(
            filing_has_10b5
            or any(
                (
                    "10b5-1"
                    in footnotes.get(
                        footnote_id,
                        "",
                    ).lower()
                )
                for footnote_id
                in coding_ids
            )
        )

        ownership_numbers = [
            str(
                number
            )
            for number
            in (
                _footnote_number(
                    footnote_id
                )
                for footnote_id
                in ownership_ids
            )
            if number is not None
        ]

        transactions.append(
            {
                "filing_id": filing_id,
                "issuer_name": (
                    issuer_name
                ),
                "ticker": ticker,
                "reporting_owner_name": (
                    owner_name
                ),
                "transaction_index": int(
                    retained_index
                ),
                "transaction_date": _value(
                    transaction,
                    "transactionDate",
                ),
                "security_title": _value(
                    transaction,
                    "securityTitle",
                ),
                "transaction_code": (
                    code
                ),
                "shares": int(
                    round(
                        shares
                    )
                ),
                "price_per_share": (
                    float(
                        price
                    )
                ),
                "gross_value": float(
                    shares
                    * price
                ),
                "acquired_disposed_code": (
                    _value(
                        transaction,
                        "transactionAmounts/transactionAcquiredDisposedCode",
                    )
                ),
                "shares_following_transaction": int(
                    round(
                        post
                    )
                ),
                "ownership_form": (
                    ownership_form
                ),
                "ownership_entity": (
                    entity
                ),
                "price_footnote_number": (
                    price_number
                ),
                "price_band_low": (
                    price_low
                ),
                "price_band_high": (
                    price_high
                ),
                "price_band_width_bps": (
                    price_width
                ),
                "ownership_footnotes": (
                    ";".join(
                        ownership_numbers
                    )
                ),
                "rule_10b5_1_flag": (
                    transaction_10b5
                ),
            }
        )

    static_holdings = []

    for holding in root.findall(
        ".//nonDerivativeHolding"
    ):
        post = float(
            _value(
                holding,
                "postTransactionAmounts/sharesOwnedFollowingTransaction",
                "0",
            )
        )

        ownership_form = _value(
            holding,
            "ownershipNature/directOrIndirectOwnership",
        )

        nature = _value(
            holding,
            "ownershipNature/natureOfOwnership",
        )

        ownership_ids = []

        for footnote_id in (
            _footnote_ids(
                holding.find(
                    "postTransactionAmounts"
                )
            )
            + _footnote_ids(
                holding.find(
                    "ownershipNature"
                )
            )
        ):
            if (
                footnote_id
                not in ownership_ids
            ):
                ownership_ids.append(
                    footnote_id
                )

        entity = _normalize_entity(
            ownership_form=(
                ownership_form
            ),
            nature=nature,
            ownership_footnotes=(
                ownership_ids
            ),
            footnotes=footnotes,
        )

        ownership_numbers = [
            str(
                number
            )
            for number
            in (
                _footnote_number(
                    footnote_id
                )
                for footnote_id
                in ownership_ids
            )
            if number is not None
        ]

        static_holdings.append(
            {
                "filing_id": filing_id,
                "ticker": ticker,
                "report_date": (
                    report_date
                ),
                "ownership_entity": (
                    entity
                ),
                "ownership_form": (
                    ownership_form
                ),
                "ownership_footnotes": (
                    ";".join(
                        ownership_numbers
                    )
                ),
                "starting_shares": int(
                    round(
                        post
                    )
                ),
                "sold_on_report_date": 0,
                "ending_shares": int(
                    round(
                        post
                    )
                ),
                "pct_reported_holdings_sold": (
                    0.0
                ),
                "is_seller": False,
                "is_static_holding_only": True,
            }
        )

    profile = {
        "filing_id": filing_id,
        "ticker": ticker,
        "issuer_name": issuer_name,
        "reporting_owner_name": (
            owner_name
        ),
        "report_date": report_date,
    }

    return (
        transactions,
        static_holdings,
        profile,
    )


def _build_rollforward(
    transactions: pd.DataFrame,
    static_rows: list[
        dict[
            str,
            object,
        ]
    ],
    profiles: dict[
        str,
        dict[
            str,
            str,
        ],
    ],
) -> pd.DataFrame:
    rows = []

    group_columns = [
        "filing_id",
        "ticker",
        "ownership_entity",
        "ownership_form",
        "ownership_footnotes",
    ]

    for key, bucket in transactions.groupby(
        group_columns,
        sort=False,
        dropna=False,
    ):
        (
            filing_id,
            ticker,
            entity,
            ownership_form,
            ownership_footnotes,
        ) = key

        bucket = bucket.sort_values(
            "transaction_index",
            kind="stable",
        )

        first = bucket.iloc[
            0
        ]

        last = bucket.iloc[
            -1
        ]

        starting = int(
            first[
                "shares"
            ]
            + first[
                "shares_following_transaction"
            ]
        )

        sold = int(
            bucket[
                "shares"
            ].sum()
        )

        ending = int(
            last[
                "shares_following_transaction"
            ]
        )

        rows.append(
            {
                "filing_id": filing_id,
                "ticker": ticker,
                "report_date": (
                    profiles[
                        str(
                            filing_id
                        )
                    ][
                        "report_date"
                    ]
                ),
                "ownership_entity": (
                    entity
                ),
                "ownership_form": (
                    ownership_form
                ),
                "ownership_footnotes": (
                    ""
                    if pd.isna(
                        ownership_footnotes
                    )
                    else ownership_footnotes
                ),
                "starting_shares": (
                    starting
                ),
                "sold_on_report_date": (
                    sold
                ),
                "ending_shares": (
                    ending
                ),
                "pct_reported_holdings_sold": float(
                    sold
                    / starting
                    if starting
                    else 0.0
                ),
                "is_seller": True,
                "is_static_holding_only": False,
            }
        )

    rows.extend(
        static_rows
    )

    return pd.DataFrame(
        rows,
        columns=ROLL_COLUMNS,
    )


def _context_lookup(
    market: pd.DataFrame,
    filing_id: str,
    transaction_date: str,
) -> pd.Series:
    matched = market[
        (
            market[
                "filing_id"
            ].astype(
                str
            )
            == str(
                filing_id
            )
        )
        & (
            market[
                "transaction_date"
            ].astype(
                str
            )
            == str(
                transaction_date
            )
        )
    ]

    if len(
        matched
    ) != 1:
        raise RuntimeError(
            "Market context is missing or duplicated for "
            f"{filing_id} {transaction_date}."
        )

    return matched.iloc[
        0
    ]


def _entity_day(
    transactions: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        filing_id,
        ticker,
        issuer_name,
        transaction_date,
        entity,
    ), bucket in transactions.groupby(
        [
            "filing_id",
            "ticker",
            "issuer_name",
            "transaction_date",
            "ownership_entity",
        ],
        sort=False,
    ):
        bucket = bucket.sort_values(
            "transaction_index",
            kind="stable",
        )

        first = bucket.iloc[
            0
        ]

        last = bucket.iloc[
            -1
        ]

        inventory = int(
            first[
                "shares"
            ]
            + first[
                "shares_following_transaction"
            ]
        )

        shares = int(
            bucket[
                "shares"
            ].sum()
        )

        gross = float(
            bucket[
                "gross_value"
            ].sum()
        )

        ending = int(
            last[
                "shares_following_transaction"
            ]
        )

        ctx = _context_lookup(
            market,
            str(
                filing_id
            ),
            str(
                transaction_date
            ),
        )

        adv20 = float(
            ctx[
                "adv20_shares"
            ]
        )

        close = float(
            ctx[
                "close_price"
            ]
        )

        pct_inventory = float(
            shares
            / inventory
            if inventory
            else 0.0
        )

        over_adv = float(
            shares
            / adv20
        )

        dollar_over_adv = float(
            gross
            / (
                adv20
                * close
            )
        )

        score = float(
            10000.0
            * (
                0.55
                * pct_inventory
                + 0.25
                * over_adv
                + 0.20
                * dollar_over_adv
            )
        )

        rows.append(
            {
                "filing_id": (
                    filing_id
                ),
                "ticker": ticker,
                "issuer_name": (
                    issuer_name
                ),
                "transaction_date": (
                    transaction_date
                ),
                "ownership_entity": (
                    entity
                ),
                "inventory_before_trading": (
                    inventory
                ),
                "sale_transactions": int(
                    len(
                        bucket
                    )
                ),
                "shares_sold": (
                    shares
                ),
                "gross_proceeds": (
                    gross
                ),
                "weighted_average_sale_price": float(
                    gross
                    / shares
                ),
                "ending_shares_after_last_trade": (
                    ending
                ),
                "pct_inventory_sold": (
                    pct_inventory
                ),
                "shares_sold_over_adv20": (
                    over_adv
                ),
                "dollars_sold_over_adv20_dollar_volume": (
                    dollar_over_adv
                ),
                "liquidation_score": (
                    score
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=ENTITY_COLUMNS,
    )


def _issuer_day(
    entity: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        filing_id,
        ticker,
        issuer_name,
        transaction_date,
    ), bucket in entity.groupby(
        [
            "filing_id",
            "ticker",
            "issuer_name",
            "transaction_date",
        ],
        sort=False,
    ):
        shares = int(
            bucket[
                "shares_sold"
            ].sum()
        )

        gross = float(
            bucket[
                "gross_proceeds"
            ].sum()
        )

        fractions = (
            bucket[
                "shares_sold"
            ].astype(
                float
            )
            / float(
                shares
            )
        )

        hhi = float(
            np.sum(
                fractions
                * fractions
            )
        )

        max_fraction = float(
            fractions.max()
        )

        ctx = _context_lookup(
            market,
            str(
                filing_id
            ),
            str(
                transaction_date
            ),
        )

        daily_volume = float(
            ctx[
                "daily_volume_shares"
            ]
        )

        adv20 = float(
            ctx[
                "adv20_shares"
            ]
        )

        float_shares = float(
            ctx[
                "float_shares"
            ]
        )

        outstanding = float(
            ctx[
                "shares_outstanding"
            ]
        )

        close = float(
            ctx[
                "close_price"
            ]
        )

        reported_inventory = float(
            bucket[
                "inventory_before_trading"
            ].sum()
        )

        day_ratio = float(
            shares
            / daily_volume
        )

        adv_ratio = float(
            shares
            / adv20
        )

        dollar_ratio = float(
            gross
            / (
                adv20
                * close
            )
        )

        float_ratio = float(
            shares
            / float_shares
        )

        outstanding_ratio = float(
            shares
            / outstanding
        )

        inventory_ratio = float(
            shares
            / reported_inventory
            if reported_inventory
            else 0.0
        )

        blended = float(
            10000.0
            * (
                0.24
                * day_ratio
                + 0.20
                * adv_ratio
                + 0.12
                * dollar_ratio
                + 0.24
                * inventory_ratio
                + 0.12
                * float_ratio
                + 0.08
                * hhi
            )
        )

        rows.append(
            {
                "filing_id": (
                    filing_id
                ),
                "ticker": ticker,
                "issuer_name": (
                    issuer_name
                ),
                "transaction_date": (
                    transaction_date
                ),
                "shares_sold": (
                    shares
                ),
                "gross_proceeds": (
                    gross
                ),
                "weighted_average_sale_price": float(
                    gross
                    / shares
                ),
                "seller_entities": int(
                    len(
                        bucket
                    )
                ),
                "seller_concentration_hhi": (
                    hhi
                ),
                "max_entity_pct_of_day_sales": (
                    max_fraction
                ),
                "sale_shares_over_day_volume": (
                    day_ratio
                ),
                "sale_shares_over_adv20": (
                    adv_ratio
                ),
                "sale_dollars_over_adv20_dollar_volume": (
                    dollar_ratio
                ),
                "sale_shares_over_float": (
                    float_ratio
                ),
                "sale_shares_over_outstanding": (
                    outstanding_ratio
                ),
                "sale_shares_over_reported_inventory": (
                    inventory_ratio
                ),
                "blended_pressure_score": (
                    blended
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=ISSUER_COLUMNS,
    )


def _rankings(
    issuer: pd.DataFrame,
) -> pd.DataFrame:
    frame = issuer[
        [
            "filing_id",
            "ticker",
            "transaction_date",
        ]
    ].copy()

    metrics = [
        "shares_sold",
        "gross_proceeds",
        "sale_shares_over_day_volume",
        "sale_shares_over_adv20",
        "sale_shares_over_float",
        "sale_shares_over_reported_inventory",
        "blended_pressure_score",
    ]

    rank_columns = []

    for metric in metrics:
        column = (
            f"rank_{metric}"
        )

        frame[
            column
        ] = (
            issuer[
                metric
            ]
            .rank(
                method="min",
                ascending=False,
            )
            .astype(
                int
            )
        )

        rank_columns.append(
            column
        )

    frame[
        "best_rank"
    ] = frame[
        rank_columns
    ].min(
        axis=1
    )

    frame[
        "worst_rank"
    ] = frame[
        rank_columns
    ].max(
        axis=1
    )

    frame[
        "rank_spread"
    ] = (
        frame[
            "worst_rank"
        ]
        - frame[
            "best_rank"
        ]
    )

    frame[
        "top_metric_count"
    ] = (
        frame[
            rank_columns
        ]
        == 1
    ).sum(
        axis=1
    )

    return frame[
        RANK_COLUMNS
    ]


def _decomposition(
    transactions: pd.DataFrame,
    entity: pd.DataFrame,
    roll: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        filing_id,
        ticker,
        transaction_date,
    ), bucket in transactions.groupby(
        [
            "filing_id",
            "ticker",
            "transaction_date",
        ],
        sort=False,
    ):
        total = int(
            bucket[
                "shares"
            ].sum()
        )

        direct = int(
            bucket.loc[
                bucket[
                    "ownership_form"
                ]
                == "D",
                "shares",
            ].sum()
        )

        indirect = int(
            total
            - direct
        )

        entity_bucket = entity[
            (
                entity[
                    "filing_id"
                ]
                == filing_id
            )
            & (
                entity[
                    "transaction_date"
                ]
                == transaction_date
            )
        ].copy()

        top = (
            entity_bucket
            .sort_values(
                [
                    "shares_sold",
                    "ownership_entity",
                ],
                ascending=[
                    False,
                    True,
                ],
                kind="stable",
            )
            .iloc[
                0
            ]
        )

        top_shares = int(
            top[
                "shares_sold"
            ]
        )

        covered = int(
            bucket.loc[
                bucket[
                    "price_footnote_number"
                ].notna(),
                "shares",
            ].sum()
        )

        filing_roll = roll[
            roll[
                "filing_id"
            ]
            == filing_id
        ]

        static = filing_roll[
            filing_roll[
                "is_static_holding_only"
            ].astype(
                bool
            )
        ]

        rows.append(
            {
                "filing_id": (
                    filing_id
                ),
                "ticker": ticker,
                "transaction_date": (
                    transaction_date
                ),
                "direct_sale_shares": (
                    direct
                ),
                "indirect_sale_shares": (
                    indirect
                ),
                "direct_sale_pct": float(
                    direct
                    / total
                ),
                "indirect_sale_pct": float(
                    indirect
                    / total
                ),
                "top_entity_name": str(
                    top[
                        "ownership_entity"
                    ]
                ),
                "top_entity_sale_shares": (
                    top_shares
                ),
                "top_entity_sale_pct": float(
                    top_shares
                    / total
                ),
                "tail_entity_sale_shares": int(
                    total
                    - top_shares
                ),
                "tail_entity_sale_pct": float(
                    (
                        total
                        - top_shares
                    )
                    / total
                ),
                "price_range_covered_shares": (
                    covered
                ),
                "price_range_coverage_pct": float(
                    covered
                    / total
                ),
                "reported_inventory_before_trading": int(
                    entity_bucket[
                        "inventory_before_trading"
                    ].sum()
                ),
                "seller_ending_shares": int(
                    entity_bucket[
                        "ending_shares_after_last_trade"
                    ].sum()
                ),
                "static_only_ending_shares": int(
                    static[
                        "ending_shares"
                    ].sum()
                ),
                "direct_seller_flag": bool(
                    (
                        bucket[
                            "ownership_form"
                        ]
                        == "D"
                    ).any()
                ),
                "static_holding_only_entities": (
                    "|".join(
                        static[
                            "ownership_entity"
                        ].astype(
                            str
                        ).tolist()
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=DECOMP_COLUMNS,
    )


def _summary(
    *,
    transactions: pd.DataFrame,
    roll: pd.DataFrame,
    issuer: pd.DataFrame,
    rankings: pd.DataFrame,
    decomposition: pd.DataFrame,
    profiles: dict[
        str,
        dict[
            str,
            str,
        ],
    ],
) -> dict:
    metrics = [
        "shares_sold",
        "gross_proceeds",
        "sale_shares_over_day_volume",
        "sale_shares_over_adv20",
        "sale_shares_over_float",
        "sale_shares_over_reported_inventory",
        "blended_pressure_score",
    ]

    winners = {}

    for metric in metrics:
        row = (
            issuer
            .sort_values(
                [
                    metric,
                    "shares_sold",
                    "ticker",
                ],
                ascending=[
                    False,
                    False,
                    True,
                ],
                kind="stable",
            )
            .iloc[
                0
            ]
        )

        winners[
            metric
        ] = {
            "ticker": str(
                row[
                    "ticker"
                ]
            ),
            "transaction_date": str(
                row[
                    "transaction_date"
                ]
            ),
            "value": float(
                row[
                    metric
                ]
            ),
        }

        if metric == "shares_sold":
            winners[
                metric
            ][
                "value"
            ] = int(
                row[
                    metric
                ]
            )

    spread_row = (
        rankings
        .sort_values(
            [
                "rank_spread",
                "best_rank",
                "ticker",
            ],
            ascending=[
                False,
                True,
                True,
            ],
            kind="stable",
        )
        .iloc[
            0
        ]
    )

    static = roll[
        roll[
            "is_static_holding_only"
        ].astype(
            bool
        )
    ]

    largest_static = (
        static
        .sort_values(
            [
                "ending_shares",
                "ownership_entity",
            ],
            ascending=[
                False,
                True,
            ],
            kind="stable",
        )
        .iloc[
            0
        ]
    )

    per_filing = {}

    for ticker, tx in transactions.groupby(
        "ticker",
        sort=False,
    ):
        filing_id = str(
            tx.iloc[
                0
            ][
                "filing_id"
            ]
        )

        filing_roll = roll[
            roll[
                "filing_id"
            ]
            == filing_id
        ]

        filing_static = filing_roll[
            filing_roll[
                "is_static_holding_only"
            ].astype(
                bool
            )
        ]

        entity_totals = (
            tx.groupby(
                "ownership_entity",
                sort=False,
            )[
                "shares"
            ]
            .sum()
            .reset_index()
            .sort_values(
                [
                    "shares",
                    "ownership_entity",
                ],
                ascending=[
                    False,
                    True,
                ],
                kind="stable",
            )
        )

        top_entity = (
            entity_totals.iloc[
                0
            ]
        )

        issuer_rows = issuer[
            issuer[
                "filing_id"
            ]
            == filing_id
        ]

        peak = (
            issuer_rows
            .sort_values(
                [
                    "blended_pressure_score",
                    "transaction_date",
                ],
                ascending=[
                    False,
                    True,
                ],
                kind="stable",
            )
            .iloc[
                0
            ]
        )

        decomp_rows = decomposition[
            decomposition[
                "filing_id"
            ]
            == filing_id
        ]

        per_filing[
            str(
                ticker
            )
        ] = {
            "reporting_owner_name": (
                profiles[
                    filing_id
                ][
                    "reporting_owner_name"
                ]
            ),
            "report_date": (
                profiles[
                    filing_id
                ][
                    "report_date"
                ]
            ),
            "sale_dates": sorted(
                set(
                    tx[
                        "transaction_date"
                    ].astype(
                        str
                    )
                )
            ),
            "total_sale_transactions": int(
                len(
                    tx
                )
            ),
            "total_shares_sold": int(
                tx[
                    "shares"
                ].sum()
            ),
            "total_gross_proceeds": float(
                tx[
                    "gross_value"
                ].sum()
            ),
            "seller_entities": int(
                tx[
                    "ownership_entity"
                ].nunique()
            ),
            "direct_seller_flag": bool(
                (
                    tx[
                        "ownership_form"
                    ]
                    == "D"
                ).any()
            ),
            "static_holding_only_entities": (
                filing_static[
                    "ownership_entity"
                ].astype(
                    str
                ).tolist()
            ),
            "top_entity_name": str(
                top_entity[
                    "ownership_entity"
                ]
            ),
            "top_entity_sale_shares": int(
                top_entity[
                    "shares"
                ]
            ),
            "peak_blended_pressure_date": str(
                peak[
                    "transaction_date"
                ]
            ),
            "peak_blended_pressure_score": float(
                peak[
                    "blended_pressure_score"
                ]
            ),
            "static_only_ending_shares": int(
                filing_static[
                    "ending_shares"
                ].sum()
            ),
            "reported_inventory_before_trading": int(
                decomp_rows[
                    "reported_inventory_before_trading"
                ].sum()
            ),
        }

    return {
        "filing_ids": list(
            profiles.keys()
        ),
        "tickers": [
            profiles[
                filing_id
            ][
                "ticker"
            ]
            for filing_id
            in profiles
        ],
        "issuer_day_count": int(
            len(
                issuer
            )
        ),
        "total_non_derivative_sale_transactions": int(
            len(
                transactions
            )
        ),
        "total_shares_sold": int(
            transactions[
                "shares"
            ].sum()
        ),
        "total_gross_sale_proceeds": float(
            transactions[
                "gross_value"
            ].sum()
        ),
        "metric_winners": winners,
        "highest_rank_spread": {
            "ticker": str(
                spread_row[
                    "ticker"
                ]
            ),
            "transaction_date": str(
                spread_row[
                    "transaction_date"
                ]
            ),
            "rank_spread": int(
                spread_row[
                    "rank_spread"
                ]
            ),
        },
        "largest_static_holding_entity": str(
            largest_static[
                "ownership_entity"
            ]
        ),
        "largest_static_holding_shares": int(
            largest_static[
                "ending_shares"
            ]
        ),
        "per_filing": per_filing,
    }


@dataclass(frozen=True)
class Form4SalePressureSkill:
    """Programmatic Form 4 sale-ledger reconstruction and pressure analytics."""

    name: str = "form4-sale-pressure-domain"

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
            "form 4" in normalized
            and (
                "sale pressure"
                in normalized
                or "insider sale"
                in normalized
            )
            and (
                "ownership"
                in normalized
                or "non derivative"
                in normalized
            )
            and (
                "cross sectional"
                in normalized
                or "ranking"
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
        del instruction, seed

        data_dir = (
            _find_form4_data_dir(
                task_dir
            )
        )

        all_transactions = []
        all_static = []
        profiles = {}

        for path in sorted(
            data_dir.glob(
                "form4_*.xml"
            )
        ):
            (
                transactions,
                static_rows,
                profile,
            ) = _parse_filing(
                path
            )

            all_transactions.extend(
                transactions
            )

            all_static.extend(
                static_rows
            )

            profiles[
                profile[
                    "filing_id"
                ]
            ] = profile

        if not all_transactions:
            raise RuntimeError(
                "No non-derivative Form 4 sales were reconstructed."
            )

        transactions = pd.DataFrame(
            all_transactions,
            columns=TRANSACTION_COLUMNS,
        )

        market = pd.read_csv(
            data_dir
            / "market_context.csv"
        )

        roll = _build_rollforward(
            transactions,
            all_static,
            profiles,
        )

        entity = _entity_day(
            transactions,
            market,
        )

        issuer = _issuer_day(
            entity,
            market,
        )

        rankings = _rankings(
            issuer
        )

        decomposition = (
            _decomposition(
                transactions,
                entity,
                roll,
            )
        )

        summary = _summary(
            transactions=(
                transactions
            ),
            roll=roll,
            issuer=issuer,
            rankings=rankings,
            decomposition=(
                decomposition
            ),
            profiles=profiles,
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        transactions.to_csv(
            out_dir
            / "transactions_normalized.csv",
            index=False,
        )

        roll.to_csv(
            out_dir
            / "ownership_rollforward.csv",
            index=False,
        )

        entity.to_csv(
            out_dir
            / "entity_day_sale_pressure.csv",
            index=False,
        )

        issuer.to_csv(
            out_dir
            / "issuer_pressure_scenarios.csv",
            index=False,
        )

        rankings.to_csv(
            out_dir
            / "cross_sectional_rankings.csv",
            index=False,
        )

        decomposition.to_csv(
            out_dir
            / "pressure_decomposition.csv",
            index=False,
        )

        (
            out_dir
            / "summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
