from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent.capability_bridge import rank_capabilities
from agent.offline_common import earnings, holdings, ownership, sec_filings, xbrl
from agent.offline_filing_alpha import (
    _extract_debt_amount_musd,
    _extract_event_date,
    _extract_guidance_midpoint_musd,
    _extract_item_numbers,
)
from agent.offline_form4_sale_pressure import _parse_filing


PUBLIC_UNITS = Path("/home/elliot/track1-coding-public/units")


def test_earnings_surprise_matches_mature_formulas_and_aggregates() -> None:
    first = earnings.earnings_surprise(1.12, 1.0, 0.04, 0.9)
    second = earnings.earnings_surprise(0.98, 1.0, 0.0, 1.0)
    assert first.classification == "beat"
    assert np.isclose(first.surprise_pct, 12.0)
    assert np.isclose(first.sue, 3.0)
    assert second.sue == 0.0
    summary = earnings.aggregate_earnings_surprises([first, second])
    assert np.isclose(summary["beat_rate"], 0.5)
    assert np.isclose(summary["avg_surprise_pct"], 5.0)


def test_earnings_surprise_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError):
        earnings.earnings_surprise(1.0, 1.0, 0.1, 0.9, threshold_pct=-1.0)


def test_sec_filing_extractors_match_mature_implementation() -> None:
    text = (
        "Date of Report (Date of earliest event reported): May 22, 2024 "
        "Item 2.02 Results. Item 7.01 Regulation FD. Item 9.01 Exhibits. "
        "Business Outlook: Revenue is expected to be $28.0 billion, plus or minus 2%."
    )
    assert sec_filings.extract_sec_event_date(text) == _extract_event_date(text)
    assert "|".join(sec_filings.extract_sec_item_numbers(text)) == _extract_item_numbers(text)
    assert np.isclose(
        sec_filings.revenue_guidance_midpoint_millions(text),
        _extract_guidance_midpoint_musd(text),
    )
    debt = (
        "The Company issued $1,725,000,000 principal amount of Notes. "
        "It may sell an additional $225,000,000 principal amount. "
        "The Notes issued include $225,000,000 pursuant to the full exercise."
    )
    assert np.isclose(
        sec_filings.extract_debt_principal_millions(debt),
        _extract_debt_amount_musd(debt),
    )


def test_html_text_and_executive_departure_are_generic() -> None:
    source = "<html><style>x</style><body>Chief Financial Officer stepped down.</body></html>"
    text = sec_filings.html_to_text(source)
    assert text == "Chief Financial Officer stepped down."
    assert sec_filings.classify_executive_departure(text) == "cfo"


def test_amendment_resolution_and_holdings_invariants() -> None:
    resolved = holdings.resolve_amendment_sequence(
        [
            {"accession": "a", "is_amendment": False},
            {"accession": "b", "is_amendment": True, "amendment_type": "NEW HOLDINGS"},
            {"accession": "c", "is_amendment": True, "amendment_type": "RESTATEMENT"},
        ]
    )
    assert [row.action for row in resolved] == [
        "replace_initial",
        "append_new_holdings",
        "replace_restatement",
    ]
    assert resolved[-1].effective_accessions == ("c",)

    raw = pd.DataFrame(
        {
            "CUSIP": ["AAA", "AAA", "BBB", "CCC", ""],
            "SSHPRNAMTTYPE": ["SH", "SH", "PRN", "SH", "SH"],
            "PUTCALL": ["", "", "", "PUT", ""],
            "VALUE": [10, 20, 30, 40, 50],
            "SSHPRNAMT": [1, 2, 3, 4, 5],
        }
    )
    cleaned, audit = holdings.clean_long_holdings(raw)
    assert cleaned["CUSIP"].tolist() == ["AAA"]
    assert cleaned.loc[0, "VALUE"] == 30
    assert np.isclose(cleaned.loc[0, "weight"], 1.0)
    assert sum(value for key, value in audit.items() if key.startswith("filtered_")) == 3
    assert np.isclose(holdings.portfolio_turnover({"A": 0.7, "B": 0.3}, {"A": 0.4, "C": 0.6}), 0.6)
    assert np.isclose(holdings.weighted_overlap({"A": 0.7, "B": 0.3}, {"A": 0.4, "C": 0.6}), 0.4)


def test_form4_parser_matches_mature_sale_ledger() -> None:
    path = PUBLIC_UNITS / "t1-form4-cross-sectional-sale-pressure" / "environment/data/form4_iot.xml"
    generic = ownership.parse_form4_transactions(path.read_text(encoding="utf-8"))
    mature, _static, _profile = _parse_filing(path)
    assert len(generic) == len(mature)
    assert [row.transaction_date for row in generic] == [row["transaction_date"] for row in mature]
    assert np.allclose([row.shares for row in generic], [row["shares"] for row in mature])
    assert np.allclose([row.price_per_share for row in generic], [row["price_per_share"] for row in mature])


def test_ownership_price_range_and_pressure_errors() -> None:
    parsed = ownership.parse_price_range("prices ranging from $106.73 to $107.73 per share")
    assert parsed is not None and np.isclose(parsed[0], 106.73)
    ratios = ownership.sale_pressure_ratios(10.0, 1000.0, 100.0, 1000.0, 100.0)
    assert np.isclose(ratios["pct_inventory_sold"], 0.1)
    with pytest.raises(ValueError):
        ownership.sale_pressure_ratios(10.0, 1000.0, 100.0, 0.0, 100.0)


def test_xbrl_fact_parsing_scale_sign_and_selection() -> None:
    source = """
    <html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL">
      <body>
        <ix:nonNumeric name="dei:DocumentType" contextRef="c1">10-K</ix:nonNumeric>
        <ix:nonFraction name="us-gaap:Revenue" contextRef="c2" unitRef="USD" scale="6">1,234</ix:nonFraction>
        <ix:nonFraction name="us-gaap:Loss" contextRef="c2" unitRef="USD" sign="-">5</ix:nonFraction>
      </body>
    </html>
    """
    facts = xbrl.parse_xbrl_facts(source)
    assert xbrl.facts_by_concept(facts, "DocumentType")[0].value == "10-K"
    assert xbrl.facts_by_concept(facts, "Revenue")[0].value == 1_234_000_000.0
    assert xbrl.facts_by_concept(facts, "Loss")[0].value == -5.0


@pytest.mark.parametrize(
    ("unit_name", "capability_id"),
    (
        ("t1-13f-amendment-aware-crowding", "amendment-aware-holdings"),
        ("t1-form4-cross-sectional-sale-pressure", "form4-ownership-sales"),
        ("t1-sec-10k-report-long", "xbrl-fact-parsing"),
        ("t1-sec-8k-event-alpha", "sec-filing-event-extraction"),
        ("t1-earnings-surprise-calculator", "earnings-surprise-statistics"),
    ),
)
def test_batch6a_public_units_route_narrowly(unit_name: str, capability_id: str) -> None:
    task = PUBLIC_UNITS / unit_name
    selected = rank_capabilities(instruction=(task / "instruction.md").read_text(), task_dir=task)
    assert capability_id in {item.descriptor.capability_id for item in selected}


@pytest.mark.parametrize(
    "instruction",
    (
        "Summarize this annual report in plain English.",
        "Compute portfolio overlap for two arbitrary vectors.",
        "Analyze ordinary employee stock sales.",
        "Calculate revenue growth from a CSV.",
    ),
)
def test_batch6a_generic_language_does_not_route(tmp_path: Path, instruction: str) -> None:
    task = tmp_path / "task"
    task.mkdir()
    assert rank_capabilities(instruction=instruction, task_dir=task) == ()
