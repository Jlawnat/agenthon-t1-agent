from __future__ import annotations

import json

import numpy as np
import pandas as pd

from agent.finance_composition import plan_finance_task
from agent.finance_xccy import (
    XccyConfig,
    build_projection_curve,
    clean_quote_book,
    effective_quote_rate,
    parse_date,
)


def test_clean_quote_book_deduplicates_and_normalizes():
    frame = pd.DataFrame(
        [
            {
                "quote_time": "2020-01-01T08:00:00Z",
                "curve": "GBPUSD",
                "instrument_type": "spot",
                "tenor": "SPOT",
                "start_date": "2020-01-03",
                "end_date": "2020-01-03",
                "ccy_or_pair": "GBPUSD",
                "quote_unit": "spot_rate",
                "quote_value": 1.29,
                "quote_adjustment_bps": np.nan,
            },
            {
                "quote_time": "2020-01-01T09:00:00Z",
                "curve": "GBPUSD",
                "instrument_type": "spot",
                "tenor": "SPOT",
                "start_date": "2020-01-03",
                "end_date": "2020-01-03",
                "ccy_or_pair": "GBPUSD",
                "quote_unit": "spot_rate",
                "quote_value": 1.30,
                "quote_adjustment_bps": np.nan,
            },
            {
                "quote_time": "2020-01-01T09:00:00Z",
                "curve": "GBPUSD",
                "instrument_type": "forward",
                "tenor": "1M",
                "start_date": "2020-01-03",
                "end_date": "2020-02-03",
                "ccy_or_pair": "GBPUSD",
                "quote_unit": "forward_points_pips",
                "quote_value": 5.0,
                "quote_adjustment_bps": np.nan,
            },
        ]
    )
    cleaned = clean_quote_book(frame)
    assert len(cleaned) == 2
    spot = cleaned[cleaned["instrument_type"] == "spot"].iloc[0]
    forward = cleaned[cleaned["instrument_type"] == "forward"].iloc[0]
    assert np.isclose(spot["normalized_quote"], 1.30)
    assert spot["normalized_unit"] == "fx_rate"
    assert np.isclose(forward["normalized_quote"], 0.0005)


def test_projection_curve_applies_fra_adjustment():
    config = XccyConfig(
        valuation_date=parse_date("2020-01-01"),
        spot_date=parse_date("2020-01-03"),
        usd_notional=1_000_000.0,
        inception_spot_fx=1.30,
        contract_spread_bps=25.0,
        usd_day_count="ACT/360",
        gbp_day_count="ACT/365F",
        fx_pair="GBPUSD",
        collateral_currency="USD",
    )
    quotes = pd.DataFrame(
        [
            {
                "quote_time": "2020-01-01T09:00:00Z",
                "curve": "GBP_LIBOR3M",
                "instrument_type": "deposit",
                "tenor": "3M",
                "start_date": "2020-01-03",
                "end_date": "2020-04-03",
                "ccy_or_pair": "GBP",
                "quote_unit": "rate_pct",
                "quote_value": 1.5,
                "quote_adjustment_bps": np.nan,
            },
            {
                "quote_time": "2020-01-01T09:00:00Z",
                "curve": "GBP_LIBOR3M",
                "instrument_type": "fra",
                "tenor": "3x6",
                "start_date": "2020-04-03",
                "end_date": "2020-07-03",
                "ccy_or_pair": "GBP",
                "quote_unit": "rate_pct",
                "quote_value": 1.6,
                "quote_adjustment_bps": -1.0,
            },
        ]
    )
    cleaned = clean_quote_book(quotes)
    fra = cleaned[cleaned["instrument_type"] == "fra"].iloc[0]
    assert np.isclose(effective_quote_rate(fra), 0.0159)
    curve = build_projection_curve(cleaned, config, "GBP")
    assert len(curve) == 3


def test_planner_recognizes_xccy_desk_recipe(tmp_path):
    data = tmp_path / "environment" / "data"
    data.mkdir(parents=True)

    pd.DataFrame(
        columns=[
            "quote_time",
            "curve",
            "instrument_type",
            "tenor",
            "start_date",
            "end_date",
            "ccy_or_pair",
            "quote_unit",
            "quote_value",
            "quote_adjustment_bps",
        ]
    ).to_csv(data / "quotes.csv", index=False)

    pd.DataFrame(
        columns=["fixing_date", "index_name", "rate_pct"]
    ).to_csv(data / "fixings.csv", index=False)

    pd.DataFrame(
        columns=[
            "period_id",
            "accrual_start",
            "accrual_end",
            "payment_date",
            "reset_date",
            "fixing_date",
            "current_coupon_fixed",
        ]
    ).to_csv(data / "schedule.csv", index=False)

    (data / "trade.json").write_text(
        json.dumps(
            {
                "valuation_date": "2020-02-10",
                "spot_date": "2020-02-12",
                "fx_pair": "GBPUSD",
                "collateral_currency": "USD",
                "usd_day_count": "ACT/360",
                "gbp_day_count": "ACT/365F",
                "usd_notional": 50_000_000.0,
                "contract_spread_bps": 30.0,
                "inception_spot_fx": 1.30,
            }
        ),
        encoding="utf-8",
    )

    instruction = """
    Price a USD-collateralized MTM GBP/USD cross-currency basis swap.
    Build OIS discount curves and LIBOR projection curves, reconcile FX forward
    points and implied basis, use historical fixings, reset notionals, compute
    finite-difference risk, stress scenarios and roll_down_to_2020_05_15.
    """

    plan = plan_finance_task(instruction, tmp_path)
    assert plan.executable_recipe == "xccy-desk-analysis"
    assert {
        "xccy_cashflow_engine",
        "curve_bootstrap",
        "fx_forward_curve",
        "projection_curve",
        "fixing_aware_coupons",
        "mtm_notional_resets",
        "collateralized_discounting",
        "stress_revaluation",
    }.issubset(set(plan.capabilities))
