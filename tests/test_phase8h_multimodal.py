from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent.finance_composition import plan_finance_task
from agent.finance_multimodal import (
    _results,
    normalize_cross_section,
)
from agent.finance_process_selection import select_return_process


def _task(tmp_path: Path) -> Path:
    data = tmp_path / "task" / "environment" / "data"
    data.mkdir(parents=True)
    return data.parent.parent


def test_cross_section_normalization_uses_population_std_and_median_fill() -> None:
    values = pd.Series([1.0, 2.0, np.nan, 4.0])
    out = normalize_cross_section(values)
    filled = pd.Series([1.0, 2.0, 2.0, 4.0])
    expected = (filled - filled.mean()) / filled.std(ddof=0)
    assert np.allclose(out.to_numpy(), expected.to_numpy())


def test_process_selection_is_insufficient_below_minimum() -> None:
    result = select_return_process(
        [0.01, -0.02, 0.03],
        minimum_history=6,
        holding_period_days=21,
        conviction_scale=4.0,
        jump_threshold_sigma=1.4,
    )
    assert result.selected_process == "insufficient_history"
    assert result.process_sign == 0
    assert result.process_conviction == 0.0
    assert result.gbm_mu is None


def test_process_selection_is_deterministic_and_finite() -> None:
    returns = [
        0.02, 0.04, -0.01, 0.03, 0.08, 0.01,
        -0.02, 0.05, 0.01, 0.02, -0.01, 0.03,
    ]
    one = select_return_process(
        returns,
        minimum_history=6,
        holding_period_days=21,
        conviction_scale=4.0,
        jump_threshold_sigma=1.4,
    )
    two = select_return_process(
        returns,
        minimum_history=6,
        holding_period_days=21,
        conviction_scale=4.0,
        jump_threshold_sigma=1.4,
    )
    assert one == two
    assert one.selected_process in {
        "martingale", "gbm", "ou", "merton_jump_diffusion"
    }
    assert 0.0 <= one.process_conviction <= 1.0
    assert np.isfinite(one.martingale_aic)
    assert np.isfinite(one.gbm_aic)
    assert np.isfinite(one.ou_aic)
    assert np.isfinite(one.merton_aic)


def test_planner_recognises_multimodal_committee_process_pipeline(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    data = task / "environment" / "data"

    pd.DataFrame({
        "asof_date": ["2024-01-01"],
        "ticker": ["AAA"],
        "revenue_growth_ttm": [0.1],
        "gross_margin_ttm": [0.4],
        "debt_to_assets": [0.2],
        "cash_to_assets": [0.1],
        "share_count_growth_yoy": [0.0],
        "days_since_last_10q": [30],
        "days_since_last_10k": [60],
        "count_8k_trailing_90d": [1],
    }).to_csv(data / "fund.csv", index=False)

    pd.DataFrame({
        "asof_date": ["2024-01-01"],
        "ticker": ["AAA"],
        "article_count_7d": [1],
        "article_count_30d": [2],
        "source_count_30d": [2],
        "avg_tone_7d": [0.1],
        "avg_tone_30d": [0.1],
        "theme_risk_count_30d": [0],
        "theme_supply_count_30d": [0],
        "theme_macro_count_30d": [0],
    }).to_csv(data / "news.csv", index=False)

    pd.DataFrame({
        "asof_date": ["2024-01-01"],
        "ticker": ["AAA"],
        "entry_date": ["2024-01-01"],
        "exit_date": ["2024-02-01"],
        "forward_return_21d": [0.01],
    }).to_csv(data / "returns.csv", index=False)

    (data / "params.json").write_text(json.dumps({
        "long_top_n": 3,
        "short_bottom_n": 3,
        "min_trade_signal": 0.05,
        "initial_capital": 100000,
        "transaction_cost_bps": 10,
        "annualization_factor": 12,
        "filing_weight": 0.32,
        "issuer_activity_weight": 0.2,
        "news_weight": 0.18,
        "audit_weight": 0.3,
        "disagreement_penalty": 0.28,
        "reference_lookback_days": 540,
        "reference_stale_threshold_days": 120,
        "severe_bulletin_cutoff": 2.5,
        "process_trailing_window": 12,
        "process_min_history": 6,
        "process_conviction_scale": 4.0,
        "jump_threshold_sigma": 1.4,
    }))

    plan = plan_finance_task(
        "Fuse a four-agent EDGAR, issuer activity, GDELT news and audit "
        "committee with a disagreement penalty, then select among "
        "martingale, GBM, OU and Merton jump-diffusion processes and "
        "construct a long/short portfolio.",
        task,
    )
    assert plan.executable_recipe == "multimodal-alpha-process-analysis"
    assert {
        "multimodal_issuer_panel",
        "committee_fusion",
        "audit_vintage_checks",
        "process_model_selection",
        "committee_disagreement",
        "long_short_portfolio",
    }.issubset(plan.capabilities)


def test_multimodal_results_use_population_volatility() -> None:
    portfolio = pd.DataFrame({
        "asof_date": ["2024-01-01", "2024-02-01"],
        "net_return": [0.10, -0.05],
        "turnover": [0.25, 0.50],
        "equity_curve": [110.0, 104.5],
    })
    diagnostics = pd.DataFrame({
        "process_history_ready_issuers": [0, 1],
        "news_nonzero_issuers": [1, 1],
    })
    params = {
        "initial_capital": 100.0,
        "annualization_factor": 12.0,
    }

    result = _results(
        portfolio,
        diagnostics,
        params,
        num_issuers=1,
    )

    expected = (
        np.std([0.10, -0.05], ddof=0)
        * np.sqrt(12.0)
    )

    assert abs(
        result["annualized_volatility"] - expected
    ) < 1e-15
