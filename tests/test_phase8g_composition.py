from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent.finance_composition import plan_finance_task
from agent.finance_primitives import (
    central_price_delta,
    discount_cashflow,
    down_and_out_call_price,
    normal_delta_var,
    normal_tail_multiplier,
    sma_seeded_ema,
    solve_breakeven_volatility,
)
from agent.finance_schema import TaskDataCatalog


def _task(tmp_path: Path) -> Path:
    task = tmp_path / "task" / "environment" / "data"
    task.mkdir(parents=True)
    return task.parent.parent


def test_schema_adapter_selects_by_shape_not_filename(tmp_path: Path) -> None:
    task = _task(tmp_path)
    data = task / "environment" / "data"
    pd.DataFrame({"date": ["2026-01-01"], "log_return": [0.01]}).to_csv(data / "mystery.csv", index=False)
    (data / "settings.json").write_text(json.dumps({"notional": 1000, "S0": 100, "K": 100, "B": 70, "r": 0.03, "T": 1, "participation": 1.0}))
    catalog = TaskDataCatalog.discover(task)
    assert catalog.csv_with_columns({"date", "log_return"}).name == "mystery.csv"
    assert catalog.json_with_keys({"notional", "s0", "participation"}).name == "settings.json"


def test_composer_recognises_structured_product_from_capabilities(tmp_path: Path) -> None:
    task = _task(tmp_path)
    data = task / "environment" / "data"
    pd.DataFrame({"date": ["2026-01-01"], "log_return": [0.01]}).to_csv(data / "r.csv", index=False)
    (data / "p.json").write_text(json.dumps({"notional": 1000, "S0": 100, "K": 100, "B": 70, "r": 0.03, "T": 1, "participation": 1.0}))
    plan = plan_finance_task(
        "Fit GARCH, value a structured note with an analytical barrier option, find breakeven volatility, and report VaR and ES.",
        task,
    )
    assert plan.executable_recipe == "structured-product-risk"
    assert {"garch11", "analytical_barrier_option", "root_solve", "var_es"}.issubset(plan.capabilities)


def test_composer_recognises_vol_target_pipeline(tmp_path: Path) -> None:
    task = _task(tmp_path)
    data = task / "environment" / "data"
    pd.DataFrame({"date": ["2026-01-01", "2026-01-02"], "a": [100, 101], "b": [100, 99]}).to_csv(data / "panel.csv", index=False)
    (data / "p.json").write_text(json.dumps({
        "fast_ema": 5, "slow_ema": 20, "vol_lookback": 10, "vol_target": 0.1,
        "max_leverage_per_asset": 1.0, "risk_free_annual": 0.0, "n_assets": 2,
        "regime_lo_pct": 30, "regime_hi_pct": 70, "scale_low_vol": 1.2, "scale_high_vol": 0.5,
    }))
    plan = plan_finance_task(
        "Use GARCH volatility regimes, SMA-seeded EMA signals, EWMA volatility and volatility targeting; report Sharpe, drawdown and Calmar.",
        task,
    )
    assert plan.executable_recipe == "volatility-target-strategy"


def test_composer_decomposes_other_unseen_finance_domains(tmp_path: Path) -> None:
    task = _task(tmp_path)
    cases = [
        ("Estimate DCC GARCH covariance, stationary bootstrap FHS, Kupiec and Christoffersen tests.", {"garch11", "dcc_correlation", "stationary_bootstrap", "kupiec_backtest", "christoffersen_backtest"}),
        ("Price an American option with Crank-Nicolson, PSOR and discrete cash dividends.", {"crank_nicolson_pde", "psor_projection", "cash_dividend_jump"}),
        ("Build OIS curves and FX forwards for a cross-currency XCCY cashflow valuation.", {"curve_bootstrap", "fx_forward_curve", "xccy_cashflow_engine"}),
        ("Estimate a Fama-French factor model with Newey-West HAC, GRS and rolling beta.", {"factor_ols", "newey_west_hac", "grs_joint_alpha_test", "rolling_factor_beta"}),
    ]
    for instruction, expected in cases:
        plan = plan_finance_task(instruction, task)
        assert expected.issubset(set(plan.capabilities))


def test_structured_product_primitives_match_reference_conventions() -> None:
    notional = 1000.0
    spot = 77.54975542623474
    strike = 80.65
    barrier = 54.28
    rate = 0.0216
    maturity = 0.25
    participation = 1.1
    vol = 0.2586800935963609
    bond = discount_cashflow(notional, rate, maturity)
    n_options = notional * participation / spot
    price_fn = lambda s: down_and_out_call_price(
        spot=s, strike=strike, barrier=barrier, maturity=maturity, rate=rate, volatility=vol
    )
    unit = price_fn(spot)
    delta = central_price_delta(price_fn=price_fn, spot=spot, relative_step=0.01)
    assert abs(bond - 994.6145537913911) < 1e-9
    assert abs(unit - 2.8723811969778144) < 1e-10
    assert abs(n_options - 14.184441897387014) < 1e-10
    assert abs(delta - 0.4220772133647326) < 1e-10
    note_var = normal_delta_var(
        exposure=n_options * delta * spot,
        annualized_volatility=vol,
        confidence=0.99,
    )
    note_es = note_var * normal_tail_multiplier(0.99)
    assert abs(note_var - 17.60037774602308) < 1e-10
    assert abs(note_es - 46.90877705215862) < 1e-10
    breakeven = solve_breakeven_volatility(
        target_value=notional,
        fixed_value=bond,
        units=n_options,
        option_price_at_vol=lambda sigma: down_and_out_call_price(
            spot=spot, strike=strike, barrier=barrier, maturity=maturity, rate=rate, volatility=sigma
        ),
    )
    assert abs(breakeven - 0.08266075008190174) < 1e-8


def test_sma_seeded_ema_exact_seed() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    ema = sma_seeded_ema(x, 3)
    assert np.isnan(ema[0]) and np.isnan(ema[1])
    assert ema[2] == 2.0
    assert ema[3] == 3.0


def test_schema_adapter_tracks_nested_json_paths(tmp_path: Path) -> None:
    task = _task(tmp_path)
    data = task / "environment" / "data"
    (data / "trade.json").write_text(
        json.dumps({"curves": {"usd": {"interpolation": "log_df"}}, "trade": {"notional": 1e6}})
    )
    catalog = TaskDataCatalog.discover(task)
    assert catalog.has_json_paths({"curves.usd.interpolation", "trade.notional"})
