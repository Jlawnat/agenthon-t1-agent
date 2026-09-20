from __future__ import annotations

from dataclasses import dataclass
import json
import math
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
    ewma_annualized_volatility,
    fit_garch11_zero_mean,
    log_return_performance,
    sma_seeded_ema,
    solve_breakeven_volatility,
)
from agent.finance_schema import TaskDataCatalog


def _solve_structured_product(task_dir: Path, out_dir: Path) -> None:
    catalog = TaskDataCatalog.discover(task_dir)
    returns_path = catalog.csv_with_columns({"date", "log_return"})
    params_path = catalog.json_with_keys({"notional", "s0", "k", "b", "r", "t", "participation"})
    p = json.loads(params_path.read_text(encoding="utf-8"))
    returns = pd.read_csv(returns_path)["log_return"].to_numpy(dtype=float)

    fit = fit_garch11_zero_mean(returns)
    garch_vol = float(math.sqrt(float(fit.conditional_variance[-1]) * 252.0))
    notional = float(p["notional"])
    spot = float(p["S0"])
    strike = float(p["K"])
    barrier = float(p["B"])
    rate = float(p["r"])
    maturity = float(p["T"])
    participation = float(p["participation"])

    bond_pv = discount_cashflow(notional, rate, maturity)
    n_options = notional * participation / spot

    def option_at_vol(vol: float) -> float:
        return down_and_out_call_price(
            spot=spot,
            strike=strike,
            barrier=barrier,
            maturity=maturity,
            rate=rate,
            volatility=vol,
        )

    option_unit_price = option_at_vol(garch_vol)
    option_value = n_options * option_unit_price
    fair_value = bond_pv + option_value
    markup_pct = (notional - fair_value) / fair_value * 100.0
    option_delta = central_price_delta(
        price_fn=lambda s: down_and_out_call_price(
            spot=s,
            strike=strike,
            barrier=barrier,
            maturity=maturity,
            rate=rate,
            volatility=garch_vol,
        ),
        spot=spot,
        relative_step=0.01,
    )
    note_delta = n_options * option_delta
    note_var_99 = normal_delta_var(
        exposure=note_delta * spot,
        annualized_volatility=garch_vol,
        confidence=0.99,
    )
    # This contract family reports ES as a tail multiplier applied to the
    # already-computed VaR. Keep that convention here rather than changing
    # the mathematically standard ES primitive used by other compositions.
    note_es_99 = note_var_99 * normal_tail_multiplier(0.99)
    breakeven_vol = solve_breakeven_volatility(
        target_value=notional,
        fixed_value=bond_pv,
        units=n_options,
        option_price_at_vol=option_at_vol,
    )

    results = {
        "fair_value": fair_value,
        "markup_pct": markup_pct,
        "note_var_99": note_var_99,
        "note_es_99": note_es_99,
        "breakeven_vol": breakeven_vol,
        "garch_persistence": fit.persistence,
        "long_run_variance": fit.long_run_variance,
        "garch_vol": garch_vol,
        "bond_pv": bond_pv,
        "option_unit_price": option_unit_price,
        "option_value": option_value,
        "n_options": n_options,
        "option_delta": option_delta,
        "note_delta": note_delta,
    }
    intermediates = {
        "garch_omega": {"value": fit.omega},
        "garch_alpha": {"value": fit.alpha},
        "garch_beta": {"value": fit.beta},
        "garch_persistence": {"value": fit.persistence},
        "long_run_variance": {"value": fit.long_run_variance},
        "garch_vol": {"value": garch_vol},
        "bond_pv": {"value": bond_pv},
        "option_unit_price": {"value": option_unit_price},
        "option_value": {"value": option_value},
        "n_options": {"value": n_options},
        "option_delta": {"value": option_delta},
        "note_delta": {"value": note_delta},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (out_dir / "solution.json").write_text(
        json.dumps({"intermediates": intermediates}, indent=2) + "\n",
        encoding="utf-8",
    )


def _solve_volatility_target_strategy(task_dir: Path, out_dir: Path) -> None:
    catalog = TaskDataCatalog.discover(task_dir)
    prices_path = catalog.csv_with_columns({"date"}, min_extra_columns=2)
    params_path = catalog.json_with_keys(
        {
            "fast_ema",
            "slow_ema",
            "vol_lookback",
            "vol_target",
            "max_leverage_per_asset",
            "risk_free_annual",
            "n_assets",
            "regime_lo_pct",
            "regime_hi_pct",
            "scale_low_vol",
            "scale_high_vol",
        }
    )
    p = json.loads(params_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(prices_path)
    asset_cols = [c for c in frame.columns if str(c).lower() != "date"]
    if len(asset_cols) != int(p["n_assets"]):
        raise RuntimeError("Price panel asset count does not match n_assets.")
    prices = frame[asset_cols].to_numpy(dtype=float)
    if np.any(prices <= 0.0) or not np.all(np.isfinite(prices)):
        raise RuntimeError("Price panel contains invalid prices.")
    returns = np.log(prices[1:] / prices[:-1])

    fast = int(p["fast_ema"])
    slow = int(p["slow_ema"])
    signal_full = np.empty_like(prices, dtype=float)
    for j in range(prices.shape[1]):
        fast_ema = sma_seeded_ema(prices[:, j], fast)
        slow_ema = sma_seeded_ema(prices[:, j], slow)
        signal_full[:, j] = np.sign(fast_ema - slow_ema)
    signal = np.nan_to_num(signal_full[1:], nan=0.0)

    ewma_vol = np.column_stack(
        [ewma_annualized_volatility(returns[:, j], int(p["vol_lookback"])) for j in range(returns.shape[1])]
    )
    garch_fits = [fit_garch11_zero_mean(returns[:, j]) for j in range(returns.shape[1])]
    cond_vol = np.column_stack(
        [np.sqrt(np.maximum(fit.conditional_variance, 0.0)) for fit in garch_fits]
    )
    composite_vol = np.sqrt(np.mean(cond_vol * cond_vol, axis=1)) * math.sqrt(252.0)
    lo = float(np.percentile(composite_vol, float(p["regime_lo_pct"])))
    hi = float(np.percentile(composite_vol, float(p["regime_hi_pct"])))
    low = composite_vol <= lo
    high = composite_vol >= hi
    scale = np.ones(len(composite_vol), dtype=float)
    scale[low] = float(p["scale_low_vol"])
    scale[high] = float(p["scale_high_vol"])

    target_per_asset = float(p["vol_target"]) / float(p["n_assets"])
    max_lev = float(p["max_leverage_per_asset"])
    safe_vol = np.where(ewma_vol > 1e-12, ewma_vol, np.nan)
    static_weights = signal * (target_per_asset / safe_vol)
    dynamic_weights = signal * ((target_per_asset * scale)[:, None] / safe_vol)
    static_weights = np.nan_to_num(np.clip(static_weights, -max_lev, max_lev), nan=0.0)
    dynamic_weights = np.nan_to_num(np.clip(dynamic_weights, -max_lev, max_lev), nan=0.0)

    # End-of-day signals/vol/regime at t are applied to next day's return.
    static_returns = np.sum(static_weights[:-1] * returns[1:], axis=1)
    dynamic_returns = np.sum(dynamic_weights[:-1] * returns[1:], axis=1)
    static_perf = log_return_performance(
        static_returns,
        risk_free_annual=float(p["risk_free_annual"]),
    )
    dynamic_perf = log_return_performance(
        dynamic_returns,
        risk_free_annual=float(p["risk_free_annual"]),
    )

    frac_low = float(np.mean(low))
    frac_high = float(np.mean(high))
    frac_mid = 1.0 - frac_low - frac_high
    avg_persistence = float(np.mean([fit.persistence for fit in garch_fits]))
    avg_lrv = float(np.mean([fit.long_run_variance for fit in garch_fits]))

    results = {
        "dynamic_sharpe_ratio": dynamic_perf.sharpe_ratio,
        "sharpe_improvement": dynamic_perf.sharpe_ratio - static_perf.sharpe_ratio,
        "regime_frac_high": frac_high,
        "composite_vol_mean": float(np.mean(composite_vol)),
    }
    intermediates = {
        "static_annualized_return": {"value": static_perf.annualized_return},
        "static_annualized_volatility": {"value": static_perf.annualized_volatility},
        "static_sharpe_ratio": {"value": static_perf.sharpe_ratio},
        "static_max_drawdown": {"value": static_perf.max_drawdown},
        "static_calmar_ratio": {"value": static_perf.calmar_ratio},
        "dynamic_annualized_return": {"value": dynamic_perf.annualized_return},
        "dynamic_annualized_volatility": {"value": dynamic_perf.annualized_volatility},
        "dynamic_max_drawdown": {"value": dynamic_perf.max_drawdown},
        "dynamic_calmar_ratio": {"value": dynamic_perf.calmar_ratio},
        "avg_garch_persistence": {"value": avg_persistence},
        "avg_long_run_variance": {"value": avg_lrv},
        "regime_frac_low": {"value": frac_low},
        "regime_frac_mid": {"value": frac_mid},
        "regime_thresh_lo": {"value": lo},
        "regime_thresh_hi": {"value": hi},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (out_dir / "solution.json").write_text(
        json.dumps({"intermediates": intermediates}, indent=2) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class CompositeFinanceSkill:
    name: str = "finance-composition-domain"

    def matches(self, *, instruction: str, task_dir: Path) -> bool:
        return plan_finance_task(instruction, task_dir).executable

    def solve(
        self,
        *,
        instruction: str,
        task_dir: Path,
        out_dir: Path,
        seed: int,
    ) -> None:
        del seed
        plan = plan_finance_task(instruction, task_dir)
        if plan.executable_recipe == "structured-product-risk":
            _solve_structured_product(task_dir, out_dir)
            return
        if plan.executable_recipe == "volatility-target-strategy":
            _solve_volatility_target_strategy(task_dir, out_dir)
            return
        raise RuntimeError(
            "Finance composition plan is not executable yet: "
            + ", ".join(plan.capabilities)
        )
