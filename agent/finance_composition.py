from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from agent.finance_schema import TaskDataCatalog


@dataclass(frozen=True)
class CompositionPlan:
    capabilities: tuple[str, ...]
    executable_recipe: str | None
    confidence: int

    @property
    def executable(self) -> bool:
        return self.executable_recipe is not None


def _contains(text: str, *terms: str) -> bool:
    return any(term in text for term in terms)


def plan_finance_task(instruction: str, task_dir: Path) -> CompositionPlan:
    text = re.sub(r"\s+", " ", instruction.lower())
    catalog = TaskDataCatalog.discover(task_dir)
    caps: set[str] = set()
    score = 0

    if _contains(text, "garch"):
        caps.add("garch11")
        score += 2
    if _contains(text, "dcc", "dynamic conditional correlation"):
        caps.update(("standardized_residuals", "dcc_correlation"))
        score += 3
    if _contains(text, "stationary bootstrap", "filtered historical simulation"):
        caps.add("stationary_bootstrap")
    if _contains(text, "kupiec"):
        caps.add("kupiec_backtest")
    if _contains(text, "christoffersen"):
        caps.add("christoffersen_backtest")

    if _contains(text, "crank-nicolson", "crank nicolson"):
        caps.add("crank_nicolson_pde")
        score += 3
    if _contains(text, "psor", "projected successive over-relaxation"):
        caps.add("psor_projection")
    if _contains(text, "cash dividend", "discrete dividend"):
        caps.add("cash_dividend_jump")

    if _contains(text, "cross-currency", "cross currency", "xccy"):
        caps.add("xccy_cashflow_engine")
        score += 3
    if _contains(text, "ois", "discount curve"):
        caps.add("curve_bootstrap")
    if _contains(text, "fx forward", "forward points"):
        caps.add("fx_forward_curve")

    if _contains(text, "fama-french", "fama french", "factor model"):
        caps.add("factor_ols")
        score += 3
    if _contains(text, "newey-west", "newey west", "hac"):
        caps.add("newey_west_hac")
    if _contains(text, "grs", "gibbons-ross-shanken"):
        caps.add("grs_joint_alpha_test")
    if _contains(text, "rolling", "rolling window") and "beta" in text:
        caps.add("rolling_factor_beta")

    if _contains(text, "ema"):
        caps.add("sma_seeded_ema")
    if _contains(text, "ewma") and _contains(text, "vol", "variance"):
        caps.add("ewma_volatility")
    if _contains(text, "volatility target", "vol target", "vol-target"):
        caps.add("vol_target_weights")
        score += 2
    if _contains(text, "regime"):
        caps.add("percentile_regime")
    if _contains(text, "sharpe", "calmar", "drawdown"):
        caps.add("performance_metrics")

    if _contains(text, "barrier option", "down-and-out", "down and out"):
        caps.add("analytical_barrier_option")
        score += 2
    if _contains(text, "breakeven vol", "breakeven volatility"):
        caps.add("root_solve")
    if _contains(text, "var", "value-at-risk", "value at risk"):
        caps.add("var_es")

    recipe = None
    cta_required = {
        "garch11",
        "sma_seeded_ema",
        "ewma_volatility",
        "vol_target_weights",
        "percentile_regime",
        "performance_metrics",
    }
    cta_schema = catalog.has_csv({"date"}, min_extra_columns=2) and catalog.has_json(
        {
            "fast_ema",
            "slow_ema",
            "vol_lookback",
            "vol_target",
            "n_assets",
            "regime_lo_pct",
            "regime_hi_pct",
        }
    )
    if cta_required.issubset(caps) and cta_schema:
        recipe = "volatility-target-strategy"
        score += 4

    structured_required = {
        "garch11",
        "analytical_barrier_option",
        "root_solve",
        "var_es",
    }
    structured_schema = catalog.has_csv({"date", "log_return"}) and catalog.has_json(
        {"notional", "s0", "k", "b", "r", "t", "participation"}
    )
    if structured_required.issubset(caps) and structured_schema:
        recipe = "structured-product-risk"
        score += 4

    return CompositionPlan(
        capabilities=tuple(sorted(caps)),
        executable_recipe=recipe,
        confidence=score,
    )
