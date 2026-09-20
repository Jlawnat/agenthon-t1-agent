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
    """Match semantic terms without accidental substring collisions.

    Examples:
    - "ois" matches "OIS curve"
    - "ois" does NOT match "Poisson"
    - multi-word and hyphenated phrases remain supported
    """
    for term in terms:
        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(term.lower())
            + r"(?![a-z0-9])"
        )
        if re.search(pattern, text):
            return True
    return False


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
    if _contains(
        text,
        "cash dividend",
        "cash dividends",
        "discrete dividend",
        "discrete dividends",
        "discrete cash dividend",
        "discrete cash dividends",
    ):
        caps.add("cash_dividend_jump")
    if _contains(text, "early exercise boundary", "exercise boundary"):
        caps.add("exercise_boundary")
    if _contains(text, "delta", "greeks"):
        caps.add("finite_difference_delta")
    if _contains(text, "richardson"):
        caps.add("richardson_extrapolation")

    if _contains(text, "cross-currency", "cross currency", "xccy"):
        caps.add("xccy_cashflow_engine")
        score += 3
    if _contains(text, "ois", "discount curve"):
        caps.add("curve_bootstrap")
    if _contains(
        text,
        "fx forward",
        "fx forwards",
        "forward point",
        "forward points",
    ):
        caps.add("fx_forward_curve")
    if _contains(text, "libor", "projection curve", "fra"):
        caps.add("projection_curve")
    if _contains(text, "historical fixing", "historical fixings", "fixing date"):
        caps.add("fixing_aware_coupons")
    if _contains(text, "reset notional", "reset notionals", "mtm reset", "mark-to-market"):
        caps.add("mtm_notional_resets")
    if _contains(text, "collateral", "collateralized", "collateralised"):
        caps.add("collateralized_discounting")
    if _contains(text, "stress scenario", "stress scenarios", "stress_results"):
        caps.add("stress_revaluation")

    if _contains(text, "fama-french", "fama french", "factor model"):
        caps.add("factor_ols")
        score += 3
    if _contains(text, "newey-west", "newey west", "hac"):
        caps.add("newey_west_hac")
    if _contains(text, "grs", "gibbons-ross-shanken"):
        caps.add("grs_joint_alpha_test")
    if _contains(text, "rolling", "rolling window") and "beta" in text:
        caps.add("rolling_factor_beta")
    if _contains(text, "variance inflation", "vif"):
        caps.add("vif")
    if _contains(text, "durbin-watson", "durbin watson"):
        caps.add("durbin_watson")

    if _contains(text, "ema"):
        caps.add("sma_seeded_ema")
    if _contains(text, "ewma") and _contains(
        text,
        "vol",
        "volatility",
        "variance",
    ):
        caps.add("ewma_volatility")
    if _contains(
        text,
        "volatility target",
        "volatility targeting",
        "vol target",
        "vol targeting",
        "vol-target",
        "vol-targeting",
    ):
        caps.add("vol_target_weights")
        score += 2
    if _contains(
        text,
        "regime",
        "regimes",
    ):
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

    if _contains(
        text,
        "ornstein-uhlenbeck",
        "ornstein uhlenbeck",
        "ou process",
        "ou model",
        "ou in log-space",
        "ou in log space",
    ):
        caps.add("ou_calibration")
        score += 2
    if _contains(
        text,
        "adf",
        "augmented dickey",
    ):
        caps.add("stationarity_test")
    if _contains(
        text,
        "exact ou transition",
        "exact ou transition distribution",
    ):
        caps.add("exact_ou_simulation")
    if (
        _contains(
            text,
            "funding rate",
            "funding rates",
        )
        and _contains(
            text,
            "basis carry",
            "carry trade",
        )
    ):
        caps.add("basis_carry_risk")
        score += 2

    if _contains(
        text,
        "mean-reverting jump-diffusion",
        "mean reverting jump diffusion",
    ):
        caps.add("ou_exact_calibration")
        score += 2
    if _contains(
        text,
        "poisson",
        "jump-diffusion",
        "jump diffusion",
    ):
        caps.add("compound_poisson_jumps")
    if _contains(
        text,
        "conditional moments",
    ):
        caps.add("conditional_process_moments")
    if _contains(
        text,
        "monte carlo",
        "mc simulation",
    ):
        caps.add("process_monte_carlo")

    recipe = None

    funding_required = {
        "ou_calibration",
        "stationarity_test",
        "exact_ou_simulation",
        "basis_carry_risk",
        "process_monte_carlo",
    }
    funding_schema = (
        catalog.has_csv({
            "symbol",
            "funding_time",
            "funding_rate",
        })
        and catalog.has_json({
            "target_symbol",
            "analysis_start",
            "analysis_end",
            "regime_split_date",
            "annualization_periods",
            "ou_dt",
            "mc_num_paths",
            "mc_horizon_periods",
            "mc_random_seed",
            "var_confidence",
        })
    )
    if (
        funding_required.issubset(caps)
        and funding_schema
    ):
        recipe = (
            "funding-ou-carry-analysis"
        )
        score += 5

    log_jump_required = {
        "ou_calibration",
        "ou_exact_calibration",
        "compound_poisson_jumps",
        "conditional_process_moments",
        "process_monte_carlo",
    }
    log_jump_schema = (
        catalog.has_csv({"dgs10"})
    )
    if (
        log_jump_required.issubset(caps)
        and log_jump_schema
    ):
        recipe = "log-ou-jump-analysis"
        score += 5

    xccy_required = {
        "xccy_cashflow_engine",
        "curve_bootstrap",
        "fx_forward_curve",
        "projection_curve",
        "fixing_aware_coupons",
        "mtm_notional_resets",
        "collateralized_discounting",
        "stress_revaluation",
    }
    xccy_schema = (
        catalog.has_json(
            {
                "valuation_date",
                "spot_date",
                "fx_pair",
                "collateral_currency",
                "usd_notional",
                "contract_spread_bps",
                "inception_spot_fx",
            }
        )
        and catalog.has_csv(
            {
                "quote_time",
                "curve",
                "instrument_type",
                "tenor",
                "start_date",
                "end_date",
                "ccy_or_pair",
                "quote_unit",
                "quote_value",
            }
        )
        and catalog.has_csv({"fixing_date", "index_name", "rate_pct"})
        and catalog.has_csv(
            {
                "period_id",
                "accrual_start",
                "accrual_end",
                "payment_date",
                "reset_date",
                "fixing_date",
            }
        )
    )
    if xccy_required.issubset(caps) and xccy_schema:
        recipe = "xccy-desk-analysis"
        score += 5

    fd_required = {
        "crank_nicolson_pde",
        "psor_projection",
        "cash_dividend_jump",
        "exercise_boundary",
        "finite_difference_delta",
        "richardson_extrapolation",
    }
    if fd_required.issubset(caps):
        recipe = "finite-difference-option-analysis"
        score += 4

    factor_required = {
        "factor_ols",
        "newey_west_hac",
        "grs_joint_alpha_test",
        "rolling_factor_beta",
    }
    factor_schema = (
        catalog.has_csv({"date", "mkt-rf", "smb", "hml", "rf"})
        and sum(
            1
            for artifact in catalog.artifacts
            if artifact.kind == "csv"
            and "date" in artifact.columns
            and len(artifact.columns) >= 3
            and not {"mkt-rf", "smb", "hml", "rf"}.issubset(artifact.columns)
        )
        == 1
    )
    if factor_required.issubset(caps) and factor_schema:
        recipe = "factor-regression-analysis"
        score += 4

    dcc_required = {
        "garch11",
        "standardized_residuals",
        "dcc_correlation",
        "stationary_bootstrap",
        "kupiec_backtest",
        "christoffersen_backtest",
        "var_es",
    }
    dcc_schema = False
    for price_artifact in catalog.artifacts:
        if (
            price_artifact.kind != "csv"
            or "date" not in price_artifact.columns
            or len(price_artifact.columns) < 3
        ):
            continue
        series_keys = frozenset(c for c in price_artifact.columns if c != "date")
        if any(
            artifact.kind == "json" and artifact.json_keys == series_keys
            for artifact in catalog.artifacts
        ):
            dcc_schema = True
            break
    if dcc_required.issubset(caps) and dcc_schema:
        recipe = "dcc-multivariate-risk"
        score += 4

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
