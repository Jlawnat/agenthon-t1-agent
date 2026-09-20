from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from agent.finance_composition import plan_finance_task
from agent.finance_multivariate import (
    conditional_covariance_path,
    coverage_backtest,
    filtered_historical_var_es,
    fit_dcc_qml,
    portfolio_conditional_volatility,
    standardized_residuals,
    stationary_bootstrap_rows,
)
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
from agent.finance_derivatives_mc import (
    run_cliquet_workflow,
    run_mc_greek_surface_workflow,
)
from agent.finance_processes import (
    run_funding_ou_carry_workflow,
    run_log_ou_jump_moments_workflow,
)
from agent.finance_xccy import run_xccy_workflow


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



def _find_series_weight_mapping(
    catalog: TaskDataCatalog,
    asset_columns: list[str],
) -> tuple[Path, np.ndarray]:
    wanted = {str(c).strip().lower() for c in asset_columns}
    matches: list[tuple[Path, np.ndarray]] = []
    for artifact in catalog.artifacts:
        if artifact.kind != "json" or set(artifact.json_keys) != wanted:
            continue
        try:
            value = json.loads(artifact.path.read_text(encoding="utf-8"))
            lower = {str(k).strip().lower(): float(v) for k, v in value.items()}
            weights = np.asarray([lower[str(c).strip().lower()] for c in asset_columns], dtype=float)
        except Exception:
            continue
        if np.all(np.isfinite(weights)):
            matches.append((artifact.path, weights))
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one numeric JSON mapping keyed by the price-series columns."
        )
    return matches[0]


def _solve_dcc_multivariate_risk(task_dir: Path, out_dir: Path) -> None:
    from scipy.stats import norm

    catalog = TaskDataCatalog.discover(task_dir)
    prices_path = catalog.csv_with_columns({"date"}, min_extra_columns=2)
    frame = pd.read_csv(prices_path)
    asset_cols = [c for c in frame.columns if str(c).strip().lower() != "date"]
    if len(asset_cols) < 2:
        raise RuntimeError("DCC portfolio risk requires at least two assets.")

    prices = frame[asset_cols].to_numpy(dtype=float)
    if prices.shape[0] < 20 or np.any(prices <= 0.0) or not np.all(np.isfinite(prices)):
        raise RuntimeError("Price panel contains invalid or insufficient observations.")
    dates = frame["date"].astype(str).iloc[1:].reset_index(drop=True)
    returns = np.log(prices[1:] / prices[:-1])

    _, weights = _find_series_weight_mapping(catalog, asset_cols)
    if not np.isclose(float(np.sum(weights)), 1.0, atol=1e-6):
        raise RuntimeError("Portfolio weights must sum to one.")

    garch_fits = [fit_garch11_zero_mean(returns[:, j]) for j in range(returns.shape[1])]
    conditional_variances = np.column_stack(
        [fit.conditional_variance for fit in garch_fits]
    )
    conditional_std = np.sqrt(np.maximum(conditional_variances, 1e-18))
    z = standardized_residuals(returns, conditional_variances)

    dcc = fit_dcc_qml(z)
    covariances = conditional_covariance_path(conditional_std, dcc.correlations)
    portfolio_vol = portfolio_conditional_volatility(covariances, weights)
    portfolio_returns = returns @ weights
    annualised_vol = portfolio_vol * math.sqrt(252.0)

    z_1pct = abs(float(norm.ppf(0.01)))
    var_normal = z_1pct * portfolio_vol
    es_normal = (
        float(norm.pdf(norm.ppf(0.01))) / 0.01
    ) * portfolio_vol
    violations = (portfolio_returns < -var_normal).astype(int)

    bootstrap_z = stationary_bootstrap_rows(
        z,
        n_samples=1000,
        restart_probability=1.0 / 21.0,
        seed=42,
    )
    var_fhs, es_fhs = filtered_historical_var_es(
        bootstrap_z,
        conditional_std,
        weights,
        tail_probability=0.01,
    )
    backtest = coverage_backtest(violations, expected_rate=0.01)

    garch_rows = []
    for ticker, fit in zip(asset_cols, garch_fits, strict=True):
        unconditional_vol = math.sqrt(max(fit.long_run_variance, 0.0)) * math.sqrt(252.0)
        garch_rows.append(
            {
                "ticker": ticker,
                "omega": fit.omega,
                "alpha": fit.alpha,
                "beta": fit.beta,
                "persistence": fit.persistence,
                "unconditional_vol": unconditional_vol,
            }
        )

    pair_columns: dict[str, np.ndarray] = {}
    for i in range(len(asset_cols)):
        for j in range(i + 1, len(asset_cols)):
            pair_columns[f"{asset_cols[i]}_{asset_cols[j]}"] = dcc.correlations[:, i, j]

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(garch_rows).to_csv(out_dir / "garch_params.csv", index=False)

    (out_dir / "dcc_params.json").write_text(
        json.dumps(
            {
                "dcc_alpha": round(dcc.alpha, 8),
                "dcc_beta": round(dcc.beta, 8),
                "dcc_persistence": round(dcc.persistence, 8),
                "log_likelihood": round(dcc.log_likelihood, 4),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    corr_frame = pd.DataFrame({"date": dates})
    for name, values in pair_columns.items():
        corr_frame[name] = values
    corr_frame.to_csv(out_dir / "dynamic_correlations.csv", index=False)

    pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": portfolio_returns,
            "conditional_vol": portfolio_vol,
            "annualised_vol": annualised_vol,
            "var_1pct": var_normal,
            "es_1pct": es_normal,
            "violation": violations,
            "var_1pct_fhs": var_fhs,
            "es_1pct_fhs": es_fhs,
        }
    ).to_csv(out_dir / "portfolio_var.csv", index=False)

    (out_dir / "backtest_results.json").write_text(
        json.dumps(
            {
                "num_observations": backtest.num_observations,
                "num_violations": backtest.num_violations,
                "violation_rate": round(backtest.violation_rate, 6),
                "expected_rate": 0.01,
                "kupiec_lr_stat": round(backtest.kupiec_lr_stat, 6),
                "kupiec_p_value": round(backtest.kupiec_p_value, 6),
                "reject_h0_kupiec_5pct": bool(backtest.kupiec_p_value < 0.05),
                "christoffersen_lr_ind": round(backtest.christoffersen_lr_ind, 6),
                "christoffersen_lr_cc": round(backtest.christoffersen_lr_cc, 6),
                "christoffersen_p_value": round(backtest.christoffersen_p_value, 6),
                "reject_h0_cc_5pct": bool(backtest.christoffersen_p_value < 0.05),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _extract_sector_assignments(
    instruction: str,
    tickers: list[str],
) -> dict[str, str]:
    import re

    known = set(tickers)
    mapping: dict[str, str] = {}

    for raw in instruction.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 2:
            continue
        sector = parts[0]
        if not sector or sector.lower() == "sector" or set(sector) <= {"-", ":"}:
            continue
        members = [token.strip() for token in parts[1].split(",")]
        for ticker in members:
            if ticker in known:
                mapping[ticker] = sector

    for raw in instruction.splitlines():
        match = re.match(r"^\s*-\s*([^:]+):\s*(.+?)\s*$", raw)
        if not match:
            continue
        sector = match.group(1).strip()
        members = [token.strip() for token in match.group(2).split(",")]
        for ticker in members:
            if ticker in known:
                mapping[ticker] = sector

    return mapping


def _solve_factor_regression_analysis(
    instruction: str,
    task_dir: Path,
    out_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    from agent.finance_factor_stats import (
        durbin_watson,
        grs_joint_alpha_test,
        newey_west_covariance,
        ols_with_inference,
        optimal_newey_west_lag,
        rolling_ols_coefficient,
        variance_inflation_factors,
    )

    catalog = TaskDataCatalog.discover(task_dir)

    factor_artifacts = [
        artifact
        for artifact in catalog.artifacts
        if artifact.kind == "csv"
        and {"date", "mkt-rf", "smb", "hml", "rf"}.issubset(artifact.columns)
    ]
    if len(factor_artifacts) != 1:
        raise RuntimeError("Expected exactly one daily factor CSV.")
    factor_path = factor_artifacts[0].path

    price_artifacts = [
        artifact
        for artifact in catalog.artifacts
        if artifact.kind == "csv"
        and "date" in artifact.columns
        and len(artifact.columns) >= 3
        and artifact.path != factor_path
    ]
    if len(price_artifacts) != 1:
        raise RuntimeError("Expected exactly one asset price panel CSV.")
    price_path = price_artifacts[0].path

    prices = pd.read_csv(price_path)
    factors = pd.read_csv(factor_path)

    def normalize_date_column(frame: pd.DataFrame) -> pd.DataFrame:
        date_cols = [c for c in frame.columns if str(c).strip().lower() == "date"]
        if len(date_cols) != 1:
            raise RuntimeError("Expected exactly one date column.")
        frame = frame.copy()
        frame[date_cols[0]] = pd.to_datetime(frame[date_cols[0]], errors="raise")
        return frame.set_index(date_cols[0]).sort_index()

    prices = normalize_date_column(prices)
    factors = normalize_date_column(factors)

    factor_name_map = {str(c).strip().lower(): c for c in factors.columns}
    required_factor_names = ("mkt-rf", "smb", "hml", "rf")
    if any(name not in factor_name_map for name in required_factor_names):
        raise RuntimeError("Required factor columns are missing.")

    prices = prices.apply(pd.to_numeric, errors="coerce")
    factors = factors.apply(pd.to_numeric, errors="coerce")

    returns = prices.pct_change(fill_method=None).dropna(how="any")
    common_idx = returns.index.intersection(factors.index)
    returns = returns.loc[common_idx]
    factors = factors.loc[common_idx]

    if len(common_idx) <= 260:
        raise RuntimeError("Factor regression requires more than one trading year.")

    tickers = sorted(str(c) for c in prices.columns)
    if len(tickers) < 2:
        raise RuntimeError("Factor regression requires multiple assets.")

    rf = factors[factor_name_map["rf"]]
    excess_returns = returns.subtract(rf, axis=0)

    factor_matrix = np.column_stack(
        [
            factors[factor_name_map["mkt-rf"]].to_numpy(dtype=float),
            factors[factor_name_map["smb"]].to_numpy(dtype=float),
            factors[factor_name_map["hml"]].to_numpy(dtype=float),
        ]
    )
    n_obs = len(common_idx)
    design = np.column_stack([np.ones(n_obs), factor_matrix])
    nw_lag = optimal_newey_west_lag(n_obs)

    stock_results: dict[str, dict[str, float]] = {}
    residual_matrix = np.zeros((n_obs, len(tickers)), dtype=float)
    alpha_vector = np.zeros(len(tickers), dtype=float)
    rolling_data: dict[str, np.ndarray] = {}

    for column_index, ticker in enumerate(tickers):
        y = excess_returns[ticker].to_numpy(dtype=float)
        fit = ols_with_inference(design, y)
        beta = fit.coefficients

        nw_cov = newey_west_covariance(design, fit.residuals, lag=nw_lag)
        nw_se = np.sqrt(np.maximum(np.diag(nw_cov), 0.0))
        t_alpha_nw = float(beta[0] / nw_se[0]) if nw_se[0] > 0.0 else 0.0
        p_alpha_nw = float(
            2.0 * stats.t.sf(abs(t_alpha_nw), df=fit.degrees_of_freedom)
        )

        alpha_vector[column_index] = beta[0]
        residual_matrix[:, column_index] = fit.residuals

        annual_alpha = float(beta[0] * 252.0)
        annual_excess_return = float(np.mean(y) * 252.0)
        annual_residual_std = float(np.std(fit.residuals, ddof=1) * math.sqrt(252.0))
        information_ratio = (
            annual_alpha / annual_residual_std if annual_residual_std > 0.0 else 0.0
        )
        treynor = annual_excess_return / beta[1] if beta[1] != 0.0 else 0.0

        rolling = rolling_ols_coefficient(
            design,
            y,
            window=252,
            coefficient_index=1,
        )
        rolling_data[ticker] = rolling

        stock_results[ticker] = {
            "alpha_daily": round(float(beta[0]), 8),
            "alpha_annualized": round(annual_alpha, 6),
            "beta_mkt": round(float(beta[1]), 6),
            "beta_smb": round(float(beta[2]), 6),
            "beta_hml": round(float(beta[3]), 6),
            "r_squared": round(fit.r_squared, 6),
            "adj_r_squared": round(fit.adjusted_r_squared, 6),
            "t_stat_alpha": round(float(fit.t_statistics[0]), 4),
            "p_value_alpha": round(float(fit.p_values[0]), 6),
            "t_stat_alpha_nw": round(t_alpha_nw, 4),
            "p_value_alpha_nw": round(p_alpha_nw, 6),
            "durbin_watson": round(durbin_watson(fit.residuals), 6),
            "information_ratio": round(float(information_ratio), 6),
            "treynor_ratio": round(float(treynor), 6),
            "ann_excess_return": round(annual_excess_return, 6),
            "rolling_beta_mkt_mean": round(float(np.mean(rolling)), 6),
            "rolling_beta_mkt_std": round(float(np.std(rolling, ddof=1)), 6),
            "rolling_beta_mkt_min": round(float(np.min(rolling)), 6),
            "rolling_beta_mkt_max": round(float(np.max(rolling)), 6),
        }

    alpha_ranking = sorted(
        tickers,
        key=lambda ticker: stock_results[ticker]["alpha_annualized"],
        reverse=True,
    )

    grs = grs_joint_alpha_test(alpha_vector, residual_matrix, factor_matrix)
    vif = variance_inflation_factors(factor_matrix)

    out_dir.mkdir(parents=True, exist_ok=True)

    rolling_dates = common_idx[251:]
    rolling_frame = pd.DataFrame({"date": rolling_dates})
    for ticker in tickers:
        rolling_frame[ticker] = rolling_data[ticker]
    rolling_frame.to_csv(out_dir / "rolling_betas.csv", index=False)

    results = {
        "meta": {
            "trading_days": int(n_obs),
            "start_date": str(common_idx.min().date()),
            "end_date": str(common_idx.max().date()),
            "num_stocks": len(tickers),
            "nw_lag": int(nw_lag),
        },
        "stocks": stock_results,
        "alpha_ranking": alpha_ranking,
        "grs_test": {
            "grs_statistic": round(grs.statistic, 4),
            "grs_p_value": round(grs.p_value, 6),
            "grs_rejected": bool(grs.p_value < 0.05),
            "df1": grs.df1,
            "df2": grs.df2,
        },
        "factor_diagnostics": {
            "vif_mkt_rf": round(float(vif[0]), 4),
            "vif_smb": round(float(vif[1]), 4),
            "vif_hml": round(float(vif[2]), 4),
            "nw_lag": int(nw_lag),
        },
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_rows = []
    for ticker in tickers:
        item = stock_results[ticker]
        summary_rows.append({
            "ticker": ticker,
            "alpha_annualized": item["alpha_annualized"],
            "beta_mkt": item["beta_mkt"],
            "beta_smb": item["beta_smb"],
            "beta_hml": item["beta_hml"],
            "r_squared": item["r_squared"],
            "adj_r_squared": item["adj_r_squared"],
            "t_stat_alpha": item["t_stat_alpha"],
            "p_value_alpha": item["p_value_alpha"],
            "t_stat_alpha_nw": item["t_stat_alpha_nw"],
            "p_value_alpha_nw": item["p_value_alpha_nw"],
            "durbin_watson": item["durbin_watson"],
            "information_ratio": item["information_ratio"],
            "treynor_ratio": item["treynor_ratio"],
        })
    pd.DataFrame(summary_rows).to_csv(
        out_dir / "regression_summary.csv",
        index=False,
    )

    sector_map = _extract_sector_assignments(instruction, tickers)
    if set(sector_map) != set(tickers):
        missing = sorted(set(tickers) - set(sector_map))
        raise RuntimeError(
            "Could not recover sector assignments for: " + ", ".join(missing)
        )

    sector_rows = []
    for sector in sorted(set(sector_map.values())):
        members = [ticker for ticker in tickers if sector_map[ticker] == sector]
        sector_rows.append({
            "sector": sector,
            "avg_alpha_annualized": round(
                float(np.mean([stock_results[t]["alpha_annualized"] for t in members])), 6
            ),
            "avg_beta_mkt": round(
                float(np.mean([stock_results[t]["beta_mkt"] for t in members])), 6
            ),
            "avg_r_squared": round(
                float(np.mean([stock_results[t]["r_squared"] for t in members])), 6
            ),
        })
    pd.DataFrame(sector_rows).to_csv(
        out_dir / "sector_summary.csv",
        index=False,
    )

    x = np.arange(len(tickers))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width, [stock_results[t]["beta_mkt"] for t in tickers], width, label="Mkt-RF")
    ax.bar(x, [stock_results[t]["beta_smb"] for t in tickers], width, label="SMB")
    ax.bar(x + width, [stock_results[t]["beta_hml"] for t in tickers], width, label="HML")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=45)
    ax.set_ylabel("Factor Loading")
    ax.set_title("Fama-French 3-Factor Loadings")
    ax.legend()
    ax.axhline(y=0.0, linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "factor_loadings.png", dpi=100)
    plt.close(fig)

    ranked = alpha_ranking
    alphas = [stock_results[t]["alpha_annualized"] for t in ranked]
    labels = [
        f"{ticker} *" if stock_results[ticker]["p_value_alpha_nw"] < 0.05 else ticker
        for ticker in ranked
    ]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(ranked)), alphas)
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Annualized Alpha")
    ax.set_title("Annualized Jensen's Alpha (NW significance)")
    ax.axvline(x=0.0, linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / "alpha_chart.png", dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(tickers, [stock_results[t]["r_squared"] for t in tickers])
    ax.set_ylabel("R-squared")
    ax.set_title("R-squared — Fama-French 3-Factor Model")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "r_squared_chart.png", dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 8))
    for ticker in tickers:
        ax.plot(rolling_dates, rolling_data[ticker], label=ticker, alpha=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Market Beta (252-day)")
    ax.set_title("Rolling 252-Day Market Beta")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "rolling_betas.png", dpi=100)
    plt.close(fig)


def _parse_coarse_fd_grid(instruction: str, fine_stock: int, fine_time: int) -> tuple[int, int]:
    import re

    match_s = re.search(
        r"coarser grid[^\\n]*?N_?S\\s*=\\s*([0-9]+)",
        instruction,
        re.I,
    )
    match_t = re.search(
        r"coarser grid.*?N_?T\\s*=\\s*([0-9]+)",
        instruction,
        re.I | re.S,
    )
    coarse_stock = int(match_s.group(1)) if match_s else max(fine_stock // 2, 3)
    coarse_time = int(match_t.group(1)) if match_t else max(fine_time // 2, 2)
    return coarse_stock, coarse_time


def _solve_finite_difference_option_analysis(
    instruction: str,
    out_dir: Path,
) -> None:
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from agent.finance_fd import (
        crank_nicolson_option,
        parse_fd_option_spec,
        richardson_second_order,
        with_grid_size,
    )

    spec = parse_fd_option_spec(instruction)

    am_put = crank_nicolson_option(
        spec,
        option_type="put",
        exercise_type="american",
        return_grid=True,
        return_boundary=True,
    )
    am_call = crank_nicolson_option(
        spec,
        option_type="call",
        exercise_type="american",
    )
    eu_put = crank_nicolson_option(
        spec,
        option_type="put",
        exercise_type="european",
    )
    eu_call = crank_nicolson_option(
        spec,
        option_type="call",
        exercise_type="european",
    )

    am_put_no_div = crank_nicolson_option(
        spec,
        option_type="put",
        exercise_type="american",
        dividends=(),
    )
    am_call_no_div = crank_nicolson_option(
        spec,
        option_type="call",
        exercise_type="american",
        dividends=(),
    )
    eu_put_no_div = crank_nicolson_option(
        spec,
        option_type="put",
        exercise_type="european",
        dividends=(),
    )
    eu_call_no_div = crank_nicolson_option(
        spec,
        option_type="call",
        exercise_type="european",
        dividends=(),
    )

    coarse_stock, coarse_time = _parse_coarse_fd_grid(
        instruction,
        spec.stock_steps,
        spec.time_steps,
    )
    coarse_spec = with_grid_size(
        spec,
        stock_steps=coarse_stock,
        time_steps=coarse_time,
    )
    coarse_put = crank_nicolson_option(
        coarse_spec,
        option_type="put",
        exercise_type="american",
    )

    rich = richardson_second_order(am_put.value, coarse_put.value)
    err_fine = abs(am_put.value - rich)
    err_coarse = abs(coarse_put.value - rich)
    convergence_ratio = err_coarse / err_fine if err_fine > 1e-10 else float("inf")

    out_dir.mkdir(parents=True, exist_ok=True)

    option_values = {
        "american_put": am_put.value,
        "american_call": am_call.value,
        "european_put": eu_put.value,
        "european_call": eu_call.value,
        "american_put_no_div": am_put_no_div.value,
        "american_call_no_div": am_call_no_div.value,
        "european_put_no_div": eu_put_no_div.value,
        "european_call_no_div": eu_call_no_div.value,
    }
    (out_dir / "option_values.json").write_text(
        json.dumps(option_values, indent=2) + "\n",
        encoding="utf-8",
    )

    if am_put.value_grid is None or am_put.exercise_boundary is None:
        raise RuntimeError("American put diagnostic grid was not produced.")

    stock_sample_idx = list(range(0, spec.stock_steps + 1, 5))
    if spec.stock_steps not in stock_sample_idx:
        stock_sample_idx.append(spec.stock_steps)
    time_sample_idx = list(range(0, spec.time_steps + 1, 10))
    if spec.time_steps not in time_sample_idx:
        time_sample_idx.append(spec.time_steps)

    with (out_dir / "american_put_grid.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            ["S"] + [f"{am_put.time_grid[i]:.4f}" for i in time_sample_idx]
        )
        for stock_index in stock_sample_idx:
            writer.writerow(
                [f"{am_put.stock_grid[stock_index]:.2f}"]
                + [
                    f"{am_put.value_grid[stock_index, time_index]:.6f}"
                    for time_index in time_sample_idx
                ]
            )

    with (out_dir / "early_exercise_boundary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["t", "S_star"])
        for time, boundary in zip(
            am_put.time_grid,
            am_put.exercise_boundary,
            strict=True,
        ):
            writer.writerow([f"{time:.6f}", f"{boundary:.6f}"])

    with (out_dir / "greeks.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["option_type", "delta"])
        writer.writerow(["american_put", f"{am_put.delta:.6f}"])
        writer.writerow(["american_call", f"{am_call.delta:.6f}"])
        writer.writerow(["european_put", f"{eu_put.delta:.6f}"])
        writer.writerow(["european_call", f"{eu_call.delta:.6f}"])

    with (out_dir / "convergence.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["grid", "price"])
        writer.writerow(["fine", f"{am_put.value:.6f}"])
        writer.writerow(["coarse", f"{coarse_put.value:.6f}"])
        writer.writerow(["richardson", f"{rich:.6f}"])

    fig, ax = plt.subplots(figsize=(10, 6))
    mask = am_put.exercise_boundary > 0.0
    ax.plot(
        am_put.time_grid[mask],
        am_put.exercise_boundary[mask],
        linewidth=1.5,
    )
    ax.axhline(y=spec.strike, linestyle="--", alpha=0.5, label=f"K={spec.strike:g}")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Critical Stock Price S*(t)")
    ax.set_title("Early Exercise Boundary (American Put)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "exercise_boundary.png", dpi=150)
    plt.close(fig)

    no_div_call_diff = abs(am_call_no_div.value - eu_call_no_div.value)
    summary = {
        "american_geq_european_put": bool(am_put.value >= eu_put.value),
        "american_geq_european_call": bool(am_call.value >= eu_call.value),
        "no_div_call_diff": float(no_div_call_diff),
        "early_exercise_premium_put": float(am_put.value - eu_put.value),
        "early_exercise_premium_call": float(am_call.value - eu_call.value),
        "exercise_boundary_at_T": float(am_put.exercise_boundary[-1]),
        "exercise_boundary_at_0": float(am_put.exercise_boundary[0]),
        "richardson_estimate": float(rich),
        "convergence_ratio": float(convergence_ratio),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
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
        if plan.executable_recipe == "cliquet-forward-start-analysis":
            run_cliquet_workflow(task_dir, out_dir)
            return
        if plan.executable_recipe == "mc-greek-surface-analysis":
            run_mc_greek_surface_workflow(out_dir)
            return
        if plan.executable_recipe == "funding-ou-carry-analysis":
            run_funding_ou_carry_workflow(task_dir, out_dir)
            return
        if plan.executable_recipe == "log-ou-jump-analysis":
            run_log_ou_jump_moments_workflow(task_dir, out_dir)
            return
        if plan.executable_recipe == "xccy-desk-analysis":
            run_xccy_workflow(task_dir, out_dir, instruction)
            return
        if plan.executable_recipe == "finite-difference-option-analysis":
            _solve_finite_difference_option_analysis(instruction, out_dir)
            return
        if plan.executable_recipe == "factor-regression-analysis":
            _solve_factor_regression_analysis(instruction, task_dir, out_dir)
            return
        if plan.executable_recipe == "dcc-multivariate-risk":
            _solve_dcc_multivariate_risk(task_dir, out_dir)
            return
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
