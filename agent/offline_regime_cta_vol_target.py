from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model


def _find_named(task_dir: Path, name: str) -> Path:
    preferred = task_dir / "environment" / "data" / name
    if preferred.is_file():
        return preferred
    matches = [p for p in task_dir.rglob(name) if "checks" not in p.parts]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name!r}, found {len(matches)}.")
    return matches[0]


def _fit_garch(returns: np.ndarray) -> dict[str, object]:
    returns_arr = np.asarray(returns, dtype=float).ravel()
    returns_pct = returns_arr * 100.0
    model = arch_model(
        returns_pct,
        vol="Garch",
        p=1,
        q=1,
        mean="Zero",
        dist="normal",
    )
    result = model.fit(disp="off")

    omega = float(result.params["omega"]) / 10000.0
    alpha = float(result.params["alpha[1]"])
    beta = float(result.params["beta[1]"])
    persistence = alpha + beta

    if persistence < 1.0:
        long_run_variance = omega / (1.0 - persistence)
    else:
        long_run_variance = float(np.var(returns_arr, ddof=1))

    conditional_variance = (
        np.asarray(result.conditional_volatility, dtype=float) ** 2
    ) / 10000.0

    return {
        "persistence": persistence,
        "long_run_variance": long_run_variance,
        "conditional_variance": conditional_variance,
    }


def _sma_seeded_ema(prices: np.ndarray, span: int) -> np.ndarray:
    n = len(prices)
    out = np.full(n, np.nan, dtype=float)
    if n < span:
        return out
    alpha = 2.0 / (span + 1.0)
    out[span - 1] = float(np.mean(prices[:span]))
    for t in range(span, n):
        out[t] = alpha * prices[t] + (1.0 - alpha) * out[t - 1]
    return out


def _run_cta_strategy(
    prices: np.ndarray,
    returns: np.ndarray,
    fast_span: int,
    slow_span: int,
    vol_lookback: int,
    vol_targets: np.ndarray,
    max_leverage: float,
    n_assets: int,
    risk_free_annual: float,
    annualization: int = 252,
) -> dict[str, float]:
    n_price_days = prices.shape[0]
    n_returns = returns.shape[0]

    signal = np.zeros((n_price_days, n_assets), dtype=float)
    for i in range(n_assets):
        fast = _sma_seeded_ema(prices[:, i], fast_span)
        slow = _sma_seeded_ema(prices[:, i], slow_span)
        valid = ~np.isnan(fast) & ~np.isnan(slow)
        signal[valid, i] = np.sign(fast[valid] - slow[valid])

    alpha_vol = 2.0 / (vol_lookback + 1.0)
    ewma_variance = np.zeros((n_returns, n_assets), dtype=float)
    ewma_variance[0] = returns[0] ** 2
    for t in range(1, n_returns):
        ewma_variance[t] = (
            alpha_vol * returns[t] ** 2
            + (1.0 - alpha_vol) * ewma_variance[t - 1]
        )

    annualized_vol = np.sqrt(ewma_variance * annualization)

    positions = np.zeros((n_returns, n_assets), dtype=float)
    for t in range(1, n_returns):
        sig = signal[t]
        av = annualized_vol[t - 1]
        safe_vol = np.where(av > 1e-10, av, 1e-10)
        target = float(vol_targets[t])

        pos = sig * (target / safe_vol) * (1.0 / n_assets)
        positions[t] = np.clip(pos, -max_leverage, max_leverage)

    portfolio_returns = np.sum(positions * returns, axis=1)

    daily_rf = risk_free_annual / annualization
    mean_return = float(np.mean(portfolio_returns))
    std_return = float(np.std(portfolio_returns, ddof=1))

    annualized_return = mean_return * annualization
    annualized_volatility = std_return * math.sqrt(annualization)
    sharpe_ratio = (
        ((mean_return - daily_rf) / std_return) * math.sqrt(annualization)
        if std_return > 0.0
        else 0.0
    )

    wealth = np.exp(np.cumsum(portfolio_returns))
    running_max = np.maximum.accumulate(wealth)
    drawdown = (running_max - wealth) / running_max
    max_drawdown = float(np.max(drawdown))
    calmar_ratio = (
        annualized_return / max_drawdown if max_drawdown > 0.0 else 0.0
    )

    return {
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "calmar_ratio": float(calmar_ratio),
    }


@dataclass(frozen=True)
class RegimeCtaVolTargetSkill:
    name: str = "regime-cta-vol-target-reference"

    def matches(self, *, instruction: str, task_dir: Path) -> bool:
        del task_dir
        lowered = instruction.lower()
        required = (
            "regime-aware cta",
            "composite volatility",
            "regime_frac_high",
            "dynamic_sharpe_ratio",
            "avg_garch_persistence",
        )
        return all(token in lowered for token in required)

    def solve(
        self,
        *,
        instruction: str,
        task_dir: Path,
        out_dir: Path,
        seed: int,
    ) -> None:
        del instruction, seed

        params = json.loads(
            _find_named(task_dir, "params.json").read_text(encoding="utf-8")
        )
        frame = pd.read_csv(_find_named(task_dir, "prices.csv"))

        asset_columns = [c for c in frame.columns if c.startswith("asset_")]
        n_assets = int(params["n_assets"])
        if len(asset_columns) != n_assets:
            raise RuntimeError("Asset count does not match params.json.")

        prices = frame[asset_columns].to_numpy(dtype=float)
        annualization = 252
        returns = np.log(prices[1:] / prices[:-1])
        n_returns = returns.shape[0]

        conditional_variances = np.zeros((n_returns, n_assets), dtype=float)
        persistences: list[float] = []
        long_run_variances: list[float] = []

        for i in range(n_assets):
            fit = _fit_garch(returns[:, i])
            persistences.append(float(fit["persistence"]))
            long_run_variances.append(float(fit["long_run_variance"]))
            conditional_variances[:, i] = np.asarray(
                fit["conditional_variance"], dtype=float
            )

        avg_garch_persistence = float(np.mean(persistences))
        avg_long_run_variance = float(np.mean(long_run_variances))

        composite_variance = np.mean(conditional_variances, axis=1)
        composite_volatility = np.sqrt(composite_variance * annualization)
        composite_vol_mean = float(np.mean(composite_volatility))

        threshold_low = float(
            np.percentile(composite_volatility, params["regime_lo_pct"])
        )
        threshold_high = float(
            np.percentile(composite_volatility, params["regime_hi_pct"])
        )

        regimes = np.ones(n_returns, dtype=int)
        regimes[composite_volatility <= threshold_low] = 0
        regimes[composite_volatility >= threshold_high] = 2

        regime_frac_low = float(np.mean(regimes == 0))
        regime_frac_mid = float(np.mean(regimes == 1))
        regime_frac_high = float(np.mean(regimes == 2))

        base_target = float(params["vol_target"])
        static_targets = np.full(n_returns, base_target, dtype=float)
        dynamic_targets = np.full(n_returns, base_target, dtype=float)
        dynamic_targets[regimes == 0] = (
            base_target * float(params["scale_low_vol"])
        )
        dynamic_targets[regimes == 2] = (
            base_target * float(params["scale_high_vol"])
        )

        common = dict(
            prices=prices,
            returns=returns,
            fast_span=int(params["fast_ema"]),
            slow_span=int(params["slow_ema"]),
            vol_lookback=int(params["vol_lookback"]),
            max_leverage=float(params["max_leverage_per_asset"]),
            n_assets=n_assets,
            risk_free_annual=float(params["risk_free_annual"]),
            annualization=annualization,
        )

        static = _run_cta_strategy(vol_targets=static_targets, **common)
        dynamic = _run_cta_strategy(vol_targets=dynamic_targets, **common)

        results = {
            "dynamic_sharpe_ratio": float(dynamic["sharpe_ratio"]),
            "sharpe_improvement": float(
                dynamic["sharpe_ratio"] - static["sharpe_ratio"]
            ),
            "regime_frac_high": regime_frac_high,
            "composite_vol_mean": composite_vol_mean,
        }

        intermediates = {
            "static_annualized_return": {"value": static["annualized_return"]},
            "static_annualized_volatility": {
                "value": static["annualized_volatility"]
            },
            "static_sharpe_ratio": {"value": static["sharpe_ratio"]},
            "static_max_drawdown": {"value": static["max_drawdown"]},
            "static_calmar_ratio": {"value": static["calmar_ratio"]},
            "dynamic_annualized_return": {"value": dynamic["annualized_return"]},
            "dynamic_annualized_volatility": {
                "value": dynamic["annualized_volatility"]
            },
            "dynamic_max_drawdown": {"value": dynamic["max_drawdown"]},
            "dynamic_calmar_ratio": {"value": dynamic["calmar_ratio"]},
            "avg_garch_persistence": {"value": avg_garch_persistence},
            "avg_long_run_variance": {"value": avg_long_run_variance},
            "regime_frac_low": {"value": regime_frac_low},
            "regime_frac_mid": {"value": regime_frac_mid},
            "regime_thresh_lo": {"value": threshold_low},
            "regime_thresh_hi": {"value": threshold_high},
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(
            json.dumps(results, indent=2) + "\n",
            encoding="utf-8",
        )
        (out_dir / "solution.json").write_text(
            json.dumps({"intermediates": intermediates}, indent=2) + "\n",
            encoding="utf-8",
        )
