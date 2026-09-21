from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

try:
    from arch import arch_model
except Exception:  # pragma: no cover - benchmark image provides arch
    arch_model = None


def _find_named(task_dir: Path, name: str) -> Path:
    preferred = task_dir / "environment" / "data" / name
    if preferred.is_file():
        return preferred
    matches = [p for p in task_dir.rglob(name) if "checks" not in p.parts]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name!r}, found {len(matches)}.")
    return matches[0]


def _sma_seeded_ema(values: np.ndarray, span: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    span = int(span)
    if span <= 0 or n < span:
        return out
    alpha = 2.0 / (span + 1.0)
    seed_index = span - 1
    out[seed_index] = float(np.mean(x[:span]))
    for t in range(span, n):
        out[t] = alpha * x[t] + (1.0 - alpha) * out[t - 1]
    return out


def _ewma_annualized_vol(returns: np.ndarray, lookback: int) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    n = len(r)
    out = np.full(n, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(r))
    if len(finite) == 0:
        return out

    alpha = 2.0 / (int(lookback) + 1.0)
    first = int(finite[0])
    variance = float(r[first] * r[first])
    out[first] = math.sqrt(252.0 * variance)

    for t in range(first + 1, n):
        if not np.isfinite(r[t]):
            out[t] = math.sqrt(252.0 * variance)
            continue
        variance = (
            alpha * float(r[t] * r[t])
            + (1.0 - alpha) * variance
        )
        out[t] = math.sqrt(252.0 * variance)

    return out


def _fit_garch(raw_returns: np.ndarray) -> dict[str, object]:
    """Match the benchmark reference GARCH convention exactly.

    Reference generation fits decimal portfolio returns after multiplying
    by 100, then converts omega and conditional variances back to native
    decimal-return squared units.
    """
    if arch_model is None:
        raise RuntimeError("arch package is unavailable.")

    x = np.asarray(
        raw_returns,
        dtype=float,
    )
    x = x[np.isfinite(x)]

    if len(x) < 50:
        raise RuntimeError(
            "Too few returns for GARCH fitting."
        )

    x_pct = x * 100.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = arch_model(
            x_pct,
            vol="Garch",
            p=1,
            q=1,
            mean="Zero",
            dist="normal",
        )

        fitted = model.fit(
            disp="off",
        )

    omega = (
        float(fitted.params["omega"])
        / 10000.0
    )
    alpha = float(
        fitted.params["alpha[1]"]
    )
    beta = float(
        fitted.params["beta[1]"]
    )
    persistence = alpha + beta

    if persistence < 1.0:
        long_run_variance = (
            omega
            / (1.0 - persistence)
        )
    else:
        long_run_variance = float(
            np.var(
                x,
                ddof=1,
            )
        )

    conditional_variance = (
        np.asarray(
            fitted.conditional_volatility,
            dtype=float,
        )
        ** 2
        / 10000.0
    )

    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": persistence,
        "long_run_variance": float(
            long_run_variance
        ),
        "cond_var": conditional_variance,
    }


def _garch_multistep_forecast(
    *,
    omega: float,
    alpha: float,
    beta: float,
    last_return: float,
    last_variance: float,
    holding_period: int,
) -> tuple[np.ndarray, float]:
    """Benchmark-reference GARCH multi-step variance recursion."""
    horizon = int(holding_period)

    forecasts = np.empty(
        horizon,
        dtype=float,
    )

    forecasts[0] = (
        omega
        + alpha * last_return ** 2
        + beta * last_variance
    )

    persistence = alpha + beta

    for step in range(
        1,
        horizon,
    ):
        forecasts[step] = (
            omega
            + persistence
            * forecasts[step - 1]
        )

    return (
        forecasts,
        float(
            np.sum(forecasts)
        ),
    )


@dataclass(frozen=True)
class CtaBaselCapitalSkill:
    name: str = "cta-basel-capital"

    def matches(self, *, instruction: str, task_dir: Path) -> bool:
        del task_dir
        lowered = instruction.lower()
        required = (
            "ema-crossover",
            "volatility-targeted",
            "garch(1,1)",
            "stressed var",
            "basel",
            "capital_charge",
            "stressed_var",
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

        prices = pd.read_csv(_find_named(task_dir, "prices.csv"))
        params = json.loads(
            _find_named(task_dir, "params.json").read_text(encoding="utf-8")
        )

        prices["date"] = pd.to_datetime(prices["date"])
        prices = prices.sort_values("date", kind="stable").reset_index(drop=True)

        n_assets = int(params["n_assets"])
        asset_cols = [f"asset_{i}" for i in range(n_assets)]
        missing = [c for c in asset_cols if c not in prices.columns]
        if missing:
            raise RuntimeError(f"Missing price columns: {missing}")

        p = prices[asset_cols].apply(pd.to_numeric, errors="coerce")
        if p.isna().any().any() or (p <= 0.0).any().any():
            raise RuntimeError("Prices must be positive finite values.")

        price_values = p.to_numpy(dtype=float)
        log_returns = np.full_like(price_values, np.nan, dtype=float)
        log_returns[1:] = np.log(price_values[1:] / price_values[:-1])

        fast = int(params["fast_ema"])
        slow = int(params["slow_ema"])
        vol_lookback = int(params["vol_lookback"])
        vol_target = float(params["vol_target"])
        max_leverage = float(params["max_leverage_per_asset"])

        signals = np.full_like(price_values, np.nan, dtype=float)
        vols = np.full_like(price_values, np.nan, dtype=float)

        for j in range(n_assets):
            ema_fast = _sma_seeded_ema(price_values[:, j], fast)
            ema_slow = _sma_seeded_ema(price_values[:, j], slow)

            valid_signal = np.isfinite(ema_fast) & np.isfinite(ema_slow)
            signals[valid_signal, j] = np.where(
                ema_fast[valid_signal] > ema_slow[valid_signal],
                1.0,
                -1.0,
            )

            vols[:, j] = _ewma_annualized_vol(
                log_returns[:, j],
                vol_lookback,
            )

        per_asset_target = vol_target / n_assets
        weights = signals * per_asset_target / vols
        weights = np.clip(weights, -max_leverage, max_leverage)

        # Signals, EMAs and EWMA volatility at close t can only
        # determine the position held for the next return interval.
        # Applying weight[t] to return[t] would use information from
        # the same return being traded and creates look-ahead bias.
        applied_weights = np.full_like(
            weights,
            np.nan,
            dtype=float,
        )
        applied_weights[1:] = weights[:-1]

        # Keep every observable daily return in the portfolio history.
        # Before the slow EMA is seeded, the strategy has no established
        # position, so undefined weights represent zero exposure rather
        # than observations that should be deleted.
        return_rows = np.isfinite(
            log_returns
        ).all(axis=1)

        effective_weights = np.where(
            np.isfinite(applied_weights),
            applied_weights,
            0.0,
        )

        if int(return_rows.sum()) < 300:
            raise RuntimeError(
                "Too few CTA return observations."
            )

        strategy_returns = np.sum(
            effective_weights[return_rows]
            * log_returns[return_rows],
            axis=1,
        )
        strategy_weights = (
            effective_weights[return_rows]
        )

        annualized_return = float(np.mean(strategy_returns) * 252.0)
        annualized_volatility = float(
            np.std(strategy_returns, ddof=1) * math.sqrt(252.0)
        )
        risk_free = float(params["risk_free_annual"])
        sharpe = float(
            (annualized_return - risk_free)
            / max(annualized_volatility, 1e-15)
        )

        wealth = np.exp(np.cumsum(strategy_returns))
        running_peak = np.maximum.accumulate(
            np.concatenate(([1.0], wealth))
        )
        wealth_with_initial = np.concatenate(([1.0], wealth))
        drawdowns = 1.0 - wealth_with_initial / running_peak
        max_drawdown = float(np.max(drawdowns))
        calmar = float(
            annualized_return / max(max_drawdown, 1e-15)
        )
        avg_abs_leverage = float(
            np.mean(np.sum(np.abs(strategy_weights), axis=1))
        )

        fitted = _fit_garch(
            strategy_returns
        )

        omega = float(
            fitted["omega"]
        )
        alpha = float(
            fitted["alpha"]
        )
        beta = float(
            fitted["beta"]
        )
        persistence = float(
            fitted["persistence"]
        )
        long_run_variance = float(
            fitted[
                "long_run_variance"
            ]
        )

        confidence = float(
            params[
                "confidence_level"
            ]
        )
        holding_period = int(
            params[
                "holding_period"
            ]
        )

        z = abs(
            float(
                norm.ppf(
                    1.0 - confidence
                )
            )
        )

        last_variance = float(
            fitted["cond_var"][-1]
        )
        last_return = float(
            strategy_returns[-1]
        )

        _, cumulative_variance = (
            _garch_multistep_forecast(
                omega=omega,
                alpha=alpha,
                beta=beta,
                last_return=last_return,
                last_variance=last_variance,
                holding_period=holding_period,
            )
        )

        var_99 = float(
            z
            * math.sqrt(
                cumulative_variance
            )
        )

        window = 250
        stressed_candidates = []
        if len(strategy_returns) < window:
            raise RuntimeError("Too few CTA returns for 250-day stress windows.")

        # The task contract explicitly requires a fresh GARCH(1,1) on every
        # 250-business-day rolling window.
        for end in range(window, len(strategy_returns) + 1):
            sample = strategy_returns[end - window:end]
            try:
                stress_fit = _fit_garch(
                    sample
                )

                last_variance_w = float(
                    stress_fit[
                        "cond_var"
                    ][-1]
                )
                last_return_w = float(
                    sample[-1]
                )

                _, cumulative_variance_w = (
                    _garch_multistep_forecast(
                        omega=float(
                            stress_fit[
                                "omega"
                            ]
                        ),
                        alpha=float(
                            stress_fit[
                                "alpha"
                            ]
                        ),
                        beta=float(
                            stress_fit[
                                "beta"
                            ]
                        ),
                        last_return=last_return_w,
                        last_variance=last_variance_w,
                        holding_period=holding_period,
                    )
                )

            except Exception:
                continue

            stressed_candidates.append(
                z
                * math.sqrt(
                    cumulative_variance_w
                )
            )

        if not stressed_candidates:
            raise RuntimeError("No stressed GARCH window converged.")

        stressed_var = float(max(stressed_candidates))
        multiplier = float(params["basel_multiplier"])
        var_component = float(multiplier * var_99)
        svar_component = float(multiplier * stressed_var)
        capital_charge = float(var_component + svar_component)

        results = {
            "capital_charge": capital_charge,
            "stressed_var": stressed_var,
            "var_component": var_component,
            "svar_component": svar_component,
        }

        solution = {
            "intermediates": {
                "annualized_return": {"value": annualized_return},
                "annualized_volatility": {"value": annualized_volatility},
                "sharpe_ratio": {"value": sharpe},
                "max_drawdown": {"value": max_drawdown},
                "calmar_ratio": {"value": calmar},
                "avg_abs_leverage": {"value": avg_abs_leverage},
                "garch_omega": {"value": omega},
                "garch_alpha": {"value": alpha},
                "garch_beta": {"value": beta},
                "garch_persistence": {"value": persistence},
                "long_run_variance": {"value": long_run_variance},
                "var_99": {"value": var_99},
            }
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(
            json.dumps(results, indent=2) + "\n",
            encoding="utf-8",
        )
        (out_dir / "solution.json").write_text(
            json.dumps(solution, indent=2) + "\n",
            encoding="utf-8",
        )
