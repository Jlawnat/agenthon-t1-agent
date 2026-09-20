from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _discover(task_dir: Path) -> tuple[Path, Path]:
    for directory in sorted(
        {p.parent for p in task_dir.rglob("*") if p.is_file() and "checks" not in p.parts},
        key=str,
    ):
        params = directory / "params.json"
        returns = directory / "returns.csv"
        if not (params.is_file() and returns.is_file()):
            continue
        try:
            p = json.loads(params.read_text(encoding="utf-8"))
        except Exception:
            continue
        if {
            "rolling_window",
            "risk_budget",
            "cvar_confidence",
            "cvar_window",
            "top_k_eigenvalues",
            "regime_crisis_threshold",
            "regime_risk_off_threshold",
        }.issubset(p):
            return params, returns
    raise RuntimeError("Could not discover regime risk-parity inputs.")


def _solve(task_dir: Path, out_dir: Path) -> None:
    params_path, returns_path = _discover(task_dir)
    p = json.loads(params_path.read_text(encoding="utf-8"))
    df = pd.read_csv(returns_path)

    tickers = [c for c in df.columns if c != "date"]
    dates = df["date"].astype(str).tolist()
    returns = df[tickers].to_numpy(dtype=float)

    mask = ~np.isnan(returns).any(axis=1)
    returns = returns[mask]
    dates = [d for d, keep in zip(dates, mask) if keep]
    num_valid = int(len(returns))

    estimation_window = p.get("estimation_window_days")
    if estimation_window is not None and len(returns) > int(estimation_window):
        returns = returns[-int(estimation_window):]
        dates = dates[-int(estimation_window):]

    rolling_window = int(p["rolling_window"])
    risk_budget = p["risk_budget"]
    cvar_conf = float(p["cvar_confidence"])
    cvar_window = int(p["cvar_window"])
    top_k = int(p["top_k_eigenvalues"])
    crisis_threshold = float(p["regime_crisis_threshold"])
    off_threshold = float(p["regime_risk_off_threshold"])

    n_assets = returns.shape[1]
    start_idx = rolling_window
    n_obs = returns.shape[0] - rolling_window
    mp_threshold = float((1.0 + math.sqrt(n_assets / rolling_window)) ** 2)

    eigenvalues = []
    absorption = []
    regimes = []

    for t in range(start_idx, returns.shape[0]):
        window = returns[t - rolling_window:t]
        corr = np.corrcoef(window.T)
        vals = np.linalg.eigvalsh(corr)
        eigenvalues.append(vals)

        ar = float(np.sum(np.sort(vals)[-top_k:]) / np.sum(vals))
        absorption.append(ar)

        if ar > crisis_threshold:
            regimes.append("crisis")
        elif ar > off_threshold:
            regimes.append("risk-off")
        else:
            regimes.append("risk-on")

    absorption = np.asarray(absorption, dtype=float)
    eigenvalues = np.asarray(eigenvalues, dtype=float)

    mean_ar = float(np.mean(absorption))
    std_ar = float(np.std(absorption, ddof=1))
    mean_largest = float(np.mean(eigenvalues[:, -1]))
    counts = {name: regimes.count(name) for name in ["crisis", "risk-off", "risk-on"]}

    dates_bt = dates[start_idx:]

    rebalance = []
    for i in range(len(dates_bt) - 1):
        current = datetime.strptime(dates_bt[i], "%Y-%m-%d")
        nxt = datetime.strptime(dates_bt[i + 1], "%Y-%m-%d")
        if current.month != nxt.month:
            rebalance.append(i)

    returns_bt = returns[start_idx:]
    weights = np.ones(n_assets) / n_assets
    portfolio_returns = np.zeros(len(returns_bt), dtype=float)
    weight_stock_01 = []
    rebal_set = set(rebalance)
    rebal_count = 0

    for t in range(len(returns_bt)):
        portfolio_returns[t] = float(np.sum(weights * returns_bt[t]))

        if t in rebal_set:
            rebal_count += 1
            regime = regimes[t]
            budget = float(risk_budget[regime])
            absolute_t = start_idx + t

            if absolute_t >= rolling_window:
                window = returns[absolute_t - rolling_window:absolute_t]
            else:
                window = returns[:absolute_t] if absolute_t > 1 else returns[:2]

            stds = np.std(window, axis=0, ddof=1)
            stds = np.where(stds < 1e-10, 1e-10, stds)
            inv = 1.0 / stds
            weights = inv / np.sum(inv)
            weights = weights * budget
            weight_stock_01.append(float(weights[0]))

    ann_ret = float(np.mean(portfolio_returns) * 252.0)
    ann_vol = float(np.std(portfolio_returns, ddof=1) * math.sqrt(252.0))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    wealth = np.cumprod(1.0 + portfolio_returns)
    peak = np.maximum.accumulate(wealth)
    drawdown = (peak - wealth) / peak
    max_drawdown = float(np.max(drawdown))

    trailing = portfolio_returns[-cvar_window:]
    var_quantile = float(np.percentile(trailing, (1.0 - cvar_conf) * 100.0))
    tail = trailing[trailing <= var_quantile]
    cvar = -float(np.mean(tail)) if len(tail) else 0.0
    win_rate = float(np.mean(portfolio_returns > 0))

    results = {
        "num_valid_observations": num_valid,
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "cvar": cvar,
    }

    intermediates = {
        "num_regime_observations": {"value": int(n_obs)},
        "mp_threshold": {"value": mp_threshold},
        "mean_absorption_ratio": {"value": mean_ar},
        "std_absorption_ratio": {"value": std_ar},
        "mean_largest_eigenvalue": {"value": mean_largest},
        "regime_count_crisis": {"value": int(counts["crisis"])},
        "regime_count_risk_off": {"value": int(counts["risk-off"])},
        "regime_count_risk_on": {"value": int(counts["risk-on"])},
        "num_rebalances": {"value": int(rebal_count)},
        "mean_weight_stock_01": {
            "value": float(np.mean(weight_stock_01)) if weight_stock_01 else 0.0
        },
        "portfolio_var_99": {"value": float(-var_quantile)},
        "cvar": {"value": cvar},
        "win_rate": {"value": win_rate},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (out_dir / "solution.json").write_text(
        json.dumps({"intermediates": intermediates}, indent=2) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class RegimeRiskParityCvarSkill:
    name: str = "regime-riskparity-cvar-domain"

    def matches(self, *, instruction: str, task_dir: Path) -> bool:
        lowered = instruction.lower()
        return (
            "absorption ratio" in lowered
            and ("risk-parity" in lowered or "risk parity" in lowered)
            and ("cvar" in lowered or "expected shortfall" in lowered)
        )

    def solve(self, *, instruction: str, task_dir: Path, out_dir: Path, seed: int) -> None:
        del instruction, seed
        _solve(task_dir, out_dir)
