from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm


def _find_inputs(task_dir: Path) -> tuple[Path, Path]:
    for directory in sorted({p.parent for p in task_dir.rglob("*") if p.is_file() and "checks" not in p.parts}, key=str):
        params = directory / "params.json"
        returns = directory / "returns.csv"
        if params.is_file() and returns.is_file():
            try:
                payload = json.loads(params.read_text(encoding="utf-8"))
            except Exception:
                continue
            if {"S0", "K", "B", "r", "T", "confidence_level", "n_contracts"}.issubset(payload):
                return params, returns
    raise RuntimeError("Could not discover barrier-option GARCH inputs.")


def _fit_garch(returns: np.ndarray) -> dict[str, object]:
    # Import lazily so the offline runtime remains importable in local
    # development environments where arch is not installed.
    from arch import arch_model

    values = np.asarray(returns, dtype=float).ravel()
    percent = values * 100.0
    model = arch_model(percent, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
    fitted = model.fit(disp="off")

    omega = float(fitted.params["omega"] / 10000.0)
    alpha = float(fitted.params["alpha[1]"])
    beta = float(fitted.params["beta[1]"])
    persistence = alpha + beta

    if persistence < 1.0:
        long_run_var = omega / (1.0 - persistence)
    else:
        long_run_var = float(np.var(values, ddof=1))

    cond_var = np.asarray((fitted.conditional_volatility ** 2) / 10000.0, dtype=float)
    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": persistence,
        "long_run_variance": float(long_run_var),
        "cond_var": cond_var,
    }


def _barrier_price(S: float, K: float, T: float, r: float, sigma: float, H: float, q: float = 0.0) -> float:
    if S <= H or sigma <= 0.0 or T <= 0.0:
        return 0.0

    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    vanilla = (
        S * math.exp(-q * T) * norm.cdf(d1)
        - K * math.exp(-r * T) * norm.cdf(d2)
    )

    lam = (r - q + 0.5 * sigma * sigma) / (sigma * sigma)
    ratio = H / S
    d1_h = (
        math.log(H * H / (S * K)) + (r - q + 0.5 * sigma * sigma) * T
    ) / (sigma * sqrt_t)
    d2_h = d1_h - sigma * sqrt_t

    down_in = (
        S * math.exp(-q * T) * ratio ** (2.0 * lam) * norm.cdf(d1_h)
        - K * math.exp(-r * T) * ratio ** (2.0 * lam - 2.0) * norm.cdf(d2_h)
    )
    return max(float(vanilla - down_in), 0.0)


def _solve(task_dir: Path, out_dir: Path) -> None:
    params_path, returns_path = _find_inputs(task_dir)
    p = json.loads(params_path.read_text(encoding="utf-8"))
    returns = pd.read_csv(returns_path)["log_return"].to_numpy(dtype=float)

    fit = _fit_garch(returns)
    cond_var = np.asarray(fit["cond_var"], dtype=float)
    garch_vol = float(math.sqrt(float(cond_var[-1]) * 252.0))
    cond_vol_series = np.sqrt(cond_var * 252.0)
    vol_of_vol = float(np.std(np.diff(cond_vol_series), ddof=1) * math.sqrt(252.0))

    S0 = float(p["S0"])
    K = float(p["K"])
    H = float(p["B"])
    r = float(p["r"])
    T = float(p["T"])
    confidence = float(p["confidence_level"])
    n_contracts = float(p["n_contracts"])
    fd_dS_pct = float(p.get("fd_dS_pct", 0.01))
    fd_dsigma = float(p.get("fd_dsigma", 0.01))

    price = _barrier_price(S0, K, T, r, garch_vol, H)
    dS = fd_dS_pct * S0
    up = _barrier_price(S0 + dS, K, T, r, garch_vol, H)
    down = _barrier_price(S0 - dS, K, T, r, garch_vol, H)
    delta = (up - down) / (2.0 * dS)
    gamma = (up - 2.0 * price + down) / (dS * dS)

    sig_up = _barrier_price(S0, K, T, r, garch_vol + fd_dsigma, H)
    sig_dn = _barrier_price(S0, K, T, r, max(garch_vol - fd_dsigma, 1e-6), H)
    vega = (sig_up - sig_dn) / (2.0 * fd_dsigma)

    z = abs(float(norm.ppf(1.0 - confidence)))
    daily_vol = garch_vol / math.sqrt(252.0)
    delta_s_vol = S0 * daily_vol
    daily_vol_of_vol = vol_of_vol / math.sqrt(252.0)

    delta_var = abs(n_contracts * delta * z * delta_s_vol)
    gamma_var = 0.5 * abs(n_contracts * gamma * (z * delta_s_vol) ** 2)
    vega_var = abs(n_contracts * vega * z * daily_vol_of_vol)

    total_sq = delta_var ** 2 + gamma_var ** 2 + vega_var ** 2
    portfolio_var = math.sqrt(total_sq)
    if total_sq > 0:
        delta_pct = delta_var ** 2 / total_sq * 100.0
        gamma_pct = gamma_var ** 2 / total_sq * 100.0
        vega_pct = vega_var ** 2 / total_sq * 100.0
    else:
        delta_pct = gamma_pct = vega_pct = 0.0

    results = {
        "portfolio_var": float(portfolio_var),
        "delta_var_pct": float(delta_pct),
        "gamma_var_pct": float(gamma_pct),
        "vega_var_pct": float(vega_pct),
    }
    intermediates = {
        "garch_omega": {"value": float(fit["omega"])},
        "garch_alpha": {"value": float(fit["alpha"])},
        "garch_beta": {"value": float(fit["beta"])},
        "garch_persistence": {"value": float(fit["persistence"])},
        "long_run_variance": {"value": float(fit["long_run_variance"])},
        "garch_vol": {"value": garch_vol},
        "analytical_price": {"value": price},
        "delta": {"value": float(delta)},
        "gamma": {"value": float(gamma)},
        "vega": {"value": float(vega)},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (out_dir / "solution.json").write_text(
        json.dumps({"intermediates": intermediates}, indent=2) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class BarrierGarchVarSkill:
    name: str = "barrier-garch-var-domain"

    def matches(self, *, instruction: str, task_dir: Path) -> bool:
        lowered = instruction.lower()
        return (
            ("barrier" in lowered and "garch" in lowered and "var" in lowered)
            or any(path.name == "returns.csv" for path in task_dir.rglob("*.csv"))
            and "barrier" in lowered
        )

    def solve(self, *, instruction: str, task_dir: Path, out_dir: Path, seed: int) -> None:
        del instruction, seed
        _solve(task_dir, out_dir)
