from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


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
            payload = json.loads(params.read_text(encoding="utf-8"))
        except Exception:
            continue
        if {"lambda", "confidence_level", "weights", "annualization_factor", "assets"}.issubset(payload):
            return params, returns
    raise RuntimeError("Could not discover EWMA portfolio-risk inputs.")


def _solve(task_dir: Path, out_dir: Path) -> None:
    params_path, returns_path = _discover(task_dir)
    p = json.loads(params_path.read_text(encoding="utf-8"))
    df = pd.read_csv(returns_path)

    assets = list(p["assets"])
    returns = df[assets].to_numpy(dtype=float)
    estimation_window = p.get("estimation_window_days")
    if estimation_window is not None and int(estimation_window) < len(returns):
        returns = returns[-int(estimation_window):]

    lam = float(p["lambda"])
    weights = np.asarray(p["weights"], dtype=float)
    annualization = float(p["annualization_factor"])
    confidence = float(p["confidence_level"])

    sigma = np.outer(returns[0], returns[0])
    for row in returns[1:]:
        sigma = lam * sigma + (1.0 - lam) * np.outer(row, row)

    port_var = float(weights @ sigma @ weights)
    port_vol = float(np.sqrt(port_var))
    sigma_w = sigma @ weights
    marginal = sigma_w / port_vol
    component = weights * marginal
    pct = component / port_vol

    results = {
        "portfolio_daily_vol": port_vol,
        "var_99_parametric": float(norm.ppf(confidence) * port_vol),
        "annualized_vol": float(port_vol * np.sqrt(annualization)),
    }
    solution = {
        "intermediates": {
            "ewma_cov_0_0": float(sigma[0, 0]),
            "ewma_cov_0_1": float(sigma[0, 1]),
            "ewma_cov_1_1": float(sigma[1, 1]),
            "ewma_cov_2_3": float(sigma[2, 3]),
            "ewma_cov_3_4": float(sigma[3, 4]),
            "portfolio_variance": port_var,
            "marginal_risk": {a: float(marginal[i]) for i, a in enumerate(assets)},
            "component_risk": {a: float(component[i]) for i, a in enumerate(assets)},
            "pct_risk_contribution": {a: float(pct[i]) for i, a in enumerate(assets)},
        }
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (out_dir / "solution.json").write_text(json.dumps(solution, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class EwmaPortfolioRiskDecompositionSkill:
    name: str = "ewma-portfolio-risk-decomposition-domain"

    def matches(self, *, instruction: str, task_dir: Path) -> bool:
        lowered = instruction.lower()
        return "ewma" in lowered and (
            "euler" in lowered
            or "risk decomposition" in lowered
            or "marginal risk" in lowered
        )

    def solve(self, *, instruction: str, task_dir: Path, out_dir: Path, seed: int) -> None:
        del instruction, seed
        _solve(task_dir, out_dir)
