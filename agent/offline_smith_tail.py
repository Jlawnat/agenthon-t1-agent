from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats


K_VALUES = [25, 50, 75, 100, 150, 200, 300, 500, 750, 1000]


def _find_price_csv(task_dir: Path) -> Path:
    candidates = []
    for path in sorted(task_dir.rglob("*.csv")):
        if "checks" in path.parts:
            continue
        try:
            frame = pd.read_csv(path, nrows=3)
        except Exception:
            continue
        if "close" in {str(c).lower() for c in frame.columns}:
            candidates.append(path)
    if not candidates:
        raise RuntimeError("Could not discover a close-price CSV for tail-index estimation.")
    preferred = [p for p in candidates if "sp500" in p.name.lower()]
    return preferred[0] if preferred else candidates[0]


def _gpd_negloglik(params: np.ndarray, excesses: np.ndarray) -> float:
    xi, sigma = float(params[0]), float(params[1])
    if sigma <= 0.0:
        return 1e10
    n = len(excesses)
    if abs(xi) < 1e-10:
        return float(n * np.log(sigma) + np.sum(excesses) / sigma)
    y = 1.0 + xi * excesses / sigma
    if np.any(y <= 0.0):
        return 1e10
    return float(n * np.log(sigma) + (1.0 + 1.0 / xi) * np.sum(np.log(y)))


def _fit_gpd(excesses: np.ndarray) -> tuple[float, float, float]:
    xi_init, _, sigma_init = stats.genpareto.fit(excesses, floc=0)
    result = optimize.minimize(
        _gpd_negloglik,
        [xi_init, sigma_init],
        args=(excesses,),
        method="Nelder-Mead",
        options={"maxiter": 50000, "xatol": 1e-12, "fatol": 1e-12},
    )
    xi_hat, sigma_hat = map(float, result.x)

    eps_xi = max(abs(xi_hat) * 1e-4, 1e-6)
    eps_sig = max(abs(sigma_hat) * 1e-4, 1e-8)
    steps = [eps_xi, eps_sig]
    hessian = np.zeros((2, 2), dtype=float)

    for i in range(2):
        for j in range(i, 2):
            pp = [xi_hat, sigma_hat]
            pm = [xi_hat, sigma_hat]
            mp = [xi_hat, sigma_hat]
            mm = [xi_hat, sigma_hat]
            pp[i] += steps[i]; pp[j] += steps[j]
            pm[i] += steps[i]; pm[j] -= steps[j]
            mp[i] -= steps[i]; mp[j] += steps[j]
            mm[i] -= steps[i]; mm[j] -= steps[j]
            value = (
                _gpd_negloglik(np.asarray(pp), excesses)
                - _gpd_negloglik(np.asarray(pm), excesses)
                - _gpd_negloglik(np.asarray(mp), excesses)
                + _gpd_negloglik(np.asarray(mm), excesses)
            ) / (4.0 * steps[i] * steps[j])
            hessian[i, j] = value
            hessian[j, i] = value

    try:
        covariance = np.linalg.inv(hessian)
        se_xi = float(np.sqrt(max(float(covariance[0, 0]), 0.0)))
        if not np.isfinite(se_xi) or se_xi <= 0:
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        se_xi = float(abs(xi_hat) * np.sqrt((1.0 + xi_hat) ** 2 / len(excesses))) if xi_hat > -0.5 else 0.1

    return xi_hat, sigma_hat, se_xi


def _solve(task_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(_find_price_csv(task_dir))
    close_col = next(c for c in df.columns if str(c).lower() == "close")
    prices = pd.to_numeric(df[close_col], errors="coerce").dropna().values.astype(np.float64)
    losses = -np.log(prices[1:] / prices[:-1])
    sorted_losses = np.sort(losses)[::-1]

    if len(sorted_losses) <= max(K_VALUES):
        raise RuntimeError("Price history is too short for requested tail order statistics.")

    hill_rows = []
    smith_rows = []

    for k in K_VALUES:
        threshold = sorted_losses[k]
        top_k = sorted_losses[:k]
        if threshold <= 0 or np.any(top_k <= 0):
            raise RuntimeError("Hill estimator requires positive upper-tail losses.")

        xi_hill = float(np.mean(np.log(top_k / threshold)))
        se_hill = float(xi_hill / np.sqrt(k))
        hill_rows.append(
            {
                "k": k,
                "xi_hill": xi_hill,
                "se_hill": se_hill,
                "ci_lower": xi_hill - 1.96 * se_hill,
                "ci_upper": xi_hill + 1.96 * se_hill,
            }
        )

        excesses = top_k - threshold
        xi_s, sigma_s, se_xi = _fit_gpd(excesses)
        smith_rows.append(
            {
                "k": k,
                "xi_smith": xi_s,
                "sigma_smith": sigma_s,
                "se_xi": se_xi,
                "ci_lower": xi_s - 1.96 * se_xi,
                "ci_upper": xi_s + 1.96 * se_xi,
            }
        )

    hill = pd.DataFrame(hill_rows)
    smith = pd.DataFrame(smith_rows)
    xi_hill = hill["xi_hill"].to_numpy(dtype=float)
    xi_smith = smith["xi_smith"].to_numpy(dtype=float)

    comparison = {
        "mean_xi_hill": float(np.mean(xi_hill)),
        "mean_xi_smith": float(np.mean(xi_smith)),
        "correlation": float(np.corrcoef(xi_hill, xi_smith)[0, 1]),
    }

    best_idx = int(np.argmin(smith["se_xi"].to_numpy(dtype=float)))
    summary = {
        "best_k": int(K_VALUES[best_idx]),
        "xi_hill_at_best_k": float(xi_hill[best_idx]),
        "xi_smith_at_best_k": float(xi_smith[best_idx]),
        "tail_is_heavy": bool(float(np.mean(xi_smith)) > 0.0),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    hill.to_csv(out_dir / "hill_estimates.csv", index=False)
    smith.to_csv(out_dir / "smith_estimates.csv", index=False)
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class TailIndexEstimationSkill:
    name: str = "tail-index-estimation-domain"

    def matches(self, *, instruction: str, task_dir: Path) -> bool:
        lowered = instruction.lower()
        semantic = "tail index" in lowered and ("hill" in lowered or "smith" in lowered or "gpd" in lowered)
        file_hint = any("sp500_daily" in p.name.lower() for p in task_dir.rglob("*.csv"))
        return bool(semantic or (file_hint and "tail" in lowered))

    def solve(self, *, instruction: str, task_dir: Path, out_dir: Path, seed: int) -> None:
        del instruction, seed
        _solve(task_dir, out_dir)
