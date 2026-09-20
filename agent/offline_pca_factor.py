from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np


def _find_params(task_dir: Path) -> Path:
    for path in sorted(task_dir.rglob("*.json")):
        if "checks" in path.parts:
            continue
        try:
            p = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(p, dict) and {"tickers", "monthly_returns", "n_factors", "target_weights"}.issubset(p):
            return path
    raise RuntimeError("Could not discover PCA-factor portfolio parameters.")


def _solve(task_dir: Path, out_dir: Path) -> None:
    # Import lazily so the offline runtime remains importable in local
    # development environments where scikit-learn is not installed.
    from sklearn.decomposition import PCA

    p = json.loads(_find_params(task_dir).read_text(encoding="utf-8"))
    tickers = p["tickers"]
    n_factors = int(p["n_factors"])
    r = np.asarray([p["monthly_returns"][t] for t in tickers], dtype=float).T
    w = np.asarray([p["target_weights"][t] for t in tickers], dtype=float)

    pca = PCA(n_components=n_factors)
    pca.fit(r)
    comp = pca.components_
    evr = pca.explained_variance_ratio_
    factor_exp = comp @ w

    a = np.vstack([comp, np.ones((1, len(tickers)))])
    b = np.append(-comp @ w, [0.0])
    hedge, *_ = np.linalg.lstsq(a, b, rcond=None)
    neutral = w + hedge
    neutral_exp = comp @ neutral

    ret_neutral = r @ neutral
    ret_target = r @ w
    sharpe_n = float(np.mean(ret_neutral) / np.std(ret_neutral, ddof=1) * np.sqrt(12))
    sharpe_t = float(np.mean(ret_target) / np.std(ret_target, ddof=1) * np.sqrt(12))

    pcs = [f"PC{i+1}" for i in range(n_factors)]
    result = {
        "pca_explained_variance_ratio": [round(float(v), 4) for v in evr],
        "factor_loadings": {
            pcs[i]: {ticker: round(float(comp[i, j]), 6) for j, ticker in enumerate(tickers)}
            for i in range(n_factors)
        },
        "target_factor_exposures": {pcs[i]: round(float(factor_exp[i]), 6) for i in range(n_factors)},
        "hedge_weights": {ticker: round(float(hedge[j]), 6) for j, ticker in enumerate(tickers)},
        "neutral_weights": {ticker: round(float(neutral[j]), 6) for j, ticker in enumerate(tickers)},
        "neutral_factor_exposures": {pcs[i]: round(float(neutral_exp[i]), 6) for i in range(n_factors)},
        "portfolio_sharpe_neutral": round(sharpe_n, 4),
        "portfolio_sharpe_target": round(sharpe_t, 4),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class PcaFactorPortfolioSkill:
    name: str = "pca-factor-portfolio-domain"

    def matches(self, *, instruction: str, task_dir: Path) -> bool:
        lowered = instruction.lower()
        return "pca" in lowered and "factor" in lowered and "portfolio" in lowered and "neutral" in lowered

    def solve(self, *, instruction: str, task_dir: Path, out_dir: Path, seed: int) -> None:
        del instruction, seed
        _solve(task_dir, out_dir)
