from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


PCA_COLS = ["1 Mo", "3 Mo", "6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "30 Yr"]
MATURITIES = np.array([1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 30], dtype=float)


def _find_treasury_csv(task_dir: Path) -> Path:
    for path in sorted(task_dir.rglob("*.csv")):
        if "checks" in path.parts:
            continue
        try:
            frame = pd.read_csv(path, nrows=2)
        except Exception:
            continue
        required = {"Date", "3 Mo", "2 Yr", "5 Yr", "10 Yr", "30 Yr"}
        if required.issubset(frame.columns):
            return path
    raise RuntimeError("Could not discover a Treasury yield-curve panel.")


def _nelson_siegel(tau: np.ndarray, beta0: float, beta1: float, beta2: float, lam: float = 1.5) -> np.ndarray:
    x = tau / lam
    factor1 = (1.0 - np.exp(-x)) / x
    factor2 = factor1 - np.exp(-x)
    return beta0 + beta1 * factor1 + beta2 * factor2


def _fit_ns(yields_vec: np.ndarray) -> tuple[np.ndarray, float]:
    def residuals(params: np.ndarray) -> np.ndarray:
        return _nelson_siegel(MATURITIES, *params, lam=1.5) - yields_vec

    result = least_squares(residuals, [4.0, -1.0, 1.0])
    fitted = _nelson_siegel(MATURITIES, *result.x, lam=1.5)
    rmse = float(np.sqrt(np.mean((fitted - yields_vec) ** 2)))
    return result.x, rmse


def _solve(task_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(_find_treasury_csv(task_dir))
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.sort_values("Date").reset_index(drop=True)

    for column in df.columns:
        if column != "Date":
            df[column] = pd.to_numeric(df[column], errors="coerce")

    spread = (df["10 Yr"] - df["2 Yr"]) * 100.0
    butterfly = (2.0 * df["5 Yr"] - df["2 Yr"] - df["10 Yr"]) * 100.0
    term = (df["30 Yr"] - df["3 Mo"]) * 100.0
    max_idx = spread.idxmax()

    spreads = {
        "spread_2s10s_mean_bp": float(spread.mean()),
        "spread_2s10s_std_bp": float(spread.std(ddof=1)),
        "spread_2s10s_max_date": df.loc[max_idx, "Date"].strftime("%Y-%m-%d"),
        "spread_2s10s_max_bp": float(spread.loc[max_idx]),
        "butterfly_2s5s10s_mean_bp": float(butterfly.mean()),
        "butterfly_2s5s10s_std_bp": float(butterfly.std(ddof=1)),
        "term_premium_mean_bp": float(term.mean()),
        "term_premium_std_bp": float(term.std(ddof=1)),
    }

    dy = np.diff(df[PCA_COLS].values.astype(float), axis=0) * 100.0
    covariance = np.cov(dy, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    evr = eigenvalues / eigenvalues.sum()

    if eigenvectors[:, 0].sum() < 0:
        eigenvectors[:, 0] *= -1.0

    pca = {
        "eigenvalues_top3": [float(x) for x in eigenvalues[:3]],
        "explained_variance_ratios_top3": [float(x) for x in evr[:3]],
        "cumulative_variance_top3": float(evr[:3].sum()),
        "pc1_loadings": [float(x) for x in eigenvectors[:, 0]],
    }

    last = df.iloc[-1]
    maturity_map = {"1 Mo": 1 / 12, "3 Mo": 0.25, "6 Mo": 0.5, "1 Yr": 1.0, "2 Yr": 2.0, "5 Yr": 5.0, "10 Yr": 10.0, "30 Yr": 30.0}

    def fwd(a: str, b: str) -> float:
        t1, t2 = maturity_map[a], maturity_map[b]
        return float((last[b] * t2 - last[a] * t1) / (t2 - t1))

    forward_rates = {
        "f_1y1y_pct": fwd("1 Yr", "2 Yr"),
        "f_2y3y_pct": fwd("2 Yr", "5 Yr"),
        "f_5y5y_pct": fwd("5 Yr", "10 Yr"),
        "f_10y20y_pct": fwd("10 Yr", "30 Yr"),
    }

    first_yields = df.iloc[0][PCA_COLS].values.astype(float)
    last_yields = df.iloc[-1][PCA_COLS].values.astype(float)
    first_params, first_rmse = _fit_ns(first_yields)
    last_params, last_rmse = _fit_ns(last_yields)

    ns = {
        "first_date": {
            "date": df.iloc[0]["Date"].strftime("%Y-%m-%d"),
            "beta0": float(first_params[0]),
            "beta1": float(first_params[1]),
            "beta2": float(first_params[2]),
            "rmse_pct": first_rmse,
        },
        "last_date": {
            "date": df.iloc[-1]["Date"].strftime("%Y-%m-%d"),
            "beta0": float(last_params[0]),
            "beta1": float(last_params[1]),
            "beta2": float(last_params[2]),
            "rmse_pct": last_rmse,
        },
    }

    d10 = (df["10 Yr"].diff() * 100.0).dropna()
    move_idx = d10.abs().idxmax()
    max_move = {
        "date": df.loc[move_idx, "Date"].strftime("%Y-%m-%d"),
        "change_bp": float(d10.loc[move_idx]),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "spreads": spreads,
        "pca": pca,
        "forward_rates": forward_rates,
        "nelson_siegel": ns,
        "max_daily_move": max_move,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class YieldCurvePcaDynamicsSkill:
    name: str = "yield-curve-pca-dynamics-domain"

    def matches(self, *, instruction: str, task_dir: Path) -> bool:
        lowered = instruction.lower()
        file_hint = any(path.name == "daily-treasury-rates.csv" for path in task_dir.rglob("*.csv"))
        semantic = "yield curve" in lowered and "pca" in lowered and ("principal component" in lowered or "forward rate" in lowered)
        return bool(file_hint or semantic)

    def solve(self, *, instruction: str, task_dir: Path, out_dir: Path, seed: int) -> None:
        del instruction, seed
        _solve(task_dir, out_dir)
