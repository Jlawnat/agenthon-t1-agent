from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.special import gammaln


def _discover_csv(task_dir: Path) -> Path:
    for path in sorted(task_dir.rglob("*.csv")):
        if "checks" in path.parts:
            continue
        try:
            cols = {c.lower() for c in pd.read_csv(path, nrows=3).columns}
        except Exception:
            continue
        if {"date", "symbol", "close"}.issubset(cols):
            return path
    raise RuntimeError("Could not discover long-form equity OHLCV data.")


def _target_stocks(instruction: str, available: list[str]) -> list[str]:
    preferred = ["AAPL", "JPM", "GS", "CVX", "MSFT"]
    if all(x in available for x in preferred):
        return preferred

    mentioned = []
    for token in re.findall(r"\b[A-Z][A-Z0-9.]{1,7}\b", instruction):
        if token in available and token not in mentioned:
            mentioned.append(token)
    if len(mentioned) >= 2:
        return mentioned[:5]
    return available[:5]


def _pseudo(x: np.ndarray) -> np.ndarray:
    return stats.rankdata(x) / (len(x) + 1.0)


def _gaussian_ll(u1, u2, rho):
    eps = 1e-10
    u1 = np.clip(u1, eps, 1 - eps)
    u2 = np.clip(u2, eps, 1 - eps)
    x1, x2 = stats.norm.ppf(u1), stats.norm.ppf(u2)
    rho2 = rho * rho
    if rho2 > 0.9999:
        return -1e15
    ll = (
        -0.5 * np.log(1 - rho2)
        - (rho2 * (x1 * x1 + x2 * x2) - 2 * rho * x1 * x2)
        / (2 * (1 - rho2))
    )
    return float(np.sum(ll))


def _t_ll(u1, u2, rho, nu):
    eps = 1e-10
    u1 = np.clip(u1, eps, 1 - eps)
    u2 = np.clip(u2, eps, 1 - eps)
    x1, x2 = stats.t.ppf(u1, nu), stats.t.ppf(u2, nu)
    rho2 = rho * rho
    if rho2 > 0.9999:
        return -1e15
    n = len(u1)
    ll = n * (
        gammaln((nu + 2) / 2)
        - gammaln(nu / 2)
        - np.log(nu * np.pi)
        - 0.5 * np.log(1 - rho2)
    )
    q = (x1 * x1 + x2 * x2 - 2 * rho * x1 * x2) / (nu * (1 - rho2))
    ll += np.sum(-(nu + 2) / 2 * np.log(1 + q))
    ll -= np.sum(stats.t.logpdf(x1, nu))
    ll -= np.sum(stats.t.logpdf(x2, nu))
    return float(ll)


def _clayton_ll(u1, u2, theta):
    if theta <= 0:
        return -1e15
    eps = 1e-10
    u1 = np.clip(u1, eps, 1 - eps)
    u2 = np.clip(u2, eps, 1 - eps)
    s = u1 ** (-theta) + u2 ** (-theta) - 1.0
    if np.any(s <= 0):
        return -1e15
    ll = (
        np.log(1 + theta)
        + (-1 - theta) * (np.log(u1) + np.log(u2))
        + (-1.0 / theta - 2) * np.log(s)
    )
    return float(np.sum(ll))


def _gumbel_ll(u1, u2, theta):
    if theta < 1:
        return -1e15
    eps = 1e-10
    u1 = np.clip(u1, eps, 1 - eps)
    u2 = np.clip(u2, eps, 1 - eps)
    lu1, lu2 = -np.log(u1), -np.log(u2)
    t1, t2 = lu1 ** theta, lu2 ** theta
    total = t1 + t2
    A = total ** (1.0 / theta)
    log_c = (
        -A
        - np.log(u1)
        - np.log(u2)
        + (theta - 1) * (np.log(lu1) + np.log(lu2))
        - (2 - 1.0 / theta) * np.log(total)
        + np.log(A + theta - 1)
    )
    return float(np.sum(log_c))


def _solve(instruction: str, task_dir: Path, out_dir: Path) -> None:
    path = _discover_csv(task_dir)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"])

    available = sorted(df["symbol"].dropna().astype(str).unique())
    stocks = _target_stocks(instruction, available)
    if len(stocks) != 5:
        raise RuntimeError("Copula equity fitting requires five target stocks.")

    returns = {}
    for symbol in stocks:
        sub = df[df["symbol"] == symbol].sort_values("date")
        prices = sub["close"].to_numpy(dtype=float)
        returns[symbol] = np.log(prices[1:] / prices[:-1])

    n = min(len(x) for x in returns.values())
    for symbol in stocks:
        returns[symbol] = returns[symbol][:n]

    pairs = list(itertools.combinations(stocks, 2))
    fits, best, tails = [], {}, []

    for s1, s2 in pairs:
        r1, r2 = returns[s1], returns[s2]
        u1, u2 = _pseudo(r1), _pseudo(r2)
        tau = float(stats.kendalltau(r1, r2)[0])
        rho = float(np.sin(np.pi * tau / 2.0))

        ll_g = _gaussian_ll(u1, u2, rho)
        aic_g = -2 * ll_g + 2
        bic_g = -2 * ll_g + np.log(n)

        opt = minimize_scalar(
            lambda nu: -_t_ll(u1, u2, rho, nu),
            bounds=(2.1, 100.0),
            method="bounded",
        )
        nu = float(opt.x)
        ll_t = float(-opt.fun)
        aic_t = -2 * ll_t + 4
        bic_t = -2 * ll_t + 2 * np.log(n)

        theta_c = float(2 * tau / (1 - tau)) if tau > 0 else 0.01
        ll_c = _clayton_ll(u1, u2, theta_c)
        aic_c = -2 * ll_c + 2
        bic_c = -2 * ll_c + np.log(n)

        theta_g = float(1 / (1 - tau)) if tau > 0 else 1.001
        ll_gu = _gumbel_ll(u1, u2, theta_g)
        aic_gu = -2 * ll_gu + 2
        bic_gu = -2 * ll_gu + np.log(n)

        rows = [
            ("gaussian", rho, np.nan, ll_g, aic_g, bic_g),
            ("student_t", rho, nu, ll_t, aic_t, bic_t),
            ("clayton", theta_c, np.nan, ll_c, aic_c, bic_c),
            ("gumbel", theta_g, np.nan, ll_gu, aic_gu, bic_gu),
        ]
        for cop, p1, p2, ll, aic, bic in rows:
            fits.append(
                {
                    "stock1": s1,
                    "stock2": s2,
                    "copula": cop,
                    "param1": p1,
                    "param2": p2,
                    "loglik": ll,
                    "aic": aic,
                    "bic": bic,
                }
            )

        best[f"{s1}-{s2}"] = min(rows, key=lambda x: x[4])[0]
        q = 0.95
        up1, up2 = u1 > q, u2 > q
        low = 1.0 - q
        lo1, lo2 = u1 < low, u2 < low
        tails.append(
            {
                "stock1": s1,
                "stock2": s2,
                "lambda_upper": float(np.sum(up1 & up2) / np.sum(up1)) if np.sum(up1) else 0.0,
                "lambda_lower": float(np.sum(lo1 & lo2) / np.sum(lo1)) if np.sum(lo1) else 0.0,
                "kendall_tau": tau,
            }
        )

    counts = Counter(best.values())
    summary = {
        "n_pairs": len(pairs),
        "most_common_best_copula": counts.most_common(1)[0][0],
        "mean_kendall_tau": float(np.mean([r["kendall_tau"] for r in tails])),
        "has_tail_dependence": bool(
            any(r["lambda_upper"] > 0.1 or r["lambda_lower"] > 0.1 for r in tails)
        ),
    }
    data_summary = {
        "n_stocks": len(stocks),
        "n_returns": n,
        "stock_list": stocks,
        "n_pairs": len(pairs),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data_summary.json").write_text(json.dumps(data_summary, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(fits).to_csv(out_dir / "copula_fits.csv", index=False)
    (out_dir / "best_copulas.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(tails).to_csv(out_dir / "tail_dependence.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class CopulaEquityFittingSkill:
    name: str = "copula-equity-fitting-domain"

    def matches(self, *, instruction: str, task_dir: Path) -> bool:
        lowered = instruction.lower()
        return "copula" in lowered and (
            "equity pair" in lowered
            or "copula fitting" in lowered
            or ("pseudo-observations" in lowered and "aic" in lowered)
        )

    def solve(self, *, instruction: str, task_dir: Path, out_dir: Path, seed: int) -> None:
        del seed
        _solve(instruction, task_dir, out_dir)
