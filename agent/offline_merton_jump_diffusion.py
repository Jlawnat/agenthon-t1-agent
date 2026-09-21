from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import optimize, stats

RISK_FREE_RATE = 0.05
DT = 1.0 / 252.0
N_MAX = 15
N_SERIES = 50
MATURITIES = (0.25, 0.5, 1.0)
MONEYNESS = (0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15)


def _find_spy_daily(task_dir: Path) -> Path:
    candidates = [
        path for path in sorted(task_dir.rglob("spy_daily.csv"))
        if "checks" not in path.parts
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one spy_daily.csv outside checks, found {len(candidates)}."
        )
    return candidates[0]


def _load_closes(path: Path) -> np.ndarray:
    prices = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prices.append(float(row["close"]))
    if len(prices) < 2:
        raise RuntimeError("spy_daily.csv must contain at least two close prices.")
    return np.asarray(prices, dtype=float)


def _negative_log_likelihood(params, x, dt, mu, n_max=N_MAX) -> float:
    sigma, lam, mu_j, sigma_j = [float(v) for v in params]
    if sigma <= 0.0 or lam <= 0.0 or sigma_j <= 0.0:
        return 1e12

    kbar = math.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0
    drift = (mu - 0.5 * sigma * sigma - lam * kbar) * dt
    log_likelihood = 0.0

    for xi in x:
        probability = 0.0
        log_poisson = -lam * dt

        for n in range(n_max + 1):
            mean_n = drift + n * mu_j
            variance_n = sigma * sigma * dt + n * sigma_j * sigma_j
            if variance_n <= 0.0:
                continue

            pdf_value = (
                1.0
                / math.sqrt(2.0 * math.pi * variance_n)
                * math.exp(
                    -0.5 * (float(xi) - mean_n) ** 2 / variance_n
                )
            )
            probability += math.exp(log_poisson) * pdf_value

            if n < n_max:
                log_poisson += math.log(lam * dt) - math.log(n + 1)

        log_likelihood += math.log(probability) if probability > 0.0 else -1e6

    return -float(log_likelihood)


def _calibrate(log_returns: np.ndarray):
    mu_physical = float(np.mean(log_returns)) / DT
    bounds = (
        (0.01, 1.0),
        (0.1, 12.0),
        (-0.10, 0.10),
        (0.005, 0.30),
    )

    rng = np.random.RandomState(42)
    best_result = None
    best_nll = np.inf

    for _ in range(10):
        x0 = [
            rng.uniform(0.05, 0.50),
            rng.uniform(1.0, 30.0),
            rng.uniform(-0.05, 0.01),
            rng.uniform(0.005, 0.10),
        ]

        try:
            result = optimize.minimize(
                _negative_log_likelihood,
                x0,
                args=(log_returns, DT, mu_physical, N_MAX),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 2000, "ftol": 1e-12},
            )
        except Exception:
            continue

        if float(result.fun) < best_nll:
            best_nll = float(result.fun)
            best_result = result

    if best_result is None:
        raise RuntimeError("Merton calibration failed for every restart.")

    sigma, lam, mu_j, sigma_j = [float(v) for v in best_result.x]
    return sigma, lam, mu_j, sigma_j, -best_nll


def _bs_call(spot, strike, rate, sigma, tau) -> float:
    if tau <= 0.0 or sigma <= 0.0:
        return max(spot - strike * math.exp(-rate * tau), 0.0)

    sqrt_tau = math.sqrt(tau)
    d1 = (
        math.log(spot / strike)
        + (rate + 0.5 * sigma * sigma) * tau
    ) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau

    return float(
        spot * stats.norm.cdf(d1)
        - strike * math.exp(-rate * tau) * stats.norm.cdf(d2)
    )


def _bs_vega(spot, strike, rate, sigma, tau) -> float:
    if tau <= 0.0 or sigma <= 0.0:
        return 0.0
    sqrt_tau = math.sqrt(tau)
    d1 = (
        math.log(spot / strike)
        + (rate + 0.5 * sigma * sigma) * tau
    ) / (sigma * sqrt_tau)
    return float(spot * stats.norm.pdf(d1) * sqrt_tau)


def _merton_call(
    spot,
    strike,
    rate,
    tau,
    sigma,
    lam,
    mu_j,
    sigma_j,
    n_terms=N_SERIES,
) -> float:
    kbar = math.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0
    price = 0.0
    log_poisson = -lam * tau

    for n in range(n_terms + 1):
        sigma_n = math.sqrt(
            sigma * sigma + n * sigma_j * sigma_j / tau
        )
        rate_n = (
            rate
            - lam * kbar
            + n * math.log(1.0 + kbar) / tau
        )
        price += math.exp(log_poisson) * _bs_call(
            spot,
            strike,
            rate_n,
            sigma_n,
            tau,
        )
        if n < n_terms:
            log_poisson += math.log(lam * tau) - math.log(n + 1)

    return float(price)


def _implied_vol(
    market_price,
    spot,
    strike,
    rate,
    tau,
    *,
    sigma_init,
) -> float:
    iv = float(sigma_init)

    for _ in range(100):
        price = _bs_call(spot, strike, rate, iv, tau)
        vega = _bs_vega(spot, strike, rate, iv, tau)
        if abs(vega) < 1e-12:
            break

        iv_new = iv - (price - market_price) / vega
        iv_new = max(0.001, min(float(iv_new), 5.0))

        if abs(iv_new - iv) < 1e-8:
            iv = iv_new
            break

        iv = iv_new

    return float(iv)


@dataclass(frozen=True)
class MertonJumpDiffusionSkill:
    name: str = "merton-jump-diffusion-domain"

    def matches(self, *, instruction: str, task_dir: Path) -> bool:
        lowered = instruction.lower()
        semantic = (
            "merton jump-diffusion" in lowered
            and "merton_prices.csv" in lowered
            and "mu_j" in lowered
            and "sigma_j" in lowered
            and "implied volatil" in lowered
        )
        candidates = [
            path for path in task_dir.rglob("spy_daily.csv")
            if "checks" not in path.parts
        ]
        return semantic and len(candidates) == 1

    def solve(
        self,
        *,
        instruction: str,
        task_dir: Path,
        out_dir: Path,
        seed: int,
    ) -> None:
        del instruction, seed

        closes = _load_closes(_find_spy_daily(task_dir))
        log_returns = np.diff(np.log(closes))

        n_returns = int(len(log_returns))
        return_mean = float(np.mean(log_returns))
        return_std = float(np.std(log_returns, ddof=1))
        return_skewness = float(stats.skew(log_returns))
        return_excess_kurtosis = float(stats.kurtosis(log_returns))
        spot = float(closes[-1])

        sigma, lam, mu_j, sigma_j, log_likelihood = _calibrate(log_returns)

        sigma_total = float(
            math.sqrt(
                sigma * sigma
                + lam * (mu_j * mu_j + sigma_j * sigma_j)
            )
        )

        rows = []
        for maturity in MATURITIES:
            for moneyness in MONEYNESS:
                strike = spot * moneyness
                merton_price = _merton_call(
                    spot,
                    strike,
                    RISK_FREE_RATE,
                    maturity,
                    sigma,
                    lam,
                    mu_j,
                    sigma_j,
                    N_SERIES,
                )
                bs_price = _bs_call(
                    spot,
                    strike,
                    RISK_FREE_RATE,
                    sigma_total,
                    maturity,
                )
                implied_vol = _implied_vol(
                    merton_price,
                    spot,
                    strike,
                    RISK_FREE_RATE,
                    maturity,
                    sigma_init=sigma_total,
                )

                rows.append(
                    {
                        "T": float(maturity),
                        "K": round(float(strike), 4),
                        "moneyness": float(moneyness),
                        "merton_price": float(merton_price),
                        "bs_price": float(bs_price),
                        "merton_implied_vol": float(implied_vol),
                    }
                )

        ivs_by_maturity = {}
        for row in rows:
            ivs_by_maturity.setdefault(row["T"], []).append(
                row["merton_implied_vol"]
            )

        max_iv_spread = max(
            max(values) - min(values)
            for values in ivs_by_maturity.values()
        )

        smile_present = any(
            (max(values) - min(values)) > 0.005
            for values in ivs_by_maturity.values()
        )

        atm_prices = {
            row["T"]: row["merton_price"]
            for row in rows
            if math.isclose(row["moneyness"], 1.0, abs_tol=1e-12)
        }

        ratios = [
            row["merton_price"] / row["bs_price"]
            for row in rows
            if row["bs_price"] > 0.0
        ]

        calibration = {
            "n_returns": n_returns,
            "return_mean": return_mean,
            "return_std": return_std,
            "return_skewness": return_skewness,
            "return_excess_kurtosis": return_excess_kurtosis,
            "sigma": float(sigma),
            "lambda": float(lam),
            "mu_J": float(mu_j),
            "sigma_J": float(sigma_j),
            "log_likelihood": float(log_likelihood),
            "S0": spot,
        }

        summary = {
            "sigma_total": sigma_total,
            "n_maturities": 3,
            "n_strikes": 7,
            "n_prices": 21,
            "smile_present": bool(smile_present),
            "atm_merton_price_T025": float(atm_prices[0.25]),
            "atm_merton_price_T050": float(atm_prices[0.5]),
            "atm_merton_price_T100": float(atm_prices[1.0]),
            "max_iv_spread": float(max_iv_spread),
            "mean_price_ratio_merton_over_bs": float(np.mean(ratios)),
        }

        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "calibration.json").write_text(
            json.dumps(calibration, indent=2) + "\n",
            encoding="utf-8",
        )

        with (out_dir / "merton_prices.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "T",
                    "K",
                    "moneyness",
                    "merton_price",
                    "bs_price",
                    "merton_implied_vol",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
