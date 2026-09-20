from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


S0 = 500.0
R = 0.05
D = 0.013
SQRT_2PI = np.sqrt(2.0 * np.pi)
SIGMAS = [0.10, 0.15, 0.20, 0.30, 0.40]
MATURITIES = [0.25, 0.50, 1.00]
MONEYNESS = [0.90, 0.95, 1.00, 1.05, 1.10]


def _bs_price(k: float, t: float, sigma: float) -> float:
    s_tilde = S0 * np.exp(-D * t)
    k_tilde = k * np.exp(-R * t)
    d1 = (np.log(s_tilde / k_tilde) + 0.5 * sigma * sigma * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return float(s_tilde * norm.cdf(d1) - k_tilde * norm.cdf(d2))


def _bs_vega(k: float, t: float, sigma: float) -> float:
    s_tilde = S0 * np.exp(-D * t)
    k_tilde = k * np.exp(-R * t)
    d1 = (np.log(s_tilde / k_tilde) + 0.5 * sigma * sigma * t) / (sigma * np.sqrt(t))
    return float(s_tilde * norm.pdf(d1) * np.sqrt(t))


def _brenner(v: float, t: float, m: float) -> float:
    if abs(m - 1.0) > 1e-9:
        return np.nan
    s_tilde = S0 * np.exp(-D * t)
    return float((v / s_tilde) * np.sqrt(2.0 * np.pi / t))


def _li_atm(v: float, t: float, m: float) -> float:
    if abs(m - 1.0) > 1e-9:
        return np.nan
    s_tilde = S0 * np.exp(-D * t)
    c = -3.0 * np.sqrt(np.pi) * v / (4.0 * s_tilde)
    if c < -1.0 or c > 1.0:
        return np.nan
    arg = 2.0 * np.pi / 3.0 - (1.0 / 3.0) * np.arccos(c)
    return float(4.0 * np.sqrt(2.0 / t) * np.cos(arg))


def _li_nonatm(v: float, k: float, t: float) -> float:
    s_tilde = S0 * np.exp(-D * t)
    k_tilde = k * np.exp(-R * t)
    term1 = v - (s_tilde - k_tilde) / 2.0
    discriminant = term1 * term1 - (s_tilde - k_tilde) ** 2 * (1.0 + s_tilde / k_tilde) / (2.0 * np.pi)
    if discriminant < 0:
        if discriminant > -1e-10:
            discriminant = 0.0
        else:
            return np.nan
    return float(np.sqrt(2.0 * np.pi / t) * ((term1 + np.sqrt(discriminant)) / (s_tilde + k_tilde)))


def _cmh(v: float, k: float, t: float) -> float:
    s_tilde = S0 * np.exp(-D * t)
    k_tilde = k * np.exp(-R * t)
    m = s_tilde / k_tilde
    alpha = SQRT_2PI / (m + 1.0) * (2.0 * v / k_tilde - (m - 1.0))
    beta = 0.5 * ((m - 1.0) / (m + 1.0)) ** 2
    discriminant = alpha * alpha - 8.0 * beta
    if discriminant < 0:
        if discriminant > -1e-10:
            discriminant = 0.0
        else:
            return np.nan
    return float((alpha + np.sqrt(discriminant)) / np.sqrt(t))


def _newton(v: float, k: float, t: float) -> tuple[float, bool]:
    sigma = 0.3
    for _ in range(100):
        price = _bs_price(k, t, sigma)
        vega = _bs_vega(k, t, sigma)
        if not np.isfinite(price) or not np.isfinite(vega) or vega <= 0:
            return np.nan, False
        error = price - v
        if abs(error) < 1e-10:
            return float(sigma), True
        new_sigma = sigma - error / vega
        if abs(new_sigma - sigma) < 1e-10:
            return float(new_sigma), True
        sigma = float(new_sigma)
    return float(sigma), False


def _rmse(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    return float(np.sqrt(np.mean(valid ** 2)))


def _solve(out_dir: Path) -> None:
    rows = []
    converged = []

    for sigma_true in SIGMAS:
        for t in MATURITIES:
            for m in MONEYNESS:
                k = S0 * m
                market = _bs_price(k, t, sigma_true)
                sigma_bs = _brenner(market, t, m)
                sigma_li_atm = _li_atm(market, t, m)
                sigma_li_nonatm = _li_nonatm(market, k, t)
                sigma_cmh = _cmh(market, k, t)
                sigma_nr, ok = _newton(market, k, t)
                converged.append(ok)

                rows.append(
                    {
                        "sigma_true": sigma_true,
                        "T": t,
                        "K": k,
                        "moneyness": m,
                        "bs_price": market,
                        "sigma_bs_approx": sigma_bs,
                        "sigma_li_atm": sigma_li_atm,
                        "sigma_li_nonatm": sigma_li_nonatm,
                        "sigma_cmh": sigma_cmh,
                        "sigma_newton": sigma_nr,
                        "error_bs": sigma_bs - sigma_true if not np.isnan(sigma_bs) else np.nan,
                        "error_li_atm": sigma_li_atm - sigma_true if not np.isnan(sigma_li_atm) else np.nan,
                        "error_li_nonatm": sigma_li_nonatm - sigma_true if not np.isnan(sigma_li_nonatm) else np.nan,
                        "error_cmh": sigma_cmh - sigma_true if not np.isnan(sigma_cmh) else np.nan,
                        "error_newton": sigma_nr - sigma_true if not np.isnan(sigma_nr) else np.nan,
                    }
                )

    frame = pd.DataFrame(rows).sort_values(["sigma_true", "T", "moneyness"]).reset_index(drop=True)
    atm = frame["moneyness"] == 1.0
    cmh_errors = frame["error_cmh"].to_numpy(dtype=float)
    li_errors = frame["error_li_nonatm"].to_numpy(dtype=float)

    summary = {
        "n_points": int(len(frame)),
        "n_atm_points": int(atm.sum()),
        "rmse_bs_atm": _rmse(frame.loc[atm, "error_bs"].to_numpy(dtype=float)),
        "rmse_li_atm": _rmse(frame.loc[atm, "error_li_atm"].to_numpy(dtype=float)),
        "rmse_li_nonatm": _rmse(li_errors),
        "rmse_cmh": _rmse(cmh_errors),
        "rmse_newton": _rmse(frame["error_newton"].to_numpy(dtype=float)),
        "max_abs_error_li_nonatm": float(np.nanmax(np.abs(li_errors))),
        "max_abs_error_cmh": float(np.nanmax(np.abs(cmh_errors))),
        "mean_abs_error_cmh": float(np.mean(np.abs(cmh_errors[~np.isnan(cmh_errors)]))),
        "all_newton_converged": bool(all(converged)),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "approximations.csv", index=False, float_format="%.10f")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class ImpliedVolApproximationSkill:
    name: str = "implied-vol-approximation-domain"

    def matches(self, *, instruction: str, task_dir: Path) -> bool:
        del task_dir
        lowered = instruction.lower()
        return (
            "implied volatil" in lowered
            and ("approximation" in lowered or "inversion" in lowered)
            and ("newton" in lowered or "brenner" in lowered or "corrado" in lowered or "li atm" in lowered)
        )

    def solve(self, *, instruction: str, task_dir: Path, out_dir: Path, seed: int) -> None:
        del instruction, task_dir, seed
        _solve(out_dir)
