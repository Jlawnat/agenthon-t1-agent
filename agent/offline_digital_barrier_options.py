from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def _discover_prices(task_dir: Path) -> Path:
    preferred = []
    fallback = []
    for path in sorted(task_dir.rglob("*.csv")):
        if "checks" in path.parts:
            continue
        try:
            cols = {c.lower() for c in pd.read_csv(path, nrows=3).columns}
        except Exception:
            continue
        if {"date", "close"}.issubset(cols):
            if "spy" in path.name.lower():
                preferred.append(path)
            else:
                fallback.append(path)
    if preferred:
        return preferred[0]
    if fallback:
        return fallback[0]
    raise RuntimeError("Could not discover daily underlying prices for digital/barrier pricing.")


def _solve(task_dir: Path, out_dir: Path) -> None:
    prices_path = _discover_prices(task_dir)
    df = pd.read_csv(prices_path)
    prices = df["close"].to_numpy(dtype=np.float64)
    log_rets = np.log(prices[1:] / prices[:-1])
    S0 = float(prices[-1])
    n_returns = int(len(log_rets))
    sigma = float(np.std(log_rets, ddof=1) * np.sqrt(252.0))

    r = 0.05
    D = 0.013

    def d_plus(S, K, T):
        return (np.log(S / K) + (r - D + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))

    def d_minus(S, K, T):
        return d_plus(S, K, T) - sigma * np.sqrt(T)

    def cash_call(S, K, T):
        return np.exp(-r * T) * norm.cdf(d_minus(S, K, T))

    def cash_put(S, K, T):
        return np.exp(-r * T) * norm.cdf(-d_minus(S, K, T))

    def asset_call(S, K, T):
        return S * np.exp(-D * T) * norm.cdf(d_plus(S, K, T))

    def asset_put(S, K, T):
        return S * np.exp(-D * T) * norm.cdf(-d_plus(S, K, T))

    def bs_call(S, K, T):
        return asset_call(S, K, T) - K * cash_call(S, K, T)

    def bs_put(S, K, T):
        return K * cash_put(S, K, T) - asset_put(S, K, T)

    def gap_call(S, K1, K2, T):
        dp = d_plus(S, K2, T)
        dm = d_minus(S, K2, T)
        return S * np.exp(-D * T) * norm.cdf(dp) - K1 * np.exp(-r * T) * norm.cdf(dm)

    def do_cash_call(S, K, H, T):
        mu = r - D - 0.5 * sigma * sigma
        root = sigma * np.sqrt(T)
        dK = (np.log(S / K) + mu * T) / root
        dH = (np.log(H * H / (S * K)) + mu * T) / root
        ratio = (H / S) ** (2 * mu / (sigma * sigma))
        return np.exp(-r * T) * (norm.cdf(dK) - ratio * norm.cdf(dH))

    def uo_cash_put(S, K, H, T):
        mu = r - D - 0.5 * sigma * sigma
        root = sigma * np.sqrt(T)
        dK = (np.log(S / K) + mu * T) / root
        dH = (np.log(H * H / (S * K)) + mu * T) / root
        ratio = (H / S) ** (2 * mu / (sigma * sigma))
        return np.exp(-r * T) * (norm.cdf(-dK) - ratio * norm.cdf(-dH))

    T_vals = [0.25, 0.50, 1.00]
    m_vals = [0.90, 0.95, 1.00, 1.05, 1.10]

    digital_rows = []
    for T in T_vals:
        for m in m_vals:
            K = m * S0
            digital_rows.append(
                {
                    "T": T,
                    "moneyness": m,
                    "K": K,
                    "cash_call": cash_call(S0, K, T),
                    "cash_put": cash_put(S0, K, T),
                    "asset_call": asset_call(S0, K, T),
                    "asset_put": asset_put(S0, K, T),
                    "bs_call": bs_call(S0, K, T),
                    "bs_put": bs_put(S0, K, T),
                }
            )
    digital = pd.DataFrame(digital_rows)

    gap_rows = []
    for T in T_vals:
        for tm in [0.95, 1.00, 1.05]:
            K2 = tm * S0
            gap_rows.append(
                {
                    "T": T,
                    "trigger_moneyness": tm,
                    "K2": K2,
                    "K1": S0,
                    "gap_call": gap_call(S0, S0, K2, T),
                }
            )
    gaps = pd.DataFrame(gap_rows)

    H_down = 0.85 * S0
    H_up = 1.15 * S0
    barrier_rows = []
    for T in T_vals:
        for m in m_vals:
            K = m * S0

            vanilla_cc = cash_call(S0, K, T)
            knockout = do_cash_call(S0, K, H_down, T)
            knockin = vanilla_cc - knockout
            barrier_rows.append(
                {
                    "T": T,
                    "moneyness": m,
                    "K": K,
                    "barrier_type": "down_out_cash_call",
                    "H": H_down,
                    "knockout_price": knockout,
                    "knockin_price": knockin,
                    "vanilla_price": vanilla_cc,
                    "parity_error": abs(knockout + knockin - vanilla_cc),
                }
            )

            vanilla_cp = cash_put(S0, K, T)
            knockout = uo_cash_put(S0, K, H_up, T)
            knockin = vanilla_cp - knockout
            barrier_rows.append(
                {
                    "T": T,
                    "moneyness": m,
                    "K": K,
                    "barrier_type": "up_out_cash_put",
                    "H": H_up,
                    "knockout_price": knockout,
                    "knockin_price": knockin,
                    "vanilla_price": vanilla_cp,
                    "parity_error": abs(knockout + knockin - vanilla_cp),
                }
            )
    barriers = pd.DataFrame(barrier_rows)

    cash_err = []
    asset_err = []
    bs_err = []
    for _, row in digital.iterrows():
        T = float(row["T"])
        cash_err.append(abs(row["cash_call"] + row["cash_put"] - np.exp(-r * T)))
        asset_err.append(abs(row["asset_call"] + row["asset_put"] - S0 * np.exp(-D * T)))
        bs_err.append(abs(row["bs_call"] - (row["asset_call"] - row["K"] * row["cash_call"])))

    atm = digital[np.isclose(digital["moneyness"], 1.0) & np.isclose(digital["T"], 0.50)].iloc[0]
    gap_atm = gaps[np.isclose(gaps["trigger_moneyness"], 1.0) & np.isclose(gaps["T"], 0.50)].iloc[0]
    do_atm = barriers[
        (barriers["barrier_type"] == "down_out_cash_call")
        & np.isclose(barriers["moneyness"], 1.0)
        & np.isclose(barriers["T"], 0.50)
    ].iloc[0]
    uo_atm = barriers[
        (barriers["barrier_type"] == "up_out_cash_put")
        & np.isclose(barriers["moneyness"], 1.0)
        & np.isclose(barriers["T"], 0.50)
    ].iloc[0]

    monotone = True
    for T in T_vals:
        sub = digital[np.isclose(digital["T"], T)].sort_values("moneyness")
        if not np.all(np.diff(sub["cash_call"].to_numpy()) <= 1e-10):
            monotone = False

    summary = {
        "n_digital_prices": int(len(digital)),
        "n_gap_prices": int(len(gaps)),
        "n_barrier_prices": int(len(barriers)),
        "max_cash_parity_error": float(max(cash_err)),
        "max_asset_parity_error": float(max(asset_err)),
        "max_bs_decomp_error": float(max(bs_err)),
        "max_barrier_parity_error": float(barriers["parity_error"].max()),
        "atm_cash_call_T050": float(atm["cash_call"]),
        "atm_cash_put_T050": float(atm["cash_put"]),
        "atm_asset_call_T050": float(atm["asset_call"]),
        "atm_gap_call_T050": float(gap_atm["gap_call"]),
        "atm_do_cash_call_T050": float(do_atm["knockout_price"]),
        "atm_uo_cash_put_T050": float(uo_atm["knockout_price"]),
        "cash_call_increases_with_S": bool(monotone),
        "do_leq_vanilla": bool(
            (
                barriers[barriers["barrier_type"] == "down_out_cash_call"]["knockout_price"]
                <= barriers[barriers["barrier_type"] == "down_out_cash_call"]["vanilla_price"] + 1e-10
            ).all()
        ),
    }
    calibration = {"S0": S0, "sigma": sigma, "r": r, "D": D, "n_returns": n_returns}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    digital.to_csv(out_dir / "digital_prices.csv", index=False)
    gaps.to_csv(out_dir / "gap_prices.csv", index=False)
    barriers.to_csv(out_dir / "barrier_digital_prices.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class DigitalBarrierOptionsSkill:
    name: str = "digital-barrier-options-domain"

    def matches(self, *, instruction: str, task_dir: Path) -> bool:
        lowered = instruction.lower()
        return "digital" in lowered and (
            "barrier" in lowered
            or "cash-or-nothing" in lowered
            or "gap option" in lowered
        )

    def solve(self, *, instruction: str, task_dir: Path, out_dir: Path, seed: int) -> None:
        del instruction, seed
        _solve(task_dir, out_dir)
