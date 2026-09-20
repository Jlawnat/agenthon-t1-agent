from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


def _find_data_dir(task_dir: Path) -> Path:
    required = {"BAA.csv", "AAA.csv", "DGS10.csv", "TEDRATE.csv", "T10YIE.csv"}
    for directory in sorted({p.parent for p in task_dir.rglob("*") if p.is_file() and "checks" not in p.parts}, key=str):
        names = {p.name for p in directory.iterdir() if p.is_file()}
        if required.issubset(names):
            return directory
    raise RuntimeError("Could not discover FRED credit-spread inputs.")


def _load_fred(path: Path) -> pd.Series:
    frame = pd.read_csv(path, na_values=[".", ""])
    frame.columns = ["date", "value"]
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date"]).set_index("date").sort_index()
    return frame["value"]


def _solve(task_dir: Path, out_dir: Path) -> None:
    data_dir = _find_data_dir(task_dir)
    baa = _load_fred(data_dir / "BAA.csv")
    aaa = _load_fred(data_dir / "AAA.csv")
    dgs10 = _load_fred(data_dir / "DGS10.csv")
    ted = _load_fred(data_dir / "TEDRATE.csv")
    t10yie = _load_fred(data_dir / "T10YIE.csv")

    dgs10_m = dgs10.dropna().resample("ME").last()
    ted_m = ted.dropna().resample("ME").last()
    t10yie_m = t10yie.dropna().resample("ME").last()
    baa.index = baa.index + pd.offsets.MonthEnd(0)
    aaa.index = aaa.index + pd.offsets.MonthEnd(0)

    merged = pd.DataFrame(
        {"BAA": baa, "AAA": aaa, "DGS10": dgs10_m, "TEDRATE": ted_m, "T10YIE": t10yie_m}
    ).loc["2003-01":"2024-12"]

    merged = merged.dropna(subset=["BAA", "AAA", "DGS10"])
    merged["TEDRATE"] = merged["TEDRATE"].ffill()
    merged = merged.dropna(subset=["T10YIE"])

    merged["baa_treasury_spread"] = merged["BAA"] - merged["DGS10"]
    merged["default_proxy"] = merged["BAA"] - merged["AAA"]
    merged["liquidity_proxy"] = merged["TEDRATE"]
    merged["inflation_proxy"] = merged["T10YIE"]

    window, nw_lag = 48, 3
    decomp_rows, variance_rows = [], []

    for end in range(window, len(merged) + 1):
        sample = merged.iloc[end - window:end]
        y = sample["baa_treasury_spread"].to_numpy(dtype=float)
        xf = np.column_stack(
            [
                sample["default_proxy"].to_numpy(dtype=float),
                sample["liquidity_proxy"].to_numpy(dtype=float),
                sample["inflation_proxy"].to_numpy(dtype=float),
            ]
        )
        x = np.column_stack([np.ones(window), xf])
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        fitted = x @ beta
        residuals = y - fitted
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

        s = (x.T * residuals ** 2) @ x
        for lag in range(1, nw_lag + 1):
            weight = 1.0 - lag / (nw_lag + 1.0)
            gamma = np.zeros((x.shape[1], x.shape[1]))
            for t in range(lag, window):
                gamma += residuals[t] * residuals[t - lag] * np.outer(x[t], x[t - lag])
            s += weight * (gamma + gamma.T)

        inv = np.linalg.pinv(x.T @ x)
        hac_cov = inv @ s @ inv
        hac_se = np.sqrt(np.maximum(np.diag(hac_cov), 0.0))
        hac_t = beta / hac_se

        var_y = float(np.var(y, ddof=1))
        if var_y > 1e-12:
            pct_default = float(np.cov(beta[1] * xf[:, 0], y)[0, 1] / var_y * 100.0)
            pct_liquidity = float(np.cov(beta[2] * xf[:, 1], y)[0, 1] / var_y * 100.0)
            pct_inflation = float(np.cov(beta[3] * xf[:, 2], y)[0, 1] / var_y * 100.0)
        else:
            pct_default = pct_liquidity = pct_inflation = 0.0
        pct_residual = (1.0 - r2) * 100.0

        current = merged.iloc[end - 1]
        default_comp = float(beta[1] * current["default_proxy"])
        liquidity_comp = float(beta[2] * current["liquidity_proxy"])
        inflation_comp = float(beta[3] * current["inflation_proxy"])
        resid = float(
            current["baa_treasury_spread"]
            - beta[0]
            - default_comp
            - liquidity_comp
            - inflation_comp
        )
        date = merged.index[end - 1].replace(day=1).strftime("%Y-%m-%d")

        decomp_rows.append(
            {
                "date": date,
                "baa_treasury_spread": round(float(current["baa_treasury_spread"]), 6),
                "default_component": round(default_comp, 6),
                "liquidity_component": round(liquidity_comp, 6),
                "inflation_component": round(inflation_comp, 6),
                "residual": round(resid, 6),
                "r_squared": round(r2, 6),
                "hac_tstat_default": round(float(hac_t[1]), 6),
                "hac_tstat_liquidity": round(float(hac_t[2]), 6),
                "hac_tstat_inflation": round(float(hac_t[3]), 6),
            }
        )
        variance_rows.append(
            {
                "date": date,
                "pct_default": round(pct_default, 6),
                "pct_liquidity": round(pct_liquidity, 6),
                "pct_inflation": round(pct_inflation, 6),
                "pct_residual": round(pct_residual, 6),
            }
        )

    decomp = pd.DataFrame(decomp_rows)
    variance = pd.DataFrame(variance_rows)
    breaks = decomp[decomp["r_squared"] < 0.50][["date", "r_squared"]].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    decomp.to_csv(out_dir / "decomposition_results.csv", index=False)
    variance.to_csv(out_dir / "variance_decomposition.csv", index=False)
    breaks.to_csv(out_dir / "regime_breaks.csv", index=False)


@dataclass(frozen=True)
class CreditSpreadDecompositionSkill:
    name: str = "credit-spread-decomposition-domain"

    def matches(self, *, instruction: str, task_dir: Path) -> bool:
        lowered = instruction.lower()
        files = {p.name for p in task_dir.rglob("*.csv")}
        return (
            {"BAA.csv", "AAA.csv", "DGS10.csv"}.issubset(files)
            or ("credit spread" in lowered and "newey-west" in lowered and "variance decomposition" in lowered)
        )

    def solve(self, *, instruction: str, task_dir: Path, out_dir: Path, seed: int) -> None:
        del instruction, seed
        _solve(task_dir, out_dir)
