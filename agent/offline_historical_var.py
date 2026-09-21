from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _find_named(task_dir: Path, name: str) -> Path:
    preferred = task_dir / "environment" / "data" / name
    if preferred.is_file():
        return preferred
    matches = [p for p in task_dir.rglob(name) if "checks" not in p.parts]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name!r} outside checks, found {len(matches)}.")
    return matches[0]


def _parse_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(
        series.astype(str).str.strip(),
        format="mixed",
        errors="coerce",
    )
    if parsed.isna().any():
        raise RuntimeError(f"Could not parse {int(parsed.isna().sum())} date value(s).")
    return parsed.dt.normalize()


@dataclass(frozen=True)
class HistoricalVarDataPrepSkill:
    name: str = "historical-var-data-prep"

    def matches(self, *, instruction: str, task_dir: Path) -> bool:
        del task_dir
        lowered = instruction.lower()
        required = (
            "historical",
            "value-at-risk",
            "trading_calendar.csv",
            "returns_raw.csv",
            "listwise",
            "results.json",
            "solution.json",
        )
        return all(token in lowered for token in required)

    def solve(
        self,
        *,
        instruction: str,
        task_dir: Path,
        out_dir: Path,
        seed: int,
    ) -> None:
        del instruction, seed

        returns_path = _find_named(task_dir, "returns_raw.csv")
        calendar_path = _find_named(task_dir, "trading_calendar.csv")

        raw = pd.read_csv(returns_path)
        raw.columns = [str(c).strip() for c in raw.columns]

        expected_columns = ["date", "SPY", "TLT", "GLD", "IEF", "EFA"]
        missing = [c for c in expected_columns if c not in raw.columns]
        if missing:
            raise RuntimeError(f"Missing required return columns: {missing}")

        frame = raw[expected_columns].copy()
        n_rows_raw = int(len(frame))
        frame["date"] = _parse_dates(frame["date"])

        asset_columns = expected_columns[1:]
        for column in asset_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        calendar = pd.read_csv(calendar_path)
        if "date" not in calendar.columns:
            raise RuntimeError("trading_calendar.csv is missing date.")
        calendar_dates = set(_parse_dates(calendar["date"]).tolist())

        non_trading = ~frame["date"].isin(calendar_dates)
        n_non_trading_days_removed = int(non_trading.sum())
        frame = frame.loc[~non_trading].copy()

        duplicate = frame.duplicated(subset=["date"], keep="last")
        n_duplicates_removed = int(duplicate.sum())
        frame = frame.loc[~duplicate].copy()

        outlier = frame[asset_columns].abs().gt(0.25).any(axis=1)
        n_outliers_removed = int(outlier.sum())
        frame = frame.loc[~outlier].copy()

        missing_rows = frame[asset_columns].isna().any(axis=1)
        n_missing_rows_removed = int(missing_rows.sum())
        frame = frame.loc[~missing_rows].copy()

        frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
        n_rows_after_cleaning = int(len(frame))
        if n_rows_after_cleaning == 0:
            raise RuntimeError("No observations remain after cleaning.")

        portfolio_returns = frame[asset_columns].mean(axis=1).to_numpy(dtype=float)
        var_95 = float(-np.percentile(portfolio_returns, 5, method="linear"))
        var_99 = float(-np.percentile(portfolio_returns, 1, method="linear"))

        worst_index = int(np.argmin(portfolio_returns))
        worst_return = float(portfolio_returns[worst_index])
        worst_date = pd.Timestamp(frame.loc[worst_index, "date"]).strftime("%Y-%m-%d")

        results = {
            "n_observations_clean": n_rows_after_cleaning,
            "var_95_portfolio": var_95,
            "var_99_portfolio": var_99,
            "worst_day_return": worst_return,
            "worst_day_date": worst_date,
        }
        solution = {
            "cleaning_report": {
                "n_rows_raw": n_rows_raw,
                "n_non_trading_days_removed": n_non_trading_days_removed,
                "n_duplicates_removed": n_duplicates_removed,
                "n_outliers_removed": n_outliers_removed,
                "n_missing_rows_removed": n_missing_rows_removed,
                "n_rows_after_cleaning": n_rows_after_cleaning,
            }
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(
            json.dumps(results, indent=2) + "\n",
            encoding="utf-8",
        )
        (out_dir / "solution.json").write_text(
            json.dumps(solution, indent=2) + "\n",
            encoding="utf-8",
        )
