from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ORDER = [
    "historical_mean_profile",
    "historical_median_profile",
    "ewma_profile",
    "winsorized_mean_profile",
]


def _find_named(task_dir: Path, name: str) -> Path:
    preferred = task_dir / "environment" / "data" / name
    if preferred.is_file():
        return preferred
    matches = [p for p in task_dir.rglob(name) if "checks" not in p.parts]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name!r}, found {len(matches)}.")
    return matches[0]


def _session_bars(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.sort_values("datetime", kind="stable").reset_index(drop=True)
    frame = frame.drop_duplicates(
        subset=["datetime"],
        keep="last",
    ).reset_index(drop=True)

    time = frame["datetime"].dt.time
    lower = pd.Timestamp("09:30:00").time()
    upper = pd.Timestamp("16:00:00").time()
    frame = frame[(time >= lower) & (time <= upper)].copy()

    pieces = []
    for _, day in frame.groupby(frame["datetime"].dt.normalize(), sort=True):
        day = day.set_index("datetime").sort_index()
        resampled = day.resample("5min", label="left", closed="left").agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        # Empty intervals created by resample are not genuine session bars.
        resampled = resampled[resampled["Open"].notna()].reset_index()
        pieces.append(resampled)

    if not pieces:
        raise RuntimeError("No intraday rows survived session filtering.")

    bars = pd.concat(pieces, ignore_index=True)
    t = bars["datetime"].dt.time
    last = pd.Timestamp("15:55:00").time()
    bars = bars[(t >= lower) & (t <= last)].copy()
    bars["date"] = bars["datetime"].dt.normalize()
    return bars.sort_values("datetime", kind="stable").reset_index(drop=True)


def _complete_sessions(
    bars: pd.DataFrame,
) -> tuple[list[pd.Timestamp], list[str], np.ndarray]:
    complete_dates: list[pd.Timestamp] = []
    excluded: list[str] = []
    volumes = []

    for date, day in bars.groupby("date", sort=True):
        day = day.sort_values("datetime", kind="stable")
        if len(day) != 78:
            excluded.append(pd.Timestamp(date).strftime("%Y-%m-%d"))
            continue
        complete_dates.append(pd.Timestamp(date))
        volumes.append(day["Volume"].to_numpy(dtype=float))

    if not volumes:
        raise RuntimeError("No complete 78-bar sessions.")

    return complete_dates, excluded, np.vstack(volumes)


def _normalize_profile(profile: np.ndarray) -> np.ndarray:
    x = np.asarray(profile, dtype=float)
    x = np.maximum(x, 0.0)
    total = float(np.sum(x))
    if total <= 0.0:
        raise RuntimeError("Volume profile has zero total.")
    return x / total


def _profiles(history_shares: np.ndarray) -> dict[str, np.ndarray]:
    mean_profile = _normalize_profile(np.mean(history_shares, axis=0))
    median_profile = _normalize_profile(np.median(history_shares, axis=0))

    # History is oldest -> most recent. Lag 19 belongs to the oldest row,
    # lag 0 to the most recent row.
    lags = np.arange(len(history_shares) - 1, -1, -1, dtype=float)
    weights = np.exp(-np.log(2.0) * lags / 20.0)
    ewma_profile = _normalize_profile(
        np.average(history_shares, axis=0, weights=weights)
    )

    low = np.percentile(history_shares, 5.0, axis=0)
    high = np.percentile(history_shares, 95.0, axis=0)
    winsorized = np.clip(history_shares, low, high)
    winsorized_profile = _normalize_profile(np.mean(winsorized, axis=0))

    return {
        "historical_mean_profile": mean_profile,
        "historical_median_profile": median_profile,
        "ewma_profile": ewma_profile,
        "winsorized_mean_profile": winsorized_profile,
    }


def _r2(realized: np.ndarray, predicted: np.ndarray) -> float:
    y = np.asarray(realized, dtype=float)
    p = np.asarray(predicted, dtype=float)
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    numerator = float(np.sum((y - p) ** 2))
    if denominator == 0.0:
        return 1.0 if np.array_equal(y, p) else 0.0
    return float(1.0 - numerator / denominator)


def _largest_remainder(quantity: int, weights: np.ndarray) -> np.ndarray:
    raw = int(quantity) * np.asarray(weights, dtype=float)
    base = np.floor(raw).astype(np.int64)
    remainder = int(quantity) - int(base.sum())
    if remainder > 0:
        fractional = raw - base
        indices = np.arange(len(base))
        # Primary: descending fractional remainder. Secondary: earlier bar.
        order = np.lexsort((indices, -fractional))
        base[order[:remainder]] += 1
    return base


@dataclass(frozen=True)
class IntradayVolumeExecutionSkill:
    name: str = "intraday-volume-execution"

    def matches(self, *, instruction: str, task_dir: Path) -> bool:
        del task_dir
        lowered = instruction.lower()
        required = (
            "historical_mean_profile",
            "historical_median_profile",
            "ewma_profile",
            "winsorized_mean_profile",
            "78",
            "20 complete trading days",
            "final_schedule.csv",
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

        raw = pd.read_csv(_find_named(task_dir, "Stock_Intraday_Data.csv"))
        order = json.loads(
            _find_named(task_dir, "order.json").read_text(encoding="utf-8")
        )
        order_dt = pd.Timestamp(order["datetime"])
        quantity = int(order["quantity"])
        order_day = order_dt.normalize()

        bars = _session_bars(raw)
        complete_dates, excluded, volumes = _complete_sessions(bars)

        totals = volumes.sum(axis=1)
        if np.any(totals <= 0.0):
            raise RuntimeError("Complete session with non-positive total volume.")
        shares = volumes / totals[:, None]

        if len(complete_dates) <= 20:
            raise RuntimeError("Need more than 20 complete sessions.")

        score_lists = {name: [] for name in MODEL_ORDER}

        for test_index in range(20, len(complete_dates)):
            history = shares[test_index - 20:test_index]
            actual = shares[test_index]
            candidate = _profiles(history)
            for name in MODEL_ORDER:
                score_lists[name].append(_r2(actual, candidate[name]))

        performance_rows = []
        for name in MODEL_ORDER:
            values = score_lists[name]
            performance_rows.append(
                {
                    "model_name": name,
                    "avg_r2": float(np.mean(values)),
                    "num_test_days": int(len(values)),
                }
            )
        performance = pd.DataFrame(
            performance_rows,
            columns=["model_name", "avg_r2", "num_test_days"],
        )

        # max preserves the first model in MODEL_ORDER when scores tie.
        best_name = max(
            MODEL_ORDER,
            key=lambda name: float(
                performance.loc[
                    performance["model_name"] == name,
                    "avg_r2",
                ].iloc[0]
            ),
        )
        best_row = performance.loc[
            performance["model_name"] == best_name
        ].iloc[0]

        # Match the benchmark execution convention.  The order calendar
        # date does not itself need to be present as a complete historical
        # session.  When absent, use the last complete historical session
        # as the scheduling template, then map its bar times onto order_day.
        complete_days_norm = [
            pd.Timestamp(d).normalize()
            for d in complete_dates
        ]

        if order_day in complete_days_norm:
            order_index = complete_days_norm.index(order_day)
        else:
            order_index = len(complete_days_norm)

        if order_index < 20:
            raise RuntimeError(
                "Fewer than 20 complete days are available before execution."
            )

        selected_indices = list(
            range(order_index - 20, order_index)
        )
        history = shares[selected_indices]
        final_profile = _profiles(history)[best_name]

        if order_day in complete_days_norm:
            schedule_day = order_day
        else:
            schedule_day = complete_days_norm[order_index - 1]

        day_bars = bars[
            bars["date"] == schedule_day
        ].sort_values(
            "datetime",
            kind="stable",
        ).copy()

        if len(day_bars) != 78:
            raise RuntimeError(
                "Schedule template is not a complete 78-bar session."
            )

        split_ts = pd.Timestamp.combine(
            schedule_day,
            order_dt.time(),
        )

        mask = day_bars["datetime"] > split_ts
        remaining = day_bars.loc[mask].copy()

        if remaining.empty:
            raise RuntimeError(
                "No session bars remain after order datetime."
            )

        remaining_positions = np.flatnonzero(
            mask.to_numpy()
        )

        restricted = final_profile[
            remaining_positions
        ]
        restricted = _normalize_profile(
            restricted
        )

        scheduled = _largest_remainder(
            quantity,
            restricted,
        )
        cumulative = np.cumsum(
            scheduled
        )

        # The historical schedule template supplies the intraday bar times,
        # but output timestamps belong to the actual order calendar date.
        output_datetimes = (
            order_day
            + (
                remaining["datetime"]
                - remaining["datetime"].dt.normalize()
            )
        )

        schedule = pd.DataFrame(
            {
                "datetime": output_datetimes.dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ).to_numpy(),
                "scheduled_quantity": scheduled.astype(np.int64),
                "predicted_bar_share": restricted.astype(float),
                "cumulative_scheduled_quantity": cumulative.astype(np.int64),
            }
        )

        out_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            {"date": excluded},
            columns=["date"],
        ).to_csv(
            out_dir / "excluded_days.csv",
            index=False,
        )

        performance.to_csv(
            out_dir / "model_performance.csv",
            index=False,
        )

        (out_dir / "best_model.json").write_text(
            json.dumps(
                {
                    "best_model": best_name,
                    "best_avg_r2": float(best_row["avg_r2"]),
                    "num_test_days": int(best_row["num_test_days"]),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        schedule.to_csv(
            out_dir / "final_schedule.csv",
            index=False,
        )
