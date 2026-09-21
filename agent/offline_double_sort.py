from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = (
    "identifier",
    "date",
    "betabab_1260d",
    "ret_12_7",
    "ret_lead1m",
)


def _find_panel(task_dir: Path) -> Path:
    preferred_names = (
        "stock_data.parquet",
        "stock_chars.pqt",
    )

    for name in preferred_names:
        preferred = (
            task_dir
            / "environment"
            / "data"
            / name
        )
        if preferred.is_file():
            return preferred

    candidates: list[Path] = []

    for pattern in (
        "*.parquet",
        "*.pqt",
        "*.pq",
    ):
        for path in sorted(
            task_dir.rglob(pattern)
        ):
            if "checks" in path.parts:
                continue

            try:
                columns = set(
                    pd.read_parquet(
                        path
                    ).columns
                )
            except Exception:
                continue

            if set(
                _REQUIRED_COLUMNS
            ).issubset(columns):
                candidates.append(path)

    candidates = sorted(
        set(candidates)
    )

    if len(candidates) != 1:
        raise RuntimeError(
            "Could not identify the unique "
            "double-sort input panel."
        )

    return candidates[0]


def _quintile_labels(n: int) -> np.ndarray:
    ranks = np.arange(int(n), dtype=np.int64)
    return np.minimum(5, (5 * ranks) // int(n) + 1)


@dataclass(frozen=True)
class DoubleSortCornerSkill:
    name: str = "double-sort-corner"

    def matches(self, *, instruction: str, task_dir: Path) -> bool:
        del task_dir
        lowered = instruction.lower()
        required = (
            "dependent double sort",
            "betabab_1260d",
            "ret_12_7",
            "ret_lead1m",
            "strategy_returns.csv",
            "15 basis points",
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

        panel_path = _find_panel(task_dir)
        frame = pd.read_parquet(panel_path, columns=list(_REQUIRED_COLUMNS)).copy()
        frame = frame.dropna(
            subset=["betabab_1260d", "ret_12_7", "ret_lead1m"]
        )
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values(
            ["date", "identifier"],
            kind="stable",
        ).reset_index(drop=True)

        rows: list[dict[str, object]] = []
        previous_weights: dict[int, float] | None = None

        for date, group in frame.groupby("date", sort=True):
            n = int(len(group))
            if n < 5:
                continue

            ranked_beta = (
                group.sort_values(
                    ["betabab_1260d", "identifier"],
                    kind="stable",
                )
                .reset_index(drop=True)
                .copy()
            )
            ranked_beta["beta_q"] = _quintile_labels(n)

            low_beta = ranked_beta[ranked_beta["beta_q"] == 1].copy()
            m = int(len(low_beta))
            if m < 1:
                continue

            low_beta = low_beta.sort_values(
                ["ret_12_7", "identifier"],
                kind="stable",
            ).reset_index(drop=True)
            low_beta["mom_q"] = _quintile_labels(m)

            target = low_beta[low_beta["mom_q"] == 5]
            k = int(len(target))
            if k < 1:
                continue

            weight = 1.0 / float(k)
            current_weights = {
                int(identifier): weight
                for identifier in target["identifier"].to_numpy()
            }

            if previous_weights is None:
                turnover = 1.0
            else:
                all_ids = set(previous_weights) | set(current_weights)
                turnover = float(
                    sum(
                        abs(
                            current_weights.get(identifier, 0.0)
                            - previous_weights.get(identifier, 0.0)
                        )
                        for identifier in all_ids
                    )
                )

            gross_return = float(target["ret_lead1m"].mean())
            net_return = gross_return - 0.0015 * turnover

            rows.append(
                {
                    "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                    "net_return": float(net_return),
                }
            )
            previous_weights = current_weights

        if not rows:
            raise RuntimeError("No feasible corner portfolio was emitted.")

        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            rows,
            columns=["date", "net_return"],
        ).to_csv(
            out_dir / "strategy_returns.csv",
            index=False,
        )
