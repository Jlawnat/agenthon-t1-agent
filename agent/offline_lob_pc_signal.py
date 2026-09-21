from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _find_lob(task_dir: Path) -> Path:
    preferred = task_dir / "environment" / "data" / "ADA_1min.csv"
    if preferred.is_file():
        return preferred
    matches = [p for p in task_dir.rglob("ADA_1min.csv") if "checks" not in p.parts]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one ADA_1min.csv, found {len(matches)}.")
    return matches[0]


def _weighted_ofi(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for depth in range(15):
        bid_true = (
            pd.to_numeric(frame[f"bids_limit_notional_{depth}"], errors="coerce")
            - pd.to_numeric(frame[f"bids_cancel_notional_{depth}"], errors="coerce")
            - pd.to_numeric(frame[f"bids_market_notional_{depth}"], errors="coerce")
        )
        ask_true = (
            pd.to_numeric(frame[f"asks_limit_notional_{depth}"], errors="coerce")
            - pd.to_numeric(frame[f"asks_cancel_notional_{depth}"], errors="coerce")
            - pd.to_numeric(frame[f"asks_market_notional_{depth}"], errors="coerce")
        )
        bid_distance = pd.to_numeric(
            frame[f"bids_distance_{depth}"], errors="coerce"
        ).abs()
        ask_distance = pd.to_numeric(
            frame[f"asks_distance_{depth}"], errors="coerce"
        ).abs()
        result[f"weighted_ofi_{depth}"] = (
            bid_true / (1.0 + bid_distance)
            - ask_true / (1.0 + ask_distance)
        )
    return result


def _first_pc_component(z: np.ndarray) -> np.ndarray:
    if z.ndim != 2 or z.shape[0] < 2:
        raise RuntimeError("PCA requires a 2D training matrix.")
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    return vt[0].astype(float, copy=True)


def _rolling_predictions(
    ofi: pd.DataFrame,
    target: pd.Series,
    window: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    x_all = ofi.to_numpy(dtype=float)
    y_all = target.to_numpy(dtype=float)
    n = len(ofi)
    pc_current = np.full(n, np.nan, dtype=float)
    predicted = np.full(n, np.nan, dtype=float)

    for t in range(window, n):
        x_train = x_all[t - window:t]
        y_train = y_all[t - window:t]

        valid = np.isfinite(y_train) & np.isfinite(x_train).all(axis=1)
        if int(valid.sum()) < 3 or not np.isfinite(x_all[t]).all():
            continue

        x_fit = x_train[valid]
        y_fit = y_train[valid]

        mean = x_fit.mean(axis=0)
        std = x_fit.std(axis=0, ddof=0)
        std = np.where(std > 1e-15, std, 1.0)

        z_train = (x_fit - mean) / std
        component = _first_pc_component(z_train)
        pc_train = z_train @ component

        corr = np.corrcoef(
            pc_train,
            x_fit[:, 0],
        )[0, 1]
        if np.isfinite(corr) and corr < 0.0:
            component = -component
            pc_train = -pc_train

        design = np.column_stack(
            [np.ones(len(pc_train), dtype=float), pc_train]
        )
        coef, *_ = np.linalg.lstsq(design, y_fit, rcond=None)

        z_current = (x_all[t] - mean) / std
        score = float(z_current @ component)
        pc_current[t] = score
        predicted[t] = float(coef[0] + coef[1] * score)

    return pc_current, predicted


@dataclass(frozen=True)
class LobPcSignalSkill:
    name: str = "lob-pc-signal"

    def matches(self, *, instruction: str, task_dir: Path) -> bool:
        del task_dir
        lowered = instruction.lower()
        required = (
            "weighted true ofi",
            "pca",
            "rolling regression",
            "window size of 40",
            "ada_1min.csv",
            "spearman_ic",
            "pearson_ic",
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

        frame = pd.read_csv(_find_lob(task_dir))
        midpoint = pd.to_numeric(frame["midpoint"], errors="coerce")
        midpoint_return_fwd = np.log(midpoint.shift(-1) / midpoint)
        ofi = _weighted_ofi(frame)

        pc_current, predicted = _rolling_predictions(
            ofi,
            midpoint_return_fwd,
            window=40,
        )
        target = midpoint_return_fwd.to_numpy(dtype=float)

        valid = np.isfinite(predicted) & np.isfinite(target)
        if int(valid.sum()) < 3:
            raise RuntimeError("Too few valid rolling predictions.")

        spearman = float(spearmanr(predicted[valid], target[valid]).statistic)
        pearson = float(pearsonr(predicted[valid], target[valid]).statistic)

        result: dict[str, float] = {}

        for idx in (0, 10, 20, 30):
            result[f"midpoint_return_fwd_index{idx}"] = float(
                midpoint_return_fwd.iloc[idx]
            )
            for depth in (0, 2, 4, 6):
                result[f"weighted_ofi_{depth}_index{idx}"] = float(
                    ofi.iloc[idx][f"weighted_ofi_{depth}"]
                )

        for idx in (40, 50, 60, 70):
            result[f"rolling_pc1_index{idx}"] = float(pc_current[idx])
            result[f"predicted_return_index{idx}"] = float(predicted[idx])

        result["spearman_ic"] = spearman
        result["pearson_ic"] = pearson

        if not all(np.isfinite(float(value)) for value in result.values()):
            raise RuntimeError("LOB signal output contains a non-finite value.")

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
