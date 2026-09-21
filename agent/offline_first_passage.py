from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from agent.qf_primitives import (
    brownian_joint_terminal_max_cdf,
    brownian_running_max_hit_probability,
    brownian_running_min_hit_probability,
    historical_log_return_calibration,
)


def _find_spy(task_dir: Path) -> Path:
    preferred = task_dir / "environment" / "data" / "spy_daily.csv"
    if preferred.is_file():
        return preferred
    matches = [p for p in task_dir.rglob("spy_daily.csv") if "checks" not in p.parts]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one spy_daily.csv, found {len(matches)}.")
    return matches[0]


@dataclass(frozen=True)
class FirstPassageTimeSkill:
    name: str = "first-passage-time"

    def matches(self, *, instruction: str, task_dir: Path) -> bool:
        del task_dir
        lowered = instruction.lower()
        required = (
            "first passage time",
            "running maximum",
            "running minimum",
            "reflection principle",
            "running_max_probs.csv",
            "running_min_probs.csv",
            "first_passage_time.csv",
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

        spy = pd.read_csv(_find_spy(task_dir))
        if "close" not in spy.columns:
            raise RuntimeError("SPY data is missing close.")
        closes = pd.to_numeric(spy["close"], errors="coerce").dropna().to_numpy(dtype=float)
        calibration = historical_log_return_calibration(closes, annualization=252.0)

        s0 = float(calibration["spot"])
        sigma = float(calibration["annualized_vol"])
        n_returns = int(calibration["n_returns"])
        r = 0.05
        dividend = 0.013
        mu = float(r - dividend - 0.5 * sigma * sigma)

        cal_out = {
            "S0": s0,
            "sigma": sigma,
            "r": r,
            "D": dividend,
            "mu_rn": mu,
            "n_returns": n_returns,
        }

        h_ratios = [1.05, 1.10, 1.15, 1.20, 1.30]
        l_ratios = [0.70, 0.80, 0.85, 0.90, 0.95]
        horizons = [0.25, 0.50, 1.00]

        n_paths = 50_000
        dt = 1.0 / 252.0
        n_steps = 252
        rng = np.random.RandomState(42)

        x = np.zeros(n_paths, dtype=float)
        running_max = np.zeros(n_paths, dtype=float)
        running_min = np.zeros(n_paths, dtype=float)
        b110 = math.log(1.10)
        a090 = math.log(0.90)
        hit_up = np.full(n_paths, np.inf, dtype=float)
        hit_down = np.full(n_paths, np.inf, dtype=float)

        snapshot_steps = {63: 0.25, 126: 0.50, 252: 1.00}
        max_snapshots: dict[float, np.ndarray] = {}
        min_snapshots: dict[float, np.ndarray] = {}
        terminal_x = None

        for step in range(1, n_steps + 1):
            z = rng.standard_normal(n_paths)
            x += mu * dt + sigma * math.sqrt(dt) * z
            running_max = np.maximum(running_max, x)
            running_min = np.minimum(running_min, x)

            now = step * dt
            newly_up = np.isinf(hit_up) & (x >= b110)
            newly_down = np.isinf(hit_down) & (x <= a090)
            hit_up[newly_up] = now
            hit_down[newly_down] = now

            if step in snapshot_steps:
                horizon = snapshot_steps[step]
                max_snapshots[horizon] = running_max.copy()
                min_snapshots[horizon] = running_min.copy()

            if step == n_steps:
                terminal_x = x.copy()

        max_rows = []
        for horizon in horizons:
            snapshot = max_snapshots[horizon]
            for ratio in h_ratios:
                b = math.log(ratio)
                analytical = brownian_running_max_hit_probability(
                    mu, sigma, horizon, b
                )
                mc = float(np.mean(snapshot >= b))
                max_rows.append(
                    {
                        "T": horizon,
                        "H_ratio": ratio,
                        "H": s0 * ratio,
                        "b": b,
                        "prob_analytical": analytical,
                        "prob_mc": mc,
                    }
                )

        min_rows = []
        for horizon in horizons:
            snapshot = min_snapshots[horizon]
            for ratio in l_ratios:
                a = math.log(ratio)
                analytical = brownian_running_min_hit_probability(
                    mu, sigma, horizon, a
                )
                mc = float(np.mean(snapshot <= a))
                min_rows.append(
                    {
                        "T": horizon,
                        "L_ratio": ratio,
                        "L": s0 * ratio,
                        "a": a,
                        "prob_analytical": analytical,
                        "prob_mc": mc,
                    }
                )

        fpt_rows = []
        times = [i / 10.0 for i in range(1, 11)]
        for barrier_type, barrier, hits, fn in (
            ("up_110", b110, hit_up, brownian_running_max_hit_probability),
            ("down_090", a090, hit_down, brownian_running_min_hit_probability),
        ):
            for t in times:
                analytical = float(fn(mu, sigma, t, barrier))
                mc = float(np.mean(hits <= t + 1e-12))
                fpt_rows.append(
                    {
                        "barrier_type": barrier_type,
                        "t": t,
                        "cdf_analytical": analytical,
                        "cdf_mc": mc,
                    }
                )

        joint_analytical = brownian_joint_terminal_max_cdf(
            mu=mu,
            sigma=sigma,
            horizon=1.0,
            terminal_level=0.0,
            max_barrier=b110,
        )
        if terminal_x is None:
            raise RuntimeError("MC terminal state was not recorded.")
        joint_mc = float(
            np.mean(
                (terminal_x <= 0.0)
                & (max_snapshots[1.0] <= b110)
            )
        )

        max_df = pd.DataFrame(max_rows)
        min_df = pd.DataFrame(min_rows)
        fpt_df = pd.DataFrame(fpt_rows)

        up_den = np.maximum(max_df["prob_analytical"].to_numpy(float), 1e-12)
        down_den = np.maximum(min_df["prob_analytical"].to_numpy(float), 1e-12)
        max_up_relerr = float(
            np.max(
                np.abs(
                    max_df["prob_analytical"].to_numpy(float)
                    - max_df["prob_mc"].to_numpy(float)
                )
                / up_den
            )
        )
        max_down_relerr = float(
            np.max(
                np.abs(
                    min_df["prob_analytical"].to_numpy(float)
                    - min_df["prob_mc"].to_numpy(float)
                )
                / down_den
            )
        )
        max_fpt_abserr = float(
            np.max(
                np.abs(
                    fpt_df["cdf_analytical"].to_numpy(float)
                    - fpt_df["cdf_mc"].to_numpy(float)
                )
            )
        )

        finite_up = hit_up[np.isfinite(hit_up)]
        e_tau_up_mc = float(np.mean(finite_up)) if finite_up.size else float("nan")
        e_tau_up = float(b110 / mu) if mu > 0 else float("inf")

        prob_hit_110 = float(
            brownian_running_max_hit_probability(mu, sigma, 1.0, b110)
        )
        prob_hit_090 = float(
            brownian_running_min_hit_probability(mu, sigma, 1.0, a090)
        )

        max_decreases = all(
            np.all(
                np.diff(
                    max_df[np.isclose(max_df["T"], horizon)]
                    .sort_values("H_ratio")["prob_analytical"]
                    .to_numpy(float)
                )
                <= 1e-12
            )
            for horizon in horizons
        )
        min_increases = all(
            np.all(
                np.diff(
                    min_df[np.isclose(min_df["T"], horizon)]
                    .sort_values("L_ratio")["prob_analytical"]
                    .to_numpy(float)
                )
                >= -1e-12
            )
            for horizon in horizons
        )
        prob_increases = all(
            np.all(
                np.diff(
                    max_df[np.isclose(max_df["H_ratio"], ratio)]
                    .sort_values("T")["prob_analytical"]
                    .to_numpy(float)
                )
                >= -1e-12
            )
            for ratio in h_ratios
        )

        summary = {
            "n_mc_paths": n_paths,
            "max_prob_up_relerr": max_up_relerr,
            "max_prob_down_relerr": max_down_relerr,
            "max_fpt_cdf_abserr": max_fpt_abserr,
            "E_tau_up": e_tau_up,
            "E_tau_up_mc": e_tau_up_mc,
            "prob_hit_110_T1": prob_hit_110,
            "prob_hit_090_T1": prob_hit_090,
            "joint_prob_analytical": float(joint_analytical),
            "joint_prob_mc": joint_mc,
            "max_decreases_with_barrier": bool(max_decreases),
            "min_increases_with_barrier": bool(min_increases),
            "prob_increases_with_T": bool(prob_increases),
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "calibration.json").write_text(
            json.dumps(cal_out, indent=2) + "\n", encoding="utf-8"
        )
        max_df.to_csv(out_dir / "running_max_probs.csv", index=False)
        min_df.to_csv(out_dir / "running_min_probs.csv", index=False)
        fpt_df.to_csv(out_dir / "first_passage_time.csv", index=False)
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
