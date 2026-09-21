from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def _find_named(
    task_dir: Path,
    name: str,
) -> Path:
    preferred = (
        task_dir
        / "environment"
        / "data"
        / name
    )
    if preferred.is_file():
        return preferred

    matches = [
        path
        for path in task_dir.rglob(
            name
        )
        if "checks" not in path.parts
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {name}, found {len(matches)}."
        )
    return matches[0]


def _simulate_scheme(
    *,
    rng: np.random.Generator,
    mean: np.ndarray,
    covariance: np.ndarray,
    weights: np.ndarray,
    initial_capital: float,
    n_paths: int,
    n_days: int,
) -> dict[str, float]:
    draws = rng.multivariate_normal(
        mean,
        covariance,
        size=(
            n_paths,
            n_days,
        ),
    )

    daily_returns = (
        draws
        @ weights
    )

    wealth = (
        initial_capital
        * np.cumprod(
            1.0
            + daily_returns,
            axis=1,
        )
    )

    terminal = wealth[
        :,
        -1,
    ]

    wealth_with_initial = (
        np.concatenate(
            [
                np.full(
                    (
                        n_paths,
                        1,
                    ),
                    initial_capital,
                    dtype=float,
                ),
                wealth,
            ],
            axis=1,
        )
    )

    running_peak = (
        np.maximum.accumulate(
            wealth_with_initial,
            axis=1,
        )
    )
    drawdown = (
        1.0
        - wealth_with_initial
        / running_peak
    )
    max_drawdown = np.max(
        drawdown,
        axis=1,
    )

    sharpe = float(
        np.mean(
            daily_returns
        )
        / np.std(
            daily_returns,
            ddof=1,
        )
        * math.sqrt(
            252.0
        )
    )

    return {
        "mean_terminal": float(
            np.mean(
                terminal
            )
        ),
        "median_terminal": float(
            np.median(
                terminal
            )
        ),
        "sharpe": sharpe,
        "p50_drawdown": float(
            np.median(
                max_drawdown
            )
        ),
    }


@dataclass(frozen=True)
class KellyVarSizingSkill:
    name: str = "kelly-var-sizing"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        del task_dir

        lowered = instruction.lower()
        required = (
            "kelly criterion",
            "var constraint",
            "returns.csv",
            "params.json",
            "median_terminal_wealth_full",
            "portfolio_var_full",
            "var_scale_factor",
        )

        return all(
            token in lowered
            for token in required
        )

    def solve(
        self,
        *,
        instruction: str,
        task_dir: Path,
        out_dir: Path,
        seed: int,
    ) -> None:
        del instruction, seed

        returns = pd.read_csv(
            _find_named(
                task_dir,
                "returns.csv",
            )
        )

        params = json.loads(
            _find_named(
                task_dir,
                "params.json",
            ).read_text(
                encoding="utf-8"
            )
        )

        asset_columns = [
            "asset_0",
            "asset_1",
            "asset_2",
        ]

        numeric = (
            returns[
                asset_columns
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
        )

        valid = numeric.dropna(
            how="any"
        ).reset_index(
            drop=True
        )

        num_valid = int(
            len(valid)
        )

        estimation_window = int(
            params.get(
                "estimation_window_days",
                len(valid),
            )
        )
        if estimation_window <= 0:
            raise RuntimeError(
                "estimation_window_days must be positive."
            )

        estimation = (
            valid.tail(
                estimation_window
            )
            .to_numpy(
                dtype=float
            )
        )

        mean = np.mean(
            estimation,
            axis=0,
        )
        covariance = np.cov(
            estimation,
            rowvar=False,
            ddof=1,
        )

        risk_free_daily = (
            float(
                params[
                    "risk_free_annual"
                ]
            )
            / 252.0
        )

        mean_excess = (
            mean
            - risk_free_daily
        )

        full_kelly = np.linalg.solve(
            covariance,
            mean_excess,
        )

        total_leverage = float(
            np.sum(
                np.abs(
                    full_kelly
                )
            )
        )

        confidence = float(
            params[
                "confidence_level"
            ]
        )
        z_score = float(
            norm.ppf(
                confidence
            )
        )

        portfolio_sigma = float(
            math.sqrt(
                float(
                    full_kelly
                    @ covariance
                    @ full_kelly
                )
            )
        )

        portfolio_var_full = float(
            z_score
            * portfolio_sigma
        )

        max_var_daily = float(
            params[
                "max_var_daily"
            ]
        )

        if portfolio_var_full <= 0.0:
            var_scale_factor = 1.0
        else:
            var_scale_factor = float(
                min(
                    1.0,
                    max_var_daily
                    / portfolio_var_full,
                )
            )

        var_kelly = (
            full_kelly
            * var_scale_factor
        )
        half_kelly = (
            0.5
            * full_kelly
        )
        equal_weight = np.full(
            3,
            1.0 / 3.0,
            dtype=float,
        )

        rng = np.random.default_rng(
            int(
                params["seed"]
            )
        )

        simulation_args = {
            "rng": rng,
            "mean": mean,
            "covariance": covariance,
            "initial_capital": float(
                params[
                    "initial_capital"
                ]
            ),
            "n_paths": int(
                params[
                    "n_simulation_paths"
                ]
            ),
            "n_days": int(
                params[
                    "n_days"
                ]
            ),
        }

        # Preserve the benchmark's declared four-scheme simulation order:
        # full, half, VaR-constrained, equal weight. A single Generator is
        # advanced across all four schemes.
        full_stats = _simulate_scheme(
            weights=full_kelly,
            **simulation_args,
        )
        _simulate_scheme(
            weights=half_kelly,
            **simulation_args,
        )
        var_stats = _simulate_scheme(
            weights=var_kelly,
            **simulation_args,
        )
        _simulate_scheme(
            weights=equal_weight,
            **simulation_args,
        )

        results = {
            "num_valid_observations": (
                num_valid
            ),
            "median_terminal_wealth_full": (
                full_stats[
                    "median_terminal"
                ]
            ),
            "median_terminal_wealth_var": (
                var_stats[
                    "median_terminal"
                ]
            ),
            "sharpe_full": (
                full_stats[
                    "sharpe"
                ]
            ),
            "sharpe_var": (
                var_stats[
                    "sharpe"
                ]
            ),
            "p50_drawdown_full": (
                full_stats[
                    "p50_drawdown"
                ]
            ),
            "p50_drawdown_var": (
                var_stats[
                    "p50_drawdown"
                ]
            ),
        }

        intermediates = {
            "mean_excess_0": {
                "value": float(
                    mean_excess[0]
                )
            },
            "mean_excess_1": {
                "value": float(
                    mean_excess[1]
                )
            },
            "mean_excess_2": {
                "value": float(
                    mean_excess[2]
                )
            },
            "cov_00": {
                "value": float(
                    covariance[
                        0,
                        0,
                    ]
                )
            },
            "cov_01": {
                "value": float(
                    covariance[
                        0,
                        1,
                    ]
                )
            },
            "cov_12": {
                "value": float(
                    covariance[
                        1,
                        2,
                    ]
                )
            },
            "kelly_fraction_0": {
                "value": float(
                    full_kelly[0]
                )
            },
            "kelly_fraction_1": {
                "value": float(
                    full_kelly[1]
                )
            },
            "kelly_fraction_2": {
                "value": float(
                    full_kelly[2]
                )
            },
            "total_kelly_leverage": {
                "value": (
                    total_leverage
                )
            },
            "portfolio_var_full": {
                "value": (
                    portfolio_var_full
                )
            },
            "var_scale_factor": {
                "value": (
                    var_scale_factor
                )
            },
            "mean_terminal_wealth_full": {
                "value": (
                    full_stats[
                        "mean_terminal"
                    ]
                )
            },
        }

        solution = {
            "intermediates": (
                intermediates
            )
        }

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            out_dir
            / "results.json"
        ).write_text(
            json.dumps(
                results,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (
            out_dir
            / "solution.json"
        ).write_text(
            json.dumps(
                solution,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
