from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _find_rates(task_dir: Path) -> Path:
    preferred = task_dir / "environment" / "data" / "fred_rates.csv"
    if preferred.is_file():
        return preferred
    matches = [p for p in task_dir.rglob("fred_rates.csv") if "checks" not in p.parts]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one fred_rates.csv, found {len(matches)}.")
    return matches[0]


def _fit_ar1(levels: np.ndarray, dt: float) -> dict[str, object]:
    x = levels[:-1]
    y = levels[1:]
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    intercept = float(coef[0])
    slope = float(coef[1])
    if not 0.0 < slope < 1.0:
        raise RuntimeError(f"OU AR(1) slope must be in (0,1), got {slope}.")

    fitted = intercept + slope * x
    residuals = y - fitted
    kappa = float(-math.log(slope) / dt)
    theta = float(intercept / (1.0 - slope))

    return {
        "intercept": intercept,
        "slope": slope,
        "residuals": residuals,
        "kappa": kappa,
        "theta": theta,
    }


def _continuous_sigma(resid_std: float, kappa: float, slope: float) -> float:
    return float(
        resid_std
        * math.sqrt(
            2.0 * kappa
            / max(1.0 - slope * slope, 1e-18)
        )
    )


@dataclass(frozen=True)
class OuJumpCommoditySkill:
    name: str = "ou-jump-commodity"

    def matches(self, *, instruction: str, task_dir: Path) -> bool:
        del task_dir
        lowered = instruction.lower()
        required = (
            "ornstein-uhlenbeck",
            "poisson",
            "fred_rates.csv",
            "dgs10",
            "3·std",
            "conditional_moments.csv",
            "stationary_distribution.json",
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

        frame = pd.read_csv(_find_rates(task_dir))
        if "DGS10" not in frame.columns:
            raise RuntimeError("fred_rates.csv is missing DGS10.")

        levels = pd.to_numeric(frame["DGS10"], errors="coerce").dropna().to_numpy(dtype=float)
        if levels.size < 100:
            raise RuntimeError("Too few valid DGS10 observations.")

        dt = 1.0 / 252.0
        changes = np.diff(levels)
        fit = _fit_ar1(levels, dt)
        residuals = np.asarray(fit["residuals"], dtype=float)
        kappa = float(fit["kappa"])
        theta = float(fit["theta"])
        slope = float(fit["slope"])

        residual_std_all = float(np.std(residuals, ddof=1))
        jump_mask = np.abs(residuals) > 3.0 * residual_std_all
        jump_resid = residuals[jump_mask]
        nonjump_resid = residuals[~jump_mask]

        n_jumps = int(jump_mask.sum())
        n_obs = int(changes.size)
        lambda_jump = float(n_jumps / (n_obs * dt))

        if n_jumps:
            mu_j = float(np.mean(jump_resid))
            sigma_j = float(np.std(jump_resid, ddof=1)) if n_jumps > 1 else 0.0
        else:
            mu_j = 0.0
            sigma_j = 0.0

        if nonjump_resid.size < 2:
            raise RuntimeError("Too few non-jump residuals.")
        residual_std_nonjump = float(np.std(nonjump_resid, ddof=1))
        sigma_ou = _continuous_sigma(
            residual_std_nonjump,
            kappa,
            slope,
        )

        s0 = float(levels[-1])
        mean_level = float(np.mean(levels))
        std_level = float(np.std(levels, ddof=1))

        second_jump_moment = sigma_j * sigma_j + mu_j * mu_j

        def moments(tau: float) -> tuple[float, float, float, float]:
            decay = math.exp(-kappa * tau)
            decay2 = math.exp(-2.0 * kappa * tau)

            mean_ou = theta + (s0 - theta) * decay
            var_ou = (
                sigma_ou * sigma_ou
                * (1.0 - decay2)
                / (2.0 * kappa)
            )

            jump_mean = (
                lambda_jump
                * mu_j
                * (1.0 - decay)
                / kappa
            )
            jump_var = (
                lambda_jump
                * second_jump_moment
                * (1.0 - decay2)
                / (2.0 * kappa)
            )

            return (
                float(mean_ou),
                float(var_ou),
                float(mean_ou + jump_mean),
                float(var_ou + jump_var),
            )

        taus = [1.0 / 12.0, 1.0 / 4.0, 1.0 / 2.0, 1.0]
        analytic = {tau: moments(tau) for tau in taus}

        n_paths = 100_000
        rng = np.random.RandomState(42)
        current = np.full(n_paths, s0, dtype=float)
        snapshots: dict[int, np.ndarray] = {}
        target_steps = {21, 63, 126, 252}
        p_jump = min(max(lambda_jump * dt, 0.0), 1.0)

        for step in range(1, 253):
            z = rng.standard_normal(n_paths)
            current += (
                kappa * (theta - current) * dt
                + sigma_ou * math.sqrt(dt) * z
            )

            if p_jump > 0.0:
                occurs = rng.random_sample(n_paths) < p_jump
                count = int(occurs.sum())
                if count:
                    if sigma_j > 0.0:
                        current[occurs] += rng.normal(
                            loc=mu_j,
                            scale=sigma_j,
                            size=count,
                        )
                    else:
                        current[occurs] += mu_j

            if step in target_steps:
                snapshots[step] = current.copy()

        rows = []
        mean_errors = []
        var_errors_pct = []
        step_by_tau = {
            1.0 / 12.0: 21,
            1.0 / 4.0: 63,
            1.0 / 2.0: 126,
            1.0: 252,
        }

        for tau in taus:
            mean_ou, var_ou, mean_jump, var_jump = analytic[tau]
            sample = snapshots[step_by_tau[tau]]
            mc_mean = float(np.mean(sample))
            mc_var = float(np.var(sample, ddof=1))

            mean_errors.append(abs(mc_mean - mean_jump))
            var_errors_pct.append(
                100.0 * abs(mc_var - var_jump) / max(abs(var_jump), 1e-12)
            )

            rows.append(
                {
                    "tau": tau,
                    "E_ST_ou": mean_ou,
                    "Var_ST_ou": var_ou,
                    "E_ST_jump": mean_jump,
                    "Var_ST_jump": var_jump,
                    "mc_mean": mc_mean,
                    "mc_var": mc_var,
                }
            )

        stationary_mean_ou = theta
        stationary_var_ou = sigma_ou * sigma_ou / (2.0 * kappa)
        stationary_mean_jump = theta + lambda_jump * mu_j / kappa
        stationary_var_jump = (
            sigma_ou * sigma_ou
            + lambda_jump * second_jump_moment
        ) / (2.0 * kappa)

        max_mean_error = float(max(mean_errors))
        max_var_error_pct = float(max(var_errors_pct))
        mc_validates = bool(
            max_mean_error < 0.05
            and max_var_error_pct < 20.0
        )

        calibration = {
            "n_observations": n_obs,
            "mean_level": mean_level,
            "std_level": std_level,
            "mean_daily_change": float(np.mean(changes)),
            "std_daily_change": float(np.std(changes, ddof=1)),
            "kappa": kappa,
            "theta": theta,
            "sigma_ou": sigma_ou,
            "lambda_jump": lambda_jump,
            "mu_J": mu_j,
            "sigma_J": sigma_j,
            "n_jumps_detected": n_jumps,
            "S0": s0,
        }

        stationary = {
            "stationary_mean_ou": float(stationary_mean_ou),
            "stationary_var_ou": float(stationary_var_ou),
            "stationary_mean_jump": float(stationary_mean_jump),
            "stationary_var_jump": float(stationary_var_jump),
            "empirical_mean": mean_level,
            "empirical_var": float(np.var(levels, ddof=1)),
        }

        summary = {
            "kappa": kappa,
            "theta": theta,
            "sigma_ou": sigma_ou,
            "lambda_jump": lambda_jump,
            "mu_J": mu_j,
            "sigma_J": sigma_j,
            "half_life_days": float(math.log(2.0) / kappa * 252.0),
            "n_jumps_detected": n_jumps,
            "mc_validates": mc_validates,
            "max_mc_mean_error": max_mean_error,
            "max_mc_var_error_pct": max_var_error_pct,
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "calibration.json").write_text(
            json.dumps(calibration, indent=2) + "\n",
            encoding="utf-8",
        )
        pd.DataFrame(rows).to_csv(
            out_dir / "conditional_moments.csv",
            index=False,
        )
        (out_dir / "stationary_distribution.json").write_text(
            json.dumps(stationary, indent=2) + "\n",
            encoding="utf-8",
        )
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
