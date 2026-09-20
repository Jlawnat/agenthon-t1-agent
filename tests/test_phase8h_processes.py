from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent.finance_composition import plan_finance_task
from agent.finance_processes import (
    fit_ou_euler,
    fit_ou_exact_ar1,
    log_ou_jump_moments,
    simulate_ou_exact,
)


def _task(tmp_path: Path) -> Path:
    data = tmp_path / "task" / "environment" / "data"
    data.mkdir(parents=True)
    return data.parent.parent


def test_ou_euler_recovers_plugin_mapping() -> None:
    x = np.array([
        0.02, 0.021, 0.0206, 0.0208,
        0.0204, 0.0207, 0.0205,
    ])
    fit = fit_ou_euler(x, dt=1.0 / 1095.0)
    assert np.isfinite(fit.kappa)
    assert np.isfinite(fit.mu)
    assert fit.sigma >= 0.0
    assert fit.num_observations == len(x) - 1


def test_exact_ou_ar1_mapping() -> None:
    values = [0.0]
    for _ in range(20):
        values.append(0.2 + 0.8 * values[-1])
    fit = fit_ou_exact_ar1(values, dt=0.25)
    assert abs(fit.phi - 0.8) < 1e-12
    assert abs(fit.theta - 1.0) < 1e-12
    assert abs(
        fit.kappa - (-np.log(0.8) / 0.25)
    ) < 1e-12


def test_log_ou_moments_reduce_to_standard_ou_without_jumps() -> None:
    m = log_ou_jump_moments(
        x0=-3.5,
        tau=1.0,
        kappa=0.7,
        theta=-3.0,
        sigma=0.2,
    )
    expected_mean = (
        -3.0
        + (-0.5) * np.exp(-0.7)
    )
    expected_var = (
        0.2**2
        * (1.0 - np.exp(-1.4))
        / (2.0 * 0.7)
    )
    assert abs(
        m.mean_x - expected_mean
    ) < 1e-12
    assert abs(
        m.var_x - expected_var
    ) < 1e-12
    assert m.mean_s > 0.0
    assert m.var_s >= 0.0


def test_exact_ou_simulation_is_deterministic_with_zero_sigma() -> None:
    path = simulate_ou_exact(
        initial=1.0,
        kappa=2.0,
        mu=0.5,
        sigma=0.0,
        dt=0.1,
        n_steps=4,
        n_paths=3,
        seed=42,
    )
    expected = (
        0.5
        + (1.0 - 0.5)
        * np.exp(-2.0 * 0.1)
    )
    assert path.shape == (3, 4)
    assert np.allclose(
        path[:, 0],
        expected,
    )


def test_planner_recognises_funding_ou_carry_from_semantics_and_schema(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    data = task / "environment" / "data"
    pd.DataFrame({
        "symbol": ["BTCUSDT"],
        "funding_time": [1],
        "funding_rate": [0.0001],
    }).to_csv(
        data / "series.csv",
        index=False,
    )
    (data / "settings.json").write_text(
        json.dumps({
            "target_symbol": "BTCUSDT",
            "analysis_start": "2022-01-01",
            "analysis_end": "2023-12-31",
            "regime_split_date": "2023-01-01",
            "annualization_periods": 1095,
            "ou_dt": 1.0 / 1095.0,
            "mc_num_paths": 100,
            "mc_horizon_periods": 20,
            "mc_random_seed": 42,
            "var_confidence": 0.95,
        })
    )
    plan = plan_finance_task(
        "Analyze funding rates, run an ADF stationarity test, "
        "fit an Ornstein-Uhlenbeck model, then use the exact "
        "OU transition in Monte Carlo for basis carry VaR and CVaR.",
        task,
    )
    assert (
        plan.executable_recipe
        == "funding-ou-carry-analysis"
    )
    assert {
        "ou_calibration",
        "stationarity_test",
        "exact_ou_simulation",
        "basis_carry_risk",
    }.issubset(
        plan.capabilities
    )


def test_planner_recognises_log_ou_jump_model_from_semantics_and_schema(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    data = task / "environment" / "data"
    pd.DataFrame({
        "date": [
            "2026-01-01",
            "2026-01-02",
        ],
        "DGS10": [4.0, 4.1],
    }).to_csv(
        data / "rates.csv",
        index=False,
    )
    plan = plan_finance_task(
        "Calibrate a geometric mean-reverting jump-diffusion: "
        "OU in log-space with Poisson jumps, conditional moments, "
        "Monte Carlo verification and a forward curve.",
        task,
    )
    assert (
        plan.executable_recipe
        == "log-ou-jump-analysis"
    )
    assert {
        "ou_exact_calibration",
        "compound_poisson_jumps",
        "conditional_process_moments",
        "process_monte_carlo",
    }.issubset(
        plan.capabilities
    )


def test_poisson_does_not_accidentally_trigger_ois_curve_capability(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    plan = plan_finance_task(
        "Calibrate an OU process with Poisson jumps and "
        "run Monte Carlo simulation.",
        task,
    )
    assert "curve_bootstrap" not in plan.capabilities
