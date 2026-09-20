from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from agent.finance_schema import TaskDataCatalog


@dataclass(frozen=True)
class OuEulerFit:
    intercept: float
    slope: float
    kappa: float
    mu: float
    sigma: float
    residual_std: float
    r_squared: float
    num_observations: int


@dataclass(frozen=True)
class OuExactFit:
    intercept: float
    phi: float
    kappa: float
    theta: float
    sigma: float
    residual_std: float
    r_squared: float
    residuals: np.ndarray
    num_observations: int


@dataclass(frozen=True)
class JumpParameters:
    intensity: float
    mean: float
    std: float
    count: int
    mask: np.ndarray


@dataclass(frozen=True)
class ProcessMoments:
    mean_x: float
    var_x: float
    mean_s: float
    var_s: float


def _finite_1d(values: Sequence[float], *, minimum: int = 3) -> np.ndarray:
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < minimum:
        raise RuntimeError(
            f"Process estimation requires at least {minimum} finite observations."
        )
    return x


def _ols_with_intercept(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, np.ndarray, float]:
    design = np.column_stack([np.ones(len(x), dtype=float), x])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coef
    residuals = y - fitted
    ss_res = float(residuals @ residuals)
    centered = y - float(np.mean(y))
    ss_tot = float(centered @ centered)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return float(coef[0]), float(coef[1]), residuals, float(r2)


def fit_ou_euler(values: Sequence[float], *, dt: float) -> OuEulerFit:
    """Plug-in Euler fit: dX = kappa(mu-X)dt + sigma dW."""
    x = _finite_1d(values)
    if dt <= 0.0:
        raise RuntimeError("OU dt must be positive.")
    lag = x[:-1]
    delta = x[1:] - x[:-1]
    intercept, slope, residuals, r2 = _ols_with_intercept(lag, delta)
    if abs(slope) < 1e-15:
        raise RuntimeError("OU Euler slope is numerically zero.")
    kappa = -slope / dt
    mu = -intercept / slope
    residual_std = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
    sigma = residual_std / math.sqrt(dt)
    return OuEulerFit(
        intercept=intercept,
        slope=slope,
        kappa=float(kappa),
        mu=float(mu),
        sigma=float(sigma),
        residual_std=residual_std,
        r_squared=r2,
        num_observations=int(residuals.size),
    )


def fit_ou_exact_ar1(values: Sequence[float], *, dt: float) -> OuExactFit:
    """Fit X[t+dt] = a + phi X[t] + eps and invert the exact OU transition."""
    x = _finite_1d(values)
    if dt <= 0.0:
        raise RuntimeError("OU dt must be positive.")
    lag = x[:-1]
    nxt = x[1:]
    intercept, phi, residuals, r2 = _ols_with_intercept(lag, nxt)
    if not 0.0 < phi < 1.0:
        raise RuntimeError(
            f"Exact OU calibration requires 0 < phi < 1; got {phi}."
        )
    kappa = -math.log(phi) / dt
    theta = intercept / (1.0 - phi)
    residual_std = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
    sigma = residual_std * math.sqrt(
        2.0 * kappa / max(1.0 - phi * phi, 1e-18)
    )
    return OuExactFit(
        intercept=intercept,
        phi=phi,
        kappa=float(kappa),
        theta=float(theta),
        sigma=float(sigma),
        residual_std=residual_std,
        r_squared=r2,
        residuals=np.asarray(residuals, dtype=float),
        num_observations=int(residuals.size),
    )


def ou_half_life(kappa: float) -> float:
    return float(math.log(2.0) / kappa) if kappa > 0.0 else float("nan")


def ou_stationary_std(kappa: float, sigma: float) -> float:
    return (
        float(sigma / math.sqrt(2.0 * kappa))
        if kappa > 0.0
        else float("nan")
    )


def simulate_ou_exact(
    *,
    initial: float,
    kappa: float,
    mu: float,
    sigma: float,
    dt: float,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """Generated OU states, excluding the initial state."""
    if (
        kappa <= 0.0
        or sigma < 0.0
        or dt <= 0.0
        or n_steps <= 0
        or n_paths <= 0
    ):
        raise RuntimeError("Invalid exact OU simulation parameters.")
    rng = np.random.default_rng(seed)
    decay = math.exp(-kappa * dt)
    step_std = sigma * math.sqrt(
        (1.0 - decay * decay) / (2.0 * kappa)
    )
    current = np.full(n_paths, float(initial), dtype=float)
    out = np.empty((n_paths, n_steps), dtype=float)
    for step in range(n_steps):
        current = (
            mu
            + (current - mu) * decay
            + step_std * rng.standard_normal(n_paths)
        )
        out[:, step] = current
    return out


def detect_ou_residual_jumps(
    residuals: Sequence[float],
    *,
    dt: float,
    z_threshold: float = 3.0,
) -> JumpParameters:
    r = _finite_1d(residuals, minimum=2)
    if dt <= 0.0 or z_threshold <= 0.0:
        raise RuntimeError("Invalid jump-detection parameters.")
    residual_std = float(np.std(r, ddof=1))
    if residual_std <= 0.0:
        mask = np.zeros(r.size, dtype=bool)
    else:
        mask = np.abs(r / residual_std) > z_threshold
    jumps = r[mask]
    count = int(mask.sum())
    intensity = count / (r.size * dt)
    jump_mean = float(np.mean(jumps)) if count else 0.0
    jump_std = float(np.std(jumps, ddof=1)) if count > 1 else 0.0
    return JumpParameters(
        intensity=float(intensity),
        mean=jump_mean,
        std=jump_std,
        count=count,
        mask=mask,
    )


def log_ou_jump_moments(
    *,
    x0: float,
    tau: float,
    kappa: float,
    theta: float,
    sigma: float,
    jump_intensity: float = 0.0,
    jump_mean: float = 0.0,
    jump_std: float = 0.0,
) -> ProcessMoments:
    if tau < 0.0 or kappa <= 0.0:
        raise RuntimeError(
            "Conditional moments require tau >= 0 and kappa > 0."
        )
    decay = math.exp(-kappa * tau)
    decay2 = math.exp(-2.0 * kappa * tau)
    jump_second = jump_std * jump_std + jump_mean * jump_mean
    mean_x = (
        theta
        + (x0 - theta) * decay
        + jump_intensity * jump_mean * (1.0 - decay) / kappa
    )
    var_x = (
        (sigma * sigma + jump_intensity * jump_second)
        * (1.0 - decay2)
        / (2.0 * kappa)
    )
    mean_s = math.exp(mean_x + 0.5 * var_x)
    var_s = (
        (math.exp(var_x) - 1.0)
        * math.exp(2.0 * mean_x + var_x)
    )
    return ProcessMoments(
        mean_x=float(mean_x),
        var_x=float(max(var_x, 0.0)),
        mean_s=float(mean_s),
        var_s=float(max(var_s, 0.0)),
    )


def log_ou_jump_stationary_moments(
    *,
    kappa: float,
    theta: float,
    sigma: float,
    jump_intensity: float,
    jump_mean: float,
    jump_std: float,
) -> tuple[float, float]:
    if kappa <= 0.0:
        return float("nan"), float("nan")
    second = jump_std * jump_std + jump_mean * jump_mean
    mean_x = theta + jump_intensity * jump_mean / kappa
    var_x = (
        sigma * sigma + jump_intensity * second
    ) / (2.0 * kappa)
    return float(mean_x), float(var_x)


def descriptive_statistics(
    values: Sequence[float],
) -> dict[str, float]:
    x = _finite_1d(values, minimum=2)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)),
        "skewness": float(stats.skew(x, bias=False)),
        "kurtosis": float(
            stats.kurtosis(x, fisher=True, bias=False)
        ),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "pct_positive": float(np.mean(x > 0.0) * 100.0),
    }


def sample_autocorrelation(
    values: Sequence[float],
    lag: int,
) -> float:
    x = _finite_1d(values, minimum=lag + 2)
    if lag <= 0 or lag >= x.size:
        raise RuntimeError("Invalid autocorrelation lag.")
    a = x[:-lag]
    b = x[lag:]
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def adf_constant(
    values: Sequence[float],
    *,
    maxlag: int = 24,
) -> tuple[float, float]:
    from statsmodels.tsa.stattools import adfuller

    x = _finite_1d(values, minimum=maxlag + 10)
    result = adfuller(
        x,
        maxlag=maxlag,
        regression="c",
        autolag="AIC",
    )
    return float(result[0]), float(result[1])


def empirical_loss_var_cvar_strict(
    annual_returns: Sequence[float],
    *,
    confidence: float,
) -> tuple[float, float]:
    r = _finite_1d(annual_returns, minimum=2)
    if not 0.5 < confidence < 1.0:
        raise RuntimeError("Confidence must lie in (0.5, 1).")
    losses = -r
    ordered = np.sort(losses)
    index = int(math.ceil(confidence * len(ordered)) - 1)
    index = min(max(index, 0), len(ordered) - 1)
    var = max(float(ordered[index]), 0.0)
    tail = losses[losses > var]
    cvar = float(np.mean(tail)) if tail.size else var
    return float(var), float(max(cvar, 0.0))


def _iso_z(value: pd.Timestamp) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.floor("s").strftime("%Y-%m-%dT%H:%M:%SZ")


def _regime_statistics(
    values: np.ndarray,
    *,
    dt: float,
) -> tuple[dict[str, float], OuEulerFit]:
    desc = descriptive_statistics(values)
    ac1 = sample_autocorrelation(values, 1)
    adf_stat, adf_p = adf_constant(values, maxlag=24)
    fit = fit_ou_euler(values, dt=dt)
    return {
        **desc,
        "autocorr_lag1": ac1,
        "adf_statistic": adf_stat,
        "adf_p_value": adf_p,
    }, fit


def run_funding_ou_carry_workflow(
    task_dir: Path,
    out_dir: Path,
) -> None:
    catalog = TaskDataCatalog.discover(task_dir)
    csv_path = catalog.csv_with_columns(
        {"symbol", "funding_time", "funding_rate"}
    )
    params_path = catalog.json_with_keys({
        "target_symbol",
        "analysis_start",
        "analysis_end",
        "regime_split_date",
        "annualization_periods",
        "ou_dt",
        "mc_num_paths",
        "mc_horizon_periods",
        "mc_random_seed",
        "var_confidence",
    })
    p = json.loads(params_path.read_text(encoding="utf-8"))

    frame = pd.read_csv(csv_path)
    frame["funding_time"] = pd.to_numeric(
        frame["funding_time"],
        errors="coerce",
    )
    frame["funding_rate"] = pd.to_numeric(
        frame["funding_rate"],
        errors="coerce",
    )
    frame = frame.dropna(
        subset=["funding_time", "funding_rate"]
    ).copy()
    frame["datetime"] = pd.to_datetime(
        frame["funding_time"].astype("int64"),
        unit="ms",
        utc=True,
        errors="coerce",
    ).dt.floor("s")
    frame = frame.dropna(subset=["datetime"])

    target = str(p["target_symbol"])
    start = pd.Timestamp(str(p["analysis_start"]), tz="UTC")
    end = (
        pd.Timestamp(str(p["analysis_end"]), tz="UTC")
        + pd.Timedelta(days=1)
    )
    frame = frame[
        (frame["symbol"].astype(str) == target)
        & (frame["datetime"] >= start)
        & (frame["datetime"] < end)
    ].sort_values("datetime").reset_index(drop=True)
    if len(frame) < 100:
        raise RuntimeError(
            "Funding-rate workflow found insufficient observations."
        )

    rates = frame["funding_rate"].to_numpy(dtype=float)
    annualization = float(p["annualization_periods"])
    dt = float(p["ou_dt"])

    desc = descriptive_statistics(rates)
    desc["annualized_mean"] = float(
        desc["mean"] * annualization
    )
    desc["annualized_volatility"] = float(
        desc["std"] * math.sqrt(annualization)
    )
    autocorr = {
        f"lag_{lag}": sample_autocorrelation(rates, lag)
        for lag in (1, 3, 6, 12, 24)
    }
    adf_stat, adf_p = adf_constant(rates, maxlag=24)
    full_fit = fit_ou_euler(rates, dt=dt)

    split = pd.Timestamp(
        str(p["regime_split_date"]),
        tz="UTC",
    )
    mask1 = frame["datetime"] < split
    mask2 = ~mask1
    rates1 = frame.loc[
        mask1, "funding_rate"
    ].to_numpy(dtype=float)
    rates2 = frame.loc[
        mask2, "funding_rate"
    ].to_numpy(dtype=float)
    reg1, fit1 = _regime_statistics(rates1, dt=dt)
    reg2, fit2 = _regime_statistics(rates2, dt=dt)

    n_paths = int(p["mc_num_paths"])
    n_steps = int(p["mc_horizon_periods"])
    seed = int(p["mc_random_seed"])
    simulated = simulate_ou_exact(
        initial=float(rates[-1]),
        kappa=full_fit.kappa,
        mu=full_fit.mu,
        sigma=full_fit.sigma,
        dt=dt,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )
    path_average_rates = np.mean(simulated, axis=1)
    cumulative_income = np.sum(simulated, axis=1)
    annual_returns = (
        cumulative_income
        * annualization
        / float(n_steps)
    )
    confidence = float(p["var_confidence"])
    var, cvar = empirical_loss_var_cvar_strict(
        annual_returns,
        confidence=confidence,
    )
    ret_pct = np.percentile(
        annual_returns,
        [5, 25, 50, 75, 95],
    )
    avg_pct = np.percentile(
        path_average_rates,
        [5, 50, 95],
    )

    half_years = ou_half_life(full_fit.kappa)
    half_periods = (
        half_years / dt
        if np.isfinite(half_years)
        else float("nan")
    )
    half_days = (
        half_years * 365.0
        if np.isfinite(half_years)
        else float("nan")
    )
    stationary_std = ou_stationary_std(
        full_fit.kappa,
        full_fit.sigma,
    )

    period1_label = f"bear_{start.year}"
    period2_label = f"recovery_{split.year}"
    regime_analysis = {
        "period1": {
            "label": period1_label,
            "mean": reg1["mean"],
            "std": reg1["std"],
            "skewness": reg1["skewness"],
            "kurtosis": reg1["kurtosis"],
            "pct_positive": reg1["pct_positive"],
            "autocorr_lag1": reg1["autocorr_lag1"],
            "adf_statistic": reg1["adf_statistic"],
            "adf_p_value": reg1["adf_p_value"],
            "ou_kappa": fit1.kappa,
            "ou_sigma": fit1.sigma,
        },
        "period2": {
            "label": period2_label,
            "mean": reg2["mean"],
            "std": reg2["std"],
            "skewness": reg2["skewness"],
            "kurtosis": reg2["kurtosis"],
            "pct_positive": reg2["pct_positive"],
            "autocorr_lag1": reg2["autocorr_lag1"],
            "adf_statistic": reg2["adf_statistic"],
            "adf_p_value": reg2["adf_p_value"],
            "ou_kappa": fit2.kappa,
            "ou_sigma": fit2.sigma,
        },
        "mean_ratio_p2_over_p1": float(
            reg2["mean"] / reg1["mean"]
        ),
        "autocorr_diff": float(
            reg2["autocorr_lag1"]
            - reg1["autocorr_lag1"]
        ),
    }

    mc_section = {
        "mean_annual_return": float(
            np.mean(annual_returns)
        ),
        "std_annual_return": float(
            np.std(annual_returns, ddof=0)
        ),
        "var_95": var,
        "cvar_95": cvar,
        "pct_negative_return": float(
            np.mean(annual_returns < 0.0) * 100.0
        ),
        "percentiles": {
            "p5": float(ret_pct[0]),
            "p25": float(ret_pct[1]),
            "p50": float(ret_pct[2]),
            "p75": float(ret_pct[3]),
            "p95": float(ret_pct[4]),
        },
        "mc_num_paths": n_paths,
        "mc_seed": seed,
        "initial_funding_rate": float(rates[-1]),
    }

    results = {
        "symbol": target,
        "num_observations": int(len(frame)),
        "date_range": {
            "start": _iso_z(
                frame["datetime"].iloc[0]
            ),
            "end": _iso_z(
                frame["datetime"].iloc[-1]
            ),
        },
        "descriptive_stats": desc,
        "autocorrelation": autocorr,
        "adf_test": {
            "statistic": adf_stat,
            "p_value": adf_p,
            "is_stationary": bool(adf_p < 0.05),
        },
        "regime_analysis": regime_analysis,
        "ou_parameters": {
            "kappa": full_fit.kappa,
            "mu": full_fit.mu,
            "sigma": full_fit.sigma,
            "half_life_periods": half_periods,
            "half_life_days": half_days,
            "stationary_std": stationary_std,
            "ols_intercept": full_fit.intercept,
            "ols_slope": full_fit.slope,
            "residual_std": full_fit.residual_std,
            "num_observations_fit": (
                full_fit.num_observations
            ),
        },
        "mc_basis_carry": mc_section,
    }

    solution = {
        "intermediates": {
            "num_observations": {
                "value": int(len(frame))
            },
            "mean_funding_rate": {
                "value": desc["mean"]
            },
            "std_funding_rate": {
                "value": desc["std"]
            },
            "annualized_mean": {
                "value": desc["annualized_mean"]
            },
            "adf_statistic": {
                "value": adf_stat
            },
            "regime1_mean": {
                "value": reg1["mean"]
            },
            "regime2_mean": {
                "value": reg2["mean"]
            },
            "ou_kappa": {
                "value": full_fit.kappa
            },
            "ou_mu": {
                "value": full_fit.mu
            },
            "ou_sigma": {
                "value": full_fit.sigma
            },
            "half_life_periods": {
                "value": half_periods
            },
            "half_life_days": {
                "value": half_days
            },
            "stationary_std": {
                "value": stationary_std
            },
            "mc_mean_annual_return": {
                "value": mc_section[
                    "mean_annual_return"
                ]
            },
            "mc_var_95": {"value": var},
            "mc_cvar_95": {"value": cvar},
            "mc_pct_negative": {
                "value": mc_section[
                    "pct_negative_return"
                ]
            },
        }
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    clean = pd.DataFrame({
        "datetime": frame["datetime"].map(_iso_z),
        "symbol": frame["symbol"].astype(str),
        "funding_rate": (
            frame["funding_rate"].astype(float)
        ),
        "funding_rate_bps": (
            frame["funding_rate"].astype(float)
            * 10000.0
        ),
    })
    clean.to_csv(
        out_dir / "funding_clean.csv",
        index=False,
    )

    regime_rows = []
    for period, label, mask, reg, fit in (
        ("period1", period1_label, mask1, reg1, fit1),
        ("period2", period2_label, mask2, reg2, fit2),
    ):
        subset = frame.loc[mask]
        regime_rows.append({
            "period": period,
            "label": label,
            "start_date": str(
                subset["datetime"].iloc[0].date()
            ),
            "end_date": str(
                subset["datetime"].iloc[-1].date()
            ),
            "num_obs": int(len(subset)),
            "mean": reg["mean"],
            "std": reg["std"],
            "skewness": reg["skewness"],
            "kurtosis": reg["kurtosis"],
            "pct_positive": reg["pct_positive"],
            "autocorr_lag1": (
                reg["autocorr_lag1"]
            ),
            "adf_statistic": (
                reg["adf_statistic"]
            ),
            "adf_p_value": reg["adf_p_value"],
            "ou_kappa": fit.kappa,
            "ou_sigma": fit.sigma,
        })
    pd.DataFrame(regime_rows).to_csv(
        out_dir / "regime_stats.csv",
        index=False,
    )

    ou_diag = {
        "full_sample": {
            "kappa": full_fit.kappa,
            "mu": full_fit.mu,
            "sigma": full_fit.sigma,
            "ols_intercept": full_fit.intercept,
            "ols_slope": full_fit.slope,
            "ols_r_squared": full_fit.r_squared,
            "residual_std": full_fit.residual_std,
            "num_observations": (
                full_fit.num_observations
            ),
        },
        "period1_bear": {
            "kappa": fit1.kappa,
            "sigma": fit1.sigma,
            "num_observations": fit1.num_observations,
        },
        "period2_recovery": {
            "kappa": fit2.kappa,
            "sigma": fit2.sigma,
            "num_observations": fit2.num_observations,
        },
    }
    mc_stats = {
        "mc_num_paths": n_paths,
        "mc_random_seed": seed,
        "mc_horizon_periods": n_steps,
        "initial_funding_rate": float(rates[-1]),
        "ou_params_used": {
            "kappa": full_fit.kappa,
            "mu": full_fit.mu,
            "sigma": full_fit.sigma,
        },
        "simulated_funding_rate_distribution": {
            "mean": float(
                np.mean(path_average_rates)
            ),
            "std": float(
                np.std(
                    path_average_rates,
                    ddof=0,
                )
            ),
            "p5": float(avg_pct[0]),
            "p50": float(avg_pct[1]),
            "p95": float(avg_pct[2]),
        },
        "annual_carry_return_distribution": {
            "mean": mc_section[
                "mean_annual_return"
            ],
            "std": mc_section[
                "std_annual_return"
            ],
            "var_95": var,
            "cvar_95": cvar,
            "pct_negative": mc_section[
                "pct_negative_return"
            ],
            **mc_section["percentiles"],
        },
    }

    for name, payload in (
        ("results.json", results),
        ("solution.json", solution),
        ("ou_diagnostics.json", ou_diag),
        ("mc_stats.json", mc_stats),
    ):
        (out_dir / name).write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )


def _simulate_log_ou_jump_euler(
    *,
    x0: float,
    horizon: float,
    dt: float,
    n_paths: int,
    kappa: float,
    theta: float,
    sigma: float,
    jump_intensity: float,
    jump_mean: float,
    jump_std: float,
) -> np.ndarray:
    n_steps = int(round(horizon / dt))
    x = np.full(n_paths, float(x0), dtype=float)
    root_dt = math.sqrt(dt)
    for _ in range(n_steps):
        diffusion = (
            sigma
            * root_dt
            * np.random.normal(size=n_paths)
        )
        counts = np.random.poisson(
            jump_intensity * dt,
            size=n_paths,
        )
        jump = np.zeros(n_paths, dtype=float)
        active = counts > 0
        if np.any(active):
            c = counts[active].astype(float)
            jump[active] = np.random.normal(
                loc=c * jump_mean,
                scale=np.sqrt(c) * jump_std,
            )
        x = (
            x
            + kappa * (theta - x) * dt
            + diffusion
            + jump
        )
    return x


def run_log_ou_jump_moments_workflow(
    task_dir: Path,
    out_dir: Path,
) -> None:
    catalog = TaskDataCatalog.discover(task_dir)
    csv_path = catalog.csv_with_columns({"dgs10"})
    frame = pd.read_csv(csv_path)
    rates = (
        pd.to_numeric(
            frame["DGS10"],
            errors="coerce",
        )
        / 100.0
    )
    rates = rates[
        np.isfinite(rates) & (rates > 0.0)
    ].to_numpy(dtype=float)
    if rates.size < 100:
        raise RuntimeError(
            "Log-OU jump workflow found "
            "insufficient positive rates."
        )

    dt = 1.0 / 252.0
    x = np.log(rates)
    fit = fit_ou_exact_ar1(x, dt=dt)
    jumps = detect_ou_residual_jumps(
        fit.residuals,
        dt=dt,
        z_threshold=3.0,
    )
    stationary_mean, stationary_var = (
        log_ou_jump_stationary_moments(
            kappa=fit.kappa,
            theta=fit.theta,
            sigma=fit.sigma,
            jump_intensity=jumps.intensity,
            jump_mean=jumps.mean,
            jump_std=jumps.std,
        )
    )

    horizons = (0.25, 0.50, 1.00, 2.00, 5.00)
    x0 = float(x[-1])
    n_paths = 10000

    np.random.seed(42)
    moment_rows = []
    forward_rows = []
    mean_errors = []
    var_errors = []

    for tau in horizons:
        analytical = log_ou_jump_moments(
            x0=x0,
            tau=tau,
            kappa=fit.kappa,
            theta=fit.theta,
            sigma=fit.sigma,
            jump_intensity=jumps.intensity,
            jump_mean=jumps.mean,
            jump_std=jumps.std,
        )
        no_jump = log_ou_jump_moments(
            x0=x0,
            tau=tau,
            kappa=fit.kappa,
            theta=fit.theta,
            sigma=fit.sigma,
        )
        terminal = _simulate_log_ou_jump_euler(
            x0=x0,
            horizon=tau,
            dt=dt,
            n_paths=n_paths,
            kappa=fit.kappa,
            theta=fit.theta,
            sigma=fit.sigma,
            jump_intensity=jumps.intensity,
            jump_mean=jumps.mean,
            jump_std=jumps.std,
        )
        mc_mean = float(np.mean(terminal))
        mc_var = float(np.var(terminal, ddof=0))
        moment_rows.append({
            "tau": tau,
            "E_X": analytical.mean_x,
            "Var_X": analytical.var_x,
            "E_S": analytical.mean_s,
            "Var_S": analytical.var_s,
            "E_X_mc": mc_mean,
            "Var_X_mc": mc_var,
        })
        forward_rows.append({
            "tau": tau,
            "forward_rate_jd": (
                analytical.mean_s
            ),
            "forward_rate_ou": no_jump.mean_s,
        })
        mean_errors.append(
            abs(
                analytical.mean_x - mc_mean
            )
            / max(
                abs(analytical.mean_x),
                1e-12,
            )
        )
        var_errors.append(
            abs(
                analytical.var_x - mc_var
            )
            / max(
                analytical.var_x,
                1e-12,
            )
        )

    one_year = next(
        row
        for row in forward_rows
        if abs(row["tau"] - 1.0) < 1e-12
    )
    half_life = ou_half_life(fit.kappa)

    calibration = {
        "n_obs": int(rates.size),
        "n_changes": int(rates.size - 1),
        "rate_mean": float(np.mean(rates)),
        "rate_std": float(
            np.std(rates, ddof=1)
        ),
        "log_rate_mean": float(np.mean(x)),
        "log_rate_std": float(
            np.std(x, ddof=1)
        ),
        "kappa": fit.kappa,
        "theta": fit.theta,
        "sigma": fit.sigma,
        "lambda": jumps.intensity,
        "mu_J": jumps.mean,
        "sigma_J": jumps.std,
        "n_jumps": jumps.count,
        "x0": x0,
        "S0": float(rates[-1]),
        "stationary_mean_X": stationary_mean,
        "stationary_var_X": stationary_var,
    }

    summary = {
        "kappa": fit.kappa,
        "theta": fit.theta,
        "sigma": fit.sigma,
        "lambda_annual": jumps.intensity,
        "half_life_years": half_life,
        "stationary_vol_X": float(
            math.sqrt(
                max(stationary_var, 0.0)
            )
        ),
        "feller_ratio": (
            float(
                2.0
                * fit.kappa
                * fit.theta
                / (fit.sigma * fit.sigma)
            )
            if fit.sigma > 0.0
            else float("nan")
        ),
        "max_mc_mean_relerr": float(
            max(mean_errors)
        ),
        "max_mc_var_relerr": float(
            max(var_errors)
        ),
        "jump_impact_on_forward": float(
            abs(
                one_year["forward_rate_jd"]
                - one_year["forward_rate_ou"]
            )
            / max(
                abs(
                    one_year[
                        "forward_rate_ou"
                    ]
                ),
                1e-12,
            )
        ),
        "mean_reversion_confirmed": bool(
            fit.kappa > 0.0
            and half_life < 20.0
        ),
        "jumps_detected": bool(
            jumps.count > 0
        ),
    }

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    (out_dir / "calibration.json").write_text(
        json.dumps(
            calibration,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(moment_rows).to_csv(
        out_dir / "conditional_moments.csv",
        index=False,
    )
    pd.DataFrame(forward_rows).to_csv(
        out_dir / "forward_curve.csv",
        index=False,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )
