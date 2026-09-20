from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _find_named_file(
    task_dir: Path,
    filename: str,
) -> Path:
    matches = [
        path
        for path in task_dir.rglob(filename)
        if path.is_file() and "checks" not in path.parts
    ]
    if not matches:
        raise RuntimeError(
            f"Offline CIR skill could not find {filename}."
        )
    return sorted(matches)[0]


def _clip(
    value: float,
    lower: float,
    upper: float,
) -> float:
    return float(min(upper, max(lower, value)))


def _fallback_parameters(
    rates: np.ndarray,
    dts: np.ndarray,
) -> tuple[float, float, float]:
    previous = rates[:-1]
    following = rates[1:]
    dt_mean = float(np.mean(dts))

    delta = following - previous
    X = np.column_stack(
        (np.ones(len(previous)), previous)
    )
    beta, *_ = np.linalg.lstsq(
        X,
        delta,
        rcond=None,
    )

    intercept = float(beta[0])
    slope = float(beta[1])

    kappa = _clip(
        -slope / max(dt_mean, 1e-12),
        0.01,
        1.0,
    )

    theta = _clip(
        (
            intercept
            / (
                kappa
                * max(dt_mean, 1e-12)
            )
        )
        if kappa > 0.0
        else float(np.mean(rates)),
        0.001,
        0.10,
    )

    fitted = (
        kappa
        * (theta - previous)
        * dts
    )
    residual = delta - fitted

    scaled = (
        residual
        / np.sqrt(
            np.maximum(previous, 1e-8)
            * np.maximum(dts, 1e-12)
        )
    )

    sigma = _clip(
        float(np.std(scaled, ddof=1)),
        0.001,
        0.20,
    )

    return kappa, theta, sigma


def _cir_log_likelihood(
    parameters: tuple[float, float, float] | np.ndarray,
    rates: np.ndarray,
    dts: np.ndarray,
) -> float:
    from scipy.stats import ncx2

    kappa = float(parameters[0])
    theta = float(parameters[1])
    sigma = float(parameters[2])

    if kappa <= 0.0 or theta <= 0.0 or sigma <= 0.0:
        return float("-inf")

    previous = np.maximum(rates[:-1], 1e-12)
    following = np.maximum(rates[1:], 1e-12)

    exp_term = np.exp(-kappa * dts)
    one_minus = 1.0 - exp_term

    if np.any(one_minus <= 0.0):
        return float("-inf")

    scale = (
        sigma
        * sigma
        * one_minus
        / (4.0 * kappa)
    )

    degrees_freedom = (
        4.0
        * kappa
        * theta
        / (sigma * sigma)
    )

    noncentrality = (
        4.0
        * kappa
        * exp_term
        * previous
        / (
            sigma
            * sigma
            * one_minus
        )
    )

    x = following / scale

    log_density = (
        ncx2.logpdf(
            x,
            degrees_freedom,
            noncentrality,
        )
        - np.log(scale)
    )

    if not np.all(np.isfinite(log_density)):
        return float("-inf")

    return float(np.sum(log_density))


def _calibrate_cir(
    *,
    rates: np.ndarray,
    dates: pd.Series,
    seed: int,
) -> tuple[float, float, float, float]:
    from scipy.optimize import minimize

    day_differences = (
        dates.diff()
        .dt.total_seconds()
        .to_numpy()
        / 86400.0
    )[1:]

    dts = day_differences / 365.0

    if np.any(dts <= 0.0):
        raise RuntimeError(
            "CIR calibration dates must be strictly increasing."
        )

    fallback = _fallback_parameters(
        rates,
        dts,
    )

    rng = np.random.default_rng(seed)

    starts: list[tuple[float, float, float]] = [
        fallback,
        (
            0.30,
            _clip(
                float(np.mean(rates)),
                0.001,
                0.10,
            ),
            0.04,
        ),
    ]

    while len(starts) < 7:
        starts.append(
            (
                float(rng.uniform(0.01, 1.0)),
                float(rng.uniform(0.001, 0.10)),
                float(rng.uniform(0.001, 0.20)),
            )
        )

    bounds = (
        (0.01, 1.0),
        (0.001, 0.10),
        (0.001, 0.20),
    )

    best_parameters = None
    best_log_likelihood = float("-inf")

    def objective(parameters: np.ndarray) -> float:
        log_likelihood = _cir_log_likelihood(
            parameters,
            rates,
            dts,
        )
        if not math.isfinite(log_likelihood):
            return 1e100
        return float(-log_likelihood)

    for start in starts:
        result = minimize(
            objective,
            x0=np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 500,
                "ftol": 1e-10,
            },
        )

        parameters = (
            tuple(float(value) for value in result.x)
            if result.x is not None and len(result.x) == 3
            else start
        )

        log_likelihood = _cir_log_likelihood(
            parameters,
            rates,
            dts,
        )

        if (
            math.isfinite(log_likelihood)
            and log_likelihood > best_log_likelihood
        ):
            best_parameters = parameters
            best_log_likelihood = log_likelihood

    if best_parameters is None:
        best_parameters = fallback
        best_log_likelihood = _cir_log_likelihood(
            fallback,
            rates,
            dts,
        )

    if not math.isfinite(best_log_likelihood):
        best_log_likelihood = -999999.0

    return (
        float(best_parameters[0]),
        float(best_parameters[1]),
        float(best_parameters[2]),
        float(best_log_likelihood),
    )


def _bond_price(
    *,
    tau: float,
    r_current: float,
    kappa: float,
    theta: float,
    sigma: float,
) -> float:
    gamma = math.sqrt(
        kappa * kappa
        + 2.0 * sigma * sigma
    )

    exponential = math.exp(gamma * tau)

    denominator = (
        (gamma + kappa)
        * (exponential - 1.0)
        + 2.0 * gamma
    )

    B = (
        2.0
        * (exponential - 1.0)
        / denominator
    )

    A_base = (
        2.0
        * gamma
        * math.exp(
            (kappa + gamma)
            * tau
            / 2.0
        )
        / denominator
    )

    exponent = (
        2.0
        * kappa
        * theta
        / (sigma * sigma)
    )

    A = A_base ** exponent

    return float(
        A
        * math.exp(
            -B * r_current
        )
    )


@dataclass(frozen=True)
class CirBondPricingSkill:
    name: str = "cir-bond-pricing"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        del task_dir

        lowered = instruction.lower()

        required = (
            "cox-ingersoll-ross",
            "zero-coupon",
            "feller",
            "bond_prices.csv",
            "forward_rates.csv",
            "dff",
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
        del instruction

        rates_path = _find_named_file(
            task_dir,
            "fred_rates.csv",
        )

        frame = pd.read_csv(rates_path)

        by_lower = {
            str(column).lower(): str(column)
            for column in frame.columns
        }

        if "date" not in by_lower or "dff" not in by_lower:
            raise RuntimeError(
                "fred_rates.csv requires date and DFF columns."
            )

        dates = pd.to_datetime(
            frame[by_lower["date"]],
            errors="coerce",
        )

        dff = pd.to_numeric(
            frame[by_lower["dff"]],
            errors="coerce",
        )

        prepared = pd.DataFrame(
            {
                "date": dates,
                "DFF": dff,
            }
        )

        prepared = prepared[
            prepared["date"]
            >= pd.Timestamp("2016-01-01")
        ].dropna(
            subset=["date", "DFF"]
        ).sort_values(
            "date",
            kind="stable",
        ).reset_index(drop=True)

        if len(prepared) < 2:
            raise RuntimeError(
                "Not enough DFF observations for CIR calibration."
            )

        rates = (
            prepared["DFF"]
            .astype(float)
            .to_numpy()
            / 100.0
        )

        (
            kappa,
            theta,
            sigma,
            log_likelihood,
        ) = _calibrate_cir(
            rates=rates,
            dates=prepared["date"],
            seed=seed,
        )

        r_current = float(rates[-1])
        r_mean = float(np.mean(rates))
        r_std = float(np.std(rates, ddof=1))
        r_min = float(np.min(rates))
        r_max = float(np.max(rates))

        feller_numerator = 2.0 * kappa * theta
        feller_denominator = sigma * sigma

        feller_ratio = float(
            feller_numerator
            / feller_denominator
        )

        feller_satisfied = bool(
            feller_numerator
            >= feller_denominator
        )

        maturities = (
            0.25,
            0.50,
            1.0,
            2.0,
            3.0,
            5.0,
            7.0,
            10.0,
        )

        market_columns = {
            1.0: "DGS1",
            2.0: "DGS2",
            5.0: "DGS5",
            10.0: "DGS10",
        }

        last_source_row = frame.iloc[-1]

        bond_rows = []
        price_by_tau = {}
        yield_errors = []

        for tau in maturities:
            price = _bond_price(
                tau=tau,
                r_current=r_current,
                kappa=kappa,
                theta=theta,
                sigma=sigma,
            )

            if (
                not math.isfinite(price)
                or price <= 0.0
            ):
                raise RuntimeError(
                    f"Invalid CIR bond price at tau={tau}: {price}"
                )

            zero_rate = float(-math.log(price))
            model_yield = float(zero_rate / tau)

            market_yield = float("nan")

            market_column = market_columns.get(float(tau))

            if (
                market_column is not None
                and market_column in frame.columns
            ):
                raw_market = pd.to_numeric(
                    pd.Series(
                        [
                            last_source_row[
                                market_column
                            ]
                        ]
                    ),
                    errors="coerce",
                ).iloc[0]

                if pd.notna(raw_market):
                    market_yield = float(raw_market) / 100.0

            yield_diff = float("nan")

            if math.isfinite(market_yield):
                yield_diff = float(
                    model_yield
                    - market_yield
                )
                yield_errors.append(
                    abs(yield_diff)
                )

            price_by_tau[float(tau)] = price

            bond_rows.append(
                {
                    "tau": float(tau),
                    "zcb_price": price,
                    "zero_rate": zero_rate,
                    "model_yield": model_yield,
                    "market_yield": market_yield,
                    "yield_diff": yield_diff,
                }
            )

        forward_rows = []

        for tau1, tau2 in zip(
            maturities[:-1],
            maturities[1:],
        ):
            P1 = price_by_tau[float(tau1)]
            P2 = price_by_tau[float(tau2)]

            forward_rate = float(
                -(
                    math.log(P2)
                    - math.log(P1)
                )
                / (tau2 - tau1)
            )

            forward_rows.append(
                {
                    "tau1": float(tau1),
                    "tau2": float(tau2),
                    "forward_rate": forward_rate,
                }
            )

        calibration = {
            "n_obs": int(len(prepared)),
            "kappa": kappa,
            "theta": theta,
            "sigma": sigma,
            "log_likelihood": log_likelihood,
            "feller_condition_satisfied": feller_satisfied,
            "feller_ratio": feller_ratio,
            "r_current": r_current,
            "r_mean": r_mean,
            "r_std": r_std,
            "r_min": r_min,
            "r_max": r_max,
        }

        mean_yield_error = (
            float(np.mean(yield_errors))
            if yield_errors
            else 0.0
        )

        max_yield_error = (
            float(np.max(yield_errors))
            if yield_errors
            else 0.0
        )

        summary = {
            "kappa": kappa,
            "theta": theta,
            "sigma": sigma,
            "r_current": r_current,
            "feller_satisfied": feller_satisfied,
            "mean_yield_error": mean_yield_error,
            "max_yield_error": max_yield_error,
            "yield_curve_upward_sloping": bool(
                bond_rows[-1]["model_yield"]
                > bond_rows[0]["model_yield"]
            ),
        }

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            out_dir
            / "calibration.json"
        ).write_text(
            json.dumps(
                calibration,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        pd.DataFrame(
            bond_rows
        ).to_csv(
            out_dir
            / "bond_prices.csv",
            index=False,
        )

        pd.DataFrame(
            forward_rows
        ).to_csv(
            out_dir
            / "forward_rates.csv",
            index=False,
        )

        (
            out_dir
            / "summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
