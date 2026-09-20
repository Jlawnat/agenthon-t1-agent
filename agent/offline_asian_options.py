from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats


def _find_daily_price_csv(
    task_dir: Path,
) -> Path:
    for path in sorted(
        task_dir.rglob("*.csv")
    ):
        if "checks" in path.parts:
            continue

        try:
            sample = pd.read_csv(
                path,
                nrows=5,
            )
        except Exception:
            continue

        lower = {
            str(column).lower()
            for column
            in sample.columns
        }

        if {
            "date",
            "close",
        }.issubset(
            lower
        ):
            return path

    raise RuntimeError(
        "No daily date/close price CSV was found."
    )


def calibrate_gbm(
    frame: pd.DataFrame,
) -> dict[str, float | int]:
    by_lower = {
        str(column).lower(): str(column)
        for column
        in frame.columns
    }

    if not {
        "date",
        "close",
    }.issubset(
        by_lower
    ):
        raise RuntimeError(
            "GBM calibration requires date and close columns."
        )

    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                frame[
                    by_lower[
                        "date"
                    ]
                ],
                errors="raise",
            ),
            "close": pd.to_numeric(
                frame[
                    by_lower[
                        "close"
                    ]
                ],
                errors="raise",
            ).astype(float),
        }
    ).sort_values(
        "date",
        kind="stable",
    ).drop_duplicates(
        subset=["date"],
        keep="last",
    ).reset_index(
        drop=True,
    )

    if (
        len(data) < 3
        or (
            data["close"]
            <= 0.0
        ).any()
    ):
        raise RuntimeError(
            "GBM calibration requires at least three positive close prices."
        )

    close = data[
        "close"
    ].to_numpy(
        dtype=float
    )

    returns = np.log(
        close[
            1:
        ]
        / close[
            :-1
        ]
    )

    return {
        "n_prices": int(
            len(close)
        ),
        "sigma": float(
            np.std(
                returns,
                ddof=1,
            )
            * math.sqrt(
                252.0
            )
        ),
        "S0": float(
            close[
                -1
            ]
        ),
        "return_mean": float(
            np.mean(
                returns
            )
        ),
        "return_std": float(
            np.std(
                returns,
                ddof=1,
            )
        ),
    }


def monitoring_times(
    T: float,
    n_monitoring: int,
) -> np.ndarray:
    return np.linspace(
        T / n_monitoring,
        T,
        n_monitoring,
        dtype=float,
    )


def geometric_asian_call(
    *,
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_monitoring: int,
) -> float:
    times = monitoring_times(
        T,
        n_monitoring,
    )

    covariance = np.minimum.outer(
        times,
        times,
    )

    mu_log_g = float(
        math.log(
            S0
        )
        + (
            r
            - 0.5
            * sigma
            * sigma
        )
        * float(
            np.mean(
                times
            )
        )
    )

    var_log_g = float(
        sigma
        * sigma
        * np.sum(
            covariance
        )
        / (
            n_monitoring
            * n_monitoring
        )
    )

    if var_log_g <= 0.0:
        forward_g = math.exp(
            mu_log_g
        )
        return float(
            math.exp(
                -r
                * T
            )
            * max(
                forward_g
                - K,
                0.0,
            )
        )

    root_var = math.sqrt(
        var_log_g
    )

    d1 = (
        mu_log_g
        - math.log(
            K
        )
        + var_log_g
    ) / root_var

    d2 = (
        d1
        - root_var
    )

    expected_g = math.exp(
        mu_log_g
        + 0.5
        * var_log_g
    )

    return float(
        math.exp(
            -r
            * T
        )
        * (
            expected_g
            * stats.norm.cdf(
                d1
            )
            - K
            * stats.norm.cdf(
                d2
            )
        )
    )


def arithmetic_moments(
    *,
    S0: float,
    T: float,
    r: float,
    sigma: float,
    n_monitoring: int,
) -> tuple[float, float]:
    times = monitoring_times(
        T,
        n_monitoring,
    )

    m1 = float(
        S0
        * np.mean(
            np.exp(
                r
                * times
            )
        )
    )

    ti = times[
        :,
        None
    ]
    tj = times[
        None,
        :
    ]

    m2 = float(
        S0
        * S0
        * np.mean(
            np.exp(
                r
                * (
                    ti
                    + tj
                )
                + sigma
                * sigma
                * np.minimum(
                    ti,
                    tj,
                )
            )
        )
    )

    return (
        m1,
        m2,
    )


def levy_asian_call(
    *,
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_monitoring: int,
) -> float:
    m1, m2 = (
        arithmetic_moments(
            S0=S0,
            T=T,
            r=r,
            sigma=sigma,
            n_monitoring=n_monitoring,
        )
    )

    variance_log = float(
        math.log(
            max(
                m2
                / (
                    m1
                    * m1
                ),
                1.0,
            )
        )
    )

    if variance_log <= 1e-16:
        return float(
            math.exp(
                -r
                * T
            )
            * max(
                m1
                - K,
                0.0,
            )
        )

    root_var = math.sqrt(
        variance_log
    )

    d1 = (
        math.log(
            m1
            / K
        )
        + 0.5
        * variance_log
    ) / root_var

    d2 = (
        d1
        - root_var
    )

    return float(
        math.exp(
            -r
            * T
        )
        * (
            m1
            * stats.norm.cdf(
                d1
            )
            - K
            * stats.norm.cdf(
                d2
            )
        )
    )


def curran_asian_call(
    *,
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_monitoring: int,
) -> float:
    times = monitoring_times(
        T,
        n_monitoring,
    )

    n = int(
        n_monitoring
    )

    cov_w = (
        sigma
        * sigma
        * np.minimum.outer(
            times,
            times,
        )
    )

    mu_x = (
        math.log(
            S0
        )
        + (
            r
            - 0.5
            * sigma
            * sigma
        )
        * times
    )

    mu_y = float(
        np.mean(
            mu_x
        )
    )

    cov_xy = np.sum(
        cov_w,
        axis=1,
    ) / n

    var_y = float(
        np.sum(
            cov_w
        )
        / (
            n
            * n
        )
    )

    var_x = (
        sigma
        * sigma
        * times
    )

    if var_y <= 0.0:
        return levy_asian_call(
            S0=S0,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            n_monitoring=n_monitoring,
        )

    slopes = (
        cov_xy
        / var_y
    )

    conditional_intercepts = (
        mu_x
        - slopes
        * mu_y
        + 0.5
        * (
            var_x
            - (
                cov_xy
                * cov_xy
                / var_y
            )
        )
    )

    def conditional_average(
        y_value: float,
    ) -> float:
        return float(
            np.mean(
                np.exp(
                    conditional_intercepts
                    + slopes
                    * y_value
                )
            )
        )

    root_var_y = math.sqrt(
        var_y
    )

    lower = (
        mu_y
        - 12.0
        * root_var_y
    )
    upper = (
        mu_y
        + 12.0
        * root_var_y
    )

    lower_value = (
        conditional_average(
            lower
        )
        - K
    )
    upper_value = (
        conditional_average(
            upper
        )
        - K
    )

    if lower_value >= 0.0:
        y_star = lower
    elif upper_value <= 0.0:
        y_star = upper
    else:
        y_star = float(
            optimize.brentq(
                lambda y: (
                    conditional_average(
                        y
                    )
                    - K
                ),
                lower,
                upper,
                xtol=1e-12,
                rtol=1e-12,
            )
        )

    asset_tail_expectations = (
        S0
        * np.exp(
            r
            * times
        )
        * stats.norm.cdf(
            (
                mu_y
                + cov_xy
                - y_star
            )
            / root_var_y
        )
    )

    exercise_probability = float(
        stats.norm.cdf(
            (
                mu_y
                - y_star
            )
            / root_var_y
        )
    )

    price = (
        math.exp(
            -r
            * T
        )
        * (
            float(
                np.mean(
                    asset_tail_expectations
                )
            )
            - K
            * exercise_probability
        )
    )

    return float(
        max(
            price,
            0.0,
        )
    )


def monte_carlo_asian(
    *,
    S0: float,
    strikes: list[float],
    T: float,
    r: float,
    sigma: float,
    n_monitoring: int,
    n_paths: int,
    rng: np.random.Generator,
) -> list[
    tuple[
        float,
        float,
        float,
    ]
]:
    dt = (
        T
        / n_monitoring
    )

    normals = rng.standard_normal(
        (
            n_paths,
            n_monitoring,
        )
    )

    log_increments = (
        (
            r
            - 0.5
            * sigma
            * sigma
        )
        * dt
        + sigma
        * math.sqrt(
            dt
        )
        * normals
    )

    log_paths = (
        math.log(
            S0
        )
        + np.cumsum(
            log_increments,
            axis=1,
        )
    )

    paths = np.exp(
        log_paths
    )

    arithmetic_average = np.mean(
        paths,
        axis=1,
    )

    geometric_average = np.exp(
        np.mean(
            log_paths,
            axis=1,
        )
    )

    discount = math.exp(
        -r
        * T
    )

    results = []

    for strike in strikes:
        arithmetic_payoff = np.maximum(
            arithmetic_average
            - strike,
            0.0,
        )

        geometric_payoff = np.maximum(
            geometric_average
            - strike,
            0.0,
        )

        discounted_arithmetic = (
            discount
            * arithmetic_payoff
        )

        mc_arith = float(
            np.mean(
                discounted_arithmetic
            )
        )

        mc_geo = float(
            discount
            * np.mean(
                geometric_payoff
            )
        )

        standard_error = float(
            np.std(
                discounted_arithmetic,
                ddof=1,
            )
            / math.sqrt(
                float(
                    n_paths
                )
            )
        )

        results.append(
            (
                mc_arith,
                mc_geo,
                standard_error,
            )
        )

    return results


@dataclass(frozen=True)
class AsianOptionSkill:
    name: str = "asian-option-domain"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        del task_dir

        lowered = (
            instruction.lower()
        )

        return (
            "asian" in lowered
            and "levy" in lowered
            and "curran" in lowered
            and (
                "geometric average" in lowered
                or "geometric asian" in lowered
            )
            and "monte carlo" in lowered
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

        path = _find_daily_price_csv(
            task_dir
        )

        calibration = calibrate_gbm(
            pd.read_csv(
                path
            )
        )

        S0 = float(
            calibration[
                "S0"
            ]
        )
        sigma = float(
            calibration[
                "sigma"
            ]
        )

        r = 0.05

        maturities = (
            0.5,
            1.0,
        )

        monitoring_counts = (
            4,
            12,
            52,
        )

        moneyness_values = (
            0.95,
            1.00,
            1.05,
        )

        n_paths = 100000

        rng = np.random.default_rng(
            seed
        )

        rows = []

        for T in maturities:
            for n_monitoring in monitoring_counts:
                strikes = [
                    S0
                    * moneyness
                    for moneyness
                    in moneyness_values
                ]

                mc_values = monte_carlo_asian(
                    S0=S0,
                    strikes=strikes,
                    T=T,
                    r=r,
                    sigma=sigma,
                    n_monitoring=n_monitoring,
                    n_paths=n_paths,
                    rng=rng,
                )

                for (
                    strike,
                    moneyness,
                    (
                        mc_arith,
                        mc_geo,
                        mc_std_err,
                    ),
                ) in zip(
                    strikes,
                    moneyness_values,
                    mc_values,
                ):
                    geo_exact = (
                        geometric_asian_call(
                            S0=S0,
                            K=strike,
                            T=T,
                            r=r,
                            sigma=sigma,
                            n_monitoring=n_monitoring,
                        )
                    )

                    levy = levy_asian_call(
                        S0=S0,
                        K=strike,
                        T=T,
                        r=r,
                        sigma=sigma,
                        n_monitoring=n_monitoring,
                    )

                    curran = curran_asian_call(
                        S0=S0,
                        K=strike,
                        T=T,
                        r=r,
                        sigma=sigma,
                        n_monitoring=n_monitoring,
                    )

                    rows.append(
                        {
                            "T": float(
                                T
                            ),
                            "n_monitoring": int(
                                n_monitoring
                            ),
                            "K": float(
                                strike
                            ),
                            "moneyness": float(
                                moneyness
                            ),
                            "geo_exact": float(
                                geo_exact
                            ),
                            "levy_arith": float(
                                levy
                            ),
                            "curran_arith": float(
                                curran
                            ),
                            "mc_arith": float(
                                mc_arith
                            ),
                            "mc_geo": float(
                                mc_geo
                            ),
                            "mc_std_err": float(
                                mc_std_err
                            ),
                        }
                    )

        frame = pd.DataFrame(
            rows,
            columns=[
                "T",
                "n_monitoring",
                "K",
                "moneyness",
                "geo_exact",
                "levy_arith",
                "curran_arith",
                "mc_arith",
                "mc_geo",
                "mc_std_err",
            ],
        ).sort_values(
            [
                "T",
                "n_monitoring",
                "K",
            ],
            kind="stable",
        ).reset_index(
            drop=True
        )

        levy_errors = np.abs(
            frame[
                "levy_arith"
            ]
            - frame[
                "mc_arith"
            ]
        )

        curran_errors = np.abs(
            frame[
                "curran_arith"
            ]
            - frame[
                "mc_arith"
            ]
        )

        geo_less_pct = float(
            100.0
            * np.mean(
                frame[
                    "geo_exact"
                ]
                < frame[
                    "mc_arith"
                ]
            )
        )

        summary = {
            "n_rows": int(
                len(
                    frame
                )
            ),
            "geo_less_than_arith_pct": (
                geo_less_pct
            ),
            "levy_vs_mc_max_error": float(
                np.max(
                    levy_errors
                )
            ),
            "curran_vs_mc_max_error": float(
                np.max(
                    curran_errors
                )
            ),
            "curran_better_than_levy": bool(
                np.max(
                    curran_errors
                )
                < np.max(
                    levy_errors
                )
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

        frame.to_csv(
            out_dir
            / "asian_prices.csv",
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
