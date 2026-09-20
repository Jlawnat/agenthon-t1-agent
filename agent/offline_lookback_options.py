from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def _find_price_csv(task_dir: Path) -> Path:
    candidates = []

    for path in sorted(task_dir.rglob("*.csv")):
        if "checks" in path.parts:
            continue

        try:
            frame = pd.read_csv(path, nrows=5)
        except Exception:
            continue

        lower = {
            str(column).lower()
            for column in frame.columns
        }

        if (
            ("close" in lower or "adj_close" in lower)
            and (
                "date" in lower
                or "datetime" in lower
                or "timestamp" in lower
            )
        ):
            candidates.append(path)

    if not candidates:
        raise RuntimeError(
            "Could not discover a daily price CSV for lookback calibration."
        )

    return candidates[0]


def calibrate_from_prices(
    price_path: Path,
) -> dict[str, float | int]:
    frame = pd.read_csv(price_path)

    by_lower = {
        str(column).lower(): str(column)
        for column in frame.columns
    }

    close_column = (
        by_lower.get("adj_close")
        or by_lower.get("close")
    )

    if close_column is None:
        raise RuntimeError(
            "Lookback calibration requires close or adj_close prices."
        )

    close = pd.to_numeric(
        frame[close_column],
        errors="coerce",
    ).dropna().astype(float)

    if len(close) < 2:
        raise RuntimeError(
            "Lookback calibration requires at least two prices."
        )

    returns = np.log(
        close.to_numpy()[1:]
        / close.to_numpy()[:-1]
    )

    return {
        "n_returns": int(len(returns)),
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns, ddof=1)),
        "sigma": float(
            np.std(returns, ddof=1)
            * math.sqrt(252.0)
        ),
        "S0": float(close.iloc[-1]),
        "r": 0.05,
        "D": 0.0,
    }


def floating_lookback_call(
    *,
    spot: float,
    running_min: float,
    maturity: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    b = (
        risk_free_rate
        - dividend_yield
    )

    if abs(b) < 1e-12:
        raise RuntimeError(
            "Zero cost-of-carry floating lookback limit is not implemented."
        )

    sqrt_t = math.sqrt(maturity)

    a1 = (
        math.log(
            spot
            / running_min
        )
        + (
            b
            + 0.5
            * volatility
            * volatility
        )
        * maturity
    ) / (
        volatility
        * sqrt_t
    )

    a2 = (
        a1
        - volatility
        * sqrt_t
    )

    a3 = (
        a1
        - 2.0
        * b
        * sqrt_t
        / volatility
    )

    disc_r = math.exp(
        -risk_free_rate
        * maturity
    )

    disc_q = math.exp(
        -dividend_yield
        * maturity
    )

    ratio_term = (
        running_min
        / spot
    ) ** (
        2.0
        * b
        / (
            volatility
            * volatility
        )
    )

    correction = (
        spot
        * disc_r
        * volatility
        * volatility
        / (
            2.0
            * b
        )
        * (
            ratio_term
            * norm.cdf(
                -a1
                + 2.0
                * b
                * sqrt_t
                / volatility
            )
            - math.exp(
                b
                * maturity
            )
            * norm.cdf(
                -a1
            )
        )
    )

    return float(
        spot
        * disc_q
        * norm.cdf(
            a1
        )
        - running_min
        * disc_r
        * norm.cdf(
            a2
        )
        + correction
    )


def floating_lookback_put(
    *,
    spot: float,
    running_max: float,
    maturity: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    b = (
        risk_free_rate
        - dividend_yield
    )

    if abs(b) < 1e-12:
        raise RuntimeError(
            "Zero cost-of-carry floating lookback limit is not implemented."
        )

    sqrt_t = math.sqrt(maturity)

    b1 = (
        math.log(
            spot
            / running_max
        )
        + (
            b
            + 0.5
            * volatility
            * volatility
        )
        * maturity
    ) / (
        volatility
        * sqrt_t
    )

    b2 = (
        b1
        - volatility
        * sqrt_t
    )

    disc_r = math.exp(
        -risk_free_rate
        * maturity
    )

    disc_q = math.exp(
        -dividend_yield
        * maturity
    )

    ratio_term = (
        spot
        / running_max
    ) ** (
        -2.0
        * b
        / (
            volatility
            * volatility
        )
    )

    correction = (
        spot
        * disc_r
        * volatility
        * volatility
        / (
            2.0
            * b
        )
        * (
            -ratio_term
            * norm.cdf(
                b1
                - 2.0
                * b
                * sqrt_t
                / volatility
            )
            + math.exp(
                b
                * maturity
            )
            * norm.cdf(
                b1
            )
        )
    )

    return float(
        -spot
        * disc_q
        * norm.cdf(
            -b1
        )
        + running_max
        * disc_r
        * norm.cdf(
            -b2
        )
        + correction
    )


def fixed_strike_lookback_call(
    *,
    spot: float,
    running_max: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    b = (
        risk_free_rate
        - dividend_yield
    )

    if abs(b) < 1e-12:
        raise RuntimeError(
            "Zero cost-of-carry fixed lookback limit is not implemented."
        )

    sqrt_t = math.sqrt(maturity)

    disc_r = math.exp(
        -risk_free_rate
        * maturity
    )

    disc_q = math.exp(
        -dividend_yield
        * maturity
    )

    if strike > running_max:
        d1 = (
            math.log(
                spot
                / strike
            )
            + (
                b
                + 0.5
                * volatility
                * volatility
            )
            * maturity
        ) / (
            volatility
            * sqrt_t
        )

        d2 = (
            d1
            - volatility
            * sqrt_t
        )

        correction = (
            spot
            * disc_r
            * volatility
            * volatility
            / (
                2.0
                * b
            )
            * (
                -(
                    spot
                    / strike
                ) ** (
                    -2.0
                    * b
                    / (
                        volatility
                        * volatility
                    )
                )
                * norm.cdf(
                    d1
                    - 2.0
                    * b
                    * sqrt_t
                    / volatility
                )
                + math.exp(
                    b
                    * maturity
                )
                * norm.cdf(
                    d1
                )
            )
        )

        return float(
            spot
            * disc_q
            * norm.cdf(
                d1
            )
            - strike
            * disc_r
            * norm.cdf(
                d2
            )
            + correction
        )

    e1 = (
        math.log(
            spot
            / running_max
        )
        + (
            b
            + 0.5
            * volatility
            * volatility
        )
        * maturity
    ) / (
        volatility
        * sqrt_t
    )

    e2 = (
        e1
        - volatility
        * sqrt_t
    )

    correction = (
        spot
        * disc_r
        * volatility
        * volatility
        / (
            2.0
            * b
        )
        * (
            -(
                spot
                / running_max
            ) ** (
                -2.0
                * b
                / (
                    volatility
                    * volatility
                )
            )
            * norm.cdf(
                e1
                - 2.0
                * b
                * sqrt_t
                / volatility
            )
            + math.exp(
                b
                * maturity
            )
            * norm.cdf(
                e1
            )
        )
    )

    return float(
        disc_r
        * (
            running_max
            - strike
        )
        + spot
        * disc_q
        * norm.cdf(
            e1
        )
        - running_max
        * disc_r
        * norm.cdf(
            e2
        )
        + correction
    )


def _simulate_continuous_extrema(
    *,
    spot: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    n_paths: int,
    seed: int,
) -> dict[
    float,
    tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
]:
    if n_paths % 2 != 0:
        raise RuntimeError(
            "Antithetic lookback simulation requires an even path count."
        )

    dt = (
        1.0
        / 252.0
    )

    drift = (
        risk_free_rate
        - dividend_yield
        - 0.5
        * volatility
        * volatility
    ) * dt

    diffusion = (
        volatility
        * math.sqrt(
            dt
        )
    )

    bridge_variance = (
        volatility
        * volatility
        * dt
    )

    rng = np.random.RandomState(
        int(seed)
    )

    half = (
        n_paths
        // 2
    )

    log_spot = float(
        math.log(
            spot
        )
    )

    current = np.full(
        n_paths,
        log_spot,
        dtype=float,
    )

    running_max = current.copy()
    running_min = current.copy()

    record_steps = {
        63: 0.25,
        126: 0.50,
        252: 1.00,
    }

    records: dict[
        float,
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
    ] = {}

    for step in range(
        1,
        253,
    ):
        base_z = rng.standard_normal(
            half
        )

        z = np.concatenate(
            (
                base_z,
                -base_z,
            )
        )

        previous = current

        next_value = (
            previous
            + drift
            + diffusion
            * z
        )

        base_u_max = np.clip(
            rng.random_sample(
                half
            ),
            1e-15,
            1.0
            - 1e-15,
        )

        u_max = np.concatenate(
            (
                base_u_max,
                1.0
                - base_u_max,
            )
        )

        bridge_max = 0.5 * (
            previous
            + next_value
            + np.sqrt(
                (
                    next_value
                    - previous
                )
                ** 2
                - 2.0
                * bridge_variance
                * np.log(
                    u_max
                )
            )
        )

        base_u_min = np.clip(
            rng.random_sample(
                half
            ),
            1e-15,
            1.0
            - 1e-15,
        )

        u_min = np.concatenate(
            (
                base_u_min,
                1.0
                - base_u_min,
            )
        )

        bridge_min = 0.5 * (
            previous
            + next_value
            - np.sqrt(
                (
                    next_value
                    - previous
                )
                ** 2
                - 2.0
                * bridge_variance
                * np.log(
                    u_min
                )
            )
        )

        running_max = np.maximum(
            running_max,
            bridge_max,
        )

        running_min = np.minimum(
            running_min,
            bridge_min,
        )

        current = next_value

        if step in record_steps:
            maturity = (
                record_steps[
                    step
                ]
            )

            records[
                maturity
            ] = (
                np.exp(
                    current.copy()
                ),
                np.exp(
                    running_max.copy()
                ),
                np.exp(
                    running_min.copy()
                ),
            )

    return records


def _mc_mean_and_se(
    payoff: np.ndarray,
    *,
    maturity: float,
    risk_free_rate: float,
) -> tuple[
    float,
    float,
]:
    discount = math.exp(
        -risk_free_rate
        * maturity
    )

    return (
        float(
            discount
            * np.mean(
                payoff
            )
        ),
        float(
            discount
            * np.std(
                payoff,
                ddof=1,
            )
            / math.sqrt(
                len(
                    payoff
                )
            )
        ),
    )


@dataclass(frozen=True)
class LookbackOptionSkill:
    """Closed-form lookback pricing with Monte Carlo validation."""

    name: str = "lookback-option-domain"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        del task_dir

        normalized = (
            instruction.lower()
            .replace(
                "-",
                " ",
            )
            .replace(
                "_",
                " ",
            )
        )

        return (
            "lookback option" in normalized
            and (
                "fixed strike" in normalized
                or "floating strike" in normalized
            )
            and (
                "monte carlo" in normalized
                or "running maximum" in normalized
                or "running minimum" in normalized
            )
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

        calibration = (
            calibrate_from_prices(
                _find_price_csv(
                    task_dir
                )
            )
        )

        spot = float(
            calibration[
                "S0"
            ]
        )

        sigma = float(
            calibration[
                "sigma"
            ]
        )

        risk_free_rate = float(
            calibration[
                "r"
            ]
        )

        dividend_yield = float(
            calibration[
                "D"
            ]
        )

        maturities = (
            0.25,
            0.50,
            1.00,
        )

        moneyness_values = (
            0.90,
            0.95,
            1.00,
            1.05,
            1.10,
        )

        n_paths = 200_000

        records = (
            _simulate_continuous_extrema(
                spot=spot,
                risk_free_rate=(
                    risk_free_rate
                ),
                dividend_yield=(
                    dividend_yield
                ),
                volatility=sigma,
                n_paths=n_paths,
                seed=int(
                    seed
                ),
            )
        )

        fixed_rows: list[
            dict[
                str,
                float,
            ]
        ] = []

        floating_rows: list[
            dict[
                str,
                object,
            ]
        ] = []

        fixed_validation: list[
            tuple[
                float,
                float,
            ]
        ] = []

        floating_validation: list[
            tuple[
                float,
                float,
            ]
        ] = []

        for maturity in maturities:
            (
                terminal,
                running_max,
                running_min,
            ) = records[
                maturity
            ]

            for moneyness in (
                moneyness_values
            ):
                strike = (
                    spot
                    * moneyness
                )

                cf_price = (
                    fixed_strike_lookback_call(
                        spot=spot,
                        running_max=spot,
                        strike=strike,
                        maturity=maturity,
                        risk_free_rate=(
                            risk_free_rate
                        ),
                        dividend_yield=(
                            dividend_yield
                        ),
                        volatility=sigma,
                    )
                )

                payoff = np.maximum(
                    running_max
                    - strike,
                    0.0,
                )

                (
                    mc_price,
                    mc_std_err,
                ) = _mc_mean_and_se(
                    payoff,
                    maturity=maturity,
                    risk_free_rate=(
                        risk_free_rate
                    ),
                )

                fixed_rows.append(
                    {
                        "T": float(
                            maturity
                        ),
                        "K": float(
                            strike
                        ),
                        "moneyness": float(
                            moneyness
                        ),
                        "cf_price": float(
                            cf_price
                        ),
                        "mc_price": float(
                            mc_price
                        ),
                        "mc_std_err": float(
                            mc_std_err
                        ),
                    }
                )

                fixed_validation.append(
                    (
                        abs(
                            cf_price
                            - mc_price
                        ),
                        mc_std_err,
                    )
                )

            float_call_cf = (
                floating_lookback_call(
                    spot=spot,
                    running_min=spot,
                    maturity=maturity,
                    risk_free_rate=(
                        risk_free_rate
                    ),
                    dividend_yield=(
                        dividend_yield
                    ),
                    volatility=sigma,
                )
            )

            (
                float_call_mc,
                float_call_se,
            ) = _mc_mean_and_se(
                terminal
                - running_min,
                maturity=maturity,
                risk_free_rate=(
                    risk_free_rate
                ),
            )

            floating_rows.append(
                {
                    "T": float(
                        maturity
                    ),
                    "type": "call",
                    "cf_price": float(
                        float_call_cf
                    ),
                    "mc_price": float(
                        float_call_mc
                    ),
                    "mc_std_err": float(
                        float_call_se
                    ),
                }
            )

            floating_validation.append(
                (
                    abs(
                        float_call_cf
                        - float_call_mc
                    ),
                    float_call_se,
                )
            )

            float_put_cf = (
                floating_lookback_put(
                    spot=spot,
                    running_max=spot,
                    maturity=maturity,
                    risk_free_rate=(
                        risk_free_rate
                    ),
                    dividend_yield=(
                        dividend_yield
                    ),
                    volatility=sigma,
                )
            )

            (
                float_put_mc,
                float_put_se,
            ) = _mc_mean_and_se(
                running_max
                - terminal,
                maturity=maturity,
                risk_free_rate=(
                    risk_free_rate
                ),
            )

            floating_rows.append(
                {
                    "T": float(
                        maturity
                    ),
                    "type": "put",
                    "cf_price": float(
                        float_put_cf
                    ),
                    "mc_price": float(
                        float_put_mc
                    ),
                    "mc_std_err": float(
                        float_put_se
                    ),
                }
            )

            floating_validation.append(
                (
                    abs(
                        float_put_cf
                        - float_put_mc
                    ),
                    float_put_se,
                )
            )

        fixed = pd.DataFrame(
            fixed_rows,
            columns=[
                "T",
                "K",
                "moneyness",
                "cf_price",
                "mc_price",
                "mc_std_err",
            ],
        )

        floating = pd.DataFrame(
            floating_rows,
            columns=[
                "T",
                "type",
                "cf_price",
                "mc_price",
                "mc_std_err",
            ],
        )

        max_mc_error_fixed = float(
            max(
                error
                for error, _se
                in fixed_validation
            )
        )

        max_mc_error_floating = float(
            max(
                error
                for error, _se
                in floating_validation
            )
        )

        mc_validates = bool(
            all(
                error
                < 2.0
                * se
                for error, se
                in (
                    fixed_validation
                    + floating_validation
                )
            )
        )

        def fixed_at(
            maturity: float,
        ) -> float:
            row = fixed[
                np.isclose(
                    fixed[
                        "T"
                    ],
                    maturity,
                )
                & np.isclose(
                    fixed[
                        "moneyness"
                    ],
                    1.0,
                )
            ]

            return float(
                row.iloc[
                    0
                ][
                    "cf_price"
                ]
            )

        def floating_at(
            maturity: float,
            option_type: str,
        ) -> float:
            row = floating[
                np.isclose(
                    floating[
                        "T"
                    ],
                    maturity,
                )
                & (
                    floating[
                        "type"
                    ]
                    == option_type
                )
            ]

            return float(
                row.iloc[
                    0
                ][
                    "cf_price"
                ]
            )

        summary = {
            "sigma": sigma,
            "S0": spot,
            "n_fixed_strike": int(
                len(
                    fixed
                )
            ),
            "n_floating_strike": int(
                len(
                    floating
                )
            ),
            "max_mc_error_fixed": (
                max_mc_error_fixed
            ),
            "max_mc_error_floating": (
                max_mc_error_floating
            ),
            "mc_validates": (
                mc_validates
            ),
            "atm_fixed_T025": (
                fixed_at(
                    0.25
                )
            ),
            "atm_fixed_T050": (
                fixed_at(
                    0.50
                )
            ),
            "atm_fixed_T100": (
                fixed_at(
                    1.00
                )
            ),
            "float_call_T050": (
                floating_at(
                    0.50,
                    "call",
                )
            ),
            "float_put_T050": (
                floating_at(
                    0.50,
                    "put",
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

        fixed.to_csv(
            out_dir
            / "fixed_strike_lookback.csv",
            index=False,
        )

        floating.to_csv(
            out_dir
            / "floating_strike_lookback.csv",
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
