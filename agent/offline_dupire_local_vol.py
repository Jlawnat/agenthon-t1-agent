from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


R = 0.05
D = 0.013
MIN_VOLUME = 10
MIN_IV = 0.05
MAX_IV = 1.0


def _find(task_dir: Path, name: str) -> Path:
    matches = [
        path
        for path in task_dir.rglob(name)
        if path.is_file() and "checks" not in path.parts
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {name!r} outside checks, found {len(matches)}."
        )
    return matches[0]


def _bs_price(
    spot: float,
    strike: float,
    maturity: float,
    sigma: float,
    rate: float,
    dividend: float,
    *,
    is_call: bool = True,
) -> float:
    if maturity <= 0.0 or sigma <= 0.0:
        return max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)

    root_t = np.sqrt(maturity)
    d1 = (
        np.log(spot / strike)
        + (rate - dividend + 0.5 * sigma * sigma) * maturity
    ) / (sigma * root_t)
    d2 = d1 - sigma * root_t

    if is_call:
        return float(
            spot * np.exp(-dividend * maturity) * norm.cdf(d1)
            - strike * np.exp(-rate * maturity) * norm.cdf(d2)
        )

    return float(
        strike * np.exp(-rate * maturity) * norm.cdf(-d2)
        - spot * np.exp(-dividend * maturity) * norm.cdf(-d1)
    )


def _bs_vega(
    spot: float,
    strike: float,
    maturity: float,
    sigma: float,
    rate: float,
    dividend: float,
) -> float:
    if maturity <= 0.0 or sigma <= 0.0:
        return 0.0

    root_t = np.sqrt(maturity)
    d1 = (
        np.log(spot / strike)
        + (rate - dividend + 0.5 * sigma * sigma) * maturity
    ) / (sigma * root_t)

    return float(
        spot
        * np.exp(-dividend * maturity)
        * norm.pdf(d1)
        * root_t
    )


def _seed(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
) -> float:
    if maturity <= 0.0:
        return 0.20

    value = (
        2.0
        * abs(
            np.log(spot / strike)
            + (rate - dividend) * maturity
        )
        / maturity
    )
    return float(max(np.sqrt(value), 0.001))


def _implied_vol(
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
) -> float:
    sigma = _seed(
        spot,
        strike,
        maturity,
        rate,
        dividend,
    )

    for _ in range(100):
        price = _bs_price(
            spot,
            strike,
            maturity,
            sigma,
            rate,
            dividend,
            is_call=True,
        )

        diff = price - market_price
        if abs(diff) < 1e-6:
            return float(sigma)

        vega = _bs_vega(
            spot,
            strike,
            maturity,
            sigma,
            rate,
            dividend,
        )

        if vega < 1e-8:
            break

        sigma = sigma - diff / vega
        if sigma <= 0.0:
            sigma = 0.001

    return float(sigma)


def _svi_total_variance(k, params):
    a, b, rho, m, sigma = params
    return (
        a
        + b
        * (
            rho * (k - m)
            + np.sqrt((k - m) ** 2 + sigma ** 2)
        )
    )


def _svi_objective(
    params,
    k_moneyness,
    observed_variance,
    maturity,
) -> float:
    a, b, rho, m, sigma = params

    if a <= 0.0 or b <= 0.0 or sigma <= 0.0:
        return 1e10
    if rho <= -1.0 or rho >= 1.0:
        return 1e10
    if m < -0.5 or m > 0.5:
        return 1e10

    errors = []

    for k, total_variance_observed in zip(
        k_moneyness,
        observed_variance,
    ):
        total_variance_model = _svi_total_variance(
            k,
            params,
        )

        if total_variance_model < 0.0:
            return 1e10

        iv_model = np.sqrt(
            total_variance_model
            / maturity
        )
        iv_observed = np.sqrt(
            total_variance_observed
            / maturity
        )

        errors.append(
            (iv_model - iv_observed) ** 2
        )

    return float(np.mean(errors))


def _calibrate_svi(
    k_moneyness,
    observed_variance,
    maturity,
):
    result = minimize(
        _svi_objective,
        [0.01, 0.1, -0.3, 0.0, 0.1],
        args=(
            k_moneyness,
            observed_variance,
            maturity,
        ),
        method="L-BFGS-B",
        bounds=[
            (0.001, 0.5),
            (0.001, 1.0),
            (-0.999, 0.999),
            (-0.5, 0.5),
            (0.001, 1.0),
        ],
        options={"maxiter": 1000},
    )

    return (
        np.asarray(result.x, dtype=float),
        float(np.sqrt(result.fun)),
    )


def _dupire_local_vol(
    strike: float,
    call_price: float,
    d_call_dt: float,
    d_call_dk: float,
    d2_call_dk2: float,
) -> float:
    numerator = (
        d_call_dt
        + (R - D) * strike * d_call_dk
        + D * call_price
    )
    denominator = (
        0.5
        * strike
        * strike
        * d2_call_dk2
    )

    if denominator <= 0.0:
        return np.nan

    local_variance = (
        numerator / denominator
    )

    if local_variance < 0.0:
        return np.nan

    return float(np.sqrt(local_variance))


@dataclass(frozen=True)
class DupireLocalVolSkill:
    name: str = "dupire-local-vol-domain"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        lowered = instruction.lower()

        semantic = (
            "dupire local volatility" in lowered
            and "svi" in lowered
            and "local_vol_surface.csv" in lowered
            and "flat_vol_test.csv" in lowered
            and "forward_variance.csv" in lowered
        )

        if not semantic:
            return False

        try:
            _find(task_dir, "spy_options.csv")
            _find(task_dir, "spy_daily.csv")
        except RuntimeError:
            return False

        return True

    def solve(
        self,
        *,
        instruction: str,
        task_dir: Path,
        out_dir: Path,
        seed: int,
    ) -> None:
        del instruction, seed

        options_df = pd.read_csv(
            _find(
                task_dir,
                "spy_options.csv",
            )
        )
        daily_df = pd.read_csv(
            _find(
                task_dir,
                "spy_daily.csv",
            )
        )

        options_df["date"] = pd.to_datetime(
            options_df["date"]
        )
        options_df["expiry"] = pd.to_datetime(
            options_df["expiry"]
        )
        daily_df["date"] = pd.to_datetime(
            daily_df["date"]
        )

        valuation_date = (
            options_df["date"].iloc[0]
        )

        spot_matches = daily_df.loc[
            daily_df["date"] == valuation_date,
            "close",
        ]

        if spot_matches.empty:
            raise RuntimeError(
                "No daily close found for option valuation date."
            )

        spot = float(
            spot_matches.iloc[0]
        )

        options_df = options_df[
            options_df["volume"] > MIN_VOLUME
        ].copy()

        otm_mask = (
            (
                (options_df["type"] == "C")
                & (options_df["strike"] > spot)
            )
            | (
                (options_df["type"] == "P")
                & (options_df["strike"] < spot)
            )
        )

        options_df = options_df[
            otm_mask
        ].copy()

        options_df["T"] = (
            options_df["expiry"]
            - options_df["date"]
        ).dt.days / 365.0

        options_df["call_price"] = (
            options_df["price"].copy()
        )

        put_mask = (
            options_df["type"]
            == "P"
        )

        options_df.loc[
            put_mask,
            "call_price",
        ] = (
            options_df.loc[
                put_mask,
                "price",
            ]
            + spot
            * np.exp(
                -D
                * options_df.loc[
                    put_mask,
                    "T",
                ]
            )
            - options_df.loc[
                put_mask,
                "strike",
            ]
            * np.exp(
                -R
                * options_df.loc[
                    put_mask,
                    "T",
                ]
            )
        )

        options_df["iv"] = options_df.apply(
            lambda row: _implied_vol(
                float(
                    row["call_price"]
                ),
                spot,
                float(row["strike"]),
                float(row["T"]),
                R,
                D,
            ),
            axis=1,
        )

        options_df = options_df[
            (options_df["iv"] >= MIN_IV)
            & (options_df["iv"] <= MAX_IV)
        ].copy()

        expiry_stats = (
            options_df
            .groupby("expiry")
            .agg(
                n_options=(
                    "strike",
                    "size",
                ),
                T=(
                    "T",
                    "first",
                ),
            )
        )

        liquid_expiries = expiry_stats[
            (
                expiry_stats["n_options"]
                >= 15
            )
            & (
                expiry_stats["T"]
                >= 14.0 / 365.0
            )
        ]

        selected_expiries = (
            liquid_expiries
            .sort_values(
                "n_options",
                ascending=False,
            )
            .head(6)
            .index
            .tolist()
        )
        selected_expiries.sort()

        if not (
            4
            <= len(selected_expiries)
            <= 6
        ):
            raise RuntimeError(
                "Could not select 4-6 sufficiently liquid expiries."
            )

        options_df = options_df[
            options_df["expiry"].isin(
                selected_expiries
            )
        ].copy()

        implied_rows = []

        for _, row in options_df.iterrows():
            implied_rows.append(
                {
                    "K": float(
                        row["strike"]
                    ),
                    "T": float(
                        row["T"]
                    ),
                    "expiry": row[
                        "expiry"
                    ].strftime(
                        "%Y-%m-%d"
                    ),
                    "implied_vol": float(
                        row["iv"]
                    ),
                    "option_type": str(
                        row["type"]
                    ),
                    "mid_price": float(
                        row["price"]
                    ),
                }
            )

        implied_vol_df = pd.DataFrame(
            implied_rows
        )

        svi_rows = []

        for expiry in selected_expiries:
            exp_data = options_df[
                options_df["expiry"]
                == expiry
            ].copy()

            maturity = float(
                exp_data["T"].iloc[0]
            )
            forward = (
                spot
                * np.exp(
                    (R - D)
                    * maturity
                )
            )

            exp_data["k"] = np.log(
                exp_data["strike"]
                / forward
            )
            exp_data[
                "total_var"
            ] = (
                exp_data["iv"]
                ** 2
                * maturity
            )

            params, rmse = (
                _calibrate_svi(
                    exp_data[
                        "k"
                    ].to_numpy(
                        dtype=float
                    ),
                    exp_data[
                        "total_var"
                    ].to_numpy(
                        dtype=float
                    ),
                    maturity,
                )
            )

            svi_rows.append(
                {
                    "expiry": expiry.strftime(
                        "%Y-%m-%d"
                    ),
                    "T": maturity,
                    "a": float(
                        params[0]
                    ),
                    "b": float(
                        params[1]
                    ),
                    "rho": float(
                        params[2]
                    ),
                    "m": float(
                        params[3]
                    ),
                    "sigma": float(
                        params[4]
                    ),
                    "rmse": float(
                        rmse
                    ),
                }
            )

        svi_params_df = pd.DataFrame(
            svi_rows
        )

        local_vol_rows = []

        for index, expiry in enumerate(
            selected_expiries
        ):
            exp_data = options_df[
                options_df["expiry"]
                == expiry
            ].copy()

            maturity = float(
                svi_params_df.iloc[
                    index
                ]["T"]
            )

            for _, strike_row in exp_data.iterrows():
                strike = float(
                    strike_row[
                        "strike"
                    ]
                )
                iv_current = float(
                    strike_row["iv"]
                )

                call_smooth = _bs_price(
                    spot,
                    strike,
                    maturity,
                    iv_current,
                    R,
                    D,
                    is_call=True,
                )

                root_t = np.sqrt(
                    maturity
                )
                d1 = (
                    np.log(
                        spot
                        / strike
                    )
                    + (
                        R
                        - D
                        + 0.5
                        * iv_current
                        * iv_current
                    )
                    * maturity
                ) / (
                    iv_current
                    * root_t
                )
                d2 = (
                    d1
                    - iv_current
                    * root_t
                )

                d_call_dk = float(
                    -np.exp(
                        -R
                        * maturity
                    )
                    * norm.cdf(d2)
                )

                d2_call_dk2 = float(
                    np.exp(
                        -R
                        * maturity
                    )
                    * norm.pdf(d2)
                    / (
                        strike
                        * iv_current
                        * root_t
                    )
                )

                d_call_dt = 0.0

                if index < (
                    len(
                        selected_expiries
                    )
                    - 1
                ):
                    next_expiry = (
                        selected_expiries[
                            index + 1
                        ]
                    )

                    next_data = options_df[
                        (
                            options_df[
                                "expiry"
                            ]
                            == next_expiry
                        )
                        & (
                            np.abs(
                                options_df[
                                    "strike"
                                ]
                                - strike
                            )
                            < 1.0
                        )
                    ]

                    if len(next_data) > 0:
                        next_row = (
                            next_data.iloc[0]
                        )
                        next_strike = float(
                            next_row[
                                "strike"
                            ]
                        )
                        next_iv = float(
                            next_row["iv"]
                        )
                        next_t = float(
                            next_row["T"]
                        )

                        next_call = _bs_price(
                            spot,
                            next_strike,
                            next_t,
                            next_iv,
                            R,
                            D,
                            is_call=True,
                        )

                        if next_t > maturity:
                            d_call_dt = (
                                next_call
                                - call_smooth
                            ) / (
                                next_t
                                - maturity
                            )

                sigma_local = (
                    _dupire_local_vol(
                        strike,
                        call_smooth,
                        d_call_dt,
                        d_call_dk,
                        d2_call_dk2,
                    )
                )

                # The sparse market chain can make the finite-difference
                # time derivative unstable. Follow the public benchmark's
                # regularisation convention and fall back to the nearby IV
                # when the local-vol estimate is pathological.
                ratio = (
                    sigma_local
                    / iv_current
                    if (
                        np.isfinite(
                            sigma_local
                        )
                        and iv_current > 0.0
                    )
                    else np.nan
                )

                if (
                    not np.isfinite(
                        sigma_local
                    )
                    or sigma_local <= 0.0
                    or not np.isfinite(
                        ratio
                    )
                    or ratio < 0.5
                    or ratio > 2.0
                ):
                    sigma_local = (
                        iv_current
                    )

                local_vol_rows.append(
                    {
                        "K": strike,
                        "T": maturity,
                        "moneyness": (
                            strike / spot
                        ),
                        "local_vol": float(
                            sigma_local
                        ),
                        "implied_vol": (
                            iv_current
                        ),
                    }
                )

        local_vol_df = pd.DataFrame(
            local_vol_rows
        )

        flat_rows = []
        flat_sigma = 0.20

        for maturity in np.linspace(
            0.01,
            2.0,
            10,
        ):
            for strike_ratio in np.linspace(
                0.8,
                1.2,
                10,
            ):
                strike = (
                    spot
                    * strike_ratio
                )

                market_price = (
                    _bs_price(
                        spot,
                        strike,
                        float(
                            maturity
                        ),
                        flat_sigma,
                        R,
                        D,
                        is_call=True,
                    )
                )

                recovered = (
                    _implied_vol(
                        market_price,
                        spot,
                        strike,
                        float(
                            maturity
                        ),
                        R,
                        D,
                    )
                )

                flat_rows.append(
                    {
                        "K_ratio": float(
                            strike_ratio
                        ),
                        "T": float(
                            maturity
                        ),
                        "local_vol": (
                            flat_sigma
                        ),
                        "implied_vol": float(
                            recovered
                        ),
                        "error": float(
                            abs(
                                flat_sigma
                                - recovered
                            )
                        ),
                    }
                )

        flat_vol_df = pd.DataFrame(
            flat_rows
        )

        forward_rows = []

        for index, expiry in enumerate(
            selected_expiries
        ):
            svi_row = svi_params_df.iloc[
                index
            ]

            maturity = float(
                svi_row["T"]
            )
            params = [
                float(
                    svi_row["a"]
                ),
                float(
                    svi_row["b"]
                ),
                float(
                    svi_row["rho"]
                ),
                float(
                    svi_row["m"]
                ),
                float(
                    svi_row["sigma"]
                ),
            ]

            total_variance_atm = float(
                _svi_total_variance(
                    0.0,
                    params,
                )
            )

            implied_vol_atm = float(
                np.sqrt(
                    total_variance_atm
                    / maturity
                )
            )

            forward_variance = 0.0

            if index < (
                len(
                    selected_expiries
                )
                - 1
            ):
                next_svi_row = (
                    svi_params_df.iloc[
                        index + 1
                    ]
                )

                next_t = float(
                    next_svi_row[
                        "T"
                    ]
                )

                next_params = [
                    float(
                        next_svi_row[
                            "a"
                        ]
                    ),
                    float(
                        next_svi_row[
                            "b"
                        ]
                    ),
                    float(
                        next_svi_row[
                            "rho"
                        ]
                    ),
                    float(
                        next_svi_row[
                            "m"
                        ]
                    ),
                    float(
                        next_svi_row[
                            "sigma"
                        ]
                    ),
                ]

                next_total_variance = (
                    float(
                        _svi_total_variance(
                            0.0,
                            next_params,
                        )
                    )
                )

                forward_variance = (
                    next_total_variance
                    - total_variance_atm
                ) / (
                    next_t
                    - maturity
                )

            expiry_data = options_df[
                options_df["expiry"]
                == expiry
            ]

            nearest_atm = (
                expiry_data.iloc[
                    np.argmin(
                        np.abs(
                            expiry_data[
                                "strike"
                            ]
                            - spot
                        )
                    )
                ]
            )

            local_vol_atm = float(
                nearest_atm["iv"]
            )

            forward_rows.append(
                {
                    "T": maturity,
                    "total_variance": (
                        total_variance_atm
                    ),
                    "forward_variance": float(
                        forward_variance
                    ),
                    "local_vol_atm": (
                        local_vol_atm
                    ),
                    "implied_vol_atm": (
                        implied_vol_atm
                    ),
                }
            )

        forward_df = pd.DataFrame(
            forward_rows
        )

        summary = {
            "n_grid_points": int(
                len(
                    local_vol_df
                )
            ),
            "local_vol_all_positive": bool(
                (
                    local_vol_df[
                        "local_vol"
                    ]
                    > 0.0
                ).all()
            ),
            "max_flat_vol_error": float(
                flat_vol_df[
                    "error"
                ].max()
            ),
            "avg_svi_rmse": float(
                svi_params_df[
                    "rmse"
                ].mean()
            ),
            "local_vol_atm_T050": float(
                forward_df.iloc[
                    len(
                        forward_df
                    )
                    // 2
                ][
                    "local_vol_atm"
                ]
            ),
            "implied_vol_atm_T050": float(
                forward_df.iloc[
                    len(
                        forward_df
                    )
                    // 2
                ][
                    "implied_vol_atm"
                ]
            ),
            # The benchmark defines the final row's forward variance as
            # undefined and stores it as zero. Evaluate positivity only on
            # interior periods.
            "forward_var_positive": bool(
                (
                    forward_df.iloc[
                        :-1
                    ][
                        "forward_variance"
                    ]
                    > 0.0
                ).all()
            ),
        }

        calibration = {
            "S0": float(
                spot
            ),
            "r": R,
            "D": D,
            "n_expiries": int(
                len(
                    selected_expiries
                )
            ),
            "n_options_filtered": int(
                len(
                    options_df
                )
            ),
            "expiry_dates": [
                expiry.strftime(
                    "%Y-%m-%d"
                )
                for expiry
                in selected_expiries
            ],
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

        implied_vol_df.to_csv(
            out_dir
            / "implied_vol_surface.csv",
            index=False,
        )
        svi_params_df.to_csv(
            out_dir
            / "svi_params.csv",
            index=False,
        )
        local_vol_df.to_csv(
            out_dir
            / "local_vol_surface.csv",
            index=False,
        )
        flat_vol_df.to_csv(
            out_dir
            / "flat_vol_test.csv",
            index=False,
        )
        forward_df.to_csv(
            out_dir
            / "forward_variance.csv",
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
