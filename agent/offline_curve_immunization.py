from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize


KEY_RATES = (2, 5, 10)


def _find_curve_data_dir(task_dir: Path) -> Path:
    required = {
        "par_yields.csv",
        "bonds.csv",
        "liability.json",
    }

    directories = sorted(
        {
            path.parent
            for path in task_dir.rglob("*")
            if path.is_file() and "checks" not in path.parts
        },
        key=lambda path: str(path),
    )

    for directory in directories:
        names = {
            path.name
            for path in directory.iterdir()
            if path.is_file()
        }
        if required.issubset(names):
            return directory

    raise RuntimeError(
        "Could not discover yield-curve immunization input files."
    )


def bootstrap_discount_factors(
    par_yields: pd.DataFrame,
) -> pd.DataFrame:
    data = (
        par_yields.copy()
        .sort_values(
            "maturity_years",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    maturities = (
        data["maturity_years"]
        .astype(int)
        .to_numpy()
    )

    if list(maturities) != list(
        range(1, len(data) + 1)
    ):
        raise RuntimeError(
            "Par curve must contain consecutive integer maturities starting at 1."
        )

    discounts: list[float] = []

    for row in data.itertuples(index=False):
        maturity = int(row.maturity_years)
        par_yield = float(row.par_yield)

        previous_sum = float(
            sum(discounts)
        )

        discount = (
            1.0
            - par_yield
            * previous_sum
        ) / (
            1.0
            + par_yield
        )

        if not (
            0.0
            < discount
            < 1.0
        ):
            raise RuntimeError(
                f"Invalid discount factor at maturity {maturity}."
            )

        discounts.append(
            float(discount)
        )

    discounts_array = np.asarray(
        discounts,
        dtype=float,
    )

    times = maturities.astype(float)

    spots = (
        -np.log(
            discounts_array
        )
        / times
    )

    forwards = np.empty(
        len(
            discounts_array
        ),
        dtype=float,
    )

    forwards[0] = float(
        -math.log(
            discounts_array[0]
        )
    )

    for index in range(
        1,
        len(
            discounts_array
        ),
    ):
        forwards[
            index
        ] = float(
            -math.log(
                discounts_array[
                    index
                ]
                / discounts_array[
                    index - 1
                ]
            )
        )

    return pd.DataFrame(
        {
            "maturity_years": (
                maturities
            ),
            "discount_factor": (
                discounts_array
            ),
            "spot_rate": spots,
            "forward_rate": forwards,
        }
    )


def _bond_cash_flows(
    face_value: float,
    coupon_rate: float,
    maturity_years: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    times = np.arange(
        1,
        maturity_years + 1,
        dtype=int,
    )

    cash_flows = np.full(
        maturity_years,
        face_value
        * coupon_rate,
        dtype=float,
    )

    cash_flows[-1] += face_value

    return (
        times,
        cash_flows,
    )


def _discount_cash_flows(
    times: np.ndarray,
    cash_flows: np.ndarray,
    spot_rates: np.ndarray,
) -> float:
    rates = spot_rates[
        times
        - 1
    ]

    return float(
        np.sum(
            cash_flows
            * np.exp(
                -rates
                * times
            )
        )
    )


def _krd(
    times: np.ndarray,
    cash_flows: np.ndarray,
    spot_rates: np.ndarray,
    key_rate: int,
) -> float:
    base = _discount_cash_flows(
        times,
        cash_flows,
        spot_rates,
    )

    bumped = spot_rates.copy()

    if (
        key_rate
        >= 1
        and key_rate
        <= len(
            bumped
        )
    ):
        bumped[
            key_rate
            - 1
        ] += 1e-4

    stressed = _discount_cash_flows(
        times,
        cash_flows,
        bumped,
    )

    if base <= 0.0:
        return 0.0

    return float(
        (
            base
            - stressed
        )
        / (
            base
            * 1e-4
        )
    )


def _ytm(
    price: float,
    times: np.ndarray,
    cash_flows: np.ndarray,
) -> float:
    def residual(yield_rate: float) -> float:
        return float(
            np.sum(
                cash_flows
                / (
                    (
                        1.0
                        + yield_rate
                    )
                    ** times
                )
            )
            - price
        )

    return float(
        brentq(
            residual,
            1e-6,
            0.30,
            xtol=1e-14,
            rtol=1e-14,
        )
    )


def _z_spread(
    price: float,
    times: np.ndarray,
    cash_flows: np.ndarray,
    spot_rates: np.ndarray,
) -> float:
    def residual(
        spread: float,
    ) -> float:
        return float(
            np.sum(
                cash_flows
                * np.exp(
                    -(
                        spot_rates[
                            times
                            - 1
                        ]
                        + spread
                    )
                    * times
                )
            )
            - price
        )

    return float(
        brentq(
            residual,
            -0.05,
            0.50,
            xtol=1e-14,
            rtol=1e-14,
        )
    )


def _bond_analytics(
    bonds: pd.DataFrame,
    spot_rates: np.ndarray,
) -> pd.DataFrame:
    rows = []

    for row in bonds.itertuples(
        index=False
    ):
        bond_id = str(
            row.bond_id
        )
        face = float(
            row.face_value
        )
        coupon = float(
            row.coupon_rate
        )
        maturity = int(
            row.maturity_years
        )

        times, cash_flows = (
            _bond_cash_flows(
                face,
                coupon,
                maturity,
            )
        )

        price = _discount_cash_flows(
            times,
            cash_flows,
            spot_rates,
        )

        ytm = _ytm(
            price,
            times,
            cash_flows,
        )

        pv_components = (
            cash_flows
            * np.exp(
                -spot_rates[
                    times
                    - 1
                ]
                * times
            )
        )

        macaulay = float(
            np.sum(
                times
                * pv_components
            )
            / price
        )

        modified = float(
            macaulay
            / (
                1.0
                + ytm
            )
        )

        convexity = float(
            np.sum(
                times
                * (
                    times
                    + 1.0
                )
                * cash_flows
                / (
                    (
                        1.0
                        + ytm
                    )
                    ** (
                        times
                        + 2.0
                    )
                )
            )
            / price
        )

        krds = {
            key_rate: _krd(
                times,
                cash_flows,
                spot_rates,
                key_rate,
            )
            for key_rate
            in KEY_RATES
        }

        z_spread = _z_spread(
            price,
            times,
            cash_flows,
            spot_rates,
        )

        rows.append(
            {
                "bond_id": bond_id,
                "price": price,
                "ytm": ytm,
                "macaulay_duration": (
                    macaulay
                ),
                "modified_duration": (
                    modified
                ),
                "convexity": convexity,
                "krd_2": krds[2],
                "krd_5": krds[5],
                "krd_10": krds[10],
                "z_spread": z_spread,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "bond_id",
            "price",
            "ytm",
            "macaulay_duration",
            "modified_duration",
            "convexity",
            "krd_2",
            "krd_5",
            "krd_10",
            "z_spread",
        ],
    )


def _liability_analytics(
    liability: dict,
    spot_rates: np.ndarray,
) -> tuple[
    dict[str, float],
    np.ndarray,
    np.ndarray,
]:
    cash_flow_rows = liability[
        "cash_flows"
    ]

    times = np.asarray(
        [
            int(
                row[
                    "years"
                ]
            )
            for row
            in cash_flow_rows
        ],
        dtype=int,
    )

    cash_flows = np.asarray(
        [
            float(
                row[
                    "amount"
                ]
            )
            for row
            in cash_flow_rows
        ],
        dtype=float,
    )

    pv = _discount_cash_flows(
        times,
        cash_flows,
        spot_rates,
    )

    pv_components = (
        cash_flows
        * np.exp(
            -spot_rates[
                times
                - 1
            ]
            * times
        )
    )

    macaulay = float(
        np.sum(
            times
            * pv_components
        )
        / pv
    )

    krds = {
        key_rate: _krd(
            times,
            cash_flows,
            spot_rates,
            key_rate,
        )
        for key_rate
        in KEY_RATES
    }

    return (
        {
            "pv_liability": (
                pv
            ),
            "macaulay_duration": (
                macaulay
            ),
            "krd_2": krds[2],
            "krd_5": krds[5],
            "krd_10": krds[10],
        },
        times,
        cash_flows,
    )


def _immunize(
    bond_analytics: pd.DataFrame,
    liability_analytics: dict[str, float],
) -> dict[str, object]:
    bonds = (
        bond_analytics.copy()
        .set_index(
            "bond_id"
        )
    )

    bond_ids = [
        "A",
        "B",
        "C",
        "D",
    ]

    if not set(
        bond_ids
    ).issubset(
        bonds.index
    ):
        raise RuntimeError(
            "Immunization engine expects bonds A, B, C, D."
        )

    target_pv = float(
        liability_analytics[
            "pv_liability"
        ]
    )

    krd2 = bonds.loc[
        bond_ids,
        "krd_2",
    ].to_numpy(
        dtype=float
    )

    krd5 = bonds.loc[
        bond_ids,
        "krd_5",
    ].to_numpy(
        dtype=float
    )

    x0 = np.full(
        4,
        target_pv
        / 4.0,
        dtype=float,
    )

    target_krd2_dollars = (
        target_pv
        * float(
            liability_analytics[
                "krd_2"
            ]
        )
    )

    target_krd5_dollars = (
        target_pv
        * float(
            liability_analytics[
                "krd_5"
            ]
        )
    )

    constraints = [
        {
            "type": "eq",
            "fun": lambda x: (
                float(
                    np.sum(
                        x
                    )
                )
                - target_pv
            ),
        },
        {
            "type": "eq",
            "fun": lambda x: (
                float(
                    np.dot(
                        x,
                        krd2,
                    )
                )
                - target_krd2_dollars
            ),
        },
        {
            "type": "eq",
            "fun": lambda x: (
                float(
                    np.dot(
                        x,
                        krd5,
                    )
                )
                - target_krd5_dollars
            ),
        },
    ]

    def objective(
        allocations: np.ndarray,
    ) -> float:
        scaled = (
            allocations
            - x0
        ) / target_pv

        return float(
            scaled
            @ scaled
        )

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[
            (
                0.0,
                None,
            )
        ]
        * 4,
        constraints=constraints,
        options={
            "ftol": 1e-10,
            "maxiter": 5000,
        },
    )

    if not result.success:
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=[
                (
                    0.0,
                    None,
                )
            ]
            * 4,
            constraints=constraints,
            options={
                "ftol": 1e-8,
                "maxiter": 5000,
            },
        )

    allocations = np.maximum(
        np.asarray(
            result.x,
            dtype=float,
        ),
        0.0,
    )

    portfolio_pv = float(
        np.sum(
            allocations
        )
    )

    if portfolio_pv <= 0.0:
        raise RuntimeError(
            "Immunization optimization produced zero portfolio value."
        )

    portfolio_krd2 = float(
        np.dot(
            allocations,
            bonds.loc[
                bond_ids,
                "krd_2",
            ].to_numpy(
                dtype=float
            ),
        )
        / portfolio_pv
    )

    portfolio_krd5 = float(
        np.dot(
            allocations,
            bonds.loc[
                bond_ids,
                "krd_5",
            ].to_numpy(
                dtype=float
            ),
        )
        / portfolio_pv
    )

    portfolio_krd10 = float(
        np.dot(
            allocations,
            bonds.loc[
                bond_ids,
                "krd_10",
            ].to_numpy(
                dtype=float
            ),
        )
        / portfolio_pv
    )

    output: dict[str, object] = {}

    for index, bond_id in enumerate(
        bond_ids
    ):
        price = float(
            bonds.loc[
                bond_id,
                "price",
            ]
        )

        output[
            f"bond_{bond_id}"
        ] = {
            "units": float(
                allocations[
                    index
                ]
                / price
            ),
            "dollar_allocation": float(
                allocations[
                    index
                ]
            ),
        }

    output[
        "portfolio_pv"
    ] = portfolio_pv

    output[
        "portfolio_krd_2"
    ] = portfolio_krd2

    output[
        "portfolio_krd_5"
    ] = portfolio_krd5

    output[
        "portfolio_krd_10"
    ] = portfolio_krd10

    return output


def _svensson_rates(
    maturities: np.ndarray,
    params: np.ndarray,
) -> np.ndarray:
    (
        beta_0,
        beta_1,
        beta_2,
        beta_3,
        lambda_1,
        lambda_2,
    ) = params

    def loading(
        lam: float,
    ) -> np.ndarray:
        x = (
            maturities
            / lam
        )

        return (
            1.0
            - np.exp(
                -x
            )
        ) / x

    f1 = loading(
        float(
            lambda_1
        )
    )

    f2 = loading(
        float(
            lambda_2
        )
    )

    return (
        beta_0
        + beta_1
        * f1
        + beta_2
        * (
            f1
            - np.exp(
                -maturities
                / lambda_1
            )
        )
        + beta_3
        * (
            f2
            - np.exp(
                -maturities
                / lambda_2
            )
        )
    )


def _fit_svensson(
    spot_rates: np.ndarray,
) -> dict[str, float]:
    maturities = np.arange(
        1,
        len(
            spot_rates
        )
        + 1,
        dtype=float,
    )

    starts = [
        np.asarray(
            [
                float(
                    spot_rates[
                        -1
                    ]
                ),
                float(
                    spot_rates[
                        0
                    ]
                    - spot_rates[
                        -1
                    ]
                ),
                0.5,
                -0.5,
                1.0,
                5.0,
            ],
            dtype=float,
        ),
        np.asarray(
            [
                0.03,
                -0.01,
                0.1,
                -0.1,
                2.0,
                3.0,
            ],
            dtype=float,
        ),
        np.asarray(
            [
                float(
                    np.mean(
                        spot_rates
                    )
                ),
                0.0,
                0.0,
                0.0,
                0.5,
                2.0,
            ],
            dtype=float,
        ),
    ]

    bounds = [
        (
            1e-10,
            None,
        ),
        (
            None,
            None,
        ),
        (
            None,
            None,
        ),
        (
            None,
            None,
        ),
        (
            1e-10,
            None,
        ),
        (
            1e-10,
            None,
        ),
    ]

    def objective(
        params: np.ndarray,
    ) -> float:
        residuals = (
            _svensson_rates(
                maturities,
                params,
            )
            - spot_rates
        )

        return float(
            residuals
            @ residuals
        )

    candidates = []

    for start in starts:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "ftol": 1e-14,
                "gtol": 1e-10,
                "maxiter": 5000,
            },
        )

        candidates.append(
            result
        )

    best = min(
        candidates,
        key=lambda result: float(
            result.fun
        ),
    )

    params = np.asarray(
        best.x,
        dtype=float,
    )

    fitted = _svensson_rates(
        maturities,
        params,
    )

    rmse = float(
        math.sqrt(
            float(
                np.mean(
                    (
                        fitted
                        - spot_rates
                    )
                    ** 2
                )
            )
        )
    )

    return {
        "beta_0": float(
            params[0]
        ),
        "beta_1": float(
            params[1]
        ),
        "beta_2": float(
            params[2]
        ),
        "beta_3": float(
            params[3]
        ),
        "lambda_1": float(
            params[4]
        ),
        "lambda_2": float(
            params[5]
        ),
        "rmse": rmse,
    }


def _stress_shifts() -> dict[str, np.ndarray]:
    maturities = np.arange(
        1,
        11,
        dtype=float,
    )

    parallel = np.full(
        10,
        0.01,
        dtype=float,
    )

    steepener = np.linspace(
        -0.005,
        0.005,
        10,
        dtype=float,
    )

    butterfly_bps = np.interp(
        maturities,
        [
            2.0,
            5.0,
            10.0,
        ],
        [
            30.0,
            -30.0,
            30.0,
        ],
        left=30.0,
        right=30.0,
    )

    butterfly = (
        butterfly_bps
        * 1e-4
    )

    return {
        "parallel_up_100": (
            parallel
        ),
        "steepener": (
            steepener
        ),
        "butterfly": (
            butterfly
        ),
    }


def _stress_test(
    *,
    bonds: pd.DataFrame,
    liability_times: np.ndarray,
    liability_cash_flows: np.ndarray,
    spot_rates: np.ndarray,
    immunization: dict[str, object],
    baseline_liability_pv: float,
) -> pd.DataFrame:
    rows = []

    units = {
        bond_id: float(
            immunization[
                f"bond_{bond_id}"
            ][
                "units"
            ]
        )
        for bond_id
        in (
            "A",
            "B",
            "C",
            "D",
        )
    }

    baseline_portfolio_pv = float(
        immunization[
            "portfolio_pv"
        ]
    )

    for scenario, shift in _stress_shifts().items():
        stressed_spots = (
            spot_rates
            + shift
        )

        portfolio_pv = 0.0

        for row in bonds.itertuples(
            index=False
        ):
            bond_id = str(
                row.bond_id
            )

            times, cash_flows = (
                _bond_cash_flows(
                    float(
                        row.face_value
                    ),
                    float(
                        row.coupon_rate
                    ),
                    int(
                        row.maturity_years
                    ),
                )
            )

            stressed_price = (
                _discount_cash_flows(
                    times,
                    cash_flows,
                    stressed_spots,
                )
            )

            portfolio_pv += (
                units[
                    bond_id
                ]
                * stressed_price
            )

        liability_pv = (
            _discount_cash_flows(
                liability_times,
                liability_cash_flows,
                stressed_spots,
            )
        )

        rows.append(
            {
                "scenario": scenario,
                "portfolio_pv": float(
                    portfolio_pv
                ),
                "liability_pv": float(
                    liability_pv
                ),
                "portfolio_pv_change": float(
                    portfolio_pv
                    - baseline_portfolio_pv
                ),
                "liability_pv_change": float(
                    liability_pv
                    - baseline_liability_pv
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "scenario",
            "portfolio_pv",
            "liability_pv",
            "portfolio_pv_change",
            "liability_pv_change",
        ],
    )


@dataclass(frozen=True)
class CurveImmunizationSkill:
    name: str = "curve-immunization-domain"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        del task_dir

        lowered = instruction.lower()

        return (
            "yield curve" in lowered
            and "bootstrap" in lowered
            and "immunization" in lowered
            and (
                "key rate duration" in lowered
                or "key rate durations" in lowered
            )
            and (
                "svensson" in lowered
                or "nelson-siegel" in lowered
                or "nelson siegel" in lowered
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
        del instruction, seed

        data_dir = _find_curve_data_dir(
            task_dir
        )

        par_yields = pd.read_csv(
            data_dir
            / "par_yields.csv"
        )

        bonds = pd.read_csv(
            data_dir
            / "bonds.csv"
        )

        liability = json.loads(
            (
                data_dir
                / "liability.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        curve = (
            bootstrap_discount_factors(
                par_yields
            )
        )

        spot_rates = curve[
            "spot_rate"
        ].to_numpy(
            dtype=float
        )

        bond_analytics = (
            _bond_analytics(
                bonds,
                spot_rates,
            )
        )

        (
            liability_analytics,
            liability_times,
            liability_cash_flows,
        ) = _liability_analytics(
            liability,
            spot_rates,
        )

        immunization = _immunize(
            bond_analytics,
            liability_analytics,
        )

        svensson = _fit_svensson(
            spot_rates
        )

        stress = _stress_test(
            bonds=bonds,
            liability_times=(
                liability_times
            ),
            liability_cash_flows=(
                liability_cash_flows
            ),
            spot_rates=spot_rates,
            immunization=immunization,
            baseline_liability_pv=float(
                liability_analytics[
                    "pv_liability"
                ]
            ),
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        curve.to_csv(
            out_dir
            / "discount_factors.csv",
            index=False,
        )

        bond_analytics.to_csv(
            out_dir
            / "bond_analytics.csv",
            index=False,
        )

        liability_output = {
            "pv_liability": round(
                float(
                    liability_analytics[
                        "pv_liability"
                    ]
                ),
                4,
            ),
            "macaulay_duration": round(
                float(
                    liability_analytics[
                        "macaulay_duration"
                    ]
                ),
                6,
            ),
            "krd_2": round(
                float(
                    liability_analytics[
                        "krd_2"
                    ]
                ),
                6,
            ),
            "krd_5": round(
                float(
                    liability_analytics[
                        "krd_5"
                    ]
                ),
                6,
            ),
            "krd_10": round(
                float(
                    liability_analytics[
                        "krd_10"
                    ]
                ),
                6,
            ),
        }

        (
            out_dir
            / "liability_analytics.json"
        ).write_text(
            json.dumps(
                liability_output,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        immunization_output: dict[
            str,
            object,
        ] = {}

        for bond_id in (
            "A",
            "B",
            "C",
            "D",
        ):
            values = immunization[
                f"bond_{bond_id}"
            ]

            immunization_output[
                f"bond_{bond_id}"
            ] = {
                "units": round(
                    float(
                        values[
                            "units"
                        ]
                    ),
                    6,
                ),
                "dollar_allocation": round(
                    float(
                        values[
                            "dollar_allocation"
                        ]
                    ),
                    4,
                ),
            }

        for field, decimals in (
            (
                "portfolio_pv",
                4,
            ),
            (
                "portfolio_krd_2",
                6,
            ),
            (
                "portfolio_krd_5",
                6,
            ),
            (
                "portfolio_krd_10",
                6,
            ),
        ):
            immunization_output[
                field
            ] = round(
                float(
                    immunization[
                        field
                    ]
                ),
                decimals,
            )

        (
            out_dir
            / "immunization_portfolio.json"
        ).write_text(
            json.dumps(
                immunization_output,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        svensson_output = {
            "beta_0": round(
                svensson[
                    "beta_0"
                ],
                8,
            ),
            "beta_1": round(
                svensson[
                    "beta_1"
                ],
                8,
            ),
            "beta_2": round(
                svensson[
                    "beta_2"
                ],
                8,
            ),
            "beta_3": round(
                svensson[
                    "beta_3"
                ],
                8,
            ),
            "lambda_1": round(
                svensson[
                    "lambda_1"
                ],
                8,
            ),
            "lambda_2": round(
                svensson[
                    "lambda_2"
                ],
                8,
            ),
            "rmse": round(
                svensson[
                    "rmse"
                ],
                10,
            ),
        }

        (
            out_dir
            / "nelson_siegel.json"
        ).write_text(
            json.dumps(
                svensson_output,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        stress.to_csv(
            out_dir
            / "stress_test.csv",
            index=False,
        )
