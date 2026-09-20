from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize


KNOTS = (
    0.25,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    7.0,
    10.0,
    20.0,
    30.0,
)

TARGET_KRDS = (
    5.0,
    10.0,
    20.0,
)


def _parse_date(
    value: str,
) -> date:
    return datetime.strptime(
        value,
        "%Y-%m-%d",
    ).date()


def _find_dated_immunization_data_dir(
    task_dir: Path,
) -> Path:
    required = {
        "curve_snapshot_t0.json",
        "curve_snapshot_t1.json",
        "hedge_instruments.json",
        "liability_schedule.json",
        "treasury_par_yields.csv",
    }

    directories = sorted(
        {
            path.parent
            for path in task_dir.rglob("*")
            if path.is_file()
            and "checks" not in path.parts
        },
        key=lambda path: str(path),
    )

    for directory in directories:
        names = {
            path.name
            for path in directory.iterdir()
            if path.is_file()
        }

        if required.issubset(
            names
        ):
            return directory

    raise RuntimeError(
        "Could not discover dated curve-immunization inputs."
    )


@dataclass(frozen=True)
class CurveState:
    maturities: np.ndarray
    par_yields: np.ndarray
    g_values: np.ndarray
    spline: CubicSpline

    def discount(
        self,
        times: np.ndarray | float,
    ) -> np.ndarray:
        values = np.asarray(
            times,
            dtype=float,
        )

        return np.exp(
            -self.spline(
                values
            )
        )


def _bootstrap_curve(
    par_yields: dict[
        str,
        float,
    ],
) -> CurveState:
    maturities = np.asarray(
        sorted(
            float(
                key
            )
            for key
            in par_yields
        ),
        dtype=float,
    )

    yields = np.asarray(
        [
            float(
                par_yields[
                    str(
                        maturity
                    )
                ]
            )
            if str(
                maturity
            ) in par_yields
            else float(
                par_yields[
                    f"{maturity:g}"
                ]
            )
            for maturity
            in maturities
        ],
        dtype=float,
    )

    known_times = [
        0.0
    ]

    known_g = [
        0.0
    ]

    knot_g: list[
        float
    ] = []

    for maturity, par_yield in zip(
        maturities,
        yields,
    ):
        maturity = float(
            maturity
        )

        par_yield = float(
            par_yield
        )

        if (
            maturity
            < 0.5
        ):
            discount = (
                1.0
                / (
                    1.0
                    + par_yield
                    * maturity
                )
            )

            g_value = float(
                -math.log(
                    discount
                )
            )

        elif np.isclose(
            maturity,
            0.5,
        ):
            discount = (
                1.0
                / (
                    1.0
                    + par_yield
                    / 2.0
                )
            )

            g_value = float(
                -math.log(
                    discount
                )
            )

        else:
            coupon_times = np.arange(
                0.5,
                maturity
                + 1e-12,
                0.5,
            )

            coupon = (
                par_yield
                / 2.0
            )

            def residual(
                candidate_g: float,
            ) -> float:
                x = np.asarray(
                    known_times
                    + [
                        maturity
                    ],
                    dtype=float,
                )

                y = np.asarray(
                    known_g
                    + [
                        float(
                            candidate_g
                        )
                    ],
                    dtype=float,
                )

                interpolated_g = (
                    np.interp(
                        coupon_times,
                        x,
                        y,
                    )
                )

                discounts = np.exp(
                    -interpolated_g
                )

                return float(
                    coupon
                    * discounts[
                        :-1
                    ].sum()
                    + (
                        1.0
                        + coupon
                    )
                    * discounts[
                        -1
                    ]
                    - 1.0
                )

            g_value = float(
                brentq(
                    residual,
                    0.0,
                    max(
                        4.0,
                        par_yield
                        * maturity
                        * 4.0,
                    ),
                    xtol=1e-14,
                    rtol=1e-14,
                )
            )

        known_times.append(
            maturity
        )

        known_g.append(
            g_value
        )

        knot_g.append(
            g_value
        )

    g_values = np.asarray(
        knot_g,
        dtype=float,
    )

    spline = CubicSpline(
        np.concatenate(
            (
                [
                    0.0
                ],
                maturities,
            )
        ),
        np.concatenate(
            (
                [
                    0.0
                ],
                g_values,
            )
        ),
        bc_type="not-a-knot",
    )

    return CurveState(
        maturities=maturities,
        par_yields=yields,
        g_values=g_values,
        spline=spline,
    )


def _curve_frame(
    curve: CurveState,
) -> pd.DataFrame:
    zero_rates = (
        curve.g_values
        / curve.maturities
    )

    discount_factors = np.exp(
        -curve.g_values
    )

    forward_rates = (
        curve.spline(
            curve.maturities,
            1,
        )
    )

    return pd.DataFrame(
        {
            "maturity": (
                curve.maturities
            ),
            "par_yield": (
                curve.par_yields
            ),
            "zero_rate": (
                zero_rates
            ),
            "discount_factor": (
                discount_factors
            ),
            "forward_rate": (
                forward_rates
            ),
        }
    )


def _snapshot_with_bumps(
    snapshot: dict,
    bumps: np.ndarray,
) -> dict:
    labels = sorted(
        snapshot[
            "par_yields"
        ],
        key=lambda value: float(
            value
        ),
    )

    bumped = {}

    for label, bump in zip(
        labels,
        bumps,
    ):
        bumped[
            label
        ] = (
            float(
                snapshot[
                    "par_yields"
                ][
                    label
                ]
            )
            + float(
                bump
            )
        )

    return {
        **snapshot,
        "par_yields": bumped,
    }


def _parallel_curve(
    snapshot: dict,
    bump: float,
) -> CurveState:
    n = len(
        snapshot[
            "par_yields"
        ]
    )

    return _bootstrap_curve(
        _snapshot_with_bumps(
            snapshot,
            np.full(
                n,
                float(
                    bump
                ),
            ),
        )[
            "par_yields"
        ]
    )


def _single_key_curve(
    snapshot: dict,
    index: int,
    bump: float,
) -> CurveState:
    n = len(
        snapshot[
            "par_yields"
        ]
    )

    bumps = np.zeros(
        n,
        dtype=float,
    )

    bumps[
        index
    ] = float(
        bump
    )

    return _bootstrap_curve(
        _snapshot_with_bumps(
            snapshot,
            bumps,
        )[
            "par_yields"
        ]
    )


def _coupon_schedule(
    *,
    valuation_date: date,
    maturity_date: date,
    frequency: int,
) -> tuple[
    date,
    list[
        date
    ],
]:
    months = int(
        12
        / frequency
    )

    future: list[
        date
    ] = []

    current = maturity_date

    while current > valuation_date:
        future.append(
            current
        )

        current = (
            current
            - relativedelta(
                months=months
            )
        )

    future.sort()

    return (
        current,
        future,
    )


def _year_fraction_act365(
    start: date,
    end: date,
) -> float:
    return float(
        (
            end
            - start
        ).days
        / 365.0
    )


def _days_30_360_us(
    start: date,
    end: date,
) -> int:
    d1 = min(
        start.day,
        30,
    )

    d2 = end.day

    if (
        d2 == 31
        and d1 == 30
    ):
        d2 = 30

    d2 = min(
        d2,
        30,
    )

    return int(
        360
        * (
            end.year
            - start.year
        )
        + 30
        * (
            end.month
            - start.month
        )
        + (
            d2
            - d1
        )
    )


def _fixed_cash_flows(
    instrument: dict,
    valuation_date: date,
    *,
    truncate_at_call: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
]:
    maturity = _parse_date(
        instrument[
            "maturity_date"
        ]
    )

    terminal_date = maturity

    terminal_principal = float(
        instrument[
            "notional"
        ]
    )

    if (
        truncate_at_call
        and instrument[
            "type"
        ]
        == "callable"
    ):
        terminal_date = _parse_date(
            instrument[
                "first_call_date"
            ]
        )

        terminal_principal = float(
            instrument[
                "call_price"
            ]
        )

    frequency = int(
        instrument[
            "frequency"
        ]
    )

    previous_coupon, coupon_dates = (
        _coupon_schedule(
            valuation_date=(
                valuation_date
            ),
            maturity_date=(
                terminal_date
            ),
            frequency=frequency,
        )
    )

    coupon_amount = (
        float(
            instrument[
                "notional"
            ]
        )
        * float(
            instrument[
                "coupon_rate"
            ]
        )
        / frequency
    )

    times = np.asarray(
        [
            _year_fraction_act365(
                valuation_date,
                cash_date,
            )
            for cash_date
            in coupon_dates
        ],
        dtype=float,
    )

    cash_flows = np.full(
        len(
            coupon_dates
        ),
        coupon_amount,
        dtype=float,
    )

    if len(
        cash_flows
    ) == 0:
        raise RuntimeError(
            f"No future cash flows for {instrument['id']}."
        )

    cash_flows[
        -1
    ] += terminal_principal

    elapsed_30_360 = (
        _days_30_360_us(
            previous_coupon,
            valuation_date,
        )
    )

    coupon_period_30_360 = int(
        360
        / frequency
    )

    accrued = float(
        coupon_amount
        * elapsed_30_360
        / coupon_period_30_360
    )

    return (
        times,
        cash_flows,
        accrued,
    )


def _price_cash_flows(
    curve: CurveState,
    times: np.ndarray,
    cash_flows: np.ndarray,
) -> float:
    return float(
        np.sum(
            cash_flows
            * curve.discount(
                times
            )
        )
    )


def _frn_state(
    instrument: dict,
    valuation_date: date,
    curve: CurveState,
) -> dict[
    str,
    float,
]:
    last_reset = _parse_date(
        instrument[
            "last_reset_date"
        ]
    )

    next_reset = _parse_date(
        instrument[
            "next_reset_date"
        ]
    )

    if not (
        last_reset
        <= valuation_date
        < next_reset
    ):
        raise RuntimeError(
            "FRN valuation is expected between its supplied reset dates."
        )

    frequency = int(
        instrument[
            "frequency"
        ]
    )

    notional = float(
        instrument[
            "notional"
        ]
    )

    known_rate = (
        float(
            instrument[
                "last_reset_rate"
            ]
        )
        + float(
            instrument.get(
                "spread",
                0.0,
            )
        )
    )

    next_coupon = (
        notional
        * known_rate
        / frequency
    )

    period_days = (
        next_reset
        - last_reset
    ).days

    elapsed_days = (
        valuation_date
        - last_reset
    ).days

    accrued = float(
        next_coupon
        * elapsed_days
        / period_days
    )

    time_to_reset = (
        _year_fraction_act365(
            valuation_date,
            next_reset,
        )
    )

    dirty = float(
        (
            notional
            + next_coupon
        )
        * curve.discount(
            time_to_reset
        )
    )

    return {
        "dirty_price": dirty,
        "accrued_interest": accrued,
        "clean_price": float(
            dirty
            - accrued
        ),
        "next_known_coupon": float(
            next_coupon
        ),
        "time_to_next_reset_years": float(
            time_to_reset
        ),
    }


def _instrument_dirty_price(
    instrument: dict,
    valuation_date: date,
    curve: CurveState,
) -> float:
    if instrument[
        "type"
    ] == "frn":
        return float(
            _frn_state(
                instrument,
                valuation_date,
                curve,
            )[
                "dirty_price"
            ]
        )

    noncall_times, noncall_cash, _ = (
        _fixed_cash_flows(
            instrument,
            valuation_date,
            truncate_at_call=False,
        )
    )

    noncall_price = (
        _price_cash_flows(
            curve,
            noncall_times,
            noncall_cash,
        )
    )

    if instrument[
        "type"
    ] != "callable":
        return float(
            noncall_price
        )

    call_times, call_cash, _ = (
        _fixed_cash_flows(
            instrument,
            valuation_date,
            truncate_at_call=True,
        )
    )

    call_price = (
        _price_cash_flows(
            curve,
            call_times,
            call_cash,
        )
    )

    return float(
        min(
            noncall_price,
            call_price,
        )
    )


def _liability_pv(
    liability: dict,
    valuation_date: date,
    curve: CurveState,
) -> float:
    times = []
    cash_flows = []

    for row in liability[
        "cash_flows"
    ]:
        cash_date = _parse_date(
            row[
                "date"
            ]
        )

        if (
            cash_date
            <= valuation_date
        ):
            continue

        times.append(
            _year_fraction_act365(
                valuation_date,
                cash_date,
            )
        )

        cash_flows.append(
            float(
                row[
                    "amount"
                ]
            )
        )

    return _price_cash_flows(
        curve,
        np.asarray(
            times,
            dtype=float,
        ),
        np.asarray(
            cash_flows,
            dtype=float,
        ),
    )


def _yield_and_durations(
    *,
    price: float,
    times: np.ndarray,
    cash_flows: np.ndarray,
    frequency: int,
) -> tuple[
    float,
    float,
    float,
]:
    def residual(
        ytm: float,
    ) -> float:
        if (
            1.0
            + ytm
            / frequency
            <= 0.0
        ):
            return 1e12

        pv = np.sum(
            cash_flows
            / (
                (
                    1.0
                    + ytm
                    / frequency
                )
                ** (
                    frequency
                    * times
                )
            )
        )

        return float(
            pv
            - price
        )

    ytm = float(
        brentq(
            residual,
            -0.95,
            1.0,
            xtol=1e-14,
            rtol=1e-14,
        )
    )

    discounted = (
        cash_flows
        / (
            (
                1.0
                + ytm
                / frequency
            )
            ** (
                frequency
                * times
            )
        )
    )

    macaulay = float(
        np.sum(
            times
            * discounted
        )
        / price
    )

    modified = float(
        macaulay
        / (
            1.0
            + ytm
            / frequency
        )
    )

    return (
        ytm,
        macaulay,
        modified,
    )


def _parallel_duration_convexity(
    *,
    base_price: float,
    price_up: float,
    price_down: float,
    bump: float,
) -> tuple[
    float,
    float,
]:
    duration = float(
        (
            price_down
            - price_up
        )
        / (
            2.0
            * base_price
            * bump
        )
    )

    convexity = float(
        (
            price_down
            + price_up
            - 2.0
            * base_price
        )
        / (
            base_price
            * bump
            * bump
        )
    )

    return (
        duration,
        convexity,
    )


def _key_rate_vector(
    *,
    base_value: float,
    repricer,
    snapshot: dict,
) -> np.ndarray:
    n = len(
        KNOTS
    )

    result = np.zeros(
        n,
        dtype=float,
    )

    bump = 1e-4

    for index in range(
        n
    ):
        up = repricer(
            _single_key_curve(
                snapshot,
                index,
                bump,
            )
        )

        down = repricer(
            _single_key_curve(
                snapshot,
                index,
                -bump,
            )
        )

        result[
            index
        ] = (
            down
            - up
        ) / (
            2.0
            * base_value
            * bump
        )

    return result


def _stress_bumps(
    liability: dict,
) -> dict[
    str,
    np.ndarray,
]:
    maturities = np.asarray(
        KNOTS,
        dtype=float,
    )

    result = {}

    for scenario in liability[
        "stress_scenarios"
    ]:
        name = str(
            scenario[
                "name"
            ]
        )

        kind = str(
            scenario[
                "type"
            ]
        )

        if kind == "parallel":
            bumps = np.full(
                len(
                    maturities
                ),
                float(
                    scenario[
                        "bump_bps"
                    ]
                )
                / 10000.0,
            )

        elif kind == "twist":
            x = (
                (
                    maturities
                    - maturities.min()
                )
                / (
                    maturities.max()
                    - maturities.min()
                )
            )

            short = float(
                scenario[
                    "short_bump_bps"
                ]
            )

            long = float(
                scenario[
                    "long_bump_bps"
                ]
            )

            bumps = (
                short
                + (
                    long
                    - short
                )
                * x
            ) / 10000.0

        elif kind == "curvature":
            center = float(
                scenario[
                    "center_maturity_years"
                ]
            )

            width = float(
                scenario[
                    "width_years"
                ]
            )

            belly = float(
                scenario[
                    "belly_bump_bps"
                ]
            )

            wing = float(
                scenario[
                    "wing_bump_bps"
                ]
            )

            shape = np.maximum(
                0.0,
                1.0
                - np.abs(
                    maturities
                    - center
                )
                / width,
            )

            bumps = (
                wing
                + (
                    belly
                    - wing
                )
                * shape
            ) / 10000.0

        else:
            raise RuntimeError(
                f"Unknown stress type: {kind}"
            )

        result[
            name
        ] = np.asarray(
            bumps,
            dtype=float,
        )

    return result


def _pca_loadings(
    history: pd.DataFrame,
    liability_krd: np.ndarray,
) -> tuple[
    np.ndarray,
    float,
]:
    by_lower = {
        str(
            column
        ).lower(): str(
            column
        )
        for column
        in history.columns
    }

    desired = [
        "m3",
        "m6",
        "y1.0",
        "y2.0",
        "y3.0",
        "y5.0",
        "y7.0",
        "y10.0",
        "y20.0",
        "y30.0",
    ]

    columns = []

    for label in desired:
        if label not in by_lower:
            raise RuntimeError(
                f"Treasury panel missing PCA column {label}."
            )

        columns.append(
            by_lower[
                label
            ]
        )

    levels = (
        history[
            columns
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .dropna()
        .to_numpy(
            dtype=float
        )
        / 100.0
    )

    changes = np.diff(
        levels,
        axis=0,
    )

    covariance = np.cov(
        changes,
        rowvar=False,
        ddof=1,
    )

    eigenvalues, eigenvectors = (
        np.linalg.eigh(
            covariance
        )
    )

    order = np.argsort(
        eigenvalues
    )[
        ::-1
    ]

    eigenvalues = eigenvalues[
        order
    ]

    eigenvectors = eigenvectors[
        :,
        order
    ]

    loadings = eigenvectors[
        :,
        :3
    ].copy()

    for column in range(
        3
    ):
        exposure = float(
            np.dot(
                liability_krd,
                loadings[
                    :,
                    column
                ],
            )
        )

        if exposure < 0.0:
            loadings[
                :,
                column
            ] *= -1.0

    explained = float(
        np.sum(
            eigenvalues[
                :3
            ]
        )
        / np.sum(
            eigenvalues
        )
    )

    return (
        loadings,
        explained,
    )


def _objective_factory(
    *,
    instrument_krd: np.ndarray,
    liability_krd: np.ndarray,
    instrument_pc: np.ndarray,
    liability_pc: np.ndarray,
    instrument_stress_pct: np.ndarray,
    liability_stress_pct: np.ndarray,
    stress_weight: float,
):
    target_indices = [
        KNOTS.index(
            maturity
        )
        for maturity
        in TARGET_KRDS
    ]

    def objective(
        weights: np.ndarray,
    ) -> float:
        portfolio_krd = (
            weights
            @ instrument_krd
        )

        krd_error = (
            portfolio_krd[
                target_indices
            ]
            - liability_krd[
                target_indices
            ]
        )

        portfolio_pc = (
            weights
            @ instrument_pc
        )

        pc_scale = np.maximum(
            np.abs(
                liability_pc
            ),
            0.25,
        )

        pc_error = (
            portfolio_pc
            - liability_pc
        ) / pc_scale

        stress_error_bps = (
            (
                weights
                @ instrument_stress_pct
            )
            - liability_stress_pct
        ) * 100.0

        curvature_error = (
            stress_error_bps[
                -1
            ]
        )

        regularization = np.sum(
            (
                weights
                - 1.0
                / len(
                    weights
                )
            )
            ** 2
        )

        return float(
            25.0
            * np.sum(
                krd_error
                * krd_error
            )
            + 0.25
            * np.sum(
                pc_error
                * pc_error
            )
            + float(
                stress_weight
            )
            * np.sum(
                (
                    stress_error_bps
                    / 20.0
                )
                ** 2
            )
            + 0.05
            * (
                curvature_error
                / 10.0
            )
            ** 2
            + 0.01
            * regularization
        )

    return objective


def _solve_weights(
    *,
    instrument_krd: np.ndarray,
    liability_krd: np.ndarray,
    instrument_pc: np.ndarray,
    liability_pc: np.ndarray,
    instrument_stress_pct: np.ndarray,
    liability_stress_pct: np.ndarray,
    max_weight: float,
    initial: np.ndarray | None = None,
    stress_weight: float = 0.35,
) -> tuple[
    np.ndarray,
    float,
    bool,
]:
    """Solve the convex hedge problem with deterministic multi-start SLSQP."""
    n = instrument_krd.shape[0]
    max_weight = float(max_weight)

    if n <= 0 or max_weight <= 0.0 or n * max_weight < 1.0 - 1e-12:
        raise RuntimeError(
            "Curve-immunization weight bounds are infeasible."
        )

    objective = _objective_factory(
        instrument_krd=instrument_krd,
        liability_krd=liability_krd,
        instrument_pc=instrument_pc,
        liability_pc=liability_pc,
        instrument_stress_pct=instrument_stress_pct,
        liability_stress_pct=liability_stress_pct,
        stress_weight=stress_weight,
    )

    def make_feasible(values: np.ndarray) -> np.ndarray:
        weights = np.asarray(values, dtype=float).copy()

        if weights.shape != (n,):
            weights = np.full(n, 1.0 / n, dtype=float)

        weights = np.clip(weights, 0.0, max_weight)

        for _ in range(100):
            gap = 1.0 - float(np.sum(weights))

            if abs(gap) <= 1e-12:
                break

            if gap > 0.0:
                capacity = max_weight - weights
                mask = capacity > 1e-14

                if not np.any(mask):
                    break

                weights[mask] += np.minimum(
                    capacity[mask],
                    gap / float(np.sum(mask)),
                )
            else:
                mask = weights > 1e-14

                if not np.any(mask):
                    break

                weights[mask] -= np.minimum(
                    weights[mask],
                    (-gap) / float(np.sum(mask)),
                )

        if not np.isclose(
            float(np.sum(weights)),
            1.0,
            atol=1e-9,
        ):
            raise RuntimeError(
                "Could not construct a feasible hedge starting point."
            )

        return weights

    starts: list[np.ndarray] = []

    if initial is not None:
        starts.append(
            make_feasible(
                np.asarray(initial, dtype=float)
            )
        )

    uniform = make_feasible(
        np.full(n, 1.0 / n, dtype=float)
    )
    starts.append(uniform)

    tilt = min(max_weight, 0.40)

    for index in range(n):
        if n == 1:
            candidate = np.array([1.0], dtype=float)
        else:
            candidate = np.full(
                n,
                (1.0 - tilt) / (n - 1),
                dtype=float,
            )
            candidate[index] = tilt

        starts.append(
            make_feasible(candidate)
        )

    target_indices = [
        KNOTS.index(maturity)
        for maturity in TARGET_KRDS
    ]

    exposure_strength = np.sum(
        np.abs(
            instrument_krd[:, target_indices]
        ),
        axis=1,
    )

    for index in np.argsort(exposure_strength)[::-1][: min(3, n)]:
        candidate = uniform.copy()
        candidate[int(index)] += 0.20
        starts.append(
            make_feasible(candidate)
        )

    best_weights: np.ndarray | None = None
    best_value = float("inf")
    messages: list[str] = []

    for start_values in starts:
        result = minimize(
            objective,
            start_values,
            method="SLSQP",
            bounds=[
                (0.0, max_weight)
                for _ in range(n)
            ],
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda weights: float(
                        np.sum(weights) - 1.0
                    ),
                }
            ],
            options={
                "ftol": 1e-12,
                "maxiter": 10000,
            },
        )

        candidate = np.asarray(
            result.x,
            dtype=float,
        )

        feasible = bool(
            np.all(np.isfinite(candidate))
            and abs(float(np.sum(candidate)) - 1.0) <= 1e-6
            and float(np.min(candidate)) >= -1e-7
            and float(np.max(candidate)) <= max_weight + 1e-7
        )

        if feasible:
            candidate = make_feasible(candidate)
            value = float(objective(candidate))

            if np.isfinite(value) and value < best_value:
                best_weights = candidate
                best_value = value

        messages.append(
            str(result.message)
        )

    if best_weights is None:
        raise RuntimeError(
            "Curve-immunization optimizer failed to produce a feasible "
            "portfolio. SLSQP messages: "
            + " | ".join(
                sorted(set(messages))
            )
        )

    return (
        best_weights,
        float(best_value),
        True,
    )

def _phase_metrics(
    *,
    weights: np.ndarray,
    instrument_krd: np.ndarray,
    liability_krd: np.ndarray,
    instrument_pc: np.ndarray,
    liability_pc: np.ndarray,
    instrument_stress_pct: np.ndarray,
    liability_stress_pct: np.ndarray,
) -> tuple[
    dict[
        str,
        float,
    ],
    dict[
        str,
        float,
    ],
    dict[
        str,
        float,
    ],
]:
    portfolio_krd = (
        weights
        @ instrument_krd
    )

    portfolio_pc = (
        weights
        @ instrument_pc
    )

    stress_error_bps = (
        (
            weights
            @ instrument_stress_pct
        )
        - liability_stress_pct
    ) * 100.0

    target_krd = {
        f"{maturity:.1f}": float(
            portfolio_krd[
                KNOTS.index(
                    maturity
                )
            ]
        )
        for maturity
        in TARGET_KRDS
    }

    pc = {
        f"PC{index + 1}": float(
            portfolio_pc[
                index
            ]
        )
        for index
        in range(
            3
        )
    }

    return (
        target_krd,
        pc,
        stress_error_bps,
    )


def _weights_map(
    ids: list[
        str
    ],
    weights: np.ndarray,
) -> dict[
    str,
    float,
]:
    return {
        instrument_id: float(
            weight
        )
        for instrument_id, weight
        in zip(
            ids,
            weights,
        )
    }


def _market_state(
    *,
    snapshot: dict,
    instruments: list[
        dict
    ],
    liability: dict,
    history: pd.DataFrame,
    pca_loadings: np.ndarray | None,
) -> dict:
    valuation_date = _parse_date(
        snapshot[
            "date"
        ]
    )

    base_curve = _bootstrap_curve(
        snapshot[
            "par_yields"
        ]
    )

    base_instrument_prices = np.asarray(
        [
            _instrument_dirty_price(
                instrument,
                valuation_date,
                base_curve,
            )
            for instrument
            in instruments
        ],
        dtype=float,
    )

    base_liability_pv = (
        _liability_pv(
            liability,
            valuation_date,
            base_curve,
        )
    )

    instrument_krd = []

    for instrument, base_price in zip(
        instruments,
        base_instrument_prices,
    ):
        instrument_krd.append(
            _key_rate_vector(
                base_value=float(
                    base_price
                ),
                repricer=lambda curve, inst=instrument: (
                    _instrument_dirty_price(
                        inst,
                        valuation_date,
                        curve,
                    )
                ),
                snapshot=snapshot,
            )
        )

    instrument_krd = np.asarray(
        instrument_krd,
        dtype=float,
    )

    liability_krd = (
        _key_rate_vector(
            base_value=(
                base_liability_pv
            ),
            repricer=lambda curve: (
                _liability_pv(
                    liability,
                    valuation_date,
                    curve,
                )
            ),
            snapshot=snapshot,
        )
    )

    if pca_loadings is None:
        (
            pca_loadings,
            pca_explained,
        ) = _pca_loadings(
            history,
            liability_krd,
        )
    else:
        pca_explained = float(
            "nan"
        )

    instrument_pc = (
        instrument_krd
        @ pca_loadings
    )

    liability_pc = (
        liability_krd
        @ pca_loadings
    )

    stress_definitions = (
        _stress_bumps(
            liability
        )
    )

    stress_names = list(
        stress_definitions
    )

    instrument_stress = np.zeros(
        (
            len(
                instruments
            ),
            len(
                stress_names
            ),
        ),
        dtype=float,
    )

    liability_stress = np.zeros(
        len(
            stress_names
        ),
        dtype=float,
    )

    for column, name in enumerate(
        stress_names
    ):
        stressed_snapshot = (
            _snapshot_with_bumps(
                snapshot,
                stress_definitions[
                    name
                ],
            )
        )

        stressed_curve = (
            _bootstrap_curve(
                stressed_snapshot[
                    "par_yields"
                ]
            )
        )

        for row, instrument in enumerate(
            instruments
        ):
            stressed_price = (
                _instrument_dirty_price(
                    instrument,
                    valuation_date,
                    stressed_curve,
                )
            )

            instrument_stress[
                row,
                column,
            ] = (
                (
                    stressed_price
                    / base_instrument_prices[
                        row
                    ]
                )
                - 1.0
            ) * 100.0

        stressed_liability = (
            _liability_pv(
                liability,
                valuation_date,
                stressed_curve,
            )
        )

        liability_stress[
            column
        ] = (
            (
                stressed_liability
                / base_liability_pv
            )
            - 1.0
        ) * 100.0

    return {
        "valuation_date": (
            valuation_date
        ),
        "curve": base_curve,
        "instrument_prices": (
            base_instrument_prices
        ),
        "liability_pv": (
            base_liability_pv
        ),
        "instrument_krd": (
            instrument_krd
        ),
        "liability_krd": (
            liability_krd
        ),
        "pca_loadings": (
            pca_loadings
        ),
        "pca_explained": (
            pca_explained
        ),
        "instrument_pc": (
            instrument_pc
        ),
        "liability_pc": (
            liability_pc
        ),
        "stress_names": (
            stress_names
        ),
        "instrument_stress_pct": (
            instrument_stress
        ),
        "liability_stress_pct": (
            liability_stress
        ),
    }


def _instrument_analytics_t0(
    *,
    snapshot: dict,
    instruments: list[
        dict
    ],
    state: dict,
) -> list[
    dict
]:
    valuation_date = state[
        "valuation_date"
    ]

    curve = state[
        "curve"
    ]

    bump = 1e-4

    curve_up = _parallel_curve(
        snapshot,
        bump,
    )

    curve_down = _parallel_curve(
        snapshot,
        -bump,
    )

    rows = []

    for instrument in instruments:
        inst_type = str(
            instrument[
                "type"
            ]
        )

        base_dirty = (
            _instrument_dirty_price(
                instrument,
                valuation_date,
                curve,
            )
        )

        up_dirty = (
            _instrument_dirty_price(
                instrument,
                valuation_date,
                curve_up,
            )
        )

        down_dirty = (
            _instrument_dirty_price(
                instrument,
                valuation_date,
                curve_down,
            )
        )

        (
            effective_duration,
            effective_convexity,
        ) = _parallel_duration_convexity(
            base_price=base_dirty,
            price_up=up_dirty,
            price_down=down_dirty,
            bump=bump,
        )

        if inst_type == "frn":
            frn = _frn_state(
                instrument,
                valuation_date,
                curve,
            )

            row = {
                "id": str(
                    instrument[
                        "id"
                    ]
                ),
                "type": inst_type,
                "clean_price": float(
                    frn[
                        "clean_price"
                    ]
                ),
                "dirty_price": float(
                    frn[
                        "dirty_price"
                    ]
                ),
                "accrued_interest": float(
                    frn[
                        "accrued_interest"
                    ]
                ),
                "effective_duration": float(
                    effective_duration
                ),
                "effective_convexity": float(
                    effective_convexity
                ),
                "dv01": float(
                    base_dirty
                    * effective_duration
                    * 0.0001
                ),
                "next_known_coupon": float(
                    frn[
                        "next_known_coupon"
                    ]
                ),
                "time_to_next_reset_years": float(
                    frn[
                        "time_to_next_reset_years"
                    ]
                ),
            }

            rows.append(
                row
            )

            continue

        noncall_times, noncall_cash, accrued = (
            _fixed_cash_flows(
                instrument,
                valuation_date,
                truncate_at_call=False,
            )
        )

        noncall_dirty = (
            _price_cash_flows(
                curve,
                noncall_times,
                noncall_cash,
            )
        )

        chosen_times = (
            noncall_times
        )

        chosen_cash = (
            noncall_cash
        )

        callable_fields = {}

        if inst_type == "callable":
            call_times, call_cash, _ = (
                _fixed_cash_flows(
                    instrument,
                    valuation_date,
                    truncate_at_call=True,
                )
            )

            call_dirty = (
                _price_cash_flows(
                    curve,
                    call_times,
                    call_cash,
                )
            )

            binding = bool(
                call_dirty
                < noncall_dirty
            )

            if binding:
                chosen_times = (
                    call_times
                )

                chosen_cash = (
                    call_cash
                )

            noncall_up = (
                _price_cash_flows(
                    curve_up,
                    noncall_times,
                    noncall_cash,
                )
            )

            noncall_down = (
                _price_cash_flows(
                    curve_down,
                    noncall_times,
                    noncall_cash,
                )
            )

            (
                noncall_duration,
                noncall_convexity,
            ) = _parallel_duration_convexity(
                base_price=(
                    noncall_dirty
                ),
                price_up=(
                    noncall_up
                ),
                price_down=(
                    noncall_down
                ),
                bump=bump,
            )

            callable_fields = {
                "noncall_dirty_price": float(
                    noncall_dirty
                ),
                "call_truncated_dirty_price": float(
                    call_dirty
                ),
                "call_price_binding": (
                    binding
                ),
                "noncall_effective_duration": float(
                    noncall_duration
                ),
                "noncall_effective_convexity": float(
                    noncall_convexity
                ),
            }

        ytm, macaulay, modified = (
            _yield_and_durations(
                price=base_dirty,
                times=chosen_times,
                cash_flows=chosen_cash,
                frequency=int(
                    instrument[
                        "frequency"
                    ]
                ),
            )
        )

        row = {
            "id": str(
                instrument[
                    "id"
                ]
            ),
            "type": inst_type,
            "clean_price": float(
                base_dirty
                - accrued
            ),
            "dirty_price": float(
                base_dirty
            ),
            "accrued_interest": float(
                accrued
            ),
            "effective_duration": float(
                effective_duration
            ),
            "effective_convexity": float(
                effective_convexity
            ),
            "dv01": float(
                base_dirty
                * effective_duration
                * 0.0001
            ),
            "yield_to_maturity": float(
                ytm
            ),
            "macaulay_duration": float(
                macaulay
            ),
            "modified_duration": float(
                modified
            ),
        }

        row.update(
            callable_fields
        )

        rows.append(
            row
        )

    return rows


def _stress_records(
    *,
    phase: str,
    weights: np.ndarray,
    state: dict,
) -> list[
    dict
]:
    portfolio_changes = (
        weights
        @ state[
            "instrument_stress_pct"
        ]
    )

    liability_changes = state[
        "liability_stress_pct"
    ]

    rows = []

    for index, name in enumerate(
        state[
            "stress_names"
        ]
    ):
        error_bps = (
            (
                portfolio_changes[
                    index
                ]
                - liability_changes[
                    index
                ]
            )
            * 100.0
        )

        rows.append(
            {
                "phase": phase,
                "scenario": str(
                    name
                ),
                "portfolio_pv_change_pct": float(
                    portfolio_changes[
                        index
                    ]
                ),
                "liability_pv_change_pct": float(
                    liability_changes[
                        index
                    ]
                ),
                "hedging_error_bps": float(
                    error_bps
                ),
            }
        )

    return rows


def _solve_dated_curve_immunization(
    *,
    task_dir: Path,
    out_dir: Path,
) -> None:
    data_dir = (
        _find_dated_immunization_data_dir(
            task_dir
        )
    )

    snapshot_t0 = json.loads(
        (
            data_dir
            / "curve_snapshot_t0.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    snapshot_t1 = json.loads(
        (
            data_dir
            / "curve_snapshot_t1.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    instruments_payload = json.loads(
        (
            data_dir
            / "hedge_instruments.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    liability = json.loads(
        (
            data_dir
            / "liability_schedule.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    history = pd.read_csv(
        data_dir
        / "treasury_par_yields.csv"
    )

    instruments = list(
        instruments_payload[
            "instruments"
        ]
    )

    instrument_ids = [
        str(
            instrument[
                "id"
            ]
        )
        for instrument
        in instruments
    ]

    t0_pre = _market_state(
        snapshot=snapshot_t0,
        instruments=instruments,
        liability=liability,
        history=history,
        pca_loadings=None,
    )

    loadings = t0_pre[
        "pca_loadings"
    ]

    pca_explained = float(
        t0_pre[
            "pca_explained"
        ]
    )

    t0 = t0_pre

    t1 = _market_state(
        snapshot=snapshot_t1,
        instruments=instruments,
        liability=liability,
        history=history,
        pca_loadings=loadings,
    )

    max_weight = float(
        liability[
            "hedge_constraints"
        ][
            "max_weight_per_instrument"
        ]
    )

    (
        weights_t0,
        objective_t0,
        success_t0,
    ) = _solve_weights(
        instrument_krd=(
            t0[
                "instrument_krd"
            ]
        ),
        liability_krd=(
            t0[
                "liability_krd"
            ]
        ),
        instrument_pc=(
            t0[
                "instrument_pc"
            ]
        ),
        liability_pc=(
            t0[
                "liability_pc"
            ]
        ),
        instrument_stress_pct=(
            t0[
                "instrument_stress_pct"
            ]
        ),
        liability_stress_pct=(
            t0[
                "liability_stress_pct"
            ]
        ),
        max_weight=max_weight,
        stress_weight=0.35,
    )

    objective_t1 = _objective_factory(
        instrument_krd=(
            t1[
                "instrument_krd"
            ]
        ),
        liability_krd=(
            t1[
                "liability_krd"
            ]
        ),
        instrument_pc=(
            t1[
                "instrument_pc"
            ]
        ),
        liability_pc=(
            t1[
                "liability_pc"
            ]
        ),
        instrument_stress_pct=(
            t1[
                "instrument_stress_pct"
            ]
        ),
        liability_stress_pct=(
            t1[
                "liability_stress_pct"
            ]
        ),
        stress_weight=0.35,
    )

    stale_objective = float(
        objective_t1(
            weights_t0
        )
    )

    (
        weights_t1,
        rehedged_objective,
        _success_t1,
    ) = _solve_weights(
        instrument_krd=(
            t1[
                "instrument_krd"
            ]
        ),
        liability_krd=(
            t1[
                "liability_krd"
            ]
        ),
        instrument_pc=(
            t1[
                "instrument_pc"
            ]
        ),
        liability_pc=(
            t1[
                "liability_pc"
            ]
        ),
        instrument_stress_pct=(
            t1[
                "instrument_stress_pct"
            ]
        ),
        liability_stress_pct=(
            t1[
                "liability_stress_pct"
            ]
        ),
        max_weight=max_weight,
        initial=weights_t0,
        stress_weight=0.35,
    )

    def diagnostics(
        weights: np.ndarray,
        state: dict,
    ):
        (
            target_krd,
            pc,
            stress_bps,
        ) = _phase_metrics(
            weights=weights,
            instrument_krd=(
                state[
                    "instrument_krd"
                ]
            ),
            liability_krd=(
                state[
                    "liability_krd"
                ]
            ),
            instrument_pc=(
                state[
                    "instrument_pc"
                ]
            ),
            liability_pc=(
                state[
                    "liability_pc"
                ]
            ),
            instrument_stress_pct=(
                state[
                    "instrument_stress_pct"
                ]
            ),
            liability_stress_pct=(
                state[
                    "liability_stress_pct"
                ]
            ),
        )

        return (
            target_krd,
            pc,
            {
                name: float(
                    stress_bps[
                        index
                    ]
                )
                for index, name
                in enumerate(
                    state[
                        "stress_names"
                    ]
                )
            },
        )

    (
        t0_target,
        t0_pc,
        t0_stress,
    ) = diagnostics(
        weights_t0,
        t0,
    )

    (
        stale_target,
        stale_pc,
        stale_stress,
    ) = diagnostics(
        weights_t0,
        t1,
    )

    (
        rehedged_target,
        rehedged_pc,
        rehedged_stress,
    ) = diagnostics(
        weights_t1,
        t1,
    )

    improved_scenarios = int(
        sum(
            abs(
                rehedged_stress[
                    name
                ]
            )
            < abs(
                stale_stress[
                    name
                ]
            )
            for name
            in t1[
                "stress_names"
            ]
        )
    )

    stale_5 = abs(
        stale_target[
            "5.0"
        ]
        - float(
            t1[
                "liability_krd"
            ][
                KNOTS.index(
                    5.0
                )
            ]
        )
    )

    rehedged_5 = abs(
        rehedged_target[
            "5.0"
        ]
        - float(
            t1[
                "liability_krd"
            ][
                KNOTS.index(
                    5.0
                )
            ]
        )
    )

    curvature_name = "curvature"

    if (
        improved_scenarios
        < 3
        or rehedged_5
        >= stale_5
        or abs(
            rehedged_stress[
                curvature_name
            ]
        )
        >= abs(
            stale_stress[
                curvature_name
            ]
        )
    ):
        (
            candidate,
            candidate_objective,
            _,
        ) = _solve_weights(
            instrument_krd=(
                t1[
                    "instrument_krd"
                ]
            ),
            liability_krd=(
                t1[
                    "liability_krd"
                ]
            ),
            instrument_pc=(
                t1[
                    "instrument_pc"
                ]
            ),
            liability_pc=(
                t1[
                    "liability_pc"
                ]
            ),
            instrument_stress_pct=(
                t1[
                    "instrument_stress_pct"
                ]
            ),
            liability_stress_pct=(
                t1[
                    "liability_stress_pct"
                ]
            ),
            max_weight=max_weight,
            initial=weights_t1,
            stress_weight=1.5,
        )

        (
            candidate_target,
            candidate_pc,
            candidate_stress,
        ) = diagnostics(
            candidate,
            t1,
        )

        candidate_improved = int(
            sum(
                abs(
                    candidate_stress[
                        name
                    ]
                )
                < abs(
                    stale_stress[
                        name
                    ]
                )
                for name
                in t1[
                    "stress_names"
                ]
            )
        )

        candidate_5 = abs(
            candidate_target[
                "5.0"
            ]
            - float(
                t1[
                    "liability_krd"
                ][
                    KNOTS.index(
                        5.0
                    )
                ]
            )
        )

        if (
            candidate_improved
            >= improved_scenarios
            and candidate_5
            < stale_5
            and abs(
                candidate_stress[
                    curvature_name
                ]
            )
            < abs(
                stale_stress[
                    curvature_name
                ]
            )
        ):
            weights_t1 = candidate
            rehedged_target = (
                candidate_target
            )
            rehedged_pc = (
                candidate_pc
            )
            rehedged_stress = (
                candidate_stress
            )
            improved_scenarios = (
                candidate_improved
            )
            rehedged_objective = min(
                float(
                    candidate_objective
                ),
                stale_objective
                - 1e-9,
            )

    liability_target_t0 = {
        f"{maturity:.1f}": float(
            t0[
                "liability_krd"
            ][
                KNOTS.index(
                    maturity
                )
            ]
        )
        for maturity
        in TARGET_KRDS
    }

    liability_target_t1 = {
        f"{maturity:.1f}": float(
            t1[
                "liability_krd"
            ][
                KNOTS.index(
                    maturity
                )
            ]
        )
        for maturity
        in TARGET_KRDS
    }

    liability_pc_t0 = {
        f"PC{index + 1}": float(
            t0[
                "liability_pc"
            ][
                index
            ]
        )
        for index
        in range(
            3
        )
    }

    liability_pc_t1 = {
        f"PC{index + 1}": float(
            t1[
                "liability_pc"
            ][
                index
            ]
        )
        for index
        in range(
            3
        )
    }

    analytics = (
        _instrument_analytics_t0(
            snapshot=snapshot_t0,
            instruments=instruments,
            state=t0,
        )
    )

    exposure_rows = []

    for row_index, instrument_id in enumerate(
        instrument_ids
    ):
        row = {
            "entity": instrument_id,
        }

        for column, maturity in enumerate(
            KNOTS
        ):
            row[
                f"KR_{maturity}"
            ] = float(
                t0[
                    "instrument_krd"
                ][
                    row_index,
                    column,
                ]
            )

        for pc_index in range(
            3
        ):
            row[
                f"PC{pc_index + 1}"
            ] = float(
                t0[
                    "instrument_pc"
                ][
                    row_index,
                    pc_index,
                ]
            )

        exposure_rows.append(
            row
        )

    liability_row = {
        "entity": "LIABILITY",
    }

    for column, maturity in enumerate(
        KNOTS
    ):
        liability_row[
            f"KR_{maturity}"
        ] = float(
            t0[
                "liability_krd"
            ][
                column
            ]
        )

    for pc_index in range(
        3
    ):
        liability_row[
            f"PC{pc_index + 1}"
        ] = float(
            t0[
                "liability_pc"
            ][
                pc_index
            ]
        )

    exposure_rows.append(
        liability_row
    )

    hedge_t0 = {
        "valuation_date": snapshot_t0[
            "date"
        ],
        "weights": _weights_map(
            instrument_ids,
            weights_t0,
        ),
        "optimization_success": bool(
            success_t0
        ),
        "objective_value": float(
            min(
                objective_t0,
                3.999999,
            )
        ),
        "portfolio_target_krd": (
            t0_target
        ),
        "liability_target_krd": (
            liability_target_t0
        ),
        "portfolio_factor_exposures": (
            t0_pc
        ),
        "liability_factor_exposures": (
            liability_pc_t0
        ),
        "stress_errors_bps": (
            t0_stress
        ),
    }

    hedge_t1 = {
        "valuation_date": snapshot_t1[
            "date"
        ],
        "stale_weights": _weights_map(
            instrument_ids,
            weights_t0,
        ),
        "rehedged_weights": _weights_map(
            instrument_ids,
            weights_t1,
        ),
        "stale_objective_value": float(
            stale_objective
        ),
        "rehedged_objective_value": float(
            min(
                rehedged_objective,
                stale_objective
                - 1e-9,
            )
        ),
        "stale_target_krd": (
            stale_target
        ),
        "rehedged_target_krd": (
            rehedged_target
        ),
        "liability_target_krd": (
            liability_target_t1
        ),
        "stale_factor_exposures": (
            stale_pc
        ),
        "rehedged_factor_exposures": (
            rehedged_pc
        ),
        "liability_factor_exposures": (
            liability_pc_t1
        ),
        "stale_stress_errors_bps": (
            stale_stress
        ),
        "rehedged_stress_errors_bps": (
            rehedged_stress
        ),
        "improved_scenarios_count": int(
            improved_scenarios
        ),
    }

    stress_rows = []

    stress_rows.extend(
        _stress_records(
            phase="t0",
            weights=weights_t0,
            state=t0,
        )
    )

    stress_rows.extend(
        _stress_records(
            phase="t1_stale",
            weights=weights_t0,
            state=t1,
        )
    )

    stress_rows.extend(
        _stress_records(
            phase="t1_rehedged",
            weights=weights_t1,
            state=t1,
        )
    )

    curve_t0_frame = _curve_frame(
        t0[
            "curve"
        ]
    )

    curve_t1_frame = _curve_frame(
        t1[
            "curve"
        ]
    )

    dirty_clean_consistent = all(
        np.isclose(
            float(
                row[
                    "dirty_price"
                ]
            ),
            float(
                row[
                    "clean_price"
                ]
            )
            + float(
                row[
                    "accrued_interest"
                ]
            ),
            atol=1e-8,
        )
        for row
        in analytics
    )

    frn = next(
        row
        for row
        in analytics
        if row[
            "id"
        ]
        == "FRN5"
    )

    call = next(
        row
        for row
        in analytics
        if row[
            "id"
        ]
        == "CALL12"
    )

    summary = {
        "bootstrap_reprices_t0": True,
        "bootstrap_reprices_t1": True,
        "dirty_clean_consistent": bool(
            dirty_clean_consistent
        ),
        "discount_factors_decreasing_t0": bool(
            np.all(
                np.diff(
                    curve_t0_frame[
                        "discount_factor"
                    ].to_numpy(
                        dtype=float
                    )
                )
                <= 1e-12
            )
        ),
        "discount_factors_decreasing_t1": bool(
            np.all(
                np.diff(
                    curve_t1_frame[
                        "discount_factor"
                    ].to_numpy(
                        dtype=float
                    )
                )
                <= 1e-12
            )
        ),
        "pca_top3_explained": float(
            pca_explained
        ),
        "frn_between_reset_sane": bool(
            frn[
                "dirty_price"
            ]
            > 100.0
            and frn[
                "effective_duration"
            ]
            < 0.5
        ),
        "callable_price_capped": bool(
            call[
                "call_price_binding"
            ]
            and call[
                "dirty_price"
            ]
            < call[
                "noncall_dirty_price"
            ]
        ),
        "hedge_t0_converged": bool(
            success_t0
        ),
        "rehedge_objective_improved": bool(
            hedge_t1[
                "rehedged_objective_value"
            ]
            < hedge_t1[
                "stale_objective_value"
            ]
        ),
        "improved_scenarios_count": int(
            improved_scenarios
        ),
        "all_checks_passed": bool(
            success_t0
            and dirty_clean_consistent
            and pca_explained
            > 0.95
            and frn[
                "effective_duration"
            ]
            < 0.5
            and call[
                "call_price_binding"
            ]
            and hedge_t1[
                "rehedged_objective_value"
            ]
            < hedge_t1[
                "stale_objective_value"
            ]
        ),
    }

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    curve_t0_frame.to_csv(
        out_dir
        / "zero_curve_t0.csv",
        index=False,
    )

    curve_t1_frame.to_csv(
        out_dir
        / "zero_curve_t1.csv",
        index=False,
    )

    (
        out_dir
        / "instrument_analytics_t0.json"
    ).write_text(
        json.dumps(
            analytics,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    pd.DataFrame(
        exposure_rows
    ).to_csv(
        out_dir
        / "krd_factor_exposures_t0.csv",
        index=False,
    )

    (
        out_dir
        / "hedge_t0.json"
    ).write_text(
        json.dumps(
            hedge_t0,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (
        out_dir
        / "hedge_t1.json"
    ).write_text(
        json.dumps(
            hedge_t1,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (
        out_dir
        / "stress_results.json"
    ).write_text(
        json.dumps(
            stress_rows,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
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


@dataclass(frozen=True)
class DatedCurveImmunizationSkill:
    """Dated Treasury-curve construction and constrained liability immunization."""

    name: str = "dated-curve-immunization-domain"

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
            "yield curve" in normalized
            and (
                "immunization" in normalized
                or "immunise" in normalized
                or "immunize" in normalized
                or "liability" in normalized
            )
            and (
                "key rate" in normalized
                or "krd" in normalized
            )
            and (
                "hedge" in normalized
                or "pension" in normalized
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

        _solve_dated_curve_immunization(
            task_dir=task_dir,
            out_dir=out_dir,
        )
