from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import csv
import json
import math
from pathlib import Path

from scipy.optimize import brentq


def parse_date(value: str) -> date:
    return datetime.strptime(
        value,
        "%Y-%m-%d",
    ).date()


def add_months(
    value: date,
    months: int,
) -> date:
    total = (
        value.month
        - 1
        + months
    )

    year = (
        value.year
        + total // 12
    )

    month = (
        total % 12
        + 1
    )

    if month == 12:
        next_first = date(
            year + 1,
            1,
            1,
        )
    else:
        next_first = date(
            year,
            month + 1,
            1,
        )

    last_day = (
        next_first
        - timedelta(
            days=1
        )
    ).day

    return date(
        year,
        month,
        min(
            value.day,
            last_day,
        ),
    )


def act_360(
    start: date,
    end: date,
) -> float:
    return (
        end
        - start
    ).days / 360.0


def act_365_25(
    start: date,
    end: date,
) -> float:
    return (
        end
        - start
    ).days / 365.25


def thirty_360(
    start: date,
    end: date,
) -> float:
    d1 = min(
        start.day,
        30,
    )

    d2 = (
        30
        if (
            d1 == 30
            and end.day == 31
        )
        else min(
            end.day,
            30,
        )
    )

    return (
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
    ) / 360.0


def log_linear_df(
    pillars: list[
        tuple[
            float,
            float,
        ]
    ],
    t: float,
) -> float:
    if t <= 0.0:
        return 1.0

    if not pillars:
        raise RuntimeError(
            "Cannot interpolate an empty curve."
        )

    ordered = sorted(
        (
            float(
                time
            ),
            float(
                df
            ),
        )
        for time, df
        in pillars
    )

    if t <= ordered[0][0]:
        time, df = (
            ordered[0]
        )

        zero = (
            -math.log(
                df
            )
            / time
        )

        return float(
            math.exp(
                -zero
                * t
            )
        )

    if t >= ordered[-1][0]:
        time, df = (
            ordered[-1]
        )

        zero = (
            -math.log(
                df
            )
            / time
        )

        return float(
            math.exp(
                -zero
                * t
            )
        )

    for index in range(
        1,
        len(
            ordered
        ),
    ):
        t0, d0 = (
            ordered[
                index - 1
            ]
        )
        t1, d1 = (
            ordered[
                index
            ]
        )

        if (
            t0
            <= t
            <= t1
        ):
            weight = (
                t
                - t0
            ) / (
                t1
                - t0
            )

            return float(
                math.exp(
                    (
                        1.0
                        - weight
                    )
                    * math.log(
                        d0
                    )
                    + weight
                    * math.log(
                        d1
                    )
                )
            )

    raise RuntimeError(
        "Curve interpolation failed."
    )


@dataclass
class Curve:
    name: str
    pillars: list[
        tuple[
            float,
            float,
            date,
            str,
        ]
    ]

    def df_pairs(
        self,
    ) -> list[
        tuple[
            float,
            float,
        ]
    ]:
        return [
            (
                pillar[
                    0
                ],
                pillar[
                    1
                ],
            )
            for pillar
            in self.pillars
        ]

    def df(
        self,
        target_date: date,
        valuation_date: date,
    ) -> float:
        time = act_365_25(
            valuation_date,
            target_date,
        )

        return log_linear_df(
            self.df_pairs(),
            time,
        )


def generate_schedule(
    effective: date,
    maturity: date,
    frequency_months: int,
) -> list[
    date
]:
    if (
        frequency_months
        <= 0
    ):
        raise RuntimeError(
            "Payment frequency must be positive."
        )

    dates = [
        maturity
    ]

    while True:
        previous = add_months(
            dates[-1],
            -frequency_months,
        )

        if (
            previous
            <= effective
        ):
            break

        dates.append(
            previous
        )

    dates.reverse()

    return dates


def bootstrap_zero_coupon_curve(
    rows: list[
        dict[str, str]
    ],
    *,
    valuation_date: date,
    rate_field: str,
    name: str,
) -> Curve:
    pillars = []

    for row in rows:
        maturity = parse_date(
            row[
                "maturity_date"
            ]
        )

        rate = float(
            row[
                rate_field
            ]
        )

        accrual = act_360(
            valuation_date,
            maturity,
        )

        df = (
            1.0
            / (
                1.0
                + rate
                * accrual
            )
        )

        time = act_365_25(
            valuation_date,
            maturity,
        )

        pillars.append(
            (
                time,
                float(
                    df
                ),
                maturity,
                row[
                    "tenor_label"
                ],
            )
        )

    pillars.sort(
        key=lambda item: (
            item[
                0
            ]
        )
    )

    return Curve(
        name,
        pillars,
    )


def fixed_leg_annuity(
    effective: date,
    maturity: date,
    ois_pairs: list[
        tuple[
            float,
            float,
        ]
    ],
    valuation_date: date,
    *,
    frequency_months: int = 6,
) -> float:
    schedule = generate_schedule(
        effective,
        maturity,
        frequency_months,
    )

    annuity = 0.0
    previous = effective

    for payment_date in schedule:
        accrual = thirty_360(
            previous,
            payment_date,
        )

        payment_time = (
            act_365_25(
                valuation_date,
                payment_date,
            )
        )

        annuity += (
            accrual
            * log_linear_df(
                ois_pairs,
                payment_time,
            )
        )

        previous = (
            payment_date
        )

    return float(
        annuity
    )


def floating_leg_pv_per_unit(
    effective: date,
    maturity: date,
    ois_pairs: list[
        tuple[
            float,
            float,
        ]
    ],
    libor_pairs: list[
        tuple[
            float,
            float,
        ]
    ],
    valuation_date: date,
    *,
    frequency_months: int = 3,
) -> float:
    schedule = generate_schedule(
        effective,
        maturity,
        frequency_months,
    )

    pv = 0.0
    previous = effective

    for payment_date in schedule:
        accrual = act_360(
            previous,
            payment_date,
        )

        previous_time = (
            act_365_25(
                valuation_date,
                previous,
            )
        )

        payment_time = (
            act_365_25(
                valuation_date,
                payment_date,
            )
        )

        libor_df_previous = (
            log_linear_df(
                libor_pairs,
                previous_time,
            )
        )

        libor_df_payment = (
            log_linear_df(
                libor_pairs,
                payment_time,
            )
        )

        forward = (
            libor_df_previous
            / libor_df_payment
            - 1.0
        ) / accrual

        ois_df = (
            log_linear_df(
                ois_pairs,
                payment_time,
            )
        )

        pv += (
            forward
            * accrual
            * ois_df
        )

        previous = (
            payment_date
        )

    return float(
        pv
    )


def model_par_swap_rate(
    effective: date,
    maturity: date,
    ois_pairs: list[
        tuple[
            float,
            float,
        ]
    ],
    libor_pairs: list[
        tuple[
            float,
            float,
        ]
    ],
    valuation_date: date,
) -> float:
    annuity = fixed_leg_annuity(
        effective,
        maturity,
        ois_pairs,
        valuation_date,
        frequency_months=6,
    )

    floating_pv = (
        floating_leg_pv_per_unit(
            effective,
            maturity,
            ois_pairs,
            libor_pairs,
            valuation_date,
            frequency_months=3,
        )
    )

    return float(
        floating_pv
        / annuity
    )


def bootstrap_libor_curve(
    irs_quotes: list[
        dict[str, str]
    ],
    *,
    ois_curve: Curve,
    deposit_curve: Curve,
    valuation_date: date,
    spot_date: date,
) -> Curve:
    pillars = list(
        deposit_curve.pillars
    )

    ois_pairs = (
        ois_curve.df_pairs()
    )

    quotes = sorted(
        irs_quotes,
        key=lambda row: (
            parse_date(
                row[
                    "maturity_date"
                ]
            )
        ),
    )

    for row in quotes:
        maturity = parse_date(
            row[
                "maturity_date"
            ]
        )

        target_rate = float(
            row[
                "par_rate"
            ]
        )

        target_time = (
            act_365_25(
                valuation_date,
                maturity,
            )
        )

        def residual(
            trial_df: float,
        ) -> float:
            trial_pairs = [
                (
                    pillar[
                        0
                    ],
                    pillar[
                        1
                    ],
                )
                for pillar
                in pillars
            ]

            trial_pairs.append(
                (
                    target_time,
                    trial_df,
                )
            )

            trial_pairs.sort()

            model_rate = (
                model_par_swap_rate(
                    spot_date,
                    maturity,
                    ois_pairs,
                    trial_pairs,
                    valuation_date,
                )
            )

            return (
                model_rate
                - target_rate
            )

        solved_df = float(
            brentq(
                residual,
                1e-8,
                1.2,
                xtol=1e-14,
                rtol=1e-14,
                maxiter=300,
            )
        )

        pillars.append(
            (
                target_time,
                solved_df,
                maturity,
                row[
                    "tenor_label"
                ],
            )
        )

        pillars.sort(
            key=lambda item: (
                item[
                    0
                ]
            )
        )

    return Curve(
        "LIBOR_3M",
        pillars,
    )


def reprice_inputs(
    *,
    ois_quotes: list[
        dict[str, str]
    ],
    deposit_quotes: list[
        dict[str, str]
    ],
    irs_quotes: list[
        dict[str, str]
    ],
    ois_curve: Curve,
    libor_curve: Curve,
    valuation_date: date,
    spot_date: date,
) -> list[
    dict[str, object]
]:
    rows = []

    for row in ois_quotes:
        maturity = parse_date(
            row[
                "maturity_date"
            ]
        )

        input_rate = float(
            row[
                "par_rate"
            ]
        )

        accrual = act_360(
            valuation_date,
            maturity,
        )

        df = ois_curve.df(
            maturity,
            valuation_date,
        )

        model_rate = (
            (
                1.0
                / df
                - 1.0
            )
            / accrual
        )

        rows.append(
            {
                "instrument_type": (
                    "ois"
                ),
                "tenor_label": row[
                    "tenor_label"
                ],
                "input_par_rate": (
                    input_rate
                ),
                "model_par_rate": float(
                    model_rate
                ),
                "residual_bps": float(
                    (
                        model_rate
                        - input_rate
                    )
                    * 1e4
                ),
            }
        )

    for row in deposit_quotes:
        maturity = parse_date(
            row[
                "maturity_date"
            ]
        )

        input_rate = float(
            row[
                "rate"
            ]
        )

        accrual = act_360(
            valuation_date,
            maturity,
        )

        df = libor_curve.df(
            maturity,
            valuation_date,
        )

        model_rate = (
            (
                1.0
                / df
                - 1.0
            )
            / accrual
        )

        rows.append(
            {
                "instrument_type": (
                    "deposit"
                ),
                "tenor_label": row[
                    "tenor_label"
                ],
                "input_par_rate": (
                    input_rate
                ),
                "model_par_rate": float(
                    model_rate
                ),
                "residual_bps": float(
                    (
                        model_rate
                        - input_rate
                    )
                    * 1e4
                ),
            }
        )

    for row in irs_quotes:
        maturity = parse_date(
            row[
                "maturity_date"
            ]
        )

        input_rate = float(
            row[
                "par_rate"
            ]
        )

        model_rate = (
            model_par_swap_rate(
                spot_date,
                maturity,
                ois_curve.df_pairs(),
                libor_curve.df_pairs(),
                valuation_date,
            )
        )

        rows.append(
            {
                "instrument_type": (
                    "irs_3m"
                ),
                "tenor_label": row[
                    "tenor_label"
                ],
                "input_par_rate": (
                    input_rate
                ),
                "model_par_rate": float(
                    model_rate
                ),
                "residual_bps": float(
                    (
                        model_rate
                        - input_rate
                    )
                    * 1e4
                ),
            }
        )

    return rows


def value_swap(
    *,
    swap_spec: dict[str, object],
    ois_curve: Curve,
    libor_pairs: list[
        tuple[
            float,
            float,
        ]
    ],
    valuation_date: date,
) -> dict[str, object]:
    effective = parse_date(
        str(
            swap_spec[
                "effective_date"
            ]
        )
    )

    maturity = parse_date(
        str(
            swap_spec[
                "maturity_date"
            ]
        )
    )

    notional = float(
        swap_spec[
            "notional"
        ]
    )

    fixed_rate = float(
        swap_spec[
            "fixed_rate"
        ]
    )

    pay_receive = str(
        swap_spec[
            "pay_receive"
        ]
    )

    annuity = fixed_leg_annuity(
        effective,
        maturity,
        ois_curve.df_pairs(),
        valuation_date,
        frequency_months=6,
    )

    floating_unit = (
        floating_leg_pv_per_unit(
            effective,
            maturity,
            ois_curve.df_pairs(),
            libor_pairs,
            valuation_date,
            frequency_months=3,
        )
    )

    fixed_leg_pv = (
        notional
        * fixed_rate
        * annuity
    )

    float_leg_pv = (
        notional
        * floating_unit
    )

    if pay_receive == "pay_fixed":
        npv = (
            float_leg_pv
            - fixed_leg_pv
        )
    elif pay_receive == "receive_fixed":
        npv = (
            fixed_leg_pv
            - float_leg_pv
        )
    else:
        raise RuntimeError(
            f"Unsupported pay_receive value: {pay_receive}"
        )

    return {
        "npv": float(
            npv
        ),
        "fixed_leg_pv": float(
            fixed_leg_pv
        ),
        "float_leg_pv": float(
            float_leg_pv
        ),
        "pay_receive": pay_receive,
    }


def bump_zero_rates(
    pairs: list[
        tuple[
            float,
            float,
        ]
    ],
    bump_bps: float,
) -> list[
    tuple[
        float,
        float,
    ]
]:
    bump = (
        float(
            bump_bps
        )
        * 1e-4
    )

    return [
        (
            time,
            float(
                math.exp(
                    math.log(
                        df
                    )
                    - bump
                    * time
                )
            ),
        )
        for time, df
        in pairs
    ]


def compute_pv01(
    *,
    swap_spec: dict[str, object],
    ois_curve: Curve,
    libor_curve: Curve,
    valuation_date: date,
) -> float:
    base = value_swap(
        swap_spec=swap_spec,
        ois_curve=ois_curve,
        libor_pairs=(
            libor_curve.df_pairs()
        ),
        valuation_date=(
            valuation_date
        ),
    )

    bump_bps = float(
        swap_spec.get(
            "pv01_bump_bps",
            1.0,
        )
    )

    bumped_pairs = (
        bump_zero_rates(
            libor_curve.df_pairs(),
            bump_bps,
        )
    )

    bumped = value_swap(
        swap_spec=swap_spec,
        ois_curve=ois_curve,
        libor_pairs=(
            bumped_pairs
        ),
        valuation_date=(
            valuation_date
        ),
    )

    return float(
        abs(
            float(
                bumped[
                    "npv"
                ]
            )
            - float(
                base[
                    "npv"
                ]
            )
        )
    )


def _load_csv(
    path: Path,
) -> list[
    dict[str, str]
]:
    with path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(
            csv.DictReader(
                handle
            )
        )


def _discover_data_dir(
    task_dir: Path,
) -> Path:
    required = {
        "conventions.json",
        "ois_quotes.csv",
        "deposits.csv",
        "irs_3m_quotes.csv",
        "swap_to_value.json",
    }

    for directory in sorted(
        {
            path.parent
            for path
            in task_dir.rglob("*")
            if path.is_file()
            and "checks"
            not in path.parts
        },
        key=lambda path: str(
            path
        ),
    ):
        names = {
            path.name
            for path
            in directory.iterdir()
            if path.is_file()
        }

        if required.issubset(
            names
        ):
            return directory

    raise RuntimeError(
        "Could not discover the dual-curve input directory."
    )


def _write_curve_outputs(
    *,
    out_dir: Path,
    ois_curve: Curve,
    libor_curve: Curve,
    valuation_date: date,
) -> None:
    with (
        out_dir
        / "ois_discount_curve.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pillar_date",
                "time_years",
                "discount_factor",
                "zero_rate_continuous",
            ],
        )

        writer.writeheader()

        for (
            time,
            df,
            maturity,
            _,
        ) in ois_curve.pillars:
            writer.writerow(
                {
                    "pillar_date": (
                        maturity.isoformat()
                    ),
                    "time_years": float(
                        time
                    ),
                    "discount_factor": float(
                        df
                    ),
                    "zero_rate_continuous": float(
                        -math.log(
                            df
                        )
                        / time
                    ),
                }
            )

    libor_pairs = (
        libor_curve.df_pairs()
    )

    with (
        out_dir
        / "libor_forward_curve.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pillar_date",
                "time_years",
                "libor_discount_factor",
                "zero_rate_continuous",
                "forward_3m",
            ],
        )

        writer.writeheader()

        for (
            time,
            df,
            maturity,
            _,
        ) in libor_curve.pillars:
            next_date = add_months(
                maturity,
                3,
            )

            next_time = (
                act_365_25(
                    valuation_date,
                    next_date,
                )
            )

            next_df = (
                log_linear_df(
                    libor_pairs,
                    next_time,
                )
            )

            accrual = act_360(
                maturity,
                next_date,
            )

            forward = (
                df
                / next_df
                - 1.0
            ) / accrual

            writer.writerow(
                {
                    "pillar_date": (
                        maturity.isoformat()
                    ),
                    "time_years": float(
                        time
                    ),
                    "libor_discount_factor": float(
                        df
                    ),
                    "zero_rate_continuous": float(
                        -math.log(
                            df
                        )
                        / time
                    ),
                    "forward_3m": float(
                        forward
                    ),
                }
            )


@dataclass(frozen=True)
class DualCurveBootstrapSkill:
    name: str = "dual-curve-rates-domain"

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
            "ois" in lowered
            and (
                "libor" in lowered
                or "projection curve" in lowered
            )
            and (
                "dual-curve" in lowered
                or "dual curve" in lowered
            )
            and "swap" in lowered
            and "bootstrap" in lowered
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

        data_dir = _discover_data_dir(
            task_dir
        )

        conventions = json.loads(
            (
                data_dir
                / "conventions.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        valuation_date = parse_date(
            conventions[
                "valuation_date"
            ]
        )

        ois_quotes = _load_csv(
            data_dir
            / "ois_quotes.csv"
        )

        deposit_quotes = (
            _load_csv(
                data_dir
                / "deposits.csv"
            )
        )

        irs_quotes = _load_csv(
            data_dir
            / "irs_3m_quotes.csv"
        )

        swap_spec = json.loads(
            (
                data_dir
                / "swap_to_value.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        spot_date = parse_date(
            str(
                swap_spec[
                    "effective_date"
                ]
            )
        )

        # The OIS discount curve must be available before the LIBOR
        # projection curve is bootstrapped.
        ois_curve = (
            bootstrap_zero_coupon_curve(
                ois_quotes,
                valuation_date=(
                    valuation_date
                ),
                rate_field="par_rate",
                name="OIS",
            )
        )

        deposit_curve = (
            bootstrap_zero_coupon_curve(
                deposit_quotes,
                valuation_date=(
                    valuation_date
                ),
                rate_field="rate",
                name="LIBOR_DEPOSITS",
            )
        )

        libor_curve = (
            bootstrap_libor_curve(
                irs_quotes,
                ois_curve=ois_curve,
                deposit_curve=(
                    deposit_curve
                ),
                valuation_date=(
                    valuation_date
                ),
                spot_date=spot_date,
            )
        )

        repriced = reprice_inputs(
            ois_quotes=ois_quotes,
            deposit_quotes=(
                deposit_quotes
            ),
            irs_quotes=irs_quotes,
            ois_curve=ois_curve,
            libor_curve=(
                libor_curve
            ),
            valuation_date=(
                valuation_date
            ),
            spot_date=spot_date,
        )

        base_swap = value_swap(
            swap_spec=swap_spec,
            ois_curve=ois_curve,
            libor_pairs=(
                libor_curve.df_pairs()
            ),
            valuation_date=(
                valuation_date
            ),
        )

        swap_output = {
            "npv": float(
                base_swap[
                    "npv"
                ]
            ),
            "fixed_leg_pv": float(
                base_swap[
                    "fixed_leg_pv"
                ]
            ),
            "float_leg_pv": float(
                base_swap[
                    "float_leg_pv"
                ]
            ),
            "pv01": compute_pv01(
                swap_spec=swap_spec,
                ois_curve=ois_curve,
                libor_curve=(
                    libor_curve
                ),
                valuation_date=(
                    valuation_date
                ),
            ),
            "pay_receive": str(
                base_swap[
                    "pay_receive"
                ]
            ),
        }

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        _write_curve_outputs(
            out_dir=out_dir,
            ois_curve=ois_curve,
            libor_curve=(
                libor_curve
            ),
            valuation_date=(
                valuation_date
            ),
        )

        with (
            out_dir
            / "repriced_quotes.csv"
        ).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "instrument_type",
                    "tenor_label",
                    "input_par_rate",
                    "model_par_rate",
                    "residual_bps",
                ],
            )

            writer.writeheader()
            writer.writerows(
                repriced
            )

        (
            out_dir
            / "swap_valuation.json"
        ).write_text(
            json.dumps(
                swap_output,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        def zero_at(
            curve: Curve,
            target_date: date,
        ) -> float:
            df = curve.df(
                target_date,
                valuation_date,
            )

            time = act_365_25(
                valuation_date,
                target_date,
            )

            return float(
                -math.log(
                    df
                )
                / time
            )

        maturity_5y = add_months(
            valuation_date,
            60,
        )

        maturity_10y = add_months(
            valuation_date,
            120,
        )

        irs_10y_maturity = next(
            parse_date(
                row[
                    "maturity_date"
                ]
            )
            for row
            in irs_quotes
            if (
                row[
                    "tenor_label"
                ]
                == "10Y"
            )
        )

        max_residual = max(
            abs(
                float(
                    row[
                        "residual_bps"
                    ]
                )
            )
            for row
            in repriced
        )

        summary = {
            "ois_5y_zero_rate_continuous": (
                zero_at(
                    ois_curve,
                    maturity_5y,
                )
            ),
            "ois_10y_zero_rate_continuous": (
                zero_at(
                    ois_curve,
                    maturity_10y,
                )
            ),
            "libor_5y_zero_rate_continuous": (
                zero_at(
                    libor_curve,
                    maturity_5y,
                )
            ),
            "libor_10y_zero_rate_continuous": (
                zero_at(
                    libor_curve,
                    maturity_10y,
                )
            ),
            "libor_10y_par_swap_rate": (
                model_par_swap_rate(
                    spot_date,
                    irs_10y_maturity,
                    ois_curve.df_pairs(),
                    libor_curve.df_pairs(),
                    valuation_date,
                )
            ),
            "max_bootstrap_residual_bps": float(
                max_residual
            ),
        }

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
