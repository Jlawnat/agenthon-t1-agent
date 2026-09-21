from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


TASK_ID = "residual-momentum"
NW_LAG = 6
BURN_IN_MONTHS = 120

CONTROLS = [
    "rvol_21d",
    "ivol_capm_21d",
    "ivol_capm_252d",
    "ivol_ff3_21d",
    "ivol_hxz4_21d",
    "beta_60m",
    "beta_dimson_21d",
    "betabab_1260d",
    "betadown_252d",
    "corr_1260d",
    "rskew_21d",
    "coskew_21d",
    "iskew_capm_21d",
    "iskew_ff3_21d",
    "iskew_hxz4_21d",
]
RAW_SIGNALS = [
    "ret_12_7",
    "ret_6_1",
]
RES_SIGNALS = [
    "resff3_12_1",
    "resff3_6_1",
]
MOMENTUM_ALL = (
    RAW_SIGNALS
    + RES_SIGNALS
)
MODEL_REGS = {
    "A": CONTROLS + RAW_SIGNALS,
    "B": CONTROLS + RES_SIGNALS,
    "C": CONTROLS + RAW_SIGNALS + RES_SIGNALS,
}

SPREAD_COLS = [
    "date",
    "model",
    "n_assets",
    "q1_count",
    "q5_count",
    "q1_return",
    "q5_return",
    "q5_minus_q1",
    "ic",
]
COEF_COLS = [
    "date",
    "model",
    "n_assets",
    "alpha",
    "ret_12_7",
    "ret_6_1",
    "resff3_12_1",
    "resff3_6_1",
]


def _find_panel(task_dir: Path) -> Path:
    matches = [
        path
        for path in task_dir.rglob("stock_chars.pqt")
        if path.is_file() and "checks" not in path.parts
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one stock_chars.pqt outside checks, found {len(matches)}."
        )
    return matches[0]


def _nw_tstat(x, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    t = len(x)

    if t == 0:
        return float("nan")

    mean = x.mean()
    residual = x - mean
    long_run_var = float(
        (residual * residual).sum() / t
    )

    for k in range(
        1,
        lag + 1,
    ):
        if k >= t:
            break

        weight = (
            1.0
            - k / (lag + 1.0)
        )
        gamma_k = float(
            (
                residual[k:]
                * residual[:-k]
            ).sum()
            / t
        )
        long_run_var += (
            2.0
            * weight
            * gamma_k
        )

    var_mean = (
        long_run_var
        / t
    )

    if (
        not np.isfinite(
            var_mean
        )
        or var_mean <= 0.0
    ):
        return float("nan")

    return float(
        mean
        / math.sqrt(
            var_mean
        )
    )


def _spearman_average_ranks(
    a,
    b,
) -> float:
    rank_a = (
        pd.Series(a)
        .rank(
            method="average"
        )
        .to_numpy()
    )
    rank_b = (
        pd.Series(b)
        .rank(
            method="average"
        )
        .to_numpy()
    )

    if (
        np.std(rank_a) == 0.0
        or np.std(rank_b) == 0.0
    ):
        return float("nan")

    value = float(
        np.corrcoef(
            rank_a,
            rank_b,
        )[0, 1]
    )

    return (
        value
        if np.isfinite(
            value
        )
        else float("nan")
    )


def _zscore_ddof0(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    z = (
        frame[
            columns
        ]
        .copy()
        .astype(float)
    )

    for column in columns:
        values = z[
            column
        ].to_numpy()

        mean = values.mean()
        std = values.std(
            ddof=0
        )

        if (
            std == 0.0
            or not np.isfinite(
                std
            )
        ):
            z[
                column
            ] = 0.0
        else:
            z[
                column
            ] = (
                values
                - mean
            ) / std

    return z


def _fit_monthly(
    frame: pd.DataFrame,
    regressors: list[str],
):
    k = len(
        regressors
    )
    min_n = max(
        80,
        4 * (k + 1),
    )

    output = {}

    for date, group in frame.groupby(
        "date",
        sort=True,
    ):
        sub = group[
            [
                "identifier",
                "ret_lead1m",
            ]
            + regressors
        ].dropna()

        if len(sub) < min_n:
            continue

        sub = (
            sub.sort_values(
                "identifier"
            )
            .reset_index(
                drop=True
            )
        )

        x_z = (
            _zscore_ddof0(
                sub,
                regressors,
            )[
                regressors
            ]
            .to_numpy()
        )

        x = np.column_stack(
            [
                np.ones(
                    len(sub)
                ),
                x_z,
            ]
        )

        y = sub[
            "ret_lead1m"
        ].to_numpy(
            dtype=float
        )

        beta, *_ = (
            np.linalg.lstsq(
                x,
                y,
                rcond=None,
            )
        )

        output[
            date
        ] = {
            "beta": beta,
            "Xz": x_z,
            "y": y,
            "ids": sub[
                "identifier"
            ].to_numpy(),
            "n": len(sub),
        }

    return output


def _compute_oos(
    monthly,
    date_to_idx,
    burn_in: int = BURN_IN_MONTHS,
):
    ic_list = []
    spread_rows = []
    prior_betas = []

    for date in sorted(
        monthly.keys()
    ):
        data = monthly[
            date
        ]
        month_index = (
            date_to_idx[
                pd.Timestamp(
                    date
                )
            ]
        )

        if (
            month_index
            >= burn_in
            and len(
                prior_betas
            )
            > 0
        ):
            mean_beta = np.mean(
                np.vstack(
                    prior_betas
                ),
                axis=0,
            )

            x = np.column_stack(
                [
                    np.ones(
                        data[
                            "n"
                        ]
                    ),
                    data[
                        "Xz"
                    ],
                ]
            )

            prediction = (
                x
                @ mean_beta
            )
            y = data[
                "y"
            ]
            identifiers = data[
                "ids"
            ]

            ic = (
                _spearman_average_ranks(
                    prediction,
                    y,
                )
            )

            order = np.lexsort(
                (
                    identifiers,
                    prediction,
                )
            )

            rank_array = np.empty(
                data[
                    "n"
                ],
                dtype=int,
            )
            rank_array[
                order
            ] = np.arange(
                data[
                    "n"
                ]
            )

            n_assets = data[
                "n"
            ]

            quintile = np.minimum(
                5,
                (
                    5
                    * rank_array
                )
                // n_assets
                + 1,
            )

            q1 = (
                quintile
                == 1
            )
            q5 = (
                quintile
                == 5
            )

            q1_count = int(
                q1.sum()
            )
            q5_count = int(
                q5.sum()
            )

            if (
                q1_count > 0
                and q5_count > 0
            ):
                q1_return = float(
                    y[
                        q1
                    ].mean()
                )
                q5_return = float(
                    y[
                        q5
                    ].mean()
                )

                spread_rows.append(
                    {
                        "date": pd.Timestamp(
                            date
                        ),
                        "n_assets": int(
                            n_assets
                        ),
                        "q1_count": (
                            q1_count
                        ),
                        "q5_count": (
                            q5_count
                        ),
                        "q1_return": (
                            q1_return
                        ),
                        "q5_return": (
                            q5_return
                        ),
                        "q5_minus_q1": (
                            q5_return
                            - q1_return
                        ),
                        "ic": (
                            ic
                            if np.isfinite(
                                ic
                            )
                            else float(
                                "nan"
                            )
                        ),
                    }
                )

                if np.isfinite(
                    ic
                ):
                    ic_list.append(
                        ic
                    )

        prior_betas.append(
            data[
                "beta"
            ]
        )

    return (
        ic_list,
        spread_rows,
    )


@dataclass(frozen=True)
class ResidualMomentumSkill:
    name: str = "residual-momentum-domain"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        lowered = instruction.lower()

        semantic = (
            "residual momentum" in lowered
            and "fama" in lowered
            and "macbeth" in lowered
            and "resff3_12_1" in lowered
            and "monthly_oos_spread_returns.csv" in lowered
            and "monthly_fmb_coefficients.csv" in lowered
        )

        if not semantic:
            return False

        try:
            _find_panel(
                task_dir
            )
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

        raw = pd.read_parquet(
            _find_panel(
                task_dir
            )
        )

        n_input_rows = int(
            len(raw)
        )

        frame = raw.copy()
        frame[
            "date"
        ] = pd.to_datetime(
            frame[
                "date"
            ]
        )

        # Normalize calendar-month keys to pandas Timestamp objects.
        # pandas versions can otherwise return np.datetime64 from
        # Series.unique() while groupby("date") yields Timestamp keys.
        all_months = [
            pd.Timestamp(date)
            for date in np.sort(
                frame[
                    "date"
                ].unique()
            )
        ]

        date_to_idx = {
            pd.Timestamp(date): index
            for index, date
            in enumerate(
                all_months
            )
        }

        per_model = {}

        for model, regressors in MODEL_REGS.items():
            monthly = (
                _fit_monthly(
                    frame,
                    regressors,
                )
            )

            (
                ic_list,
                spread_rows,
            ) = _compute_oos(
                monthly,
                date_to_idx,
            )

            per_model[
                model
            ] = {
                "monthly": monthly,
                "ic_list": (
                    ic_list
                ),
                "spread_rows": (
                    spread_rows
                ),
            }

        results = {
            "task_id": TASK_ID,
            "nw_lag": int(
                NW_LAG
            ),
            "n_input_rows": (
                n_input_rows
            ),
        }

        for model in (
            "A",
            "B",
            "C",
        ):
            results[
                f"n_months_model_{model.lower()}"
            ] = int(
                len(
                    per_model[
                        model
                    ][
                        "monthly"
                    ]
                )
            )

            results[
                f"n_oos_months_model_{model.lower()}"
            ] = int(
                len(
                    per_model[
                        model
                    ][
                        "spread_rows"
                    ]
                )
            )

        coef_names_c = (
            ["alpha"]
            + MODEL_REGS[
                "C"
            ]
        )

        monthly_c = per_model[
            "C"
        ][
            "monthly"
        ]

        for signal in MOMENTUM_ALL:
            signal_index = (
                coef_names_c.index(
                    signal
                )
            )

            series = np.array(
                [
                    data[
                        "beta"
                    ][
                        signal_index
                    ]
                    for _, data
                    in sorted(
                        monthly_c.items()
                    )
                ],
                dtype=float,
            )

            results[
                f"avg_beta_model_c_{signal}"
            ] = float(
                series.mean()
            )

            results[
                f"tstat_model_c_{signal}"
            ] = float(
                _nw_tstat(
                    series,
                    NW_LAG,
                )
            )

        ic_means = {}
        spread_means = {}

        for model in (
            "A",
            "B",
            "C",
        ):
            ic_values = np.array(
                per_model[
                    model
                ][
                    "ic_list"
                ],
                dtype=float,
            )

            spread_values = np.array(
                [
                    row[
                        "q5_minus_q1"
                    ]
                    for row
                    in per_model[
                        model
                    ][
                        "spread_rows"
                    ]
                ],
                dtype=float,
            )

            ic_means[
                model
            ] = float(
                ic_values.mean()
            )

            spread_means[
                model
            ] = float(
                spread_values.mean()
            )

        raw_max = max(
            abs(
                results[
                    "tstat_model_c_ret_12_7"
                ]
            ),
            abs(
                results[
                    "tstat_model_c_ret_6_1"
                ]
            ),
        )

        residual_max = max(
            abs(
                results[
                    "tstat_model_c_resff3_12_1"
                ]
            ),
            abs(
                results[
                    "tstat_model_c_resff3_6_1"
                ]
            ),
        )

        results[
            "residual_beats_raw"
        ] = bool(
            (
                ic_means[
                    "B"
                ]
                > ic_means[
                    "A"
                ]
            )
            and (
                spread_means[
                    "B"
                ]
                > spread_means[
                    "A"
                ]
            )
            and (
                residual_max
                > raw_max
            )
        )

        for key, value in results.items():
            if (
                isinstance(
                    value,
                    float,
                )
                and not math.isfinite(
                    value
                )
            ):
                raise RuntimeError(
                    f"non-finite value for {key}: {value}"
                )

        spread_records = []

        for model in (
            "A",
            "B",
            "C",
        ):
            for row in per_model[
                model
            ][
                "spread_rows"
            ]:
                spread_records.append(
                    {
                        "date": row[
                            "date"
                        ].strftime(
                            "%Y-%m-%d"
                        ),
                        "model": (
                            model
                        ),
                        "n_assets": int(
                            row[
                                "n_assets"
                            ]
                        ),
                        "q1_count": int(
                            row[
                                "q1_count"
                            ]
                        ),
                        "q5_count": int(
                            row[
                                "q5_count"
                            ]
                        ),
                        "q1_return": float(
                            row[
                                "q1_return"
                            ]
                        ),
                        "q5_return": float(
                            row[
                                "q5_return"
                            ]
                        ),
                        "q5_minus_q1": float(
                            row[
                                "q5_minus_q1"
                            ]
                        ),
                        "ic": (
                            float(
                                row[
                                    "ic"
                                ]
                            )
                            if np.isfinite(
                                row[
                                    "ic"
                                ]
                            )
                            else np.nan
                        ),
                    }
                )

        spread_df = pd.DataFrame(
            spread_records,
            columns=SPREAD_COLS,
        )
        spread_df = spread_df.assign(
            _date_sort=pd.to_datetime(
                spread_df[
                    "date"
                ]
            )
        )
        spread_df = (
            spread_df
            .sort_values(
                [
                    "_date_sort",
                    "model",
                ]
            )
            .drop(
                columns=[
                    "_date_sort"
                ]
            )
            .reset_index(
                drop=True
            )
        )

        coef_records = []

        for model in (
            "A",
            "B",
            "C",
        ):
            regressors = MODEL_REGS[
                model
            ]
            coef_names = (
                ["alpha"]
                + regressors
            )

            for date, data in sorted(
                per_model[
                    model
                ][
                    "monthly"
                ].items()
            ):
                row = {
                    "date": (
                        pd.Timestamp(
                            date
                        ).strftime(
                            "%Y-%m-%d"
                        )
                    ),
                    "model": model,
                    "n_assets": int(
                        data[
                            "n"
                        ]
                    ),
                    "alpha": float(
                        data[
                            "beta"
                        ][0]
                    ),
                }

                for signal in MOMENTUM_ALL:
                    if signal in regressors:
                        row[
                            signal
                        ] = float(
                            data[
                                "beta"
                            ][
                                coef_names.index(
                                    signal
                                )
                            ]
                        )
                    else:
                        row[
                            signal
                        ] = np.nan

                coef_records.append(
                    row
                )

        coef_df = pd.DataFrame(
            coef_records,
            columns=COEF_COLS,
        )
        coef_df = coef_df.assign(
            _date_sort=pd.to_datetime(
                coef_df[
                    "date"
                ]
            )
        )
        coef_df = (
            coef_df
            .sort_values(
                [
                    "_date_sort",
                    "model",
                ]
            )
            .drop(
                columns=[
                    "_date_sort"
                ]
            )
            .reset_index(
                drop=True
            )
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            out_dir
            / "fmb_results.json"
        ).write_text(
            json.dumps(
                results,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

        spread_df.to_csv(
            out_dir
            / "monthly_oos_spread_returns.csv",
            index=False,
            na_rep="",
        )

        coef_df.to_csv(
            out_dir
            / "monthly_fmb_coefficients.csv",
            index=False,
            na_rep="",
        )
