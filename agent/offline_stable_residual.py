from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


TASK_ID = "stable-residual"

SIGNALS = [
    "ret_12_7", "ret_6_1", "resff3_12_1", "resff3_6_1",
    "seas_1_1an", "seas_1_1na", "seas_2_5an", "seas_2_5na",
    "seas_6_10an", "seas_6_10na", "seas_11_15an", "seas_11_15na",
    "seas_16_20an", "seas_16_20na",
]
CONTROLS = [
    "rvol_21d", "ivol_capm_21d", "ivol_capm_252d", "ivol_ff3_21d",
    "ivol_hxz4_21d", "beta_60m", "beta_dimson_21d", "betabab_1260d",
    "betadown_252d", "rskew_21d", "coskew_21d", "iskew_capm_21d",
    "iskew_ff3_21d", "iskew_hxz4_21d", "corr_1260d",
]

REBALANCE_START_IDX = 96
LOOKBACK_MONTHS = 36
SIGN_AGREE_MIN = 24
QUANTILE_FRAC = 0.15
MIN_LEG = 10
MIN_FORMATION_N = 80
TURNOVER_CAP = 0.30
TURNOVER_TOL = 0.300000001
COST_BPS = 0.0010
ANN = 12

MONTHLY_COLS = [
    "date", "n_eligible", "n_long", "n_short", "n_coeff_months_used",
    "gross_return", "transaction_cost", "net_return", "turnover",
    "beta_exposure", "dollar_exposure", "gross_leverage",
]
COEF_COLS = [
    "signal", "median_coef", "sign_agreement_count", "stable_coef",
]
WEIGHT_COLS = [
    "date", "identifier", "alpha", "prev_weight", "target_weight",
    "final_weight", "beta_60m", "ret_lead1m",
]


def _find_panel(task_dir: Path) -> Path:
    matches = [
        p for p in task_dir.rglob("stock_chars.pqt")
        if p.is_file() and "checks" not in p.parts
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one stock_chars.pqt outside checks, found {len(matches)}."
        )
    return matches[0]


def _ols_pinv(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(x.T @ x) @ (x.T @ y)


def _zscore_ddof0(values: np.ndarray) -> np.ndarray:
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if std == 0.0 or not np.isfinite(std):
        return np.zeros_like(values)
    return (values - mean) / std


def _residualize_month(group: pd.DataFrame):
    needed = SIGNALS + CONTROLS
    universe = group.dropna(subset=needed).copy()
    formation_size = int(len(universe))

    if formation_size == 0:
        return None, 0

    winsorized = universe[SIGNALS + CONTROLS].astype(float).copy()

    for column in SIGNALS + CONTROLS:
        lo = winsorized[column].quantile(0.01)
        hi = winsorized[column].quantile(0.99)
        winsorized[column] = winsorized[column].clip(lo, hi)

    x_controls = np.column_stack(
        [np.ones(formation_size)]
        + [winsorized[c].values for c in CONTROLS]
    )

    panel = pd.DataFrame(
        {
            "identifier": universe["identifier"].astype(int).values,
            "beta_60m": universe["beta_60m"].astype(float).values,
            "ret_lead1m": universe["ret_lead1m"].astype(float).values,
        }
    )

    for signal in SIGNALS:
        y = winsorized[signal].values.astype(float)
        beta = _ols_pinv(x_controls, y)
        residual = y - x_controls @ beta
        panel[signal] = _zscore_ddof0(residual)

    return panel, formation_size


def _fit_monthly_coefs(panel: pd.DataFrame):
    sub = panel.dropna(subset=["ret_lead1m"])

    if len(sub) < MIN_FORMATION_N:
        return None

    x = np.column_stack(
        [np.ones(len(sub))]
        + [sub[s].values.astype(float) for s in SIGNALS]
    )
    y = sub["ret_lead1m"].values.astype(float)

    return _ols_pinv(x, y)[1:]


def _beta_neutral_project(
    weights: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    x = np.column_stack([np.ones(len(weights)), beta])
    return weights - x @ (
        np.linalg.pinv(x.T @ x) @ (x.T @ weights)
    )


def _turnover_one_way(a: dict[int, float], b: dict[int, float]) -> float:
    identifiers = set(a) | set(b)
    return 0.5 * sum(
        abs(a.get(i, 0.0) - b.get(i, 0.0))
        for i in identifiers
    )


@dataclass(frozen=True)
class StableResidualSkill:
    name: str = "stable-residual-domain"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        lowered = instruction.lower()

        semantic = (
            "stable residual alpha" in lowered
            and "beta-neutral" in lowered
            and "turnover" in lowered
            and "coefficients_last_rebalance.csv" in lowered
            and "final_month_weights.csv" in lowered
            and "seas_16_20na" in lowered
        )

        if not semantic:
            return False

        try:
            _find_panel(task_dir)
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

        raw = pd.read_parquet(_find_panel(task_dir))
        n_input_rows = int(len(raw))

        frame = raw.copy()
        frame["date"] = pd.to_datetime(frame["date"])

        # Keep the last file-order occurrence for duplicate (identifier, date)
        # rows, then sort deterministically.
        frame = frame.drop_duplicates(
            subset=["identifier", "date"],
            keep="last",
        )
        frame = frame.sort_values(
            ["date", "identifier"],
            kind="mergesort",
        ).reset_index(drop=True)

        if not (frame["identifier"] % 1 == 0).all():
            raise RuntimeError("identifier column contains non-integer values")

        frame["identifier"] = frame["identifier"].astype(int)

        unique_dates = [
            pd.Timestamp(d)
            for d in sorted(
                pd.to_datetime(frame["date"].unique()).tolist()
            )
        ]
        n_unique_dates = int(len(unique_dates))

        residualized_panel: dict[pd.Timestamp, pd.DataFrame] = {}
        formation_size: dict[pd.Timestamp, int] = {}

        for date, group in frame.groupby("date", sort=True):
            date_key = pd.Timestamp(date)
            panel, size = _residualize_month(group)
            formation_size[date_key] = size

            if panel is not None:
                residualized_panel[date_key] = panel

        monthly_coefs: dict[pd.Timestamp, np.ndarray] = {}

        for date, panel in residualized_panel.items():
            if formation_size[date] < MIN_FORMATION_N:
                continue

            coefs = _fit_monthly_coefs(panel)

            if coefs is not None:
                monthly_coefs[date] = coefs

        coef_dates_sorted = sorted(monthly_coefs.keys())

        records = []
        previous_weights: dict[int, float] = {}
        last_info = None

        for month_index, date in enumerate(unique_dates):
            date = pd.Timestamp(date)

            if month_index < REBALANCE_START_IDX:
                continue

            prior_dates = [
                d for d in coef_dates_sorted
                if d < date
            ]

            if len(prior_dates) < LOOKBACK_MONTHS:
                continue

            history_dates = prior_dates[-LOOKBACK_MONTHS:]
            history = np.array(
                [monthly_coefs[d] for d in history_dates],
                dtype=float,
            )

            median_coef = np.median(history, axis=0)
            median_sign = np.sign(median_coef)
            coefficient_signs = np.sign(history)

            agreement = (
                coefficient_signs
                == median_sign[None, :]
            ).sum(axis=0)

            stable_coef = np.where(
                agreement >= SIGN_AGREE_MIN,
                median_coef,
                0.0,
            )

            if (
                date not in residualized_panel
                or formation_size[date] < MIN_FORMATION_N
            ):
                continue

            panel = residualized_panel[date]
            n = int(len(panel))

            identifiers = panel["identifier"].values.astype(int)
            betas = panel["beta_60m"].values.astype(float)
            signal_matrix = panel[SIGNALS].values.astype(float)
            alpha = signal_matrix @ stable_coef

            order = np.lexsort((identifiers, -alpha))

            n_long = int(math.floor(QUANTILE_FRAC * n))
            n_short = int(math.floor(QUANTILE_FRAC * n))

            if n_long < MIN_LEG or n_short < MIN_LEG:
                continue

            long_positions = order[:n_long]
            short_positions = order[-n_short:]
            selected_positions = np.concatenate(
                [long_positions, short_positions]
            )

            selected_ids = identifiers[selected_positions]
            selected_betas = betas[selected_positions]

            raw_weights = np.concatenate(
                [
                    np.full(n_long, 1.0 / n_long),
                    np.full(n_short, -1.0 / n_short),
                ]
            )

            projected = _beta_neutral_project(
                raw_weights,
                selected_betas,
            )

            gross = float(np.sum(np.abs(projected)))

            if gross == 0.0:
                continue

            target_array = projected / gross
            target_weights = {
                int(selected_ids[i]): float(target_array[i])
                for i in range(len(selected_ids))
            }

            raw_month = frame[frame["date"] == date]

            raw_beta_map = dict(
                zip(
                    raw_month["identifier"].astype(int).tolist(),
                    raw_month["beta_60m"].astype(float).tolist(),
                )
            )
            raw_return_map = dict(
                zip(
                    raw_month["identifier"].astype(int).tolist(),
                    raw_month["ret_lead1m"].astype(float).tolist(),
                )
            )
            formation_beta_map = {
                int(identifiers[i]): float(betas[i])
                for i in range(n)
            }

            def month_beta(identifier: int) -> float:
                if identifier in formation_beta_map:
                    return formation_beta_map[identifier]

                value = raw_beta_map.get(identifier, np.nan)

                if value is None or not np.isfinite(value):
                    return 0.0

                return float(value)

            turnover = _turnover_one_way(
                previous_weights,
                target_weights,
            )

            if turnover <= TURNOVER_CAP:
                final_weights = dict(target_weights)
            else:
                lam = TURNOVER_CAP / turnover

                first_union = set(previous_weights) | set(target_weights)
                mid_weights = {
                    i: (
                        previous_weights.get(i, 0.0)
                        + lam
                        * (
                            target_weights.get(i, 0.0)
                            - previous_weights.get(i, 0.0)
                        )
                    )
                    for i in first_union
                }

                support = sorted(
                    i for i, value in mid_weights.items()
                    if value != 0.0
                )

                if not support:
                    continue

                support_beta = np.array(
                    [month_beta(i) for i in support],
                    dtype=float,
                )
                mid_array = np.array(
                    [mid_weights[i] for i in support],
                    dtype=float,
                )

                reprojected = _beta_neutral_project(
                    mid_array,
                    support_beta,
                )

                mid_gross = float(np.sum(np.abs(reprojected)))

                if mid_gross == 0.0:
                    continue

                reprojected /= mid_gross

                reprojected_weights = {
                    support[i]: float(reprojected[i])
                    for i in range(len(support))
                }

                new_turnover = _turnover_one_way(
                    previous_weights,
                    reprojected_weights,
                )

                if new_turnover > TURNOVER_TOL:
                    shrink = TURNOVER_CAP / new_turnover
                    second_union = (
                        set(previous_weights)
                        | set(reprojected_weights)
                    )

                    final_weights = {
                        i: (
                            previous_weights.get(i, 0.0)
                            + shrink
                            * (
                                reprojected_weights.get(i, 0.0)
                                - previous_weights.get(i, 0.0)
                            )
                        )
                        for i in second_union
                    }
                else:
                    final_weights = dict(reprojected_weights)

            all_names = set(final_weights) | set(previous_weights)

            gross_return = 0.0

            for identifier, weight in final_weights.items():
                if weight == 0.0:
                    continue

                realized = raw_return_map.get(identifier, float("nan"))

                if realized is not None and np.isfinite(realized):
                    gross_return += weight * realized

            transaction_cost = COST_BPS * sum(
                abs(
                    final_weights.get(i, 0.0)
                    - previous_weights.get(i, 0.0)
                )
                for i in all_names
            )

            net_return = gross_return - transaction_cost

            final_turnover = 0.5 * sum(
                abs(
                    final_weights.get(i, 0.0)
                    - previous_weights.get(i, 0.0)
                )
                for i in all_names
            )

            beta_exposure = sum(
                weight * month_beta(i)
                for i, weight in final_weights.items()
            )
            dollar_exposure = float(sum(final_weights.values()))
            gross_leverage = float(
                sum(abs(weight) for weight in final_weights.values())
            )

            records.append(
                {
                    "date": date,
                    "n_eligible": n,
                    "n_long": n_long,
                    "n_short": n_short,
                    "n_coeff_months_used": LOOKBACK_MONTHS,
                    "gross_return": float(gross_return),
                    "transaction_cost": float(transaction_cost),
                    "net_return": float(net_return),
                    "turnover": float(final_turnover),
                    "beta_exposure": float(beta_exposure),
                    "dollar_exposure": float(dollar_exposure),
                    "gross_leverage": float(gross_leverage),
                }
            )

            last_info = {
                "date": date,
                "median_coef": median_coef.copy(),
                "agreement": agreement.astype(int).copy(),
                "stable_coef": stable_coef.copy(),
                "alpha_map": {
                    int(identifiers[i]): float(alpha[i])
                    for i in range(n)
                },
                "w_target": dict(target_weights),
                "w_prev": dict(previous_weights),
                "w_final": dict(final_weights),
                "formation_beta_map": formation_beta_map,
                "raw_beta_map": raw_beta_map,
                "raw_ret_map": raw_return_map,
            }

            previous_weights = dict(final_weights)

        n_tradable = len(records)
        n_skipped = max(0, n_unique_dates - REBALANCE_START_IDX) - n_tradable

        if n_tradable == 0 or last_info is None:
            raise RuntimeError("no tradable rebalance months")

        net_series = np.array(
            [row["net_return"] for row in records],
            dtype=float,
        )
        gross_series = np.array(
            [row["gross_return"] for row in records],
            dtype=float,
        )
        turnover_series = np.array(
            [row["turnover"] for row in records],
            dtype=float,
        )
        abs_beta_series = np.array(
            [abs(row["beta_exposure"]) for row in records],
            dtype=float,
        )
        long_counts = np.array(
            [row["n_long"] for row in records],
            dtype=float,
        )
        short_counts = np.array(
            [row["n_short"] for row in records],
            dtype=float,
        )

        mean_net = float(net_series.mean())
        vol_net = float(net_series.std(ddof=0))
        annualized_return = ANN * mean_net
        annualized_volatility = math.sqrt(ANN) * vol_net
        annualized_sharpe = (
            annualized_return / annualized_volatility
            if annualized_volatility != 0.0
            else 0.0
        )

        cumulative_net_return = float(
            np.prod(1.0 + net_series) - 1.0
        )

        wealth = np.concatenate(
            [[1.0], np.cumprod(1.0 + net_series)]
        )
        peaks = np.maximum.accumulate(wealth)
        drawdowns = (peaks - wealth) / peaks
        max_drawdown = float(drawdowns.max())

        first_rebalance_date = records[0]["date"].strftime("%Y-%m-%d")
        last_rebalance_date = records[-1]["date"].strftime("%Y-%m-%d")

        summary = {
            "task_id": TASK_ID,
            "n_input_rows": int(n_input_rows),
            "n_unique_dates": int(n_unique_dates),
            "first_rebalance_date": first_rebalance_date,
            "last_rebalance_date": last_rebalance_date,
            "n_tradable_months": int(n_tradable),
            "n_skipped_months": int(n_skipped),
            "mean_monthly_gross_return": float(gross_series.mean()),
            "mean_monthly_net_return": float(mean_net),
            "monthly_volatility_net": float(vol_net),
            "annualized_return_net": float(annualized_return),
            "annualized_volatility_net": float(annualized_volatility),
            "annualized_sharpe_net": float(annualized_sharpe),
            "cumulative_net_return": float(cumulative_net_return),
            "max_drawdown_net": float(max_drawdown),
            "avg_turnover": float(turnover_series.mean()),
            "max_turnover": float(turnover_series.max()),
            "avg_abs_beta_exposure": float(abs_beta_series.mean()),
            "max_abs_beta_exposure": float(abs_beta_series.max()),
            "avg_names_long": float(long_counts.mean()),
            "avg_names_short": float(short_counts.mean()),
        }

        for key, value in summary.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise RuntimeError(f"non-finite summary value {key}={value}")

        monthly_rows = []

        for row in records:
            monthly_rows.append(
                {
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "n_eligible": int(row["n_eligible"]),
                    "n_long": int(row["n_long"]),
                    "n_short": int(row["n_short"]),
                    "n_coeff_months_used": int(row["n_coeff_months_used"]),
                    "gross_return": float(row["gross_return"]),
                    "transaction_cost": float(row["transaction_cost"]),
                    "net_return": float(row["net_return"]),
                    "turnover": float(row["turnover"]),
                    "beta_exposure": float(row["beta_exposure"]),
                    "dollar_exposure": float(row["dollar_exposure"]),
                    "gross_leverage": float(row["gross_leverage"]),
                }
            )

        coef_rows = []

        for index, signal in enumerate(SIGNALS):
            coef_rows.append(
                {
                    "signal": signal,
                    "median_coef": float(last_info["median_coef"][index]),
                    "sign_agreement_count": int(last_info["agreement"][index]),
                    "stable_coef": float(last_info["stable_coef"][index]),
                }
            )

        target_weights = last_info["w_target"]
        previous_last = last_info["w_prev"]
        final_last = last_info["w_final"]
        alpha_map = last_info["alpha_map"]
        formation_beta_map = last_info["formation_beta_map"]
        raw_beta_map = last_info["raw_beta_map"]
        raw_return_map = last_info["raw_ret_map"]

        names = set()

        for weights in (
            target_weights,
            previous_last,
            final_last,
        ):
            for identifier, value in weights.items():
                if value != 0.0:
                    names.add(identifier)

        def beta_lookup(identifier: int) -> float:
            if identifier in formation_beta_map:
                return formation_beta_map[identifier]

            value = raw_beta_map.get(identifier, np.nan)

            if value is None or not np.isfinite(value):
                return 0.0

            return float(value)

        def return_lookup(identifier: int):
            value = raw_return_map.get(identifier, np.nan)

            if value is None or not np.isfinite(value):
                return float("nan")

            return float(value)

        weight_rows = []
        final_date_string = last_info["date"].strftime("%Y-%m-%d")

        for identifier in names:
            weight_rows.append(
                {
                    "date": final_date_string,
                    "identifier_int": int(identifier),
                    "alpha": float(alpha_map.get(identifier, 0.0)),
                    "prev_weight": float(previous_last.get(identifier, 0.0)),
                    "target_weight": float(target_weights.get(identifier, 0.0)),
                    "final_weight": float(final_last.get(identifier, 0.0)),
                    "beta_60m": float(beta_lookup(identifier)),
                    "ret_lead1m": return_lookup(identifier),
                }
            )

        weight_rows.sort(
            key=lambda row: (
                -row["final_weight"],
                row["identifier_int"],
            )
        )

        weight_output = pd.DataFrame(
            [
                {
                    "date": row["date"],
                    "identifier": str(row["identifier_int"]),
                    "alpha": row["alpha"],
                    "prev_weight": row["prev_weight"],
                    "target_weight": row["target_weight"],
                    "final_weight": row["final_weight"],
                    "beta_60m": row["beta_60m"],
                    "ret_lead1m": row["ret_lead1m"],
                }
                for row in weight_rows
            ],
            columns=WEIGHT_COLS,
        )

        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        pd.DataFrame(
            monthly_rows,
            columns=MONTHLY_COLS,
        ).to_csv(
            out_dir / "monthly_results.csv",
            index=False,
        )

        pd.DataFrame(
            coef_rows,
            columns=COEF_COLS,
        ).to_csv(
            out_dir / "coefficients_last_rebalance.csv",
            index=False,
        )

        weight_output.to_csv(
            out_dir / "final_month_weights.csv",
            index=False,
            na_rep="",
        )
