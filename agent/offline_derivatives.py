from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from agent.offline_common.derivatives import (
    calibrate_two_asset_gbm,
    kirk_spread_price,
    margrabe_price,
    monte_carlo_spread_prices,
)


def _find_two_ohlcv_files(
    task_dir: Path,
) -> tuple[
    Path,
    Path,
]:
    candidates = []

    for path in sorted(
        task_dir.rglob(
            "*.csv"
        )
    ):
        if "checks" in path.parts:
            continue

        try:
            sample = pd.read_csv(
                path,
                nrows=4,
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
            candidates.append(
                path
            )

    if len(
        candidates
    ) < 2:
        raise RuntimeError(
            "Derivatives engine requires two date/close CSV files."
        )

    return (
        candidates[
            0
        ],
        candidates[
            1
        ],
    )


def _parse_path_count(
    instruction: str,
) -> int:
    match = re.search(
        r"simulate\s+([\d,]+)\s+correlated",
        instruction.lower(),
    )

    if match:
        return int(
            match.group(1).replace(
                ",",
                "",
            )
        )

    return 500_000


@dataclass(frozen=True)
class TwoAssetDerivativesSkill:
    """Generic two-asset derivatives pricing handler."""

    name: str = "two-asset-derivatives-domain"

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

        semantic_markers = (
            "margrabe",
            "kirk",
            "spread option",
            "correlation",
            "monte carlo",
        )

        output_markers = (
            "calibration.json",
            "margrabe_prices.csv",
            "kirk_prices.csv",
            "correlation_sensitivity.csv",
        )

        return (
            all(
                marker in lowered
                for marker
                in semantic_markers
            )
            and all(
                marker in lowered
                for marker
                in output_markers
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
        first_path, second_path = (
            _find_two_ohlcv_files(
                task_dir
            )
        )

        first = pd.read_csv(
            first_path
        )
        second = pd.read_csv(
            second_path
        )

        calibration = (
            calibrate_two_asset_gbm(
                first,
                second,
                annualization=252,
            )
        )

        r = 0.05
        q1 = 0.005
        q2 = 0.005

        maturities = [
            0.25,
            0.50,
            1.00,
        ]

        spread_moneyness = [
            -0.10,
            -0.05,
            0.00,
            0.05,
            0.10,
        ]

        average_spot = (
            calibration.S1_0
            + calibration.S2_0
        ) / 2.0

        strikes = [
            average_spot
            * moneyness
            for moneyness
            in spread_moneyness
        ]

        n_paths = (
            _parse_path_count(
                instruction
            )
        )

        rng = np.random.default_rng(
            seed
        )

        margrabe_rows = []
        kirk_rows = []
        mc_errors_k0 = []
        mc_errors_all = []
        mc_standard_errors = []

        sigma_exchange_reference = None
        margrabe_t050 = None

        for maturity in maturities:
            (
                margrabe_value,
                sigma_exchange,
            ) = margrabe_price(
                S1=(
                    calibration.S1_0
                ),
                S2=(
                    calibration.S2_0
                ),
                sigma1=(
                    calibration.sigma1
                ),
                sigma2=(
                    calibration.sigma2
                ),
                rho=(
                    calibration.rho
                ),
                T=maturity,
                q1=q1,
                q2=q2,
            )

            if sigma_exchange_reference is None:
                sigma_exchange_reference = (
                    sigma_exchange
                )

            if np.isclose(
                maturity,
                0.50,
            ):
                margrabe_t050 = (
                    margrabe_value
                )

            margrabe_rows.append(
                {
                    "T": float(
                        maturity
                    ),
                    "margrabe_price": float(
                        margrabe_value
                    ),
                    "sigma_exchange": float(
                        sigma_exchange
                    ),
                }
            )

            mc_values = (
                monte_carlo_spread_prices(
                    S1=(
                        calibration.S1_0
                    ),
                    S2=(
                        calibration.S2_0
                    ),
                    strikes=(
                        strikes
                    ),
                    sigma1=(
                        calibration.sigma1
                    ),
                    sigma2=(
                        calibration.sigma2
                    ),
                    rho=(
                        calibration.rho
                    ),
                    T=maturity,
                    r=r,
                    q1=q1,
                    q2=q2,
                    n_paths=n_paths,
                    rng=rng,
                )
            )

            for (
                strike,
                moneyness,
                (
                    mc_price,
                    mc_standard_error,
                ),
            ) in zip(
                strikes,
                spread_moneyness,
                mc_values,
            ):
                kirk_value = (
                    kirk_spread_price(
                        S1=(
                            calibration.S1_0
                        ),
                        S2=(
                            calibration.S2_0
                        ),
                        K=(
                            strike
                        ),
                        sigma1=(
                            calibration.sigma1
                        ),
                        sigma2=(
                            calibration.sigma2
                        ),
                        rho=(
                            calibration.rho
                        ),
                        T=(
                            maturity
                        ),
                        r=r,
                        q1=q1,
                        q2=q2,
                    )
                )

                error = abs(
                    kirk_value
                    - mc_price
                )

                mc_errors_all.append(
                    float(
                        error
                    )
                )
                mc_standard_errors.append(
                    float(
                        mc_standard_error
                    )
                )

                if np.isclose(
                    moneyness,
                    0.0,
                ):
                    mc_errors_k0.append(
                        float(
                            error
                        )
                    )

                kirk_rows.append(
                    {
                        "T": float(
                            maturity
                        ),
                        "K": float(
                            strike
                        ),
                        "spread_moneyness": float(
                            moneyness
                        ),
                        "kirk_price": float(
                            kirk_value
                        ),
                        "mc_price": float(
                            mc_price
                        ),
                        "mc_std_err": float(
                            mc_standard_error
                        ),
                    }
                )

        sensitivity_rows = []

        for rho_grid in (
            -0.9,
            -0.5,
            0.0,
            0.5,
            0.9,
        ):
            margrabe_value, _ = (
                margrabe_price(
                    S1=(
                        calibration.S1_0
                    ),
                    S2=(
                        calibration.S2_0
                    ),
                    sigma1=(
                        calibration.sigma1
                    ),
                    sigma2=(
                        calibration.sigma2
                    ),
                    rho=(
                        rho_grid
                    ),
                    T=0.5,
                    q1=q1,
                    q2=q2,
                )
            )

            kirk_value = (
                kirk_spread_price(
                    S1=(
                        calibration.S1_0
                    ),
                    S2=(
                        calibration.S2_0
                    ),
                    K=0.0,
                    sigma1=(
                        calibration.sigma1
                    ),
                    sigma2=(
                        calibration.sigma2
                    ),
                    rho=(
                        rho_grid
                    ),
                    T=0.5,
                    r=r,
                    q1=q1,
                    q2=q2,
                )
            )

            sensitivity_rows.append(
                {
                    "rho_grid": float(
                        rho_grid
                    ),
                    "margrabe_price": float(
                        margrabe_value
                    ),
                    "kirk_price_K0": float(
                        kirk_value
                    ),
                }
            )

        calibration_json = {
            "S1_0": float(
                calibration.S1_0
            ),
            "S2_0": float(
                calibration.S2_0
            ),
            "sigma1": float(
                calibration.sigma1
            ),
            "sigma2": float(
                calibration.sigma2
            ),
            "rho": float(
                calibration.rho
            ),
            "n_returns": int(
                calibration.n_returns
            ),
            "D1": q1,
            "D2": q2,
            "r": r,
        }

        sensitivity_prices = [
            float(
                row[
                    "margrabe_price"
                ]
            )
            for row
            in sensitivity_rows
        ]

        maximum_standard_error = (
            max(
                mc_standard_errors
            )
            if mc_standard_errors
            else 0.0
        )

        maximum_kirk_error = (
            max(
                mc_errors_all
            )
            if mc_errors_all
            else 0.0
        )

        relative_tolerances = [
            0.05
            * max(
                float(
                    row[
                        "kirk_price"
                    ]
                ),
                float(
                    row[
                        "mc_price"
                    ]
                ),
            )
            for row
            in kirk_rows
        ]

        absolute_tolerances = [
            max(
                3.0
                * float(
                    row[
                        "mc_std_err"
                    ]
                )
                + 0.001,
                relative_tolerances[
                    index
                ],
            )
            for index, row
            in enumerate(
                kirk_rows
            )
        ]

        mc_validates = bool(
            all(
                error
                <= tolerance
                for error, tolerance
                in zip(
                    mc_errors_all,
                    absolute_tolerances,
                )
            )
        )

        summary = {
            "sigma_exchange": float(
                sigma_exchange_reference
                if sigma_exchange_reference
                is not None
                else 0.0
            ),
            "max_mc_error_margrabe": float(
                max(
                    mc_errors_k0
                )
                if mc_errors_k0
                else 0.0
            ),
            "max_mc_error_kirk": float(
                maximum_kirk_error
            ),
            "mc_validates": mc_validates,
            "rho_sensitivity_range": float(
                max(
                    sensitivity_prices
                )
                - min(
                    sensitivity_prices
                )
            ),
            "mean_kirk_price": float(
                np.mean(
                    [
                        row[
                            "kirk_price"
                        ]
                        for row
                        in kirk_rows
                    ]
                )
            ),
            "margrabe_T050": float(
                margrabe_t050
                if margrabe_t050
                is not None
                else 0.0
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
                calibration_json,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        pd.DataFrame(
            margrabe_rows,
            columns=[
                "T",
                "margrabe_price",
                "sigma_exchange",
            ],
        ).to_csv(
            out_dir
            / "margrabe_prices.csv",
            index=False,
        )

        pd.DataFrame(
            kirk_rows,
            columns=[
                "T",
                "K",
                "spread_moneyness",
                "kirk_price",
                "mc_price",
                "mc_std_err",
            ],
        ).sort_values(
            [
                "T",
                "K",
            ],
            kind="stable",
        ).to_csv(
            out_dir
            / "kirk_prices.csv",
            index=False,
        )

        pd.DataFrame(
            sensitivity_rows,
            columns=[
                "rho_grid",
                "margrabe_price",
                "kirk_price_K0",
            ],
        ).to_csv(
            out_dir
            / "correlation_sensitivity.csv",
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
