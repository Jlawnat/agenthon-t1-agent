from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent.offline_common.volatility import (
    annualized_volatility,
    efficiency_ratios,
    estimator_arrays,
    ohlc_variance_estimators,
    rolling_ohlc_estimators,
    validate_ohlc,
)


_ESTIMATOR_NAMES = (
    "close_to_close",
    "parkinson",
    "garman_klass",
    "rogers_satchell",
    "yang_zhang",
)


def _find_ohlcv_csv(
    task_dir: Path,
) -> Path:
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
            "open",
            "high",
            "low",
            "close",
        }.issubset(
            lower
        ):
            candidates.append(
                path
            )

    if not candidates:
        raise RuntimeError(
            "No OHLCV CSV input was found."
        )

    return candidates[
        0
    ]


def _sort_ohlcv(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    by_lower = {
        str(column).lower(): str(column)
        for column
        in frame.columns
    }

    date_column = (
        by_lower[
            "date"
        ]
    )

    result = frame.copy()

    result[
        date_column
    ] = pd.to_datetime(
        result[
            date_column
        ],
        errors="raise",
    )

    return (
        result
        .sort_values(
            date_column,
            kind="stable",
        )
        .drop_duplicates(
            subset=[
                date_column
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )


@dataclass(frozen=True)
class OhlcVolatilitySkill:
    """Generic deterministic OHLC realized-volatility handler."""

    name: str = "ohlc-volatility-domain"

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
            "parkinson",
            "garman-klass",
            "rogers-satchell",
            "yang-zhang",
            "rolling",
            "efficiency",
        )

        output_markers = (
            "full_sample_vol.json",
            "rolling_vol_stats.csv",
            "vol_term_structure.csv",
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
        del instruction, seed

        input_path = (
            _find_ohlcv_csv(
                task_dir
            )
        )

        frame = _sort_ohlcv(
            pd.read_csv(
                input_path
            )
        )

        by_lower = {
            str(column).lower(): str(column)
            for column
            in frame.columns
        }

        close_column = (
            by_lower[
                "close"
            ]
        )

        close = pd.to_numeric(
            frame[
                close_column
            ],
            errors="raise",
        ).to_numpy(
            dtype=float
        )

        calibration = {
            "n_days": int(
                len(
                    frame
                )
            ),
            "n_returns": int(
                max(
                    len(
                        frame
                    )
                    - 1,
                    0,
                )
            ),
            "S0": float(
                close[
                    -1
                ]
            ),
            "ohlc_valid": bool(
                validate_ohlc(
                    frame
                )
            ),
        }

        full = (
            ohlc_variance_estimators(
                frame
            )
        )

        full_vols = {
            "cc_vol": annualized_volatility(
                full.close_to_close
            ),
            "parkinson_vol": annualized_volatility(
                full.parkinson
            ),
            "garman_klass_vol": annualized_volatility(
                full.garman_klass
            ),
            "rogers_satchell_vol": annualized_volatility(
                full.rogers_satchell
            ),
            "yang_zhang_vol": annualized_volatility(
                full.yang_zhang
            ),
        }

        rolling = (
            rolling_ohlc_estimators(
                frame,
                window=21,
            )
        )

        rolling_variances = (
            estimator_arrays(
                rolling
            )
        )

        efficiencies = (
            efficiency_ratios(
                rolling_variances
            )
        )

        rolling_rows = []

        for name in _ESTIMATOR_NAMES:
            volatility_series = np.sqrt(
                np.maximum(
                    rolling_variances[
                        name
                    ],
                    0.0,
                )
                * 252.0
            )

            rolling_rows.append(
                {
                    "estimator": (
                        name
                    ),
                    "mean_vol": float(
                        np.mean(
                            volatility_series
                        )
                    ),
                    "std_vol": float(
                        np.std(
                            volatility_series,
                            ddof=1,
                        )
                    ),
                    "min_vol": float(
                        np.min(
                            volatility_series
                        )
                    ),
                    "max_vol": float(
                        np.max(
                            volatility_series
                        )
                    ),
                    "efficiency": float(
                        efficiencies[
                            name
                        ]
                    ),
                }
            )

        term_rows = []

        for window in (
            5,
            10,
            21,
            63,
            126,
            252,
        ):
            horizon_estimates = (
                rolling_ohlc_estimators(
                    frame,
                    window=window,
                )
            )

            horizon_arrays = (
                estimator_arrays(
                    horizon_estimates
                )
            )

            cc_vols = np.sqrt(
                np.maximum(
                    horizon_arrays[
                        "close_to_close"
                    ],
                    0.0,
                )
                * 252.0
            )

            yz_vols = np.sqrt(
                np.maximum(
                    horizon_arrays[
                        "yang_zhang"
                    ],
                    0.0,
                )
                * 252.0
            )

            term_rows.append(
                {
                    "window": int(
                        window
                    ),
                    "cc_vol": float(
                        np.mean(
                            cc_vols
                        )
                    ),
                    "yz_vol": float(
                        np.mean(
                            yz_vols
                        )
                    ),
                }
            )

        range_names = (
            "parkinson",
            "garman_klass",
            "rogers_satchell",
            "yang_zhang",
        )

        range_estimators_lower_variance = bool(
            all(
                efficiencies[
                    name
                ]
                > 1.0
                for name
                in range_names
            )
        )

        yz_most_efficient = bool(
            efficiencies[
                "yang_zhang"
            ]
            == max(
                efficiencies.values()
            )
        )

        full_values = list(
            full_vols.values()
        )

        summary = {
            "n_rolling_windows": int(
                len(
                    rolling
                )
            ),
            "cc_vol_annual": float(
                full_vols[
                    "cc_vol"
                ]
            ),
            "yz_vol_annual": float(
                full_vols[
                    "yang_zhang_vol"
                ]
            ),
            "parkinson_vol_annual": float(
                full_vols[
                    "parkinson_vol"
                ]
            ),
            "gk_vol_annual": float(
                full_vols[
                    "garman_klass_vol"
                ]
            ),
            "rs_vol_annual": float(
                full_vols[
                    "rogers_satchell_vol"
                ]
            ),
            "yz_efficiency": float(
                efficiencies[
                    "yang_zhang"
                ]
            ),
            "parkinson_efficiency": float(
                efficiencies[
                    "parkinson"
                ]
            ),
            "gk_efficiency": float(
                efficiencies[
                    "garman_klass"
                ]
            ),
            "rs_efficiency": float(
                efficiencies[
                    "rogers_satchell"
                ]
            ),
            "range_estimators_lower_variance": (
                range_estimators_lower_variance
            ),
            "yz_most_efficient": (
                yz_most_efficient
            ),
            "vol_mean_range": [
                float(
                    min(
                        full_values
                    )
                ),
                float(
                    max(
                        full_values
                    )
                ),
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

        (
            out_dir
            / "full_sample_vol.json"
        ).write_text(
            json.dumps(
                full_vols,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        pd.DataFrame(
            rolling_rows,
            columns=[
                "estimator",
                "mean_vol",
                "std_vol",
                "min_vol",
                "max_vol",
                "efficiency",
            ],
        ).to_csv(
            out_dir
            / "rolling_vol_stats.csv",
            index=False,
        )

        pd.DataFrame(
            term_rows,
            columns=[
                "window",
                "cc_vol",
                "yz_vol",
            ],
        ).to_csv(
            out_dir
            / "vol_term_structure.csv",
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
