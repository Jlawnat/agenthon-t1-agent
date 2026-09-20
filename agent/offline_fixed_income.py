from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from agent.offline_common.fixed_income import (
    bootstrap_par_curve,
)


def _find_curve_json(
    task_dir: Path,
) -> Path:
    preferred = [
        path
        for path in task_dir.rglob(
            "curve_data.json"
        )
        if (
            path.is_file()
            and "checks"
            not in path.parts
        )
    ]

    if preferred:
        return sorted(
            preferred
        )[0]

    for path in sorted(
        task_dir.rglob(
            "*.json"
        )
    ):
        if "checks" in path.parts:
            continue

        try:
            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            continue

        if (
            isinstance(
                data,
                dict,
            )
            and {
                "maturities",
                "par_rates",
                "coupon_freq",
            }.issubset(
                data
            )
        ):
            return path

    raise RuntimeError(
        "No par-curve JSON input was found."
    )


def _maturity_key(
    maturity: float,
) -> str:
    if float(
        maturity
    ).is_integer():
        return str(
            int(
                maturity
            )
        )

    return str(
        float(
            maturity
        )
    )


@dataclass(frozen=True)
class FixedIncomeCurveSkill:
    """Generic deterministic fixed-income curve handler."""

    name: str = "fixed-income-curve-domain"

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
            "bootstrap",
            "zero-coupon",
            "par coupon",
            "discount factor",
            "forward rate",
        )

        output_markers = (
            "zero_rates",
            "discount_factors",
            "forward_rates",
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
            _find_curve_json(
                task_dir
            )
        )

        data = json.loads(
            input_path.read_text(
                encoding="utf-8"
            )
        )

        maturities = [
            float(
                value
            )
            for value
            in data[
                "maturities"
            ]
        ]

        par_rates = [
            float(
                value
            )
            for value
            in data[
                "par_rates"
            ]
        ]

        coupon_frequency = int(
            data[
                "coupon_freq"
            ]
        )

        curve = (
            bootstrap_par_curve(
                maturities,
                par_rates,
                coupon_frequency=(
                    coupon_frequency
                ),
            )
        )

        results = {
            "zero_rates": {
                _maturity_key(
                    maturity
                ): round(
                    float(
                        zero_rate
                    ),
                    8,
                )
                for maturity, zero_rate
                in zip(
                    curve.maturities,
                    curve.zero_rates,
                )
            },
            "discount_factors": {
                _maturity_key(
                    maturity
                ): round(
                    float(
                        discount_factor
                    ),
                    8,
                )
                for maturity, discount_factor
                in zip(
                    curve.maturities,
                    curve.discount_factors,
                )
            },
            "forward_rates": {
                _maturity_key(
                    maturity
                ): round(
                    float(
                        forward_rate
                    ),
                    8,
                )
                for maturity, forward_rate
                in zip(
                    curve.maturities,
                    curve.forward_rates,
                )
            },
        }

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            out_dir
            / "results.json"
        ).write_text(
            json.dumps(
                results,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
