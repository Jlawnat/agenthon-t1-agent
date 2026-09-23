from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt
from pathlib import Path

import pandas as pd


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _normal_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _find_options_parquet(task_dir: Path) -> Path:
    candidates = [
        path
        for path in task_dir.rglob("options.parquet")
        if "checks" not in path.parts
    ]
    if not candidates:
        raise RuntimeError(
            "Current Black-Scholes skill could not find options.parquet."
        )
    return sorted(candidates)[0]


@dataclass(frozen=True)
class CurrentBlackScholesGreeksSkill:
    name: str = "black-scholes-greeks-current-results-parquet"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        lowered = instruction.lower()

        instruction_match = all(
            token in lowered
            for token in (
                "black-scholes",
                "greeks",
                "options.parquet",
                "results.parquet",
            )
        )

        if not instruction_match:
            return False

        return any(
            "checks" not in path.parts
            for path in task_dir.rglob("options.parquet")
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

        input_path = _find_options_parquet(task_dir)
        frame = pd.read_parquet(input_path)

        required = {
            "option_id",
            "S",
            "K",
            "T",
            "r",
            "sigma",
            "option_type",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise RuntimeError(
                f"options.parquet missing required columns: {missing}"
            )

        rows: list[dict[str, object]] = []

        for row in frame.itertuples(index=False):
            option_id = getattr(row, "option_id")
            S = float(getattr(row, "S"))
            K = float(getattr(row, "K"))
            T = float(getattr(row, "T"))
            r = float(getattr(row, "r"))
            sigma = float(getattr(row, "sigma"))
            option_type = str(
                getattr(row, "option_type")
            ).strip().lower()

            if S <= 0.0 or K <= 0.0:
                raise RuntimeError(
                    "Black-Scholes requires positive spot and strike."
                )
            if T <= 0.0:
                raise RuntimeError(
                    "Black-Scholes requires positive time to expiry."
                )
            if sigma <= 0.0:
                raise RuntimeError(
                    "Black-Scholes requires positive volatility."
                )
            if option_type not in {"call", "put"}:
                raise RuntimeError(
                    f"Unsupported option_type: {option_type!r}"
                )

            sqrt_T = sqrt(T)
            d1 = (
                log(S / K)
                + (r + 0.5 * sigma * sigma) * T
            ) / (sigma * sqrt_T)
            d2 = d1 - sigma * sqrt_T

            Nd1 = _normal_cdf(d1)
            Nd2 = _normal_cdf(d2)
            Nmd1 = _normal_cdf(-d1)
            Nmd2 = _normal_cdf(-d2)
            phi_d1 = _normal_pdf(d1)
            disc = exp(-r * T)

            if option_type == "call":
                price = S * Nd1 - K * disc * Nd2
                delta = Nd1
                theta_annual = (
                    -S * phi_d1 * sigma / (2.0 * sqrt_T)
                    - r * K * disc * Nd2
                )
            else:
                price = K * disc * Nmd2 - S * Nmd1
                delta = Nd1 - 1.0
                theta_annual = (
                    -S * phi_d1 * sigma / (2.0 * sqrt_T)
                    + r * K * disc * Nmd2
                )

            gamma = phi_d1 / (
                S * sigma * sqrt_T
            )
            vega = (
                S
                * phi_d1
                * sqrt_T
            )
            theta = theta_annual / 365.0

            rows.append(
                {
                    "option_id": option_id,
                    "price": float(price),
                    "delta": float(delta),
                    "gamma": float(gamma),
                    "vega": float(vega),
                    "theta": float(theta),
                }
            )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output = pd.DataFrame(
            rows,
            columns=[
                "option_id",
                "price",
                "delta",
                "gamma",
                "vega",
                "theta",
            ],
        )

        output.to_parquet(
            out_dir / "results.parquet",
            index=False,
        )
