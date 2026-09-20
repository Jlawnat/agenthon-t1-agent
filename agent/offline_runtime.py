from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from agent.offline_bollinger import (
    BollingerBacktestSkill,
)

from agent.offline_brinson import (
    BrinsonSectorAttributionSkill,
)

from agent.offline_cir import (
    CirBondPricingSkill,
)

from agent.offline_tca import (
    BinanceParticipationTcaSkill,
)

from agent.offline_alpha import (
    AlphaHedgeStrategySkill,
)


from agent.offline_portfolio import (
    PortfolioStrategySkill,
)

from agent.offline_risk import (
    MarketRiskSkill,
)

from agent.offline_fixed_income import (
    FixedIncomeCurveSkill,
)

from agent.offline_derivatives import (
    TwoAssetDerivativesSkill,
)

from agent.offline_event_study import (
    EventStudySkill,
)

from agent.offline_volatility import (
    OhlcVolatilitySkill,
)

class OfflineSkill(Protocol):
    name: str

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        ...

    def solve(
        self,
        *,
        instruction: str,
        task_dir: Path,
        out_dir: Path,
        seed: int,
    ) -> None:
        ...


def _read_instruction(task_dir: Path) -> str:
    path = task_dir / "instruction.md"
    if not path.is_file():
        raise RuntimeError(
            "Offline runtime requires instruction.md."
        )
    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _normal_cdf(x: float) -> float:
    return 0.5 * (
        1.0
        + math.erf(
            x / math.sqrt(2.0)
        )
    )


def _find_market_csv(task_dir: Path) -> tuple[Path, str]:
    candidates: list[tuple[Path, str]] = []

    for path in sorted(task_dir.rglob("*.csv")):
        if "checks" in path.parts:
            continue

        try:
            frame = pd.read_csv(
                path,
                nrows=5,
            )
        except Exception:
            continue

        by_lower = {
            str(column).lower(): str(column)
            for column in frame.columns
        }

        if "close" in by_lower:
            candidates.append(
                (
                    path,
                    by_lower["close"],
                )
            )

    if not candidates:
        raise RuntimeError(
            "Offline Black-Scholes skill could not find "
            "a CSV containing a Close column."
        )

    return candidates[0]


@dataclass(frozen=True)
class BlackScholesGreeksSkill:
    name: str = "black-scholes-greeks-pde"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        del task_dir

        lowered = instruction.lower()
        required = (
            "black-scholes",
            "greeks",
            "pde",
            "greeks_surface.csv",
            "pde_verification.csv",
        )
        return all(
            token in lowered
            for token in required
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

        csv_path, close_column = (
            _find_market_csv(
                task_dir
            )
        )

        frame = pd.read_csv(
            csv_path
        )

        close = pd.to_numeric(
            frame[close_column],
            errors="coerce",
        ).dropna()

        if len(close) < 2:
            raise RuntimeError(
                "Price history must contain at least two valid closes."
            )

        log_returns = np.log(
            close / close.shift(1)
        ).dropna()

        S0 = float(
            close.iloc[-1]
        )
        return_mean = float(
            log_returns.mean()
        )
        return_std = float(
            log_returns.std(ddof=1)
        )
        sigma = float(
            return_std * math.sqrt(252.0)
        )
        n_returns = int(
            len(log_returns)
        )

        r = 0.05
        D = 0.013
        K = S0

        moneyness_grid = (
            0.80,
            0.85,
            0.90,
            0.95,
            1.00,
            1.05,
            1.10,
            1.15,
            1.20,
        )
        maturity_grid = (
            0.25,
            0.50,
            1.00,
        )

        greeks_rows: list[dict[str, float]] = []
        pde_rows: list[dict[str, float]] = []

        for T in maturity_grid:
            sqrt_T = math.sqrt(T)
            disc_r = math.exp(-r * T)
            disc_D = math.exp(-D * T)

            for moneyness in moneyness_grid:
                S = moneyness * K

                d1 = (
                    math.log(S / K)
                    + (
                        r
                        - D
                        + 0.5 * sigma * sigma
                    ) * T
                ) / (
                    sigma * sqrt_T
                )
                d2 = (
                    d1
                    - sigma * sqrt_T
                )

                N_d1 = _normal_cdf(d1)
                N_d2 = _normal_cdf(d2)
                N_minus_d1 = _normal_cdf(-d1)
                N_minus_d2 = _normal_cdf(-d2)
                phi_d1 = _normal_pdf(d1)

                call_price = (
                    S * disc_D * N_d1
                    - K * disc_r * N_d2
                )
                put_price = (
                    K * disc_r * N_minus_d2
                    - S * disc_D * N_minus_d1
                )

                call_delta = (
                    disc_D * N_d1
                )
                put_delta = (
                    disc_D
                    * (N_d1 - 1.0)
                )
                gamma = (
                    disc_D
                    * phi_d1
                    / (
                        S
                        * sigma
                        * sqrt_T
                    )
                )
                vega = (
                    S
                    * disc_D
                    * phi_d1
                    * sqrt_T
                )

                diffusion_theta = (
                    -S
                    * disc_D
                    * phi_d1
                    * sigma
                    / (2.0 * sqrt_T)
                )

                call_theta = (
                    diffusion_theta
                    - r
                    * K
                    * disc_r
                    * N_d2
                    + D
                    * S
                    * disc_D
                    * N_d1
                )
                put_theta = (
                    diffusion_theta
                    + r
                    * K
                    * disc_r
                    * N_minus_d2
                    - D
                    * S
                    * disc_D
                    * N_minus_d1
                )

                call_rho = (
                    K
                    * T
                    * disc_r
                    * N_d2
                )
                put_rho = (
                    -K
                    * T
                    * disc_r
                    * N_minus_d2
                )

                call_residual = (
                    call_theta
                    + (r - D)
                    * S
                    * call_delta
                    + 0.5
                    * sigma
                    * sigma
                    * S
                    * S
                    * gamma
                    - r
                    * call_price
                )
                put_residual = (
                    put_theta
                    + (r - D)
                    * S
                    * put_delta
                    + 0.5
                    * sigma
                    * sigma
                    * S
                    * S
                    * gamma
                    - r
                    * put_price
                )

                parity_error = abs(
                    call_price
                    - put_price
                    - (
                        S * disc_D
                        - K * disc_r
                    )
                )

                greeks_rows.append(
                    {
                        "T": float(T),
                        "moneyness": float(moneyness),
                        "S": float(S),
                        "call_price": float(call_price),
                        "put_price": float(put_price),
                        "call_delta": float(call_delta),
                        "put_delta": float(put_delta),
                        "gamma": float(gamma),
                        "call_theta": float(call_theta),
                        "put_theta": float(put_theta),
                        "vega": float(vega),
                        "call_rho": float(call_rho),
                        "put_rho": float(put_rho),
                    }
                )

                pde_rows.append(
                    {
                        "T": float(T),
                        "moneyness": float(moneyness),
                        "S": float(S),
                        "call_pde_residual": float(call_residual),
                        "put_pde_residual": float(put_residual),
                        "parity_error": float(parity_error),
                    }
                )

        greeks = pd.DataFrame(
            greeks_rows
        ).sort_values(
            ["T", "moneyness"],
            kind="stable",
        ).reset_index(
            drop=True
        )

        pde = pd.DataFrame(
            pde_rows
        ).sort_values(
            ["T", "moneyness"],
            kind="stable",
        ).reset_index(
            drop=True
        )

        def _atm(T: float) -> pd.Series:
            rows = greeks[
                np.isclose(
                    greeks["T"],
                    T,
                )
                & np.isclose(
                    greeks["moneyness"],
                    1.0,
                )
            ]
            if len(rows) != 1:
                raise RuntimeError(
                    "ATM row selection failed."
                )
            return rows.iloc[0]

        T050 = greeks[
            np.isclose(
                greeks["T"],
                0.50,
            )
        ]
        max_gamma_row = (
            T050.loc[
                T050["gamma"].idxmax()
            ]
        )

        call_upper = np.exp(
            -D * greeks["T"]
        )

        calibration = {
            "S0": S0,
            "sigma": sigma,
            "r": r,
            "D": D,
            "K": K,
            "n_returns": n_returns,
            "return_mean": return_mean,
            "return_std": return_std,
        }

        summary = {
            "n_grid_points": 27,
            "n_maturities": 3,
            "n_moneyness": 9,
            "max_call_pde_residual": float(
                pde[
                    "call_pde_residual"
                ].abs().max()
            ),
            "max_put_pde_residual": float(
                pde[
                    "put_pde_residual"
                ].abs().max()
            ),
            "max_parity_error": float(
                pde[
                    "parity_error"
                ].max()
            ),
            "atm_call_price_T025": float(
                _atm(0.25)[
                    "call_price"
                ]
            ),
            "atm_call_price_T050": float(
                _atm(0.50)[
                    "call_price"
                ]
            ),
            "atm_call_price_T100": float(
                _atm(1.00)[
                    "call_price"
                ]
            ),
            "atm_call_delta_T100": float(
                _atm(1.00)[
                    "call_delta"
                ]
            ),
            "atm_gamma_T100": float(
                _atm(1.00)[
                    "gamma"
                ]
            ),
            "atm_vega_T100": float(
                _atm(1.00)[
                    "vega"
                ]
            ),
            "max_gamma_moneyness_T050": float(
                max_gamma_row[
                    "moneyness"
                ]
            ),
            "delta_call_range_valid": bool(
                (
                    (greeks["call_delta"] >= -1e-12)
                    & (
                        greeks["call_delta"]
                        <= call_upper + 1e-12
                    )
                ).all()
            ),
            "gamma_positive_everywhere": bool(
                (
                    greeks["gamma"] > 0.0
                ).all()
            ),
            "vega_positive_everywhere": bool(
                (
                    greeks["vega"] > 0.0
                ).all()
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
                calibration,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        greeks.to_csv(
            out_dir
            / "greeks_surface.csv",
            index=False,
        )

        pde.to_csv(
            out_dir
            / "pde_verification.csv",
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


_SKILLS: tuple[OfflineSkill, ...] = (
    BlackScholesGreeksSkill(),
    BollingerBacktestSkill(),
    BrinsonSectorAttributionSkill(),
    CirBondPricingSkill(),
    BinanceParticipationTcaSkill(),
    AlphaHedgeStrategySkill(),
    PortfolioStrategySkill(),
    MarketRiskSkill(),
    FixedIncomeCurveSkill(),
    TwoAssetDerivativesSkill(),
    EventStudySkill(),
    OhlcVolatilitySkill(),
)


def solve_offline(
    *,
    task_dir: Path,
    out_dir: Path,
    seed: int,
) -> str:
    instruction = _read_instruction(
        task_dir
    )

    for skill in _SKILLS:
        if skill.matches(
            instruction=instruction,
            task_dir=task_dir,
        ):
            skill.solve(
                instruction=instruction,
                task_dir=task_dir,
                out_dir=out_dir,
                seed=seed,
            )
            return skill.name

    raise RuntimeError(
        "No offline solver skill matched this task. "
        "Model runtime is unavailable."
    )
