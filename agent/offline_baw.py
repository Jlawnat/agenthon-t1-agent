from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from agent.qf_primitives import (
    historical_log_return_calibration,
)


def _find_spy(task_dir: Path) -> Path:
    preferred = task_dir / "environment" / "data" / "spy_daily.csv"
    if preferred.is_file():
        return preferred

    matches = [
        path
        for path in task_dir.rglob("spy_daily.csv")
        if "checks" not in path.parts
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one spy_daily.csv, found {len(matches)}."
        )
    return matches[0]


def _bs_price(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    volatility: float,
    maturity: float,
    kind: str,
) -> float:
    s = float(spot)
    k = float(strike)
    t = float(maturity)

    if t <= 0.0:
        return float(
            max(s - k, 0.0)
            if kind == "call"
            else max(k - s, 0.0)
        )

    root_t = math.sqrt(t)
    d1 = (
        math.log(s / k)
        + (
            rate
            - dividend
            + 0.5 * volatility * volatility
        )
        * t
    ) / (volatility * root_t)
    d2 = d1 - volatility * root_t

    if kind == "call":
        return float(
            s * math.exp(-dividend * t) * norm.cdf(d1)
            - k * math.exp(-rate * t) * norm.cdf(d2)
        )

    return float(
        k * math.exp(-rate * t) * norm.cdf(-d2)
        - s * math.exp(-dividend * t) * norm.cdf(-d1)
    )


def _critical_price(
    *,
    strike: float,
    rate: float,
    dividend: float,
    volatility: float,
    maturity: float,
    kind: str,
) -> tuple[float, float]:
    k = float(strike)
    t = float(maturity)
    sigma2 = volatility * volatility

    m = 2.0 * rate / sigma2
    n = 2.0 * (rate - dividend) / sigma2
    capital_k = 1.0 - math.exp(-rate * t)
    disc_q = math.exp(-dividend * t)

    if kind == "call":
        if dividend <= 0.0:
            return math.inf, math.inf

        q2 = (
            -(n - 1.0)
            + math.sqrt(
                (n - 1.0) ** 2
                + 4.0 * m / capital_k
            )
        ) / 2.0

        def equation(s_star: float) -> float:
            d1 = (
                math.log(s_star / k)
                + (
                    rate
                    - dividend
                    + 0.5 * sigma2
                )
                * t
            ) / (
                volatility * math.sqrt(t)
            )

            european = _bs_price(
                s_star,
                k,
                rate,
                dividend,
                volatility,
                t,
                "call",
            )

            return (
                s_star
                - k
                - european
                - (
                    1.0
                    - disc_q * norm.cdf(d1)
                )
                * s_star
                / q2
            )

        lower = k * (1.0 + 1e-12)
        upper = k * 100.0
        s_star = float(
            brentq(
                equation,
                lower,
                upper,
                maxiter=200,
            )
        )
        return s_star, float(q2)

    q1 = (
        -(n - 1.0)
        - math.sqrt(
            (n - 1.0) ** 2
            + 4.0 * m / capital_k
        )
    ) / 2.0

    def equation(s_star: float) -> float:
        d1 = (
            math.log(s_star / k)
            + (
                rate
                - dividend
                + 0.5 * sigma2
            )
            * t
        ) / (
            volatility * math.sqrt(t)
        )

        european = _bs_price(
            s_star,
            k,
            rate,
            dividend,
            volatility,
            t,
            "put",
        )

        return (
            k
            - s_star
            - european
            + (
                1.0
                - disc_q * norm.cdf(-d1)
            )
            * s_star
            / q1
        )

    lower = k * 1e-12
    upper = k * (1.0 - 1e-12)
    s_star = float(
        brentq(
            equation,
            lower,
            upper,
            maxiter=200,
        )
    )
    return s_star, float(q1)


def _baw_price(
    *,
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    volatility: float,
    maturity: float,
    kind: str,
) -> tuple[float, float]:
    european = _bs_price(
        spot,
        strike,
        rate,
        dividend,
        volatility,
        maturity,
        kind,
    )

    if kind == "call" and dividend <= 0.0:
        return european, math.inf

    s_star, q_root = _critical_price(
        strike=strike,
        rate=rate,
        dividend=dividend,
        volatility=volatility,
        maturity=maturity,
        kind=kind,
    )

    d1_star = (
        math.log(s_star / strike)
        + (
            rate
            - dividend
            + 0.5 * volatility * volatility
        )
        * maturity
    ) / (
        volatility * math.sqrt(maturity)
    )
    disc_q = math.exp(-dividend * maturity)

    if kind == "call":
        if spot >= s_star:
            return float(spot - strike), s_star

        a2 = (
            1.0
            - disc_q * norm.cdf(d1_star)
        ) * s_star / q_root

        price = (
            european
            + a2 * (spot / s_star) ** q_root
        )
        return float(price), s_star

    if spot <= s_star:
        return float(strike - spot), s_star

    a1 = -(
        1.0
        - disc_q * norm.cdf(-d1_star)
    ) * s_star / q_root

    price = (
        european
        + a1 * (spot / s_star) ** q_root
    )
    return float(price), s_star


def _crr_american(
    *,
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    volatility: float,
    maturity: float,
    kind: str,
    steps: int = 1000,
) -> float:
    n = int(steps)
    dt = maturity / n
    up = math.exp(volatility * math.sqrt(dt))
    down = 1.0 / up
    discount = math.exp(-rate * dt)

    probability = (
        math.exp(
            (rate - dividend) * dt
        )
        - down
    ) / (up - down)

    if not 0.0 < probability < 1.0:
        raise RuntimeError(
            "CRR risk-neutral probability fell outside (0,1)."
        )

    j = np.arange(
        n + 1,
        dtype=float,
    )
    stock = (
        spot
        * (up ** j)
        * (down ** (n - j))
    )

    if kind == "call":
        values = np.maximum(
            stock - strike,
            0.0,
        )
    else:
        values = np.maximum(
            strike - stock,
            0.0,
        )

    for step in range(
        n - 1,
        -1,
        -1,
    ):
        values = discount * (
            probability * values[1:]
            + (1.0 - probability)
            * values[:-1]
        )

        j = np.arange(
            step + 1,
            dtype=float,
        )
        stock = (
            spot
            * (up ** j)
            * (down ** (step - j))
        )

        if kind == "call":
            exercise = np.maximum(
                stock - strike,
                0.0,
            )
        else:
            exercise = np.maximum(
                strike - stock,
                0.0,
            )

        values = np.maximum(
            values,
            exercise,
        )

    return float(values[0])


@dataclass(frozen=True)
class BaroneAdesiWhaleySkill:
    name: str = "barone-adesi-whaley"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        del task_dir

        lowered = instruction.lower()
        required = (
            "barone-adesi",
            "american option",
            "baw_prices.csv",
            "critical_prices.csv",
            "binomial_comparison.csv",
            "dividend_sensitivity.csv",
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

        frame = pd.read_csv(
            _find_spy(task_dir)
        )
        closes = pd.to_numeric(
            frame["close"],
            errors="coerce",
        ).dropna().to_numpy(
            dtype=float,
        )

        cal = historical_log_return_calibration(
            closes,
            annualization=252.0,
        )

        s0 = float(
            cal["spot"]
        )
        sigma = float(
            cal["annualized_vol"]
        )
        n_returns = int(
            cal["n_returns"]
        )

        rate = 0.05
        dividend = 0.013

        calibration = {
            "S0": s0,
            "sigma": sigma,
            "r": rate,
            "D": dividend,
            "n_returns": n_returns,
        }

        moneyness_grid = [
            0.80,
            0.90,
            0.95,
            1.00,
            1.05,
            1.10,
            1.20,
        ]
        maturity_grid = [
            0.25,
            0.50,
            1.00,
        ]

        price_rows: list[
            dict[str, float]
        ] = []

        for maturity in maturity_grid:
            for moneyness in moneyness_grid:
                strike = (
                    s0 / moneyness
                )

                bs_call = _bs_price(
                    s0,
                    strike,
                    rate,
                    dividend,
                    sigma,
                    maturity,
                    "call",
                )
                bs_put = _bs_price(
                    s0,
                    strike,
                    rate,
                    dividend,
                    sigma,
                    maturity,
                    "put",
                )

                baw_call, star_call = (
                    _baw_price(
                        spot=s0,
                        strike=strike,
                        rate=rate,
                        dividend=dividend,
                        volatility=sigma,
                        maturity=maturity,
                        kind="call",
                    )
                )
                baw_put, star_put = (
                    _baw_price(
                        spot=s0,
                        strike=strike,
                        rate=rate,
                        dividend=dividend,
                        volatility=sigma,
                        maturity=maturity,
                        kind="put",
                    )
                )

                price_rows.append(
                    {
                        "moneyness": moneyness,
                        "T": maturity,
                        "K": strike,
                        "bs_call": bs_call,
                        "bs_put": bs_put,
                        "baw_call": baw_call,
                        "baw_put": baw_put,
                        "premium_call": max(
                            baw_call - bs_call,
                            0.0,
                        ),
                        "premium_put": max(
                            baw_put - bs_put,
                            0.0,
                        ),
                        "S_star_call": star_call,
                        "S_star_put": star_put,
                    }
                )

        baw_prices = pd.DataFrame(
            price_rows
        )

        surface_rows = []
        for maturity in (
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            1.00,
            2.00,
            5.00,
        ):
            star_call, _ = (
                _critical_price(
                    strike=s0,
                    rate=rate,
                    dividend=dividend,
                    volatility=sigma,
                    maturity=maturity,
                    kind="call",
                )
            )
            star_put, _ = (
                _critical_price(
                    strike=s0,
                    rate=rate,
                    dividend=dividend,
                    volatility=sigma,
                    maturity=maturity,
                    kind="put",
                )
            )

            surface_rows.append(
                {
                    "T": maturity,
                    "S_star_call": star_call,
                    "S_star_put": star_put,
                    "S_star_call_ratio": (
                        star_call / s0
                    ),
                    "S_star_put_ratio": (
                        star_put / s0
                    ),
                }
            )

        critical = pd.DataFrame(
            surface_rows
        )

        comparison_rows = []
        for maturity in maturity_grid:
            baw_call, _ = _baw_price(
                spot=s0,
                strike=s0,
                rate=rate,
                dividend=dividend,
                volatility=sigma,
                maturity=maturity,
                kind="call",
            )
            baw_put, _ = _baw_price(
                spot=s0,
                strike=s0,
                rate=rate,
                dividend=dividend,
                volatility=sigma,
                maturity=maturity,
                kind="put",
            )
            crr_call = _crr_american(
                spot=s0,
                strike=s0,
                rate=rate,
                dividend=dividend,
                volatility=sigma,
                maturity=maturity,
                kind="call",
            )
            crr_put = _crr_american(
                spot=s0,
                strike=s0,
                rate=rate,
                dividend=dividend,
                volatility=sigma,
                maturity=maturity,
                kind="put",
            )

            call_diff = (
                baw_call - crr_call
            )
            put_diff = (
                baw_put - crr_put
            )

            comparison_rows.append(
                {
                    "T": maturity,
                    "baw_call": baw_call,
                    "baw_put": baw_put,
                    "crr_call": crr_call,
                    "crr_put": crr_put,
                    "call_diff": call_diff,
                    "put_diff": put_diff,
                    "call_reldiff": abs(
                        call_diff
                    )
                    / max(
                        abs(crr_call),
                        1e-15,
                    ),
                    "put_reldiff": abs(
                        put_diff
                    )
                    / max(
                        abs(crr_put),
                        1e-15,
                    ),
                }
            )

        comparison = pd.DataFrame(
            comparison_rows
        )

        sensitivity_rows = []
        for q_value in (
            0.00,
            0.01,
            0.02,
            0.03,
            0.05,
        ):
            bs_call = _bs_price(
                s0,
                s0,
                rate,
                q_value,
                sigma,
                1.0,
                "call",
            )
            bs_put = _bs_price(
                s0,
                s0,
                rate,
                q_value,
                sigma,
                1.0,
                "put",
            )
            baw_call, _ = _baw_price(
                spot=s0,
                strike=s0,
                rate=rate,
                dividend=q_value,
                volatility=sigma,
                maturity=1.0,
                kind="call",
            )
            baw_put, _ = _baw_price(
                spot=s0,
                strike=s0,
                rate=rate,
                dividend=q_value,
                volatility=sigma,
                maturity=1.0,
                kind="put",
            )

            sensitivity_rows.append(
                {
                    "D": q_value,
                    "baw_call": baw_call,
                    "baw_put": baw_put,
                    "bs_call": bs_call,
                    "bs_put": bs_put,
                    "premium_call": max(
                        baw_call
                        - bs_call,
                        0.0,
                    ),
                    "premium_put": max(
                        baw_put
                        - bs_put,
                        0.0,
                    ),
                }
            )

        sensitivity = pd.DataFrame(
            sensitivity_rows
        )

        atm_t1 = baw_prices[
            np.isclose(
                baw_prices["moneyness"],
                1.0,
            )
            & np.isclose(
                baw_prices["T"],
                1.0,
            )
        ].iloc[0]

        call_premiums = (
            sensitivity.sort_values(
                "D"
            )["premium_call"]
            .to_numpy(dtype=float)
        )

        summary = {
            "baw_call_atm_T1": float(
                atm_t1["baw_call"]
            ),
            "baw_put_atm_T1": float(
                atm_t1["baw_put"]
            ),
            "premium_call_atm_T1": float(
                atm_t1["premium_call"]
            ),
            "premium_put_atm_T1": float(
                atm_t1["premium_put"]
            ),
            "S_star_call_T1": float(
                atm_t1["S_star_call"]
            ),
            "S_star_put_T1": float(
                atm_t1["S_star_put"]
            ),
            "max_crr_reldiff_call": float(
                comparison[
                    "call_reldiff"
                ].max()
            ),
            "max_crr_reldiff_put": float(
                comparison[
                    "put_reldiff"
                ].max()
            ),
            "call_premium_increases_with_D": bool(
                np.all(
                    np.diff(
                        call_premiums
                    )
                    >= -1e-10
                )
            ),
            "put_premium_positive": bool(
                (
                    baw_prices[
                        "premium_put"
                    ]
                    >= -1e-10
                ).all()
            ),
            "call_geq_euro": bool(
                (
                    baw_prices[
                        "baw_call"
                    ]
                    >= baw_prices[
                        "bs_call"
                    ]
                    - 1e-10
                ).all()
            ),
            "put_geq_euro": bool(
                (
                    baw_prices[
                        "baw_put"
                    ]
                    >= baw_prices[
                        "bs_put"
                    ]
                    - 1e-10
                ).all()
            ),
            "S_star_call_increases_with_T": bool(
                np.all(
                    np.diff(
                        critical.sort_values(
                            "T"
                        )[
                            "S_star_call"
                        ].to_numpy(
                            dtype=float
                        )
                    )
                    >= -1e-6
                )
            ),
            "S_star_put_decreases_with_T": bool(
                np.all(
                    np.diff(
                        critical.sort_values(
                            "T"
                        )[
                            "S_star_put"
                        ].to_numpy(
                            dtype=float
                        )
                    )
                    <= 1e-6
                )
            ),
            "zero_D_call_premium_negligible": bool(
                abs(
                    float(
                        sensitivity.loc[
                            np.isclose(
                                sensitivity[
                                    "D"
                                ],
                                0.0,
                            ),
                            "premium_call",
                        ].iloc[0]
                    )
                )
                < 0.01
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

        baw_prices.to_csv(
            out_dir
            / "baw_prices.csv",
            index=False,
        )
        critical.to_csv(
            out_dir
            / "critical_prices.csv",
            index=False,
        )
        comparison.to_csv(
            out_dir
            / "binomial_comparison.csv",
            index=False,
        )
        sensitivity.to_csv(
            out_dir
            / "dividend_sensitivity.csv",
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
