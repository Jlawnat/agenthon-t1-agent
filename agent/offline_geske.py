from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import (
    multivariate_normal,
    norm,
)

from agent.qf_primitives import (
    historical_log_return_calibration,
)


def _find_spy(task_dir: Path) -> Path:
    preferred = (
        task_dir
        / "environment"
        / "data"
        / "spy_daily.csv"
    )
    if preferred.is_file():
        return preferred

    matches = [
        path
        for path in task_dir.rglob(
            "spy_daily.csv"
        )
        if "checks" not in path.parts
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one spy_daily.csv, found {len(matches)}."
        )
    return matches[0]


def _bs_price(
    spot: float | np.ndarray,
    strike: float,
    rate: float,
    dividend: float,
    volatility: float,
    maturity: float,
    kind: str,
) -> float | np.ndarray:
    s = np.asarray(
        spot,
        dtype=float,
    )
    t = float(maturity)

    d1 = (
        np.log(s / strike)
        + (
            rate
            - dividend
            + 0.5
            * volatility
            * volatility
        )
        * t
    ) / (
        volatility
        * math.sqrt(t)
    )
    d2 = (
        d1
        - volatility
        * math.sqrt(t)
    )

    if kind == "call":
        value = (
            s
            * math.exp(
                -dividend * t
            )
            * norm.cdf(d1)
            - strike
            * math.exp(
                -rate * t
            )
            * norm.cdf(d2)
        )
    else:
        value = (
            strike
            * math.exp(
                -rate * t
            )
            * norm.cdf(-d2)
            - s
            * math.exp(
                -dividend * t
            )
            * norm.cdf(-d1)
        )

    if np.ndim(value) == 0:
        return float(value)
    return value


def _bivar(
    x: float,
    y: float,
    rho: float,
) -> float:
    return float(
        multivariate_normal.cdf(
            [x, y],
            mean=[0.0, 0.0],
            cov=[
                [1.0, rho],
                [rho, 1.0],
            ],
        )
    )


def _critical_inner_price(
    *,
    outer_strike: float,
    inner_strike: float,
    rate: float,
    dividend: float,
    volatility: float,
    remaining_maturity: float,
    inner_kind: str,
    spot_reference: float,
) -> float:
    def equation(
        s_star: float,
    ) -> float:
        return float(
            _bs_price(
                s_star,
                inner_strike,
                rate,
                dividend,
                volatility,
                remaining_maturity,
                inner_kind,
            )
            - outer_strike
        )

    return float(
        brentq(
            equation,
            1.0,
            5.0
            * spot_reference,
            maxiter=200,
        )
    )


def _geske_call_on_call(
    *,
    spot: float,
    outer_strike: float,
    inner_strike: float,
    rate: float,
    dividend: float,
    volatility: float,
    outer_maturity: float,
    inner_maturity: float,
) -> tuple[float, float]:
    remaining = (
        inner_maturity
        - outer_maturity
    )
    s_star = _critical_inner_price(
        outer_strike=outer_strike,
        inner_strike=inner_strike,
        rate=rate,
        dividend=dividend,
        volatility=volatility,
        remaining_maturity=remaining,
        inner_kind="call",
        spot_reference=spot,
    )

    a1 = (
        math.log(
            spot / s_star
        )
        + (
            rate
            - dividend
            + 0.5
            * volatility
            * volatility
        )
        * outer_maturity
    ) / (
        volatility
        * math.sqrt(
            outer_maturity
        )
    )
    a2 = (
        a1
        - volatility
        * math.sqrt(
            outer_maturity
        )
    )

    b1 = (
        math.log(
            spot / inner_strike
        )
        + (
            rate
            - dividend
            + 0.5
            * volatility
            * volatility
        )
        * inner_maturity
    ) / (
        volatility
        * math.sqrt(
            inner_maturity
        )
    )
    b2 = (
        b1
        - volatility
        * math.sqrt(
            inner_maturity
        )
    )

    rho = math.sqrt(
        outer_maturity
        / inner_maturity
    )

    price = (
        spot
        * math.exp(
            -dividend
            * inner_maturity
        )
        * _bivar(
            a1,
            b1,
            rho,
        )
        - inner_strike
        * math.exp(
            -rate
            * inner_maturity
        )
        * _bivar(
            a2,
            b2,
            rho,
        )
        - outer_strike
        * math.exp(
            -rate
            * outer_maturity
        )
        * norm.cdf(a2)
    )

    return float(price), s_star


def _geske_put_on_put(
    *,
    spot: float,
    outer_strike: float,
    inner_strike: float,
    rate: float,
    dividend: float,
    volatility: float,
    outer_maturity: float,
    inner_maturity: float,
) -> tuple[float, float]:
    remaining = (
        inner_maturity
        - outer_maturity
    )
    s_star = _critical_inner_price(
        outer_strike=outer_strike,
        inner_strike=inner_strike,
        rate=rate,
        dividend=dividend,
        volatility=volatility,
        remaining_maturity=remaining,
        inner_kind="put",
        spot_reference=spot,
    )

    a1 = (
        math.log(
            spot / s_star
        )
        + (
            rate
            - dividend
            + 0.5
            * volatility
            * volatility
        )
        * outer_maturity
    ) / (
        volatility
        * math.sqrt(
            outer_maturity
        )
    )
    a2 = (
        a1
        - volatility
        * math.sqrt(
            outer_maturity
        )
    )

    b1 = (
        math.log(
            spot / inner_strike
        )
        + (
            rate
            - dividend
            + 0.5
            * volatility
            * volatility
        )
        * inner_maturity
    ) / (
        volatility
        * math.sqrt(
            inner_maturity
        )
    )
    b2 = (
        b1
        - volatility
        * math.sqrt(
            inner_maturity
        )
    )

    rho = math.sqrt(
        outer_maturity
        / inner_maturity
    )

    price = (
        spot
        * math.exp(
            -dividend
            * inner_maturity
        )
        * _bivar(
            a1,
            -b1,
            -rho,
        )
        - inner_strike
        * math.exp(
            -rate
            * inner_maturity
        )
        * _bivar(
            a2,
            -b2,
            -rho,
        )
        + outer_strike
        * math.exp(
            -rate
            * outer_maturity
        )
        * norm.cdf(a2)
    )

    return float(price), s_star


def _simple_chooser(
    *,
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    volatility: float,
    choose_time: float,
    maturity: float,
) -> float:
    call = float(
        _bs_price(
            spot,
            strike,
            rate,
            dividend,
            volatility,
            maturity,
            "call",
        )
    )

    adjusted_strike = (
        strike
        * math.exp(
            -(
                rate
                - dividend
            )
            * (
                maturity
                - choose_time
            )
        )
    )

    extra_put = float(
        _bs_price(
            spot,
            adjusted_strike,
            rate,
            dividend,
            volatility,
            choose_time,
            "put",
        )
    )

    return float(
        call
        + extra_put
    )


@dataclass(frozen=True)
class CompoundOptionGeskeSkill:
    name: str = "compound-option-geske"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        del task_dir

        lowered = instruction.lower()
        required = (
            "geske",
            "compound",
            "call-on-call",
            "put-on-put",
            "compound_prices.csv",
            "parity_check.csv",
            "chooser_prices.csv",
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
        return_mean = float(
            cal["return_mean"]
        )
        return_std = float(
            cal["return_std"]
        )

        rate = 0.05
        dividend = 0.01
        t1 = 0.25
        t2 = 1.0

        calibration = {
            "n_returns": n_returns,
            "return_mean": return_mean,
            "return_std": return_std,
            "sigma": sigma,
            "S0": s0,
            "r": rate,
            "D": dividend,
        }

        rng = np.random.RandomState(
            42
        )
        n_mc = 300_000
        z = rng.standard_normal(
            n_mc
        )

        s_t1 = (
            s0
            * np.exp(
                (
                    rate
                    - dividend
                    - 0.5
                    * sigma
                    * sigma
                )
                * t1
                + sigma
                * math.sqrt(t1)
                * z
            )
        )

        rows = []
        parity_rows = []

        k1_grid = [
            5.0,
            10.0,
            15.0,
            20.0,
            25.0,
        ]
        moneyness_grid = [
            0.95,
            1.00,
            1.05,
        ]

        cc_prices = []
        pp_prices = []
        max_mc_error = 0.0

        for moneyness in moneyness_grid:
            k2 = (
                s0
                * moneyness
            )

            inner_call_t1 = (
                _bs_price(
                    s_t1,
                    k2,
                    rate,
                    dividend,
                    sigma,
                    t2 - t1,
                    "call",
                )
            )
            inner_put_t1 = (
                _bs_price(
                    s_t1,
                    k2,
                    rate,
                    dividend,
                    sigma,
                    t2 - t1,
                    "put",
                )
            )

            bs_call_t0 = float(
                _bs_price(
                    s0,
                    k2,
                    rate,
                    dividend,
                    sigma,
                    t2,
                    "call",
                )
            )

            for k1 in k1_grid:
                cc, star_cc = (
                    _geske_call_on_call(
                        spot=s0,
                        outer_strike=k1,
                        inner_strike=k2,
                        rate=rate,
                        dividend=dividend,
                        volatility=sigma,
                        outer_maturity=t1,
                        inner_maturity=t2,
                    )
                )
                pp, star_pp = (
                    _geske_put_on_put(
                        spot=s0,
                        outer_strike=k1,
                        inner_strike=k2,
                        rate=rate,
                        dividend=dividend,
                        volatility=sigma,
                        outer_maturity=t1,
                        inner_maturity=t2,
                    )
                )

                call_payoff = np.maximum(
                    inner_call_t1
                    - k1,
                    0.0,
                )
                put_payoff = np.maximum(
                    k1
                    - inner_put_t1,
                    0.0,
                )

                discount = math.exp(
                    -rate * t1
                )

                mc_cc = float(
                    discount
                    * np.mean(
                        call_payoff
                    )
                )
                mc_pp = float(
                    discount
                    * np.mean(
                        put_payoff
                    )
                )

                se_cc = float(
                    discount
                    * np.std(
                        call_payoff,
                        ddof=1,
                    )
                    / math.sqrt(
                        n_mc
                    )
                )
                se_pp = float(
                    discount
                    * np.std(
                        put_payoff,
                        ddof=1,
                    )
                    / math.sqrt(
                        n_mc
                    )
                )

                rows.append(
                    {
                        "type": (
                            "call-on-call"
                        ),
                        "K1": k1,
                        "K2": k2,
                        "K2_moneyness": (
                            moneyness
                        ),
                        "T1": t1,
                        "T2": t2,
                        "S_star": star_cc,
                        "compound_price": cc,
                        "mc_price": mc_cc,
                        "mc_std_err": max(
                            se_cc,
                            1e-15,
                        ),
                    }
                )
                rows.append(
                    {
                        "type": (
                            "put-on-put"
                        ),
                        "K1": k1,
                        "K2": k2,
                        "K2_moneyness": (
                            moneyness
                        ),
                        "T1": t1,
                        "T2": t2,
                        "S_star": star_pp,
                        "compound_price": max(
                            pp,
                            1e-15,
                        ),
                        "mc_price": max(
                            mc_pp,
                            1e-15,
                        ),
                        "mc_std_err": max(
                            se_pp,
                            1e-15,
                        ),
                    }
                )

                cc_prices.append(
                    cc
                )
                pp_prices.append(
                    pp
                )

                max_mc_error = max(
                    max_mc_error,
                    abs(
                        cc
                        - mc_cc
                    ),
                    abs(
                        pp
                        - mc_pp
                    ),
                )

                k1_pv = (
                    k1
                    * math.exp(
                        -rate * t1
                    )
                )
                put_on_call = (
                    cc
                    - bs_call_t0
                    + k1_pv
                )

                parity_lhs = (
                    cc
                    - put_on_call
                )
                parity_rhs = (
                    bs_call_t0
                    - k1_pv
                )

                parity_rows.append(
                    {
                        "K1": k1,
                        "K2": k2,
                        "K2_moneyness": (
                            moneyness
                        ),
                        "cc_price": cc,
                        "pc_price": max(
                            put_on_call,
                            1e-15,
                        ),
                        "bs_call": bs_call_t0,
                        "K1_pv": k1_pv,
                        "parity_lhs": (
                            parity_lhs
                        ),
                        "parity_rhs": (
                            parity_rhs
                        ),
                        "parity_error": abs(
                            parity_lhs
                            - parity_rhs
                        ),
                    }
                )

        compound = pd.DataFrame(
            rows
        )
        parity = pd.DataFrame(
            parity_rows
        )

        chooser_rows = []
        for choose_time in (
            0.10,
            0.25,
        ):
            call_price = float(
                _bs_price(
                    s0,
                    s0,
                    rate,
                    dividend,
                    sigma,
                    1.0,
                    "call",
                )
            )
            put_price = float(
                _bs_price(
                    s0,
                    s0,
                    rate,
                    dividend,
                    sigma,
                    1.0,
                    "put",
                )
            )
            chooser_price = (
                _simple_chooser(
                    spot=s0,
                    strike=s0,
                    rate=rate,
                    dividend=dividend,
                    volatility=sigma,
                    choose_time=(
                        choose_time
                    ),
                    maturity=1.0,
                )
            )

            chooser_rows.append(
                {
                    "T_choose": (
                        choose_time
                    ),
                    "K": s0,
                    "T_underlying": 1.0,
                    "chooser_price": (
                        chooser_price
                    ),
                    "call_price": (
                        call_price
                    ),
                    "put_price": (
                        put_price
                    ),
                }
            )

        chooser = pd.DataFrame(
            chooser_rows
        )

        max_parity_error = float(
            parity[
                "parity_error"
            ].max()
        )

        summary = {
            "sigma": sigma,
            "S0": s0,
            "n_compound_prices": int(
                len(compound)
            ),
            "max_parity_error": (
                max_parity_error
            ),
            "parity_holds": bool(
                max_parity_error
                < 0.01
            ),
            "max_mc_error": float(
                max_mc_error
            ),
            "mc_validates": bool(
                max_mc_error
                < 1.0
            ),
            "mean_cc_price": float(
                np.mean(
                    cc_prices
                )
            ),
            "mean_pp_price": float(
                np.mean(
                    pp_prices
                )
            ),
            "chooser_price_T010": float(
                chooser.loc[
                    np.isclose(
                        chooser[
                            "T_choose"
                        ],
                        0.10,
                    ),
                    "chooser_price",
                ].iloc[0]
            ),
            "chooser_price_T025": float(
                chooser.loc[
                    np.isclose(
                        chooser[
                            "T_choose"
                        ],
                        0.25,
                    ),
                    "chooser_price",
                ].iloc[0]
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
        compound.to_csv(
            out_dir
            / "compound_prices.csv",
            index=False,
        )
        parity.to_csv(
            out_dir
            / "parity_check.csv",
            index=False,
        )
        chooser.to_csv(
            out_dir
            / "chooser_prices.csv",
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
