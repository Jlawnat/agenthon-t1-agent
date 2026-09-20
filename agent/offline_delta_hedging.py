from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import norm


def _find_delta_hedge_data_dir(
    task_dir: Path,
) -> Path:
    required = {
        "stock_path.csv",
        "iv_path.csv",
        "dividends.csv",
        "fee_schedule.csv",
    }

    directories = sorted(
        {
            path.parent
            for path in task_dir.rglob("*")
            if path.is_file()
            and "checks" not in path.parts
        },
        key=lambda path: str(path),
    )

    for directory in directories:
        names = {
            path.name
            for path in directory.iterdir()
            if path.is_file()
        }

        if required.issubset(
            names
        ):
            return directory

    raise RuntimeError(
        "Could not discover delta-hedging input files."
    )


def _parse_contract_parameters(
    instruction: str,
) -> dict[
    str,
    object,
]:
    # Normalize lightweight Markdown punctuation before parsing.
    plain = (
        instruction
        .replace("`", "")
        .replace("–", "-")
        .replace("—", "-")
    )

    strike_match = re.search(
        r"Strike\s+K\s*\|\s*([0-9.]+)",
        plain,
        flags=re.IGNORECASE,
    )

    rate_match = re.search(
        r"Risk[- ]free rate\s+r\s*\|\s*([0-9.]+)",
        plain,
        flags=re.IGNORECASE,
    )

    maturity_match = re.search(
        r"T\s*=\s*([0-9.]+)\s*year\s*=\s*(\d+)\s*trading days",
        plain,
        flags=re.IGNORECASE,
    )

    frequency_match = re.search(
        r"freq_days\s*(?:∈|in)\s*\{([^}]+)\}",
        plain,
        flags=re.IGNORECASE,
    )

    if not (
        strike_match
        and rate_match
        and maturity_match
        and frequency_match
    ):
        raise RuntimeError(
            "Could not parse delta-hedging contract parameters."
        )

    frequencies = [
        int(
            token.strip()
        )
        for token
        in frequency_match.group(
            1
        ).split(",")
    ]

    return {
        "strike": float(
            strike_match.group(
                1
            )
        ),
        "risk_free_rate": float(
            rate_match.group(
                1
            )
        ),
        "maturity_years": float(
            maturity_match.group(
                1
            )
        ),
        "trading_days": int(
            maturity_match.group(
                2
            )
        ),
        "frequencies": (
            frequencies
        ),
    }


def _pv_remaining_dividends(
    *,
    day: int,
    dividends: pd.DataFrame,
    risk_free_rate: float,
    trading_days: int,
) -> float:
    remaining = dividends[
        dividends[
            "ex_div_day"
        ]
        > int(
            day
        )
    ]

    if remaining.empty:
        return 0.0

    ex_days = (
        pd.to_numeric(
            remaining[
                "ex_div_day"
            ],
            errors="raise",
        )
        .astype(int)
        .to_numpy()
    )

    amounts = (
        pd.to_numeric(
            remaining[
                "amount"
            ],
            errors="raise",
        )
        .astype(float)
        .to_numpy()
    )

    times = (
        ex_days
        - int(
            day
        )
    ) / float(
        trading_days
    )

    return float(
        np.sum(
            amounts
            * np.exp(
                -float(
                    risk_free_rate
                )
                * times
            )
        )
    )


def _escrowed_call_price_delta(
    *,
    spot: float,
    strike: float,
    tau: float,
    risk_free_rate: float,
    volatility: float,
    pv_dividends: float,
) -> tuple[
    float,
    float,
    float,
]:
    effective_spot = (
        float(
            spot
        )
        - float(
            pv_dividends
        )
    )

    if effective_spot <= 0.0:
        raise RuntimeError(
            "Effective spot must be positive."
        )

    if tau <= 0.0:
        payoff = max(
            float(
                spot
            )
            - float(
                strike
            ),
            0.0,
        )

        delta = (
            1.0
            if float(
                spot
            )
            > float(
                strike
            )
            else 0.0
        )

        return (
            float(
                payoff
            ),
            float(
                delta
            ),
            float(
                effective_spot
            ),
        )

    if volatility <= 0.0:
        raise RuntimeError(
            "Implied volatility must be positive."
        )

    sqrt_tau = math.sqrt(
        tau
    )

    d1 = (
        math.log(
            effective_spot
            / float(
                strike
            )
        )
        + (
            float(
                risk_free_rate
            )
            + 0.5
            * float(
                volatility
            )
            * float(
                volatility
            )
        )
        * tau
    ) / (
        float(
            volatility
        )
        * sqrt_tau
    )

    d2 = (
        d1
        - float(
            volatility
        )
        * sqrt_tau
    )

    price = (
        effective_spot
        * norm.cdf(
            d1
        )
        - float(
            strike
        )
        * math.exp(
            -float(
                risk_free_rate
            )
            * tau
        )
        * norm.cdf(
            d2
        )
    )

    delta = float(
        norm.cdf(
            d1
        )
    )

    return (
        float(
            price
        ),
        delta,
        float(
            effective_spot
        ),
    )


def _load_paths(
    data_dir: Path,
    trading_days: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[
        str,
        float,
    ],
]:
    stock = pd.read_csv(
        data_dir
        / "stock_path.csv"
    )

    iv = pd.read_csv(
        data_dir
        / "iv_path.csv"
    )

    dividends = pd.read_csv(
        data_dir
        / "dividends.csv"
    )

    fees = pd.read_csv(
        data_dir
        / "fee_schedule.csv"
    )

    required_stock = {
        "day",
        "close",
    }

    required_iv = {
        "day",
        "implied_vol",
    }

    required_div = {
        "ex_div_day",
        "amount",
    }

    required_fee = {
        "trade_type",
        "rate_bps",
    }

    if not required_stock.issubset(
        stock.columns
    ):
        raise RuntimeError(
            "stock_path.csv has an unexpected schema."
        )

    if not required_iv.issubset(
        iv.columns
    ):
        raise RuntimeError(
            "iv_path.csv has an unexpected schema."
        )

    if not required_div.issubset(
        dividends.columns
    ):
        raise RuntimeError(
            "dividends.csv has an unexpected schema."
        )

    if not required_fee.issubset(
        fees.columns
    ):
        raise RuntimeError(
            "fee_schedule.csv has an unexpected schema."
        )

    stock = (
        stock[
            [
                "day",
                "close",
            ]
        ]
        .copy()
        .sort_values(
            "day",
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    iv = (
        iv[
            [
                "day",
                "implied_vol",
            ]
        ]
        .copy()
        .sort_values(
            "day",
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    stock[
        "day"
    ] = pd.to_numeric(
        stock[
            "day"
        ],
        errors="raise",
    ).astype(
        int
    )

    iv[
        "day"
    ] = pd.to_numeric(
        iv[
            "day"
        ],
        errors="raise",
    ).astype(
        int
    )

    stock[
        "close"
    ] = pd.to_numeric(
        stock[
            "close"
        ],
        errors="raise",
    ).astype(
        float
    )

    iv[
        "implied_vol"
    ] = pd.to_numeric(
        iv[
            "implied_vol"
        ],
        errors="raise",
    ).astype(
        float
    )

    expected_days = np.arange(
        0,
        int(
            trading_days
        )
        + 1,
        dtype=int,
    )

    if not np.array_equal(
        stock[
            "day"
        ].to_numpy(
            dtype=int
        ),
        expected_days,
    ):
        raise RuntimeError(
            "Stock path must contain every day from 0 through maturity."
        )

    if not np.array_equal(
        iv[
            "day"
        ].to_numpy(
            dtype=int
        ),
        expected_days,
    ):
        raise RuntimeError(
            "IV path must contain every day from 0 through maturity."
        )

    dividends[
        "ex_div_day"
    ] = pd.to_numeric(
        dividends[
            "ex_div_day"
        ],
        errors="raise",
    ).astype(
        int
    )

    dividends[
        "amount"
    ] = pd.to_numeric(
        dividends[
            "amount"
        ],
        errors="raise",
    ).astype(
        float
    )

    fee_rates = {
        str(
            row.trade_type
        ).strip().lower(): (
            float(
                row.rate_bps
            )
            / 10000.0
        )
        for row
        in fees.itertuples(
            index=False
        )
    }

    if not {
        "buy",
        "sell",
    }.issubset(
        fee_rates
    ):
        raise RuntimeError(
            "Fee schedule must provide buy and sell rates."
        )

    return (
        stock,
        iv,
        dividends,
        fee_rates,
    )


def _transaction_cost(
    *,
    share_change: float,
    spot: float,
    fee_rates: dict[
        str,
        float,
    ],
) -> float:
    if abs(
        share_change
    ) <= 1e-15:
        return 0.0

    trade_type = (
        "buy"
        if share_change
        > 0.0
        else "sell"
    )

    return float(
        abs(
            share_change
            * spot
        )
        * fee_rates[
            trade_type
        ]
    )


def _simulate_frequency(
    *,
    stock: pd.DataFrame,
    iv: pd.DataFrame,
    dividends: pd.DataFrame,
    fee_rates: dict[
        str,
        float,
    ],
    strike: float,
    risk_free_rate: float,
    trading_days: int,
    frequency: int,
    t0_price: float,
    t0_delta: float,
) -> dict[
    str,
    float | int,
]:
    dt = (
        1.0
        / float(
            trading_days
        )
    )

    spots = stock[
        "close"
    ].to_numpy(
        dtype=float
    )

    vols = iv[
        "implied_vol"
    ].to_numpy(
        dtype=float
    )

    dividend_map = {
        int(
            row.ex_div_day
        ): float(
            row.amount
        )
        for row
        in dividends.itertuples(
            index=False
        )
    }

    shares = float(
        t0_delta
    )

    initial_tc = (
        _transaction_cost(
            share_change=(
                shares
            ),
            spot=float(
                spots[
                    0
                ]
            ),
            fee_rates=(
                fee_rates
            ),
        )
    )

    cash = (
        float(
            t0_price
        )
        - shares
        * float(
            spots[
                0
            ]
        )
        - initial_tc
    )

    total_tc = float(
        initial_tc
    )

    total_dividends = 0.0

    n_rebalances = 0

    growth = math.exp(
        float(
            risk_free_rate
        )
        * dt
    )

    for day in range(
        1,
        int(
            trading_days
        ),
    ):
        cash *= growth

        if day in dividend_map:
            received = (
                shares
                * dividend_map[
                    day
                ]
            )

            cash += (
                received
            )

            total_dividends += (
                received
            )

        if (
            day
            % int(
                frequency
            )
            != 0
        ):
            continue

        pv_divs = (
            _pv_remaining_dividends(
                day=day,
                dividends=(
                    dividends
                ),
                risk_free_rate=(
                    risk_free_rate
                ),
                trading_days=(
                    trading_days
                ),
            )
        )

        tau = (
            int(
                trading_days
            )
            - day
        ) / float(
            trading_days
        )

        (
            _price,
            target_delta,
            _effective_spot,
        ) = _escrowed_call_price_delta(
            spot=float(
                spots[
                    day
                ]
            ),
            strike=float(
                strike
            ),
            tau=float(
                tau
            ),
            risk_free_rate=float(
                risk_free_rate
            ),
            volatility=float(
                vols[
                    day
                ]
            ),
            pv_dividends=float(
                pv_divs
            ),
        )

        share_change = (
            target_delta
            - shares
        )

        tc = _transaction_cost(
            share_change=(
                share_change
            ),
            spot=float(
                spots[
                    day
                ]
            ),
            fee_rates=(
                fee_rates
            ),
        )

        cash -= (
            share_change
            * float(
                spots[
                    day
                ]
            )
            + tc
        )

        total_tc += (
            tc
        )

        shares = float(
            target_delta
        )

        n_rebalances += 1

    terminal_day = int(
        trading_days
    )

    cash *= growth

    if terminal_day in dividend_map:
        received = (
            shares
            * dividend_map[
                terminal_day
            ]
        )

        cash += received

        total_dividends += (
            received
        )

    final_stock = float(
        spots[
            terminal_day
        ]
    )

    liquidation_change = (
        -shares
    )

    terminal_tc = (
        _transaction_cost(
            share_change=(
                liquidation_change
            ),
            spot=final_stock,
            fee_rates=(
                fee_rates
            ),
        )
    )

    cash += (
        shares
        * final_stock
        - terminal_tc
    )

    total_tc += (
        terminal_tc
    )

    payoff = max(
        final_stock
        - float(
            strike
        ),
        0.0,
    )

    cash -= (
        payoff
    )

    return {
        "freq_days": int(
            frequency
        ),
        "n_rebalances": int(
            n_rebalances
        ),
        "total_transaction_cost": float(
            total_tc
        ),
        "total_dividends_received": float(
            total_dividends
        ),
        "hedging_pnl": float(
            cash
        ),
    }


@dataclass(frozen=True)
class DeltaHedgingPnlSkill:
    """Discrete option delta hedging with dividends, IV paths and costs."""

    name: str = "delta-hedging-pnl-domain"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        del task_dir

        normalized = (
            instruction.lower()
            .replace(
                "-",
                " ",
            )
            .replace(
                "_",
                " ",
            )
        )

        return (
            "delta hedg" in normalized
            and (
                "transaction cost"
                in normalized
            )
            and (
                "dividend"
                in normalized
            )
            and (
                "implied vol"
                in normalized
                or "iv" in normalized
            )
            and (
                "hedging pnl"
                in normalized
                or "p&l"
                in instruction.lower()
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
        del seed

        params = (
            _parse_contract_parameters(
                instruction
            )
        )

        data_dir = (
            _find_delta_hedge_data_dir(
                task_dir
            )
        )

        (
            stock,
            iv,
            dividends,
            fee_rates,
        ) = _load_paths(
            data_dir,
            int(
                params[
                    "trading_days"
                ]
            ),
        )

        strike = float(
            params[
                "strike"
            ]
        )

        risk_free_rate = float(
            params[
                "risk_free_rate"
            ]
        )

        trading_days = int(
            params[
                "trading_days"
            ]
        )

        frequencies = [
            int(
                value
            )
            for value
            in params[
                "frequencies"
            ]
        ]

        spot0 = float(
            stock.iloc[
                0
            ][
                "close"
            ]
        )

        iv0 = float(
            iv.iloc[
                0
            ][
                "implied_vol"
            ]
        )

        pv_divs0 = (
            _pv_remaining_dividends(
                day=0,
                dividends=(
                    dividends
                ),
                risk_free_rate=(
                    risk_free_rate
                ),
                trading_days=(
                    trading_days
                ),
            )
        )

        (
            t0_price,
            t0_delta,
            t0_effective_spot,
        ) = _escrowed_call_price_delta(
            spot=spot0,
            strike=strike,
            tau=float(
                params[
                    "maturity_years"
                ]
            ),
            risk_free_rate=(
                risk_free_rate
            ),
            volatility=iv0,
            pv_dividends=(
                pv_divs0
            ),
        )

        simulations = [
            _simulate_frequency(
                stock=stock,
                iv=iv,
                dividends=(
                    dividends
                ),
                fee_rates=(
                    fee_rates
                ),
                strike=strike,
                risk_free_rate=(
                    risk_free_rate
                ),
                trading_days=(
                    trading_days
                ),
                frequency=frequency,
                t0_price=t0_price,
                t0_delta=t0_delta,
            )
            for frequency
            in frequencies
        ]

        final_stock = float(
            stock.iloc[
                -1
            ][
                "close"
            ]
        )

        payoff = max(
            final_stock
            - strike,
            0.0,
        )

        results = {
            "t0": {
                "price": float(
                    t0_price
                ),
                "delta": float(
                    t0_delta
                ),
                "S_eff": float(
                    t0_effective_spot
                ),
                "pv_divs": float(
                    pv_divs0
                ),
            },
            "terminal": {
                "final_stock": float(
                    final_stock
                ),
                "call_payoff": float(
                    payoff
                ),
            },
            "simulations": (
                simulations
            ),
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
