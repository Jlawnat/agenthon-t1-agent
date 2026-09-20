from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


def _find_variance_swap_data_dir(
    task_dir: Path,
) -> Path:
    required = {
        "params.json",
        "option_chain.csv",
    }

    directories = sorted(
        {
            path.parent
            for path
            in task_dir.rglob("*")
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
        "Could not discover variance-swap input files."
    )


def black_scholes_price(
    *,
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    option_type: str,
) -> float:
    if maturity <= 0.0:
        intrinsic = (
            max(
                spot
                - strike,
                0.0,
            )
            if option_type == "call"
            else max(
                strike
                - spot,
                0.0,
            )
        )

        return float(
            intrinsic
        )

    if volatility <= 0.0:
        forward_spot = (
            spot
            * math.exp(
                -dividend_yield
                * maturity
            )
        )

        discounted_strike = (
            strike
            * math.exp(
                -risk_free_rate
                * maturity
            )
        )

        intrinsic = (
            max(
                forward_spot
                - discounted_strike,
                0.0,
            )
            if option_type == "call"
            else max(
                discounted_strike
                - forward_spot,
                0.0,
            )
        )

        return float(
            intrinsic
        )

    root_t = math.sqrt(
        maturity
    )

    d1 = (
        math.log(
            spot
            / strike
        )
        + (
            risk_free_rate
            - dividend_yield
            + 0.5
            * volatility
            * volatility
        )
        * maturity
    ) / (
        volatility
        * root_t
    )

    d2 = (
        d1
        - volatility
        * root_t
    )

    discounted_spot = (
        spot
        * math.exp(
            -dividend_yield
            * maturity
        )
    )

    discounted_strike = (
        strike
        * math.exp(
            -risk_free_rate
            * maturity
        )
    )

    if option_type == "call":
        return float(
            discounted_spot
            * norm.cdf(
                d1
            )
            - discounted_strike
            * norm.cdf(
                d2
            )
        )

    if option_type == "put":
        return float(
            discounted_strike
            * norm.cdf(
                -d2
            )
            - discounted_spot
            * norm.cdf(
                -d1
            )
        )

    raise RuntimeError(
        f"Unsupported option type: {option_type}"
    )


def implied_volatility(
    *,
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    dividend_yield: float,
    option_type: str,
) -> float:
    def residual(
        volatility: float,
    ) -> float:
        return (
            black_scholes_price(
                spot=spot,
                strike=strike,
                maturity=maturity,
                risk_free_rate=(
                    risk_free_rate
                ),
                dividend_yield=(
                    dividend_yield
                ),
                volatility=(
                    volatility
                ),
                option_type=(
                    option_type
                ),
            )
            - market_price
        )

    lower = 1e-8
    upper = 5.0

    low_value = residual(
        lower
    )

    high_value = residual(
        upper
    )

    if abs(
        low_value
    ) <= 1e-8:
        return float(
            lower
        )

    if (
        low_value
        * high_value
        > 0.0
    ):
        raise RuntimeError(
            "Could not bracket option implied volatility."
        )

    return float(
        brentq(
            residual,
            lower,
            upper,
            xtol=1e-12,
            rtol=1e-12,
            maxiter=300,
        )
    )


def clean_option_chain(
    chain: pd.DataFrame,
    *,
    target_expiry: str,
    forward_price: float,
    min_volume: int,
    max_relative_spread: float,
) -> tuple[
    pd.DataFrame,
    int,
]:
    expiry = (
        chain.loc[
            chain[
                "expiry"
            ].astype(
                str
            )
            == str(
                target_expiry
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    n_input = len(
        expiry
    )

    expiry[
        "call_mid"
    ] = (
        expiry[
            "call_bid"
        ]
        + expiry[
            "call_ask"
        ]
    ) / 2.0

    expiry[
        "put_mid"
    ] = (
        expiry[
            "put_bid"
        ]
        + expiry[
            "put_ask"
        ]
    ) / 2.0

    expiry[
        "otm_is_put"
    ] = (
        expiry[
            "strike"
        ]
        < float(
            forward_price
        )
    )

    expiry[
        "otm_price"
    ] = np.where(
        expiry[
            "otm_is_put"
        ],
        expiry[
            "put_mid"
        ],
        expiry[
            "call_mid"
        ],
    )

    expiry[
        "otm_bid"
    ] = np.where(
        expiry[
            "otm_is_put"
        ],
        expiry[
            "put_bid"
        ],
        expiry[
            "call_bid"
        ],
    )

    expiry[
        "otm_ask"
    ] = np.where(
        expiry[
            "otm_is_put"
        ],
        expiry[
            "put_ask"
        ],
        expiry[
            "call_ask"
        ],
    )

    expiry[
        "relative_spread"
    ] = (
        expiry[
            "otm_ask"
        ]
        - expiry[
            "otm_bid"
        ]
    ) / expiry[
        "otm_price"
    ]

    keep = (
        (
            expiry[
                "volume"
            ]
            >= int(
                min_volume
            )
        )
        & (
            expiry[
                "otm_bid"
            ]
            > 0.0
        )
        & (
            expiry[
                "relative_spread"
            ]
            <= float(
                max_relative_spread
            )
        )
    )

    cleaned = (
        expiry.loc[
            keep
        ]
        .sort_values(
            "strike",
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    return (
        cleaned,
        n_input,
    )


def trapezoidal_strike_widths(
    strikes: np.ndarray,
) -> np.ndarray:
    strikes = np.asarray(
        strikes,
        dtype=float,
    )

    if len(
        strikes
    ) < 2:
        raise RuntimeError(
            "Variance replication requires at least two strikes."
        )

    widths = np.empty_like(
        strikes
    )

    widths[0] = (
        strikes[
            1
        ]
        - strikes[
            0
        ]
    )

    widths[
        -1
    ] = (
        strikes[
            -1
        ]
        - strikes[
            -2
        ]
    )

    if len(
        strikes
    ) > 2:
        widths[
            1:-1
        ] = (
            strikes[
                2:
            ]
            - strikes[
                :-2
            ]
        ) / 2.0

    return widths


@dataclass(frozen=True)
class VarianceSwapReplicationSkill:
    name: str = "variance-swap-replication-domain"

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

        normalized = (
            lowered
            .replace("-", " ")
            .replace("_", " ")
        )

        return (
            "variance swap" in normalized
            and (
                "replication" in normalized
                or "log contract" in normalized
            )
            and (
                "option chain" in normalized
                or "otm option" in normalized
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

        data_dir = (
            _find_variance_swap_data_dir(
                task_dir
            )
        )

        params = json.loads(
            (
                data_dir
                / "params.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        chain = pd.read_csv(
            data_dir
            / "option_chain.csv"
        )

        target_expiry = str(
            params[
                "target_expiry"
            ]
        )

        spot = float(
            params[
                "spot"
            ]
        )

        risk_free_rate = float(
            params[
                "risk_free_rate"
            ]
        )

        dividend_yield = float(
            params[
                "dividend_yield"
            ]
        )

        maturity = float(
            params[
                "expiries"
            ][
                target_expiry
            ]
        )

        forward_price = float(
            params[
                "forward_prices"
            ][
                target_expiry
            ]
        )

        filters = params[
            "filters"
        ]

        cleaned, n_input = (
            clean_option_chain(
                chain,
                target_expiry=(
                    target_expiry
                ),
                forward_price=(
                    forward_price
                ),
                min_volume=int(
                    filters[
                        "min_volume"
                    ]
                ),
                max_relative_spread=float(
                    filters[
                        "max_relative_spread"
                    ]
                ),
            )
        )

        if len(
            cleaned
        ) < 2:
            raise RuntimeError(
                "Too few option strikes survived cleaning."
            )

        iv_values = []

        for row in cleaned.itertuples(
            index=False
        ):
            option_type = (
                "put"
                if bool(
                    row.otm_is_put
                )
                else "call"
            )

            iv_values.append(
                implied_volatility(
                    market_price=float(
                        row.otm_price
                    ),
                    spot=spot,
                    strike=float(
                        row.strike
                    ),
                    maturity=maturity,
                    risk_free_rate=(
                        risk_free_rate
                    ),
                    dividend_yield=(
                        dividend_yield
                    ),
                    option_type=(
                        option_type
                    ),
                )
            )

        cleaned[
            "iv_strike"
        ] = np.asarray(
            iv_values,
            dtype=float,
        )

        lower_candidates = (
            cleaned.loc[
                cleaned[
                    "strike"
                ]
                < forward_price
            ]
        )

        upper_candidates = (
            cleaned.loc[
                cleaned[
                    "strike"
                ]
                >= forward_price
            ]
        )

        if (
            lower_candidates.empty
            or upper_candidates.empty
        ):
            raise RuntimeError(
                "Kept strikes do not straddle the supplied forward."
            )

        lower = lower_candidates.iloc[
            -1
        ]

        upper = upper_candidates.iloc[
            0
        ]

        strike_low = float(
            lower[
                "strike"
            ]
        )

        strike_high = float(
            upper[
                "strike"
            ]
        )

        if not (
            strike_low
            < forward_price
            <= strike_high
        ):
            raise RuntimeError(
                "Forward interpolation strikes are invalid."
            )

        interpolation_weight = (
            (
                forward_price
                - strike_low
            )
            / (
                strike_high
                - strike_low
            )
        )

        iv_atm = float(
            (
                1.0
                - interpolation_weight
            )
            * float(
                lower[
                    "iv_strike"
                ]
            )
            + interpolation_weight
            * float(
                upper[
                    "iv_strike"
                ]
            )
        )

        atm_var = float(
            iv_atm
            * iv_atm
        )

        strikes = cleaned[
            "strike"
        ].to_numpy(
            dtype=float
        )

        option_prices = cleaned[
            "otm_price"
        ].to_numpy(
            dtype=float
        )

        d_k = (
            trapezoidal_strike_widths(
                strikes
            )
        )

        weights = (
            d_k
            / (
                strikes
                * strikes
            )
        )

        fair_variance = float(
            (
                2.0
                * math.exp(
                    risk_free_rate
                    * maturity
                )
                / maturity
            )
            * np.sum(
                weights
                * option_prices
            )
        )

        if fair_variance < 0.0:
            raise RuntimeError(
                "Replicated variance strike is negative."
            )

        fair_vol = float(
            math.sqrt(
                fair_variance
            )
        )

        variance_risk_premium = float(
            fair_variance
            - atm_var
        )

        fair_vol_minus_atm_iv = float(
            fair_vol
            - iv_atm
        )

        variance_notional = 10000.0

        scenario_specs = (
            (
                "low",
                0.18,
            ),
            (
                "mid",
                0.25,
            ),
            (
                "high",
                0.35,
            ),
        )

        scenarios = []

        for name, sigma_rv in scenario_specs:
            realized_variance = float(
                sigma_rv
                * sigma_rv
            )

            pnl = float(
                variance_notional
                * (
                    realized_variance
                    - fair_variance
                )
            )

            scenarios.append(
                {
                    "name": name,
                    "sigma_RV": float(
                        sigma_rv
                    ),
                    "realized_variance": (
                        realized_variance
                    ),
                    "pnl": pnl,
                }
            )

        result = {
            "target_expiry": (
                target_expiry
            ),
            "forward_price": (
                forward_price
            ),
            "n_strikes_input": int(
                n_input
            ),
            "n_strikes_kept": int(
                len(
                    cleaned
                )
            ),
            "n_otm_puts_kept": int(
                cleaned[
                    "otm_is_put"
                ].sum()
            ),
            "n_otm_calls_kept": int(
                (
                    ~cleaned[
                        "otm_is_put"
                    ]
                ).sum()
            ),
            "fair_variance_strike": (
                fair_variance
            ),
            "fair_vol": (
                fair_vol
            ),
            "iv_atm_forward": (
                iv_atm
            ),
            "atm_var": (
                atm_var
            ),
            "variance_risk_premium": (
                variance_risk_premium
            ),
            "fair_vol_minus_atm_iv": (
                fair_vol_minus_atm_iv
            ),
            "scenarios": scenarios,
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
                result,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
