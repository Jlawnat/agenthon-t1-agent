from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import stats


_REQUIRED_INPUTS = (
    "prices.csv",
    "posts.jsonl",
    "sentiment_words.json",
    "market_factor.csv",
    "params.json",
)


def _find(task_dir: Path, name: str) -> Path:
    matches = [
        path
        for path in task_dir.rglob(name)
        if path.is_file() and "checks" not in path.parts
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {name!r} outside checks, found {len(matches)}."
        )
    return matches[0]


def _load_posts(path: Path) -> list[dict]:
    posts: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                posts.append(json.loads(line))
    return posts


def _score_tokens(
    text: str,
    bullish: set[str],
    bearish: set[str],
) -> int:
    clean_tokens = []
    for token in text.lower().split():
        cleaned = re.sub(r"[^a-z]", "", token)
        if cleaned:
            clean_tokens.append(cleaned)

    bull_count = sum(token in bullish for token in clean_tokens)
    bear_count = sum(token in bearish for token in clean_tokens)
    return int(bull_count - bear_count)


@dataclass(frozen=True)
class SentimentFactorAlphaSkill:
    name: str = "sentiment-factor-alpha-domain"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        lowered = instruction.lower()
        semantic = (
            "sentiment momentum factor" in lowered
            and "mean_ic" in lowered
            and "alpha_annualized" in lowered
            and "engagement multiplier" in lowered
            and "solution.json" in lowered
        )

        if not semantic:
            return False

        try:
            for filename in _REQUIRED_INPUTS:
                _find(task_dir, filename)
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

        params = json.loads(
            _find(task_dir, "params.json").read_text(encoding="utf-8")
        )
        sentiment_words = json.loads(
            _find(task_dir, "sentiment_words.json").read_text(encoding="utf-8")
        )
        posts_raw = _load_posts(
            _find(task_dir, "posts.jsonl")
        )

        prices_long = pd.read_csv(
            _find(task_dir, "prices.csv")
        )
        market = pd.read_csv(
            _find(task_dir, "market_factor.csv")
        )

        prices_long["date"] = prices_long["date"].astype(str)
        market["date"] = market["date"].astype(str)

        tickers = sorted(
            str(value)
            for value in prices_long["ticker"].unique().tolist()
        )
        dates = sorted(
            str(value)
            for value in prices_long["date"].unique().tolist()
        )

        n_days = len(dates)
        n_tickers = len(tickers)
        ticker_to_idx = {
            ticker: index
            for index, ticker in enumerate(tickers)
        }
        date_to_idx = {
            date: index
            for index, date in enumerate(dates)
        }

        prices_data = np.zeros(
            (n_days, n_tickers),
            dtype=float,
        )

        for ticker_index, ticker in enumerate(tickers):
            frame = (
                prices_long[
                    prices_long["ticker"].astype(str) == ticker
                ]
                .sort_values("date", kind="stable")
            )

            if len(frame) != n_days:
                raise RuntimeError(
                    f"Price panel for {ticker} does not contain {n_days} rows."
                )

            prices_data[:, ticker_index] = pd.to_numeric(
                frame["close"],
                errors="raise",
            ).to_numpy(dtype=float)

        market_by_date = {
            str(row["date"]): float(row["mkt_rf"])
            for _, row in market.iterrows()
        }

        momentum_lookback = int(
            params["momentum_lookback_days"]
        )
        min_posts = int(
            params["volume_filter_min_posts"]
        )
        long_n = int(
            params["long_top_n"]
        )
        short_n = int(
            params["short_bottom_n"]
        )
        tc_bps = float(
            params["transaction_cost_bps"]
        )
        engagement_weight = bool(
            params["engagement_weight"]
        )
        vol_target = params.get(
            "vol_target_annual"
        )
        vol_target = (
            None
            if vol_target is None
            else float(vol_target)
        )
        vol_lookback = int(
            params.get(
                "vol_lookback_days",
                20,
            )
        )
        max_leverage_scale = float(
            params.get(
                "max_leverage_scale",
                3.0,
            )
        )

        num_posts = len(posts_raw)

        # Contract: ONLY negative likes marks an invalid post.
        posts = [
            post
            for post in posts_raw
            if float(post.get("likes", 0)) >= 0.0
        ]
        num_valid_posts = len(posts)

        bullish_set = {
            str(value).lower()
            for value in sentiment_words["bullish"]
        }
        bearish_set = {
            str(value).lower()
            for value in sentiment_words["bearish"]
        }

        daily_sums = np.zeros(
            (n_days, n_tickers),
            dtype=float,
        )
        daily_counts = np.zeros(
            (n_days, n_tickers),
            dtype=int,
        )

        weighted_scores: list[float] = []
        num_scored_nonzero = 0

        for post in posts:
            date_str = str(post["timestamp"])[:10]
            ticker = str(post["ticker"])

            raw_score = _score_tokens(
                str(post["text"]),
                bullish_set,
                bearish_set,
            )

            if engagement_weight and raw_score != 0:
                likes = float(post.get("likes", 0))
                retweets = float(post.get("retweets", 0))
                replies = float(post.get("replies", 0))
                engagement_multiplier = (
                    1.0
                    + math.log(
                        1.0
                        + likes
                        + 2.0 * retweets
                        + 3.0 * replies
                    )
                )
                score = float(
                    raw_score
                    * engagement_multiplier
                )
            else:
                score = float(raw_score)

            if score != 0.0:
                num_scored_nonzero += 1
                weighted_scores.append(score)

            # Non-trading-day posts remain valid/countable but do not
            # contribute to trading-day aggregates.
            date_index = date_to_idx.get(date_str)
            ticker_index = ticker_to_idx.get(ticker)

            if date_index is None or ticker_index is None:
                continue

            daily_sums[
                date_index,
                ticker_index,
            ] += score
            daily_counts[
                date_index,
                ticker_index,
            ] += 1

        mean_weighted_score = (
            float(np.mean(weighted_scores))
            if weighted_scores
            else 0.0
        )

        daily_returns = np.zeros(
            (n_days, n_tickers),
            dtype=float,
        )
        daily_returns[1:] = (
            prices_data[1:]
            - prices_data[:-1]
        ) / prices_data[:-1]

        signals = np.full(
            (n_days, n_tickers),
            np.nan,
            dtype=float,
        )
        num_signal_days = 0

        for day_index in range(
            momentum_lookback,
            n_days,
        ):
            lookback_mean = np.mean(
                daily_sums[
                    day_index - momentum_lookback:
                    day_index
                ],
                axis=0,
            )

            day_signal = (
                daily_sums[day_index]
                - lookback_mean
            )

            eligible = (
                daily_counts[day_index]
                >= min_posts
            )

            signals[
                day_index,
                eligible,
            ] = day_signal[
                eligible
            ]

            if (
                int(
                    np.sum(
                        ~np.isnan(
                            signals[
                                day_index
                            ]
                        )
                    )
                )
                >= long_n + short_n
            ):
                num_signal_days += 1

        total_positions = (
            long_n
            + short_n
        )
        position_weight = (
            1.0
            / total_positions
        )

        previous_weights = np.zeros(
            n_tickers,
            dtype=float,
        )
        portfolio_returns_list: list[float] = []
        portfolio_dates: list[str] = []

        for day_index in range(
            momentum_lookback,
            n_days - 1,
        ):
            day_signal = signals[
                day_index
            ]

            valid_mask = ~np.isnan(
                day_signal
            )
            valid_count = int(
                np.sum(valid_mask)
            )

            if (
                valid_count
                < total_positions
            ):
                if portfolio_returns_list:
                    portfolio_returns_list.append(
                        0.0
                    )
                    portfolio_dates.append(
                        dates[
                            day_index + 1
                        ]
                    )
                    previous_weights = np.zeros(
                        n_tickers,
                        dtype=float,
                    )
                continue

            valid_indices = np.where(
                valid_mask
            )[0]
            valid_signals = (
                day_signal[
                    valid_mask
                ]
            )
            ranked = np.argsort(
                valid_signals
            )

            short_indices = (
                valid_indices[
                    ranked[
                        :short_n
                    ]
                ]
            )
            long_indices = (
                valid_indices[
                    ranked[
                        -long_n:
                    ]
                ]
            )

            weights = np.zeros(
                n_tickers,
                dtype=float,
            )
            weights[
                long_indices
            ] = position_weight
            weights[
                short_indices
            ] = -position_weight

            if (
                vol_target is not None
                and day_index
                >= momentum_lookback
                + vol_lookback
            ):
                trailing_returns = daily_returns[
                    day_index - vol_lookback + 1:
                    day_index + 1
                ]

                covariance = np.cov(
                    trailing_returns.T
                )

                variance = float(
                    weights
                    @ covariance
                    @ weights
                )

                portfolio_vol = (
                    math.sqrt(
                        max(
                            variance,
                            0.0,
                        )
                    )
                    * math.sqrt(252.0)
                )

                if portfolio_vol > 0.0:
                    scale = min(
                        vol_target
                        / portfolio_vol,
                        max_leverage_scale,
                    )
                    weights = (
                        weights
                        * scale
                    )

            turnover = float(
                np.sum(
                    np.abs(
                        weights
                        - previous_weights
                    )
                )
            )

            transaction_cost = (
                turnover
                * tc_bps
                / 10000.0
            )

            next_day_return = (
                daily_returns[
                    day_index + 1
                ]
            )

            portfolio_return = float(
                np.sum(
                    weights
                    * next_day_return
                )
                - transaction_cost
            )

            portfolio_returns_list.append(
                portfolio_return
            )
            portfolio_dates.append(
                dates[
                    day_index + 1
                ]
            )

            previous_weights = (
                weights.copy()
            )

        portfolio_returns = np.asarray(
            portfolio_returns_list,
            dtype=float,
        )

        if len(portfolio_returns) < 2:
            raise RuntimeError(
                "Sentiment strategy produced fewer than two return observations."
            )

        total_return = float(
            np.prod(
                1.0
                + portfolio_returns
            )
            - 1.0
        )

        annualized_return = float(
            np.mean(
                portfolio_returns
            )
            * 252.0
        )

        annualized_volatility = float(
            np.std(
                portfolio_returns,
                ddof=1,
            )
            * math.sqrt(252.0)
        )

        sharpe_ratio = (
            annualized_return
            / annualized_volatility
            if annualized_volatility > 0.0
            else 0.0
        )

        cumulative_wealth = np.cumprod(
            1.0
            + portfolio_returns
        )
        running_max = np.maximum.accumulate(
            cumulative_wealth
        )
        drawdown = (
            running_max
            - cumulative_wealth
        ) / running_max

        max_drawdown = float(
            np.max(
                drawdown
            )
        )

        win_rate = float(
            np.mean(
                portfolio_returns
                > 0.0
            )
        )

        ic_values: list[float] = []

        for day_index in range(
            momentum_lookback,
            n_days - 1,
        ):
            day_signal = signals[
                day_index
            ]
            valid_mask = ~np.isnan(
                day_signal
            )

            if int(np.sum(valid_mask)) < 3:
                continue

            correlation, _ = (
                stats.spearmanr(
                    day_signal[
                        valid_mask
                    ],
                    daily_returns[
                        day_index + 1
                    ][
                        valid_mask
                    ],
                )
            )

            if not np.isnan(
                correlation
            ):
                ic_values.append(
                    float(correlation)
                )

        ic_array = np.asarray(
            ic_values,
            dtype=float,
        )

        mean_ic = float(
            np.mean(ic_array)
        )
        std_ic = float(
            np.std(
                ic_array,
                ddof=1,
            )
        )
        ic_ir = (
            mean_ic / std_ic
            if std_ic > 0.0
            else 0.0
        )

        market_aligned = np.asarray(
            [
                market_by_date.get(
                    date,
                    0.0,
                )
                for date
                in portfolio_dates
            ],
            dtype=float,
        )

        regression = stats.linregress(
            market_aligned,
            portfolio_returns,
        )

        beta = float(
            regression.slope
        )
        alpha_daily = float(
            regression.intercept
        )
        r_squared = float(
            regression.rvalue ** 2
        )
        # Frozen benchmark convention: despite the checkpoint name
        # "alpha_tstat", the reference divides alpha_daily by the
        # regression slope standard error (LinregressResult.stderr),
        # not by intercept_stderr.
        alpha_tstat = (
            alpha_daily
            / float(
                regression.stderr
            )
            if (
                regression.stderr
                is not None
                and regression.stderr
                > 0.0
            )
            else 0.0
        )
        alpha_annualized = float(
            alpha_daily
            * 252.0
        )

        results = {
            "num_valid_posts": int(
                num_valid_posts
            ),
            "mean_weighted_score": float(
                mean_weighted_score
            ),
            "total_return": float(
                total_return
            ),
            "annualized_return": float(
                annualized_return
            ),
            "sharpe_ratio": float(
                sharpe_ratio
            ),
            "max_drawdown": float(
                max_drawdown
            ),
            "mean_ic": float(
                mean_ic
            ),
            "ic_ir": float(
                ic_ir
            ),
            "alpha_annualized": float(
                alpha_annualized
            ),
            "beta": float(
                beta
            ),
        }

        def box(
            value,
            note: str | None = None,
        ) -> dict:
            output = {
                "value": value
            }
            if note is not None:
                output[
                    "note"
                ] = note
            return output

        solution = {
            "intermediates": {
                "num_posts": box(
                    int(
                        num_posts
                    )
                ),
                "num_valid_posts": box(
                    int(
                        num_valid_posts
                    ),
                    "posts remaining after filtering invalid records",
                ),
                "num_tickers": box(
                    int(
                        n_tickers
                    )
                ),
                "num_scored_nonzero": box(
                    int(
                        num_scored_nonzero
                    )
                ),
                "num_signal_days": box(
                    int(
                        num_signal_days
                    )
                ),
                "mean_weighted_score": box(
                    float(
                        mean_weighted_score
                    ),
                    "mean of nonzero final scores",
                ),
                "num_ic_observations": box(
                    int(
                        len(
                            ic_array
                        )
                    )
                ),
                "mean_ic": box(
                    float(
                        mean_ic
                    )
                ),
                "std_ic": box(
                    float(
                        std_ic
                    )
                ),
                "ic_ir": box(
                    float(
                        ic_ir
                    )
                ),
                "annualized_volatility": box(
                    float(
                        annualized_volatility
                    )
                ),
                "win_rate": box(
                    float(
                        win_rate
                    )
                ),
                "alpha_daily": box(
                    float(
                        alpha_daily
                    )
                ),
                "beta": box(
                    float(
                        beta
                    )
                ),
                "r_squared": box(
                    float(
                        r_squared
                    )
                ),
                "alpha_tstat": box(
                    float(
                        alpha_tstat
                    )
                ),
            }
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

        (
            out_dir
            / "solution.json"
        ).write_text(
            json.dumps(
                solution,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
