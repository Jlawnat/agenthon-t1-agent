from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _find_price_csv(task_dir: Path) -> Path:
    required = {
        "date",
        "adj_close",
        "adj_open",
    }

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

        columns = {
            str(column).lower()
            for column in frame.columns
        }

        if required.issubset(columns):
            return path

    raise RuntimeError(
        "Offline Bollinger skill could not find a CSV "
        "containing date, adj_close, and adj_open."
    )


def _round_money(value: float) -> float:
    return round(
        float(value),
        2,
    )


def _write_plotly_html(
    *,
    out_dir: Path,
    dates: list[str],
    cumulative_returns: list[float],
) -> None:
    dates_json = json.dumps(
        dates,
        ensure_ascii=False,
    )
    returns_json = json.dumps(
        [
            float(value)
            for value in cumulative_returns
        ]
    )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Bollinger Band Mean Reversion: AAPL Cumulative Returns</title>
  <script src="https://cdn.plot.ly/plotly-3.1.0.min.js"></script>
</head>
<body>
  <div id="plotly-aapl-chart" style="width:100%;height:640px;"></div>
  <script>
    const dates = {dates_json};
    const cumulativeReturn = {returns_json};

    Plotly.newPlot(
      "plotly-aapl-chart",
      [
        {{
          x: dates,
          y: cumulativeReturn,
          type: "scatter",
          mode: "lines",
          name: "AAPL"
        }}
      ],
      {{
        title: "Bollinger Band Mean Reversion: AAPL Cumulative Returns",
        xaxis: {{title: "Date"}},
        yaxis: {{title: "Cumulative Return"}},
        shapes: [
          {{
            type: "line",
            xref: "paper",
            x0: 0,
            x1: 1,
            y0: 1.0,
            y1: 1.0,
            line: {{dash: "dash"}}
          }}
        ]
      }},
      {{responsive: true}}
    );
  </script>
</body>
</html>
"""

    (
        out_dir
        / "cumulative_returns.html"
    ).write_text(
        html,
        encoding="utf-8",
    )


@dataclass(frozen=True)
class BollingerBacktestSkill:
    name: str = "bollinger-backtest-aapl"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        del task_dir

        lowered = instruction.lower()

        required = (
            "bollinger",
            "mean reversion",
            "adj_close",
            "adj_open",
            "risk-based position sizing",
            "trailing stop",
            "daily_portfolio.csv",
            "trade_analysis.json",
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

        price_path = _find_price_csv(
            task_dir
        )

        frame = pd.read_csv(
            price_path
        )

        required_columns = (
            "date",
            "adj_close",
            "adj_open",
        )

        by_lower = {
            str(column).lower(): str(column)
            for column in frame.columns
        }

        missing = [
            name
            for name in required_columns
            if name not in by_lower
        ]

        if missing:
            raise RuntimeError(
                "Bollinger input is missing required columns: "
                + ", ".join(missing)
            )

        data = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    frame[
                        by_lower["date"]
                    ],
                    errors="raise",
                ),
                "adj_close": pd.to_numeric(
                    frame[
                        by_lower["adj_close"]
                    ],
                    errors="raise",
                ).astype(float),
                "adj_open": pd.to_numeric(
                    frame[
                        by_lower["adj_open"]
                    ],
                    errors="raise",
                ).astype(float),
            }
        ).sort_values(
            "date",
            kind="stable",
        ).reset_index(
            drop=True
        )

        if data.empty:
            raise RuntimeError(
                "Bollinger input contains no rows."
            )

        data["sma"] = (
            data["adj_close"]
            .rolling(
                window=20,
                min_periods=20,
            )
            .mean()
        )

        rolling_std = (
            data["adj_close"]
            .rolling(
                window=20,
                min_periods=20,
            )
            .std(
                ddof=1
            )
        )

        data["upper_band"] = (
            data["sma"]
            + 2.0 * rolling_std
        )
        data["lower_band"] = (
            data["sma"]
            - 2.0 * rolling_std
        )

        initial_capital = 100000.0
        flat_fee = 9.99
        commission = 0.001
        risk_fraction = 0.02
        trailing_stop = 0.05

        cash = initial_capital
        shares = 0
        peak_price: float | None = None
        entry: dict[str, object] | None = None
        pending: dict[str, object] | None = None

        trade_rows: list[
            dict[str, object]
        ] = []
        analysis_rows: list[
            dict[str, object]
        ] = []
        portfolio_values: list[float] = []

        def buy_total(
            quantity: int,
            price: float,
        ) -> float:
            gross = (
                quantity
                * price
            )
            return (
                gross
                * (1.0 + commission)
                + flat_fee
            )

        def sell_net(
            quantity: int,
            price: float,
        ) -> float:
            gross = (
                quantity
                * price
            )
            return (
                gross
                * (1.0 - commission)
                - flat_fee
            )

        def holding_drawdown(
            start_index: int,
            end_index: int,
        ) -> float:
            prices = (
                data.loc[
                    start_index:end_index,
                    "adj_close",
                ]
                .astype(float)
                .to_numpy()
            )

            running_max = (
                np.maximum.accumulate(
                    prices
                )
            )

            drawdowns = (
                running_max
                - prices
            ) / running_max

            return float(
                np.max(
                    drawdowns
                )
            )

        row_count = len(
            data
        )

        for index, row in data.iterrows():
            date_text = (
                row["date"]
                .strftime(
                    "%Y-%m-%d"
                )
            )

            # Execute yesterday's signal at today's adjusted open.
            if (
                pending is not None
                and pending[
                    "exec_index"
                ]
                == index
            ):
                pending_type = str(
                    pending[
                        "type"
                    ]
                )

                if (
                    pending_type == "BUY"
                    and shares == 0
                ):
                    execution_price = float(
                        row[
                            "adj_open"
                        ]
                    )
                    lower_band = float(
                        pending[
                            "lower_band"
                        ]
                    )
                    stop_distance = (
                        execution_price
                        - lower_band
                    )

                    quantity = 0

                    if stop_distance > 0.0:
                        max_risk_amount = (
                            risk_fraction
                            * cash
                        )

                        max_shares_by_risk = (
                            math.floor(
                                max_risk_amount
                                / stop_distance
                            )
                        )

                        max_shares_by_cash = max(
                            0,
                            math.floor(
                                (
                                    cash
                                    - flat_fee
                                )
                                / (
                                    execution_price
                                    * (
                                        1.0
                                        + commission
                                    )
                                )
                            ),
                        )

                        quantity = min(
                            max_shares_by_risk,
                            max_shares_by_cash,
                        )

                    if quantity > 0:
                        cost = buy_total(
                            quantity,
                            execution_price,
                        )

                        cash -= cost
                        shares = quantity
                        peak_price = float(
                            pending[
                                "signal_close"
                            ]
                        )

                        entry = {
                            "signal_date": (
                                pending[
                                    "signal_date"
                                ]
                            ),
                            "exec_date": date_text,
                            "exec_index": index,
                            "price": execution_price,
                            "shares": quantity,
                            "buy_total": cost,
                        }

                        trade_rows.append(
                            {
                                "ticker": "AAPL",
                                "type": "BUY",
                                "signal_date": (
                                    pending[
                                        "signal_date"
                                    ]
                                ),
                                "exec_date": date_text,
                                "price": (
                                    _round_money(
                                        execution_price
                                    )
                                ),
                                "shares": quantity,
                                "pnl": None,
                            }
                        )

                elif (
                    pending_type == "SELL"
                    and shares > 0
                ):
                    if entry is None:
                        raise RuntimeError(
                            "Sell execution has no active entry."
                        )

                    execution_price = float(
                        row[
                            "adj_open"
                        ]
                    )

                    quantity = shares

                    revenue = sell_net(
                        quantity,
                        execution_price,
                    )

                    cash += revenue

                    pnl = (
                        revenue
                        - float(
                            entry[
                                "buy_total"
                            ]
                        )
                    )

                    trade_rows.append(
                        {
                            "ticker": "AAPL",
                            "type": "SELL",
                            "signal_date": (
                                pending[
                                    "signal_date"
                                ]
                            ),
                            "exec_date": date_text,
                            "price": (
                                _round_money(
                                    execution_price
                                )
                            ),
                            "shares": quantity,
                            "pnl": (
                                _round_money(
                                    pnl
                                )
                            ),
                        }
                    )

                    analysis_rows.append(
                        {
                            "trade_num": (
                                len(
                                    analysis_rows
                                )
                                + 1
                            ),
                            "entry_date": str(
                                entry[
                                    "exec_date"
                                ]
                            ),
                            "exit_date": date_text,
                            "entry_price": (
                                _round_money(
                                    float(
                                        entry[
                                            "price"
                                        ]
                                    )
                                )
                            ),
                            "exit_price": (
                                _round_money(
                                    execution_price
                                )
                            ),
                            "pnl": (
                                _round_money(
                                    pnl
                                )
                            ),
                            "exit_reason": str(
                                pending[
                                    "reason"
                                ]
                            ),
                            "intra_trade_max_drawdown": round(
                                holding_drawdown(
                                    int(
                                        entry[
                                            "exec_index"
                                        ]
                                    ),
                                    index,
                                ),
                                6,
                            ),
                        }
                    )

                    shares = 0
                    peak_price = None
                    entry = None

                pending = None

            # Signals are evaluated on today's adjusted close and,
            # when possible, execute on the next trading day's open.
            if shares > 0:
                if peak_price is None:
                    raise RuntimeError(
                        "Open position has no peak price."
                    )

                peak_price = max(
                    peak_price,
                    float(
                        row[
                            "adj_close"
                        ]
                    ),
                )

                if index < row_count - 1:
                    reason: str | None = None

                    upper_band = row[
                        "upper_band"
                    ]

                    if (
                        pd.notna(
                            upper_band
                        )
                        and float(
                            row[
                                "adj_close"
                            ]
                        )
                        > float(
                            upper_band
                        )
                    ):
                        reason = (
                            "bollinger"
                        )

                    elif (
                        float(
                            row[
                                "adj_close"
                            ]
                        )
                        < peak_price
                        * (
                            1.0
                            - trailing_stop
                        )
                    ):
                        reason = (
                            "stop_loss"
                        )

                    if reason is not None:
                        pending = {
                            "type": "SELL",
                            "signal_date": date_text,
                            "exec_index": (
                                index
                                + 1
                            ),
                            "reason": reason,
                        }

            elif index < row_count - 1:
                lower_band = row[
                    "lower_band"
                ]

                if (
                    pd.notna(
                        lower_band
                    )
                    and float(
                        row[
                            "adj_close"
                        ]
                    )
                    < float(
                        lower_band
                    )
                ):
                    pending = {
                        "type": "BUY",
                        "signal_date": date_text,
                        "signal_close": float(
                            row[
                                "adj_close"
                            ]
                        ),
                        "lower_band": float(
                            lower_band
                        ),
                        "exec_index": (
                            index
                            + 1
                        ),
                    }

            # A position still open on the final date closes at the
            # final adjusted close, including transaction costs.
            if (
                index == row_count - 1
                and shares > 0
            ):
                if entry is None:
                    raise RuntimeError(
                        "Forced close has no active entry."
                    )

                execution_price = float(
                    row[
                        "adj_close"
                    ]
                )
                quantity = shares

                revenue = sell_net(
                    quantity,
                    execution_price,
                )
                cash += revenue

                pnl = (
                    revenue
                    - float(
                        entry[
                            "buy_total"
                        ]
                    )
                )

                trade_rows.append(
                    {
                        "ticker": "AAPL",
                        "type": "SELL",
                        "signal_date": date_text,
                        "exec_date": date_text,
                        "price": (
                            _round_money(
                                execution_price
                            )
                        ),
                        "shares": quantity,
                        "pnl": (
                            _round_money(
                                pnl
                            )
                        ),
                    }
                )

                analysis_rows.append(
                    {
                        "trade_num": (
                            len(
                                analysis_rows
                            )
                            + 1
                        ),
                        "entry_date": str(
                            entry[
                                "exec_date"
                            ]
                        ),
                        "exit_date": date_text,
                        "entry_price": (
                            _round_money(
                                float(
                                    entry[
                                        "price"
                                    ]
                                )
                            )
                        ),
                        "exit_price": (
                            _round_money(
                                execution_price
                            )
                        ),
                        "pnl": (
                            _round_money(
                                pnl
                            )
                        ),
                        "exit_reason": (
                            "forced_close"
                        ),
                        "intra_trade_max_drawdown": round(
                            holding_drawdown(
                                int(
                                    entry[
                                        "exec_index"
                                    ]
                                ),
                                index,
                            ),
                            6,
                        ),
                    }
                )

                shares = 0
                peak_price = None
                entry = None
                pending = None

            portfolio_values.append(
                float(
                    cash
                    + shares
                    * float(
                        row[
                            "adj_close"
                        ]
                    )
                )
            )

        portfolio = pd.Series(
            portfolio_values,
            dtype=float,
        )

        daily_returns = (
            portfolio
            .pct_change(
                fill_method=None
            )
            .fillna(
                0.0
            )
        )

        cumulative_return = (
            1.0
            + daily_returns
        ).cumprod()

        mean_daily_return = float(
            daily_returns.mean()
        )

        annualized_return = (
            mean_daily_return
            * 252.0
        )

        annualized_volatility = float(
            daily_returns.std(
                ddof=1
            )
            * math.sqrt(
                252.0
            )
        )

        sharpe_ratio = (
            annualized_return
            / annualized_volatility
            if annualized_volatility > 0.0
            else 0.0
        )

        running_max = (
            cumulative_return
            .cummax()
        )

        drawdown = (
            running_max
            - cumulative_return
        ) / running_max

        max_drawdown = float(
            drawdown.max()
        )

        calmar_ratio = (
            annualized_return
            / max_drawdown
            if max_drawdown > 0.0
            else 0.0
        )

        sell_pnls = [
            float(
                row[
                    "pnl"
                ]
            )
            for row in trade_rows
            if (
                row[
                    "type"
                ]
                == "SELL"
                and row[
                    "pnl"
                ]
                is not None
            )
        ]

        num_trades = len(
            sell_pnls
        )

        win_rate = (
            sum(
                1
                for pnl in sell_pnls
                if pnl > 0.0
            )
            / num_trades
            if num_trades
            else 0.0
        )

        final_capital = float(
            cash
            + shares
            * float(
                data.iloc[-1][
                    "adj_close"
                ]
            )
        )

        total_return = (
            final_capital
            - initial_capital
        ) / initial_capital

        total_pnl = sum(
            sell_pnls
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        results = {
            "AAPL": {
                "total_return": float(
                    total_return
                ),
                "final_capital": (
                    _round_money(
                        final_capital
                    )
                ),
                "annualized_return": float(
                    annualized_return
                ),
                "annualized_volatility": float(
                    annualized_volatility
                ),
                "sharpe_ratio": float(
                    sharpe_ratio
                ),
                "max_drawdown": float(
                    max_drawdown
                ),
                "calmar_ratio": float(
                    calmar_ratio
                ),
                "num_trades": int(
                    num_trades
                ),
                "win_rate": float(
                    win_rate
                ),
                "total_pnl": (
                    _round_money(
                        total_pnl
                    )
                ),
                "num_trading_days": int(
                    row_count
                ),
            }
        }

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

        pd.DataFrame(
            trade_rows,
            columns=(
                "ticker",
                "type",
                "signal_date",
                "exec_date",
                "price",
                "shares",
                "pnl",
            ),
        ).to_csv(
            out_dir
            / "trades.csv",
            index=False,
        )

        date_strings = [
            value.strftime(
                "%Y-%m-%d"
            )
            for value in data[
                "date"
            ]
        ]

        cumulative_values = [
            float(value)
            for value in cumulative_return
        ]

        pd.DataFrame(
            {
                "date": date_strings,
                "cumulative_return": (
                    cumulative_values
                ),
            }
        ).to_csv(
            out_dir
            / "daily_portfolio.csv",
            index=False,
        )

        (
            out_dir
            / "trade_analysis.json"
        ).write_text(
            json.dumps(
                {
                    "trades": (
                        analysis_rows
                    )
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        _write_plotly_html(
            out_dir=out_dir,
            dates=date_strings,
            cumulative_returns=(
                cumulative_values
            ),
        )
