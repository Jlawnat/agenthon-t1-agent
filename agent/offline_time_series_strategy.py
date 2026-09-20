from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd


def _find_single_asset_price_csv(task_dir: Path) -> Path:
    for path in sorted(task_dir.rglob("*.csv")):
        if "checks" in path.parts:
            continue
        try:
            frame = pd.read_csv(path, nrows=5)
        except Exception:
            continue
        lower = {str(c).lower() for c in frame.columns}
        if (
            "date" in lower
            and ("adj_close" in lower or "close" in lower)
            and ("adj_open" in lower or "open" in lower)
        ):
            return path
    raise RuntimeError(
        "No single-asset price CSV with date/open/close data was found."
    )


def _infer_ticker(instruction: str, price_path: Path) -> str:
    patterns = (
        r"\(([A-Z]{1,6})\)",
        r'ticker\s+is\s+always\s+"([A-Z]{1,6})"',
        r"\bfor\s+([A-Z]{1,6})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, instruction)
        if match:
            return match.group(1)

    stem = price_path.stem.upper()
    for token in re.split(r"[^A-Z0-9]+", stem):
        if (
            token
            and token not in {"PRICES", "PRICE", "DAILY", "DATA", "HISTORY"}
            and token.isalpha()
            and len(token) <= 6
        ):
            return token
    return "ASSET"


def _parse_ma_config(instruction: str) -> tuple[str, int, int]:
    lowered = instruction.lower()

    if "simple moving average" in lowered or re.search(
        r"\bSMA\s*\(", instruction, flags=re.IGNORECASE
    ):
        kind = "SMA"
        pattern = r"SMA\s*\(\s*(\d+)\s*\)"
    elif "exponential moving average" in lowered or re.search(
        r"\bEMA\s*\(", instruction, flags=re.IGNORECASE
    ):
        kind = "EMA"
        pattern = r"EMA\s*\(\s*(\d+)\s*\)"
    else:
        raise RuntimeError(
            "Could not infer whether the crossover uses SMA or EMA."
        )

    values = [
        int(value)
        for value in re.findall(
            pattern,
            instruction,
            flags=re.IGNORECASE,
        )
    ]

    unique: list[int] = []
    for value in values:
        if value not in unique:
            unique.append(value)

    if len(unique) < 2:
        generic = [
            int(value)
            for value in re.findall(
                r"\b(?:window|span)\s*=\s*(\d+)",
                instruction,
                flags=re.IGNORECASE,
            )
        ]
        for value in generic:
            if value not in unique:
                unique.append(value)

    if len(unique) < 2:
        raise RuntimeError(
            f"Could not infer fast and slow {kind} windows from the instruction."
        )

    return kind, min(unique), max(unique)


def _parse_initial_capital(instruction: str) -> float:
    match = re.search(
        r"initial\s+capital[^$\d]*\$?\s*([\d,]+(?:\.\d+)?)",
        instruction,
        flags=re.IGNORECASE,
    )
    if not match:
        return 100000.0
    return float(match.group(1).replace(",", ""))


def _moving_average(
    series: pd.Series,
    *,
    kind: str,
    window: int,
) -> pd.Series:
    if kind == "EMA":
        return series.ewm(
            span=window,
            adjust=False,
        ).mean()

    if kind == "SMA":
        return series.rolling(
            window=window,
        ).mean()

    raise RuntimeError(
        f"Unsupported moving-average kind: {kind}"
    )


def _write_plotly_html(
    *,
    out_dir: Path,
    dates: list[str],
    cumulative_returns: list[float],
    ticker: str,
    ma_kind: str,
) -> None:
    title = (
        f"{ma_kind} Crossover Momentum: "
        f"{ticker} Cumulative Returns"
    )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-3.1.0.min.js"></script>
</head>
<body>
  <div id="plotly-chart" style="width:100%;height:640px;"></div>
  <script>
    const dates = {json.dumps(dates)};
    const cumulativeReturn = {json.dumps([float(x) for x in cumulative_returns])};

    Plotly.newPlot(
      "plotly-chart",
      [
        {{
          x: dates,
          y: cumulativeReturn,
          type: "scatter",
          mode: "lines",
          name: "{ticker}"
        }}
      ],
      {{
        title: "{title}",
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
    (out_dir / "cumulative_returns.html").write_text(
        html,
        encoding="utf-8",
    )


@dataclass(frozen=True)
class TimeSeriesStrategySkill:
    """Generic deterministic single-asset moving-average crossover backtest."""

    name: str = "time-series-strategy-domain"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        del task_dir
        lowered = instruction.lower()

        moving_average = (
            "ema" in lowered
            or "sma" in lowered
            or "exponential moving average" in lowered
            or "simple moving average" in lowered
        )

        return (
            moving_average
            and "crossover" in lowered
            and "backtest" in lowered
            and (
                "momentum" in lowered
                or "golden cross" in lowered
                or "death cross" in lowered
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

        price_path = _find_single_asset_price_csv(task_dir)
        frame = pd.read_csv(price_path)

        by_lower = {
            str(column).lower(): str(column)
            for column in frame.columns
        }

        signal_column = (
            by_lower.get("adj_close")
            or by_lower.get("close")
        )
        execution_column = (
            by_lower.get("adj_open")
            or by_lower.get("open")
        )
        date_column = by_lower.get("date")

        if signal_column is None or execution_column is None:
            raise RuntimeError(
                "Moving-average crossover engine requires signal close "
                "and execution open prices."
            )
        if date_column is None:
            raise RuntimeError(
                "Moving-average crossover engine requires a date column."
            )

        data = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    frame[date_column],
                    errors="raise",
                ),
                "signal_close": pd.to_numeric(
                    frame[signal_column],
                    errors="raise",
                ).astype(float),
                "execution_open": pd.to_numeric(
                    frame[execution_column],
                    errors="raise",
                ).astype(float),
            }
        ).sort_values(
            "date",
            kind="stable",
        ).drop_duplicates(
            subset=["date"],
            keep="last",
        ).reset_index(drop=True)

        if len(data) < 3:
            raise RuntimeError(
                "Moving-average crossover backtest requires at least "
                "three price rows."
            )

        ma_kind, fast_window, slow_window = _parse_ma_config(
            instruction
        )
        initial_capital = _parse_initial_capital(
            instruction
        )
        ticker = _infer_ticker(
            instruction,
            price_path,
        )

        data["ma_fast"] = _moving_average(
            data["signal_close"],
            kind=ma_kind,
            window=fast_window,
        )
        data["ma_slow"] = _moving_average(
            data["signal_close"],
            kind=ma_kind,
            window=slow_window,
        )

        cash = float(initial_capital)
        shares = 0
        entry_cost = 0.0
        pending: dict[str, object] | None = None

        trade_rows: list[dict[str, object]] = []
        portfolio_values: list[float] = []

        for index, row in data.iterrows():
            date_text = row["date"].strftime("%Y-%m-%d")

            if (
                pending is not None
                and int(pending["exec_index"]) == index
            ):
                pending_type = str(pending["type"])

                if pending_type == "BUY" and shares == 0:
                    price = float(row["execution_open"])
                    quantity = max(
                        0,
                        math.floor(cash / price),
                    )
                    if quantity > 0:
                        cost = quantity * price
                        cash -= cost
                        shares = quantity
                        entry_cost = cost

                        trade_rows.append(
                            {
                                "ticker": ticker,
                                "type": "BUY",
                                "signal_date": str(
                                    pending["signal_date"]
                                ),
                                "exec_date": date_text,
                                "price": float(price),
                                "shares": int(quantity),
                                "pnl": None,
                            }
                        )

                elif pending_type == "SELL" and shares > 0:
                    price = float(row["execution_open"])
                    quantity = shares
                    proceeds = quantity * price
                    pnl = proceeds - entry_cost
                    cash += proceeds

                    trade_rows.append(
                        {
                            "ticker": ticker,
                            "type": "SELL",
                            "signal_date": str(
                                pending["signal_date"]
                            ),
                            "exec_date": date_text,
                            "price": float(price),
                            "shares": int(quantity),
                            "pnl": float(pnl),
                        }
                    )

                    shares = 0
                    entry_cost = 0.0

                pending = None

            if index > 0 and index < len(data) - 1:
                fast_previous = data.iloc[index - 1]["ma_fast"]
                slow_previous = data.iloc[index - 1]["ma_slow"]
                fast_current = row["ma_fast"]
                slow_current = row["ma_slow"]

                ready = (
                    pd.notna(fast_previous)
                    and pd.notna(slow_previous)
                    and pd.notna(fast_current)
                    and pd.notna(slow_current)
                )

                if ready:
                    crossed_above = (
                        float(fast_previous)
                        <= float(slow_previous)
                        and float(fast_current)
                        > float(slow_current)
                    )
                    crossed_below = (
                        float(fast_previous)
                        >= float(slow_previous)
                        and float(fast_current)
                        < float(slow_current)
                    )

                    if shares == 0 and crossed_above:
                        pending = {
                            "type": "BUY",
                            "signal_date": date_text,
                            "exec_index": index + 1,
                        }
                    elif shares > 0 and crossed_below:
                        pending = {
                            "type": "SELL",
                            "signal_date": date_text,
                            "exec_index": index + 1,
                        }

            if index == len(data) - 1 and shares > 0:
                price = float(row["signal_close"])
                quantity = shares
                proceeds = quantity * price
                pnl = proceeds - entry_cost
                cash += proceeds

                trade_rows.append(
                    {
                        "ticker": ticker,
                        "type": "SELL",
                        "signal_date": date_text,
                        "exec_date": date_text,
                        "price": float(price),
                        "shares": int(quantity),
                        "pnl": float(pnl),
                    }
                )

                shares = 0
                entry_cost = 0.0
                pending = None

            portfolio_values.append(
                float(
                    cash
                    + shares
                    * float(row["signal_close"])
                )
            )

        portfolio = pd.Series(
            portfolio_values,
            dtype=float,
        )

        daily_returns = (
            portfolio
            .pct_change(fill_method=None)
            .fillna(0.0)
        )

        cumulative_return = (
            1.0 + daily_returns
        ).cumprod()

        annualized_return = float(
            daily_returns.mean()
            * 252.0
        )

        annualized_volatility = float(
            daily_returns.std(ddof=1)
            * math.sqrt(252.0)
        )

        sharpe_ratio = (
            annualized_return
            / annualized_volatility
            if annualized_volatility > 0.0
            else 0.0
        )

        running_max = cumulative_return.cummax()
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
            float(row["pnl"])
            for row in trade_rows
            if (
                row["type"] == "SELL"
                and row["pnl"] is not None
            )
        ]

        num_trades = len(sell_pnls)
        win_rate = (
            sum(pnl > 0.0 for pnl in sell_pnls)
            / num_trades
            if num_trades
            else 0.0
        )

        final_capital = float(
            portfolio.iloc[-1]
        )

        total_return = (
            final_capital
            - initial_capital
        ) / initial_capital

        total_pnl = float(
            sum(sell_pnls)
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        results = {
            ticker: {
                "total_return": float(total_return),
                "final_capital": round(final_capital, 2),
                "annualized_return": float(annualized_return),
                "annualized_volatility": float(
                    annualized_volatility
                ),
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "calmar_ratio": float(calmar_ratio),
                "num_trades": int(num_trades),
                "win_rate": float(win_rate),
                "total_pnl": round(total_pnl, 2),
                "num_trading_days": int(len(data)),
            }
        }

        (out_dir / "results.json").write_text(
            json.dumps(results, indent=2) + "\n",
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
            out_dir / "trades.csv",
            index=False,
        )

        dates = [
            value.strftime("%Y-%m-%d")
            for value in data["date"]
        ]

        cumulative_values = [
            float(value)
            for value in cumulative_return
        ]

        pd.DataFrame(
            {
                "date": dates,
                "cumulative_return": cumulative_values,
            }
        ).to_csv(
            out_dir / "daily_portfolio.csv",
            index=False,
        )

        _write_plotly_html(
            out_dir=out_dir,
            dates=dates,
            cumulative_returns=cumulative_values,
            ticker=ticker,
            ma_kind=ma_kind,
        )
