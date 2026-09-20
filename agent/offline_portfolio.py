from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from agent.offline_common.portfolio import (
    clean_price_panel,
    cross_sectional_momentum_path,
    find_price_panel_csv,
    parse_cross_sectional_momentum_spec,
    simple_returns_from_prices,
    summarize_return_path,
)


@dataclass(frozen=True)
class PortfolioStrategySkill:
    """Generic deterministic portfolio-strategy domain handler."""

    name: str = "portfolio-strategy-domain"

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
            "cross-sectional",
            "momentum",
            "long",
            "short",
            "portfolio",
            "monthly",
        )

        output_markers = (
            "results.json",
            "portfolio.csv",
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
        del seed

        spec = (
            parse_cross_sectional_momentum_spec(
                instruction
            )
        )

        price_path = (
            find_price_panel_csv(
                task_dir
            )
        )

        raw = pd.read_csv(
            price_path
        )

        date_column = next(
            (
                str(column)
                for column
                in raw.columns
                if str(
                    column
                ).lower()
                == "date"
            ),
            None,
        )

        if date_column is None:
            raise RuntimeError(
                "Price panel has no date column."
            )

        if date_column != "date":
            raw = raw.rename(
                columns={
                    date_column: "date"
                }
            )

        (
            cleaned,
            asset_columns,
        ) = clean_price_panel(
            raw,
            date_column="date",
        )

        returns = (
            simple_returns_from_prices(
                cleaned,
                asset_columns=asset_columns,
                date_column="date",
            )
        )

        portfolio = (
            cross_sectional_momentum_path(
                returns,
                asset_columns=asset_columns,
                spec=spec,
                date_column="date",
            )
        )

        performance = (
            summarize_return_path(
                portfolio,
                periods_per_year=(
                    spec.periods_per_year
                ),
            )
        )

        results = {
            "num_clean_rows": int(
                len(
                    cleaned
                )
            ),
            "num_signal_months": int(
                len(
                    portfolio
                )
            ),
            "total_return": float(
                performance[
                    "total_return"
                ]
            ),
            "annualized_return": float(
                performance[
                    "annualized_return"
                ]
            ),
            "annualized_vol": float(
                performance[
                    "annualized_vol"
                ]
            ),
            "sharpe": float(
                performance[
                    "sharpe"
                ]
            ),
            "max_drawdown": float(
                performance[
                    "max_drawdown"
                ]
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

        portfolio.to_csv(
            out_dir
            / "portfolio.csv",
            index=False,
        )
