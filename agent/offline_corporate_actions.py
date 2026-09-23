from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _find_task_file(
    task_dir: Path,
    name: str,
) -> Path:
    candidates = [
        path
        for path in task_dir.rglob(name)
        if "checks" not in path.parts
    ]

    if not candidates:
        raise RuntimeError(
            f"Corporate Action skill could not find {name}."
        )

    return sorted(candidates)[0]


def _parse_split_ratio(value: object) -> float:
    if isinstance(value, (int, float)):
        ratio = float(value)
    else:
        text = str(value).strip()

        if ":" in text:
            left, right = text.split(":", 1)
            ratio = float(left) / float(right)
        else:
            ratio = float(text)

    if ratio <= 0.0:
        raise RuntimeError(
            f"Invalid stock split ratio: {value!r}"
        )

    return ratio


@dataclass(frozen=True)
class CorporateActionAdjustmentSkill:
    name: str = "corporate-action-price-adjustment"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        lowered = instruction.lower()

        required = (
            "corporate action",
            "adjusted_prices.csv",
            "corporate_actions.json",
        )

        if not all(
            token in lowered
            for token in required
        ):
            return False

        names = {
            path.name
            for path in task_dir.rglob("*")
            if (
                path.is_file()
                and "checks" not in path.parts
            )
        }

        return {
            "prices.csv",
            "corporate_actions.json",
        }.issubset(names)

    def solve(
        self,
        *,
        instruction: str,
        task_dir: Path,
        out_dir: Path,
        seed: int,
    ) -> None:
        del instruction, seed

        prices_path = _find_task_file(
            task_dir,
            "prices.csv",
        )
        actions_path = _find_task_file(
            task_dir,
            "corporate_actions.json",
        )

        prices = pd.read_csv(prices_path)

        required_columns = {
            "date",
            "close",
            "volume",
        }
        missing = sorted(
            required_columns.difference(
                prices.columns
            )
        )
        if missing:
            raise RuntimeError(
                f"prices.csv missing required columns: {missing}"
            )

        with actions_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            payload = json.load(handle)

        actions = payload.get(
            "corporate_actions",
            [],
        )
        if not isinstance(actions, list):
            raise RuntimeError(
                "corporate_actions must be a list."
            )

        working = prices.copy()
        working["date"] = pd.to_datetime(
            working["date"],
            errors="raise",
        )
        working = working.sort_values(
            "date",
            kind="stable",
        ).reset_index(drop=True)

        close_unadjusted = pd.to_numeric(
            working["close"],
            errors="raise",
        ).astype(float)

        volume_unadjusted = pd.to_numeric(
            working["volume"],
            errors="raise",
        ).astype(float)

        close_adjusted = close_unadjusted.copy()
        volume_adjusted = volume_unadjusted.copy()

        cumulative_price_factor = 1.0
        cumulative_volume_factor = 1.0
        total_dividends = 0.0
        n_actions_applied = 0

        ordered_actions = sorted(
            actions,
            key=lambda action: str(
                action["date"]
            ),
        )

        for action in ordered_actions:
            action_date = pd.Timestamp(
                action["date"]
            )

            prior_or_same = working[
                working["date"] <= action_date
            ]

            if prior_or_same.empty:
                raise RuntimeError(
                    "Corporate action occurs before "
                    "the first available trading date."
                )

            reference_row = prior_or_same.iloc[-1]
            reference_close = float(
                reference_row["close"]
            )

            strictly_prior = (
                working["date"] < action_date
            )

            action_type = str(
                action["type"]
            ).strip().lower()

            if action_type == "split":
                ratio = _parse_split_ratio(
                    action["ratio"]
                )

                price_factor = 1.0 / ratio

                close_adjusted.loc[
                    strictly_prior
                ] *= price_factor

                volume_adjusted.loc[
                    strictly_prior
                ] *= ratio

                cumulative_price_factor *= (
                    price_factor
                )
                cumulative_volume_factor *= ratio

            elif action_type == "dividend":
                amount = float(
                    action["amount"]
                )

                if reference_close <= 0.0:
                    raise RuntimeError(
                        "Dividend adjustment requires "
                        "a positive reference close."
                    )

                price_factor = (
                    reference_close - amount
                ) / reference_close

                if price_factor <= 0.0:
                    raise RuntimeError(
                        "Dividend adjustment produced "
                        "a non-positive price factor."
                    )

                close_adjusted.loc[
                    strictly_prior
                ] *= price_factor

                cumulative_price_factor *= (
                    price_factor
                )
                total_dividends += amount

            else:
                raise RuntimeError(
                    f"Unsupported corporate action type: {action_type!r}"
                )

            n_actions_applied += 1

        adjusted = pd.DataFrame(
            {
                "date": working[
                    "date"
                ].dt.strftime("%Y-%m-%d"),
                "close_unadjusted": close_unadjusted,
                "close_adjusted": close_adjusted.round(4),
                "volume_unadjusted": np.rint(
                    volume_unadjusted
                ).astype("int64"),
                "volume_adjusted": np.rint(
                    volume_adjusted
                ).astype("int64"),
            }
        )

        results = {
            "n_actions_applied": int(
                n_actions_applied
            ),
            "cumulative_price_adjustment_factor": round(
                float(cumulative_price_factor),
                8,
            ),
            "cumulative_volume_adjustment_factor": round(
                float(cumulative_volume_factor),
                4,
            ),
            "first_date_adjusted_close": round(
                float(
                    adjusted[
                        "close_adjusted"
                    ].iloc[0]
                ),
                4,
            ),
            "last_date_unadjusted_close": round(
                float(
                    adjusted[
                        "close_unadjusted"
                    ].iloc[-1]
                ),
                4,
            ),
            "total_dividends_per_original_share": round(
                float(total_dividends),
                4,
            ),
        }

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        adjusted.to_csv(
            out_dir / "adjusted_prices.csv",
            index=False,
        )

        with (
            out_dir / "results.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                results,
                handle,
                indent=2,
            )
            handle.write("\n")
