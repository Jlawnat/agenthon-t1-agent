from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from agent.finance_process_selection import select_return_process
from agent.finance_schema import TaskDataCatalog

COMMITTEE_COLUMNS = [
    "asof_date","ticker","sector","cot_bucket",
    "filing_agent_score","issuer_activity_agent_score",
    "news_agent_score","audit_agent_score",
    "committee_score","committee_dispersion",
    "agreement_multiplier","committee_alpha",
]

PROCESS_COLUMNS = [
    "asof_date","ticker","history_count","selected_process",
    "martingale_aic",
    "gbm_mu","gbm_sigma","gbm_aic",
    "ou_theta","ou_mu","ou_sigma","ou_half_life_days","ou_aic",
    "merton_mu","merton_sigma","merton_jump_intensity",
    "merton_jump_mean","merton_jump_std","merton_aic",
    "process_expected_return","process_sign","process_conviction",
]

ALPHA_COLUMNS = [
    "asof_date","ticker","sector","cot_bucket",
    "committee_alpha","process_sign","process_conviction","final_signal",
    "alpha_rank","selection","portfolio_weight",
    "entry_date","exit_date","forward_return_21d",
]

PORTFOLIO_COLUMNS = [
    "asof_date","long_count","short_count","gross_return","turnover",
    "transaction_cost","net_return","equity_curve",
]

DIAGNOSTIC_COLUMNS = [
    "asof_date","num_issuers",
    "fundamental_missing_cells","issuer_activity_missing_cells",
    "news_missing_cells","audit_bulletin_rows_365d",
    "audit_mismatch_issuers","news_nonzero_issuers",
    "process_history_ready_issuers",
]

FILING_COMPONENTS = [
    ("revenue_growth_ttm",1.0),
    ("gross_margin_ttm",1.0),
    ("debt_to_assets",-1.0),
    ("cash_to_assets",1.0),
    ("share_count_growth_yoy",-1.0),
    ("days_since_last_10q",-1.0),
    ("days_since_last_10k",-1.0),
    ("count_8k_trailing_90d",-1.0),
]

ACTIVITY_COMPONENTS = [
    ("insider_net_buy_value_30d",1.0),
    ("insider_net_buy_value_90d",1.0),
    ("insider_buy_count_90d",1.0),
    ("insider_sell_count_90d",-1.0),
    ("officer_net_buy_value_90d",1.0),
    ("director_net_buy_value_90d",1.0),
]

AUDIT_COMPONENTS = [
    ("bulletin_severity_sum_365d",-1.0),
    ("severe_bulletin_count_365d",-1.0),
    ("days_since_last_bulletin",1.0),
    ("reference_mismatch_count",-1.0),
    ("reference_mismatch_abs_sum",-1.0),
    ("reference_stale_field_count",-1.0),
]

FUNDAMENTAL_RAW = [x for x,_ in FILING_COMPONENTS]
ACTIVITY_RAW = [x for x,_ in ACTIVITY_COMPONENTS]
NEWS_RAW = [
    "article_count_7d","article_count_30d","source_count_30d",
    "avg_tone_7d","avg_tone_30d",
    "theme_risk_count_30d","theme_supply_count_30d","theme_macro_count_30d",
]


def normalize_cross_section(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    if clean.notna().any():
        clean = clean.fillna(clean.median())
    else:
        return pd.Series(
            np.zeros(len(clean), dtype=float),
            index=clean.index,
        )
    std = float(clean.std(ddof=0))
    if std <= 0.0 or np.isnan(std):
        return pd.Series(
            np.zeros(len(clean), dtype=float),
            index=clean.index,
        )
    return ((clean - clean.mean()) / std).clip(-3.0, 3.0)


def _artifact(
    catalog: TaskDataCatalog,
    required: set[str],
) -> Path:
    matches = [
        a.path
        for a in catalog.artifacts
        if a.kind == "csv" and required.issubset(a.columns)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one CSV for {sorted(required)}, found {matches}."
        )
    return matches[0]


def _reference_vintages(
    catalog: TaskDataCatalog,
) -> tuple[Path, Path]:
    required = {"asof_date","ticker","field_name","field_value"}
    matches = sorted(
        [
            a.path
            for a in catalog.artifacts
            if a.kind == "csv" and required.issubset(a.columns)
        ],
        key=lambda p: str(p),
    )
    if len(matches) != 2:
        raise RuntimeError(
            f"Expected two reference-vintage CSVs, found {matches}."
        )
    return matches[0], matches[1]


def _load_inputs(task_dir: Path) -> dict[str, object]:
    catalog = TaskDataCatalog.discover(task_dir)
    params_path = catalog.json_with_keys({
        "long_top_n","short_bottom_n","min_trade_signal",
        "initial_capital","transaction_cost_bps","annualization_factor",
        "filing_weight","issuer_activity_weight","news_weight","audit_weight",
        "disagreement_penalty","reference_lookback_days",
        "reference_stale_threshold_days","severe_bulletin_cutoff",
        "process_trailing_window","process_min_history",
        "process_conviction_scale","jump_threshold_sigma",
    })
    ref1, ref2 = _reference_vintages(catalog)
    return {
        "universe": pd.read_csv(_artifact(catalog, {
            "ticker","cik","issuer_name","sector","cot_bucket"
        }), dtype={"cik": str}),
        "rebalance": pd.read_csv(_artifact(catalog, {"rebalance_date"})),
        "fundamental": pd.read_csv(_artifact(catalog, {
            "asof_date","ticker","revenue_growth_ttm","gross_margin_ttm",
            "debt_to_assets","cash_to_assets","share_count_growth_yoy",
            "days_since_last_10q","days_since_last_10k","count_8k_trailing_90d",
        })),
        "activity": pd.read_csv(_artifact(catalog, {
            "asof_date","ticker","insider_net_buy_value_30d",
            "insider_net_buy_value_90d","insider_buy_count_90d",
            "insider_sell_count_90d","officer_net_buy_value_90d",
            "director_net_buy_value_90d",
        })),
        "cot": pd.read_csv(_artifact(catalog, {
            "asof_date","cot_bucket","lev_money_net_oi","asset_mgr_net_oi"
        })),
        "news": pd.read_csv(_artifact(catalog, {
            "asof_date","ticker","article_count_7d","article_count_30d",
            "source_count_30d","avg_tone_7d","avg_tone_30d",
            "theme_risk_count_30d","theme_supply_count_30d",
            "theme_macro_count_30d",
        })),
        "bulletins": pd.read_csv(_artifact(catalog, {
            "bulletin_date","ticker","bulletin_type","severity"
        })),
        "reference_v1": pd.read_csv(ref1),
        "reference_v2": pd.read_csv(ref2),
        "forward": pd.read_csv(_artifact(catalog, {
            "asof_date","ticker","entry_date","exit_date","forward_return_21d"
        })),
        "params": json.loads(params_path.read_text(encoding="utf-8")),
    }


def _build_base(inputs: dict[str, object]) -> pd.DataFrame:
    rebalance = inputs["rebalance"].rename(
        columns={"rebalance_date":"asof_date"}
    )
    universe = inputs["universe"]
    base = (
        rebalance.assign(_k=1)
        .merge(universe.assign(_k=1), on="_k", how="inner")
        .drop(columns="_k")
    )
    base["asof_date"] = pd.to_datetime(
        base["asof_date"]
    ).dt.strftime("%Y-%m-%d")
    return (
        base
        .merge(inputs["fundamental"], on=["asof_date","ticker"], how="left")
        .merge(inputs["activity"], on=["asof_date","ticker"], how="left")
        .merge(inputs["news"], on=["asof_date","ticker"], how="left")
        .merge(inputs["cot"], on=["asof_date","cot_bucket"], how="left")
        .merge(
            inputs["forward"],
            on=["asof_date","ticker"],
            how="left",
            validate="one_to_one",
        )
    )


def _build_audit(
    panel: pd.DataFrame,
    bulletins: pd.DataFrame,
    ref1: pd.DataFrame,
    ref2: pd.DataFrame,
    params: dict[str, object],
) -> pd.DataFrame:
    bulletins = bulletins.copy()
    bulletins["bulletin_date"] = pd.to_datetime(
        bulletins["bulletin_date"]
    )
    ref1 = ref1.copy()
    ref2 = ref2.copy()
    ref1["asof_date"] = pd.to_datetime(ref1["asof_date"])
    ref2["asof_date"] = pd.to_datetime(ref2["asof_date"])

    lookback = int(params["reference_lookback_days"])
    stale = int(params["reference_stale_threshold_days"])
    severe = float(params["severe_bulletin_cutoff"])
    fields = sorted(
        set(ref1["field_name"].unique())
        | set(ref2["field_name"].unique())
    )

    rows = []
    for row in panel[["asof_date","ticker"]].itertuples(index=False):
        asof = pd.Timestamp(row.asof_date)
        ticker = row.ticker
        b = bulletins.loc[
            (bulletins["ticker"] == ticker)
            & (bulletins["bulletin_date"] <= asof)
            & (
                bulletins["bulletin_date"]
                > asof - pd.Timedelta(days=365)
            )
        ]
        if b.empty:
            severity_sum = 0.0
            severe_count = 0
            days_since_last = 365.0
        else:
            severity_sum = float(b["severity"].sum())
            severe_count = int((b["severity"] >= severe).sum())
            days_since_last = float(
                (asof - b["bulletin_date"].max()).days
            )

        mismatch_count = 0
        mismatch_abs_sum = 0.0
        stale_count = 0

        for field in fields:
            one = ref1.loc[
                (ref1["ticker"] == ticker)
                & (ref1["field_name"] == field)
                & (ref1["asof_date"] <= asof)
                & (
                    ref1["asof_date"]
                    > asof - pd.Timedelta(days=lookback)
                )
            ].sort_values("asof_date")
            two = ref2.loc[
                (ref2["ticker"] == ticker)
                & (ref2["field_name"] == field)
                & (ref2["asof_date"] <= asof)
                & (
                    ref2["asof_date"]
                    > asof - pd.Timedelta(days=lookback)
                )
            ].sort_values("asof_date")

            newest = []
            if not one.empty:
                newest.append(one["asof_date"].iloc[-1])
            if not two.empty:
                newest.append(two["asof_date"].iloc[-1])
            if (
                not newest
                or (asof - max(newest)).days > stale
            ):
                stale_count += 1

            if one.empty or two.empty:
                continue
            v1 = one["field_value"].iloc[-1]
            v2 = two["field_value"].iloc[-1]
            try:
                diff = abs(float(v1) - float(v2))
            except (TypeError, ValueError):
                diff = 0.0 if v1 == v2 else 1.0
            if diff > 1e-12:
                mismatch_count += 1
                mismatch_abs_sum += diff

        rows.append({
            "asof_date": asof.strftime("%Y-%m-%d"),
            "ticker": ticker,
            "bulletin_severity_sum_365d": severity_sum,
            "severe_bulletin_count_365d": severe_count,
            "days_since_last_bulletin": days_since_last,
            "reference_mismatch_count": mismatch_count,
            "reference_mismatch_abs_sum": mismatch_abs_sum,
            "reference_stale_field_count": stale_count,
        })
    return pd.DataFrame(rows)


def _build_committee(
    panel: pd.DataFrame,
    audit: pd.DataFrame,
    params: dict[str, object],
) -> pd.DataFrame:
    panel = panel.merge(
        audit,
        on=["asof_date","ticker"],
        how="left",
        validate="one_to_one",
    )
    enriched = []

    for _, group in panel.groupby("asof_date", sort=True):
        grp = group.copy()

        filing_parts = []
        for col, sign in FILING_COMPONENTS:
            name = f"n_{col}"
            grp[name] = normalize_cross_section(
                sign * pd.to_numeric(grp[col], errors="coerce")
            )
            filing_parts.append(name)
        grp["filing_agent_score"] = grp[filing_parts].mean(axis=1)

        activity_parts = []
        for col, sign in ACTIVITY_COMPONENTS:
            name = f"n_{col}"
            grp[name] = normalize_cross_section(
                sign * pd.to_numeric(grp[col], errors="coerce")
            )
            activity_parts.append(name)
        grp["issuer_activity_agent_score"] = grp[
            activity_parts
        ].mean(axis=1)

        news_parts = []
        transformations = {
            "n_news_article_count_7d": np.log1p(
                pd.to_numeric(grp["article_count_7d"], errors="coerce")
            ),
            "n_news_article_count_30d": np.log1p(
                pd.to_numeric(grp["article_count_30d"], errors="coerce")
            ),
            "n_news_source_count_30d": np.log1p(
                pd.to_numeric(grp["source_count_30d"], errors="coerce")
            ),
            "n_news_avg_tone_7d": pd.to_numeric(
                grp["avg_tone_7d"], errors="coerce"
            ),
            "n_news_avg_tone_30d": pd.to_numeric(
                grp["avg_tone_30d"], errors="coerce"
            ),
            "n_news_velocity": (
                pd.to_numeric(
                    grp["article_count_7d"], errors="coerce"
                )
                / np.maximum(
                    pd.to_numeric(
                        grp["article_count_30d"], errors="coerce"
                    ),
                    1.0,
                )
            ),
            "n_news_theme_risk_count_30d": -np.log1p(
                pd.to_numeric(
                    grp["theme_risk_count_30d"], errors="coerce"
                )
            ),
            "n_news_theme_supply_count_30d": -np.log1p(
                pd.to_numeric(
                    grp["theme_supply_count_30d"], errors="coerce"
                )
            ),
            "n_news_theme_macro_count_30d": -np.log1p(
                pd.to_numeric(
                    grp["theme_macro_count_30d"], errors="coerce"
                )
            ),
        }
        for name, values in transformations.items():
            grp[name] = normalize_cross_section(values)
            news_parts.append(name)
        grp["news_agent_score"] = grp[news_parts].mean(axis=1)

        audit_parts = []
        for col, sign in AUDIT_COMPONENTS:
            name = f"n_audit_{col}"
            grp[name] = normalize_cross_section(
                sign * pd.to_numeric(grp[col], errors="coerce")
            )
            audit_parts.append(name)
        grp["audit_agent_score"] = grp[audit_parts].mean(axis=1)

        grp["committee_score"] = (
            float(params["filing_weight"])
            * grp["filing_agent_score"]
            + float(params["issuer_activity_weight"])
            * grp["issuer_activity_agent_score"]
            + float(params["news_weight"])
            * grp["news_agent_score"]
            + float(params["audit_weight"])
            * grp["audit_agent_score"]
        )
        matrix = grp[
            [
                "filing_agent_score",
                "issuer_activity_agent_score",
                "news_agent_score",
                "audit_agent_score",
            ]
        ].to_numpy(dtype=float)
        grp["committee_dispersion"] = matrix.std(axis=1, ddof=0)
        grp["agreement_multiplier"] = (
            1.0
            - float(params["disagreement_penalty"])
            * grp["committee_dispersion"]
        ).clip(lower=0.0)
        grp["committee_alpha"] = (
            grp["committee_score"]
            * grp["agreement_multiplier"]
        )
        enriched.append(grp)

    return (
        pd.concat(enriched, ignore_index=True)[COMMITTEE_COLUMNS]
        .sort_values(["asof_date","ticker"])
        .reset_index(drop=True)
    )


def _build_process(
    panel: pd.DataFrame,
    forward: pd.DataFrame,
    params: dict[str, object],
) -> pd.DataFrame:
    history = forward[[
        "asof_date","ticker","forward_return_21d"
    ]].copy()
    history["asof_date"] = pd.to_datetime(history["asof_date"])
    rows = []

    for row in panel[["asof_date","ticker"]].itertuples(index=False):
        asof = pd.Timestamp(row.asof_date)
        ticker = row.ticker
        trailing = history.loc[
            (history["ticker"] == ticker)
            & (history["asof_date"] < asof),
            "forward_return_21d",
        ].tail(int(params["process_trailing_window"]))
        result = select_return_process(
            trailing.to_numpy(dtype=float),
            minimum_history=int(params["process_min_history"]),
            holding_period_days=int(params["forward_return_days"]),
            conviction_scale=float(params["process_conviction_scale"]),
            jump_threshold_sigma=float(params["jump_threshold_sigma"]),
        )
        rows.append({
            "asof_date": asof.strftime("%Y-%m-%d"),
            "ticker": ticker,
            **result.__dict__,
        })
    return (
        pd.DataFrame(rows)[PROCESS_COLUMNS]
        .sort_values(["asof_date","ticker"])
        .reset_index(drop=True)
    )


def _build_alpha(
    panel: pd.DataFrame,
    committee: pd.DataFrame,
    process: pd.DataFrame,
    params: dict[str, object],
) -> pd.DataFrame:
    base = panel[[
        "asof_date","ticker","sector","cot_bucket",
        "entry_date","exit_date","forward_return_21d",
    ]].merge(
        committee[["asof_date","ticker","committee_alpha"]],
        on=["asof_date","ticker"],
        how="left",
        validate="one_to_one",
    ).merge(
        process[[
            "asof_date","ticker","process_sign","process_conviction"
        ]],
        on=["asof_date","ticker"],
        how="left",
        validate="one_to_one",
    )

    base["final_signal"] = (
        base["committee_alpha"]
        * base["process_sign"]
        * base["process_conviction"]
    )
    frames = []
    for _, group in base.groupby("asof_date", sort=True):
        grp = group.sort_values(
            ["final_signal","ticker"],
            ascending=[False,True],
            kind="stable",
        ).copy()
        grp["alpha_rank"] = np.arange(1, len(grp) + 1)
        grp["selection"] = "neutral"
        grp["portfolio_weight"] = 0.0

        eligible = grp[
            grp["final_signal"].abs()
            >= float(params["min_trade_signal"])
        ]
        longs = eligible[
            eligible["final_signal"] > 0.0
        ].sort_values(
            ["alpha_rank"],
            ascending=True,
        ).head(int(params["long_top_n"]))
        shorts = eligible[
            eligible["final_signal"] < 0.0
        ].sort_values(
            ["alpha_rank"],
            ascending=False,
        ).head(int(params["short_bottom_n"]))

        if len(longs):
            grp.loc[longs.index, "selection"] = "long"
            grp.loc[longs.index, "portfolio_weight"] = (
                0.5 / len(longs)
            )
        if len(shorts):
            grp.loc[shorts.index, "selection"] = "short"
            grp.loc[shorts.index, "portfolio_weight"] = (
                -0.5 / len(shorts)
            )
        frames.append(grp)

    alpha = pd.concat(frames, ignore_index=True)
    alpha["entry_date"] = pd.to_datetime(
        alpha["entry_date"]
    ).dt.strftime("%Y-%m-%d")
    alpha["exit_date"] = pd.to_datetime(
        alpha["exit_date"]
    ).dt.strftime("%Y-%m-%d")
    return (
        alpha[ALPHA_COLUMNS]
        .sort_values(
            ["asof_date","alpha_rank","ticker"]
        )
        .reset_index(drop=True)
    )


def _build_portfolio(
    alpha: pd.DataFrame,
    universe: pd.DataFrame,
    params: dict[str, object],
) -> pd.DataFrame:
    order = universe["ticker"].tolist()
    previous = pd.Series(0.0, index=order)
    rows = []

    for asof, group in alpha.groupby("asof_date", sort=True):
        current = (
            group.set_index("ticker")["portfolio_weight"]
            .reindex(order)
            .fillna(0.0)
        )
        turnover = 0.5 * float(
            (current - previous).abs().sum()
        )
        previous = current
        gross = float(
            (
                group["portfolio_weight"]
                * group["forward_return_21d"]
            ).sum()
        )
        tc = (
            turnover
            * float(params["transaction_cost_bps"])
            / 10000.0
        )
        net = gross - tc
        equity = (
            rows[-1]["equity_curve"] * (1.0 + net)
            if rows
            else float(params["initial_capital"]) * (1.0 + net)
        )
        rows.append({
            "asof_date": asof,
            "long_count": int((group["selection"] == "long").sum()),
            "short_count": int((group["selection"] == "short").sum()),
            "gross_return": gross,
            "turnover": turnover,
            "transaction_cost": tc,
            "net_return": net,
            "equity_curve": equity,
        })

    return pd.DataFrame(rows)[PORTFOLIO_COLUMNS]


def _build_diagnostics(
    panel: pd.DataFrame,
    audit: pd.DataFrame,
    process: pd.DataFrame,
    bulletins: pd.DataFrame,
    params: dict[str, object],
) -> pd.DataFrame:
    b = bulletins.copy()
    b["bulletin_date"] = pd.to_datetime(b["bulletin_date"])
    rows = []

    for asof, group in panel.groupby("asof_date", sort=True):
        asof_ts = pd.Timestamp(asof)
        audit_group = audit[audit["asof_date"] == asof]
        process_group = process[process["asof_date"] == asof]
        relevant_b = b.loc[
            (b["bulletin_date"] <= asof_ts)
            & (
                b["bulletin_date"]
                > asof_ts - pd.Timedelta(days=365)
            )
        ]
        rows.append({
            "asof_date": asof,
            "num_issuers": int(len(group)),
            "fundamental_missing_cells": int(
                group[FUNDAMENTAL_RAW].isna().sum().sum()
            ),
            "issuer_activity_missing_cells": int(
                group[ACTIVITY_RAW].isna().sum().sum()
            ),
            "news_missing_cells": int(
                group[NEWS_RAW].isna().sum().sum()
            ),
            "audit_bulletin_rows_365d": int(len(relevant_b)),
            "audit_mismatch_issuers": int(
                (audit_group["reference_mismatch_count"] > 0).sum()
            ),
            "news_nonzero_issuers": int(
                (
                    pd.to_numeric(
                        group["article_count_30d"],
                        errors="coerce",
                    ).fillna(0.0)
                    > 0.0
                ).sum()
            ),
            "process_history_ready_issuers": int(
                (
                    process_group["selected_process"]
                    != "insufficient_history"
                ).sum()
            ),
        })
    return pd.DataFrame(rows)[DIAGNOSTIC_COLUMNS]


def _results(
    portfolio: pd.DataFrame,
    diagnostics: pd.DataFrame,
    params: dict[str, object],
    num_issuers: int,
) -> dict[str, object]:
    net = portfolio["net_return"].to_numpy(dtype=float)
    initial = float(params["initial_capital"])
    ending = float(portfolio["equity_curve"].iloc[-1])
    cumulative = ending / initial - 1.0
    n = len(net)
    annualization = float(params["annualization_factor"])
    annualized_return = (
        (1.0 + cumulative) ** (annualization / n) - 1.0
    )
    annualized_vol = (
        float(np.std(net, ddof=0)) * math.sqrt(annualization)
        if n > 1 else 0.0
    )
    arithmetic_ann = float(np.mean(net)) * annualization
    sharpe = (
        arithmetic_ann / annualized_vol
        if annualized_vol > 0.0 else 0.0
    )
    wealth = portfolio["equity_curve"].to_numpy(dtype=float)
    dd = wealth / np.maximum.accumulate(wealth) - 1.0
    return {
        "num_issuers": int(num_issuers),
        "num_rebalance_dates": int(n),
        "num_backtest_periods": int(n),
        "num_process_ready_dates": int(
            (diagnostics["process_history_ready_issuers"] > 0).sum()
        ),
        "num_news_active_dates": int(
            (diagnostics["news_nonzero_issuers"] > 0).sum()
        ),
        "cumulative_return": float(cumulative),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(np.min(dd)),
        "avg_turnover": float(portfolio["turnover"].mean()),
        "best_rebalance_date": str(
            portfolio.loc[
                portfolio["net_return"].idxmax(),
                "asof_date",
            ]
        ),
        "worst_rebalance_date": str(
            portfolio.loc[
                portfolio["net_return"].idxmin(),
                "asof_date",
            ]
        ),
    }


def _solution(
    alpha: pd.DataFrame,
    process: pd.DataFrame,
) -> dict[str, object]:
    first = str(alpha["asof_date"].min())
    last = str(alpha["asof_date"].max())

    def picked(date: str, side: str) -> list[str]:
        return (
            alpha[
                (alpha["asof_date"] == date)
                & (alpha["selection"] == side)
            ]
            .sort_values("alpha_rank")["ticker"]
            .astype(str)
            .tolist()
        )

    counts = (
        process["selected_process"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    return {
        "first_rebalance_date": first,
        "first_rebalance_longs": picked(first, "long"),
        "first_rebalance_shorts": picked(first, "short"),
        "last_rebalance_date": last,
        "last_rebalance_longs": picked(last, "long"),
        "last_rebalance_shorts": picked(last, "short"),
        "selected_process_counts": {
            str(k): int(v)
            for k, v in counts.items()
        },
    }


def run_multimodal_alpha_workflow(
    task_dir: Path,
    out_dir: Path,
) -> None:
    inputs = _load_inputs(task_dir)
    params = inputs["params"]
    panel = _build_base(inputs)
    if len(panel) != (
        len(inputs["universe"])
        * len(inputs["rebalance"])
    ):
        raise RuntimeError(
            "Base multimodal panel did not preserve the full cartesian product."
        )

    audit = _build_audit(
        panel,
        inputs["bulletins"],
        inputs["reference_v1"],
        inputs["reference_v2"],
        params,
    )
    committee = _build_committee(panel, audit, params)
    process = _build_process(
        panel,
        inputs["forward"],
        params,
    )
    alpha = _build_alpha(
        panel,
        committee,
        process,
        params,
    )
    portfolio = _build_portfolio(
        alpha,
        inputs["universe"],
        params,
    )
    diagnostics = _build_diagnostics(
        panel,
        audit,
        process,
        inputs["bulletins"],
        params,
    )
    results = _results(
        portfolio,
        diagnostics,
        params,
        num_issuers=len(inputs["universe"]),
    )
    solution = _solution(alpha, process)

    out_dir.mkdir(parents=True, exist_ok=True)
    committee.to_csv(
        out_dir / "committee_panel.csv",
        index=False,
    )
    process.to_csv(
        out_dir / "process_diagnostics.csv",
        index=False,
    )
    alpha.to_csv(
        out_dir / "alpha_panel.csv",
        index=False,
    )
    portfolio.to_csv(
        out_dir / "portfolio_returns.csv",
        index=False,
    )
    diagnostics.to_csv(
        out_dir / "modality_diagnostics.csv",
        index=False,
    )
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "solution.json").write_text(
        json.dumps(solution, indent=2) + "\n",
        encoding="utf-8",
    )
