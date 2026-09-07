from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent.skill_packs import load_skill_packs


TARGET_PACKS = [
    "software",
    "numerical",
    "data-causality",
    "accounting",
    "derivatives",
    "fixed-income",
    "credit",
    "factor-research",
    "backtesting",
    "risk-management",
    "microstructure",
    "fx",
    "nlp-finance",
    "cross-domain",
]


CAPABILITY_SIGNALS: dict[str, dict[str, tuple[str, ...]]] = {
    "software": {
        "contracts": ("schema", "output", "path", "file", "dtype"),
        "robustness": ("edge case", "nan", "inf", "determin", "error"),
        "reproducibility": ("seed", "reproduc", "determin"),
    },
    "numerical": {
        "stability": ("stability", "stable", "conditioning", "overflow", "underflow"),
        "convergence": ("convergence", "tolerance", "error", "residual"),
        "root_optimisation": ("root", "optimization", "optimisation", "solver"),
        "simulation": ("monte carlo", "simulation", "seed", "variance"),
    },
    "data-causality": {
        "lookahead": ("look-ahead", "lookahead", "leakage"),
        "timestamps": ("timestamp", "signal time", "execution time", "information time"),
        "split_discipline": ("train", "validation", "test", "cutoff"),
    },
    "accounting": {
        "nav": ("nav", "net asset"),
        "pnl": ("p&l", "pnl", "profit"),
        "cash_positions": ("cash", "position", "holding"),
        "costs": ("fee", "cost", "transaction cost"),
    },
    "derivatives": {
        "black_scholes": ("black-scholes", "black scholes"),
        "greeks": ("delta", "gamma", "vega", "theta"),
        "parity_bounds": ("put-call", "parity", "arbitrage", "bounds"),
        "trees_pde": ("binomial", "finite difference", "pde"),
        "monte_carlo": ("monte carlo", "simulation"),
        "calibration": ("calibration", "implied volatility", "root"),
    },
    "fixed-income": {
        "discounting": ("discount factor", "present value", "cashflow"),
        "curve": ("yield curve", "term structure", "bootstrap", "bootstrapp"),
        "duration_dv01": ("duration", "dv01", "convexity"),
        "rates": ("zero rate", "forward rate", "spot rate"),
    },
    "credit": {
        "default_survival": ("default probability", "survival probability", "hazard"),
        "recovery": ("recovery", "loss given default", "lgd"),
        "cds": ("cds", "credit default swap", "protection leg", "premium leg"),
        "migration": ("migration", "transition matrix", "rating"),
    },
    "factor-research": {
        "ic": ("information coefficient", " ic ", "spearman"),
        "neutrality": ("neutral", "dollar-neutral", "beta-neutral"),
        "ranking": ("rank", "quantile", "cross-sectional"),
        "lookahead": ("look-ahead", "lookahead", "leakage"),
        "factor_portfolio": ("factor", "portfolio", "weight"),
    },
    "backtesting": {
        "self_financing": ("self-financing", "self financing"),
        "costs_turnover": ("transaction cost", "turnover", "fee"),
        "positions_fills": ("position", "fill", "trade"),
        "lookahead": ("look-ahead", "lookahead", "leakage"),
        "performance": ("sharpe", "drawdown", "return"),
    },
    "risk-management": {
        "var": ("var", "value at risk"),
        "es": ("expected shortfall", "cvar", "conditional var"),
        "stress": ("stress", "scenario"),
        "covariance": ("covariance", "correlation"),
        "aggregation": ("portfolio", "exposure", "aggregate"),
    },
    "microstructure": {
        "spread": ("bid", "ask", "spread"),
        "vwap": ("vwap", "twap"),
        "implementation_shortfall": ("implementation shortfall", "slippage"),
        "execution": ("execution", "participation", "schedule"),
        "orderbook": ("order book", "lob", "depth"),
    },
    "fx": {
        "cip": ("covered interest parity", "cip"),
        "forwards": ("forward rate", "fx forward"),
        "triangular": ("triangular", "cross rate"),
        "quote_convention": ("quote", "base currency", "counter currency"),
        "fx_options": ("garman-kohlhagen", "garman", "fx option"),
    },
    "nlp-finance": {
        "sentiment": ("sentiment", "tone"),
        "temporal": ("timestamp", "look-ahead", "lookahead", "publication"),
        "text_processing": ("token", "text", "document", "entity"),
        "finance_context": ("finance", "filing", "earnings", "fomc"),
    },
    "cross-domain": {
        "union": ("union", "multiple", "cross-domain", "constituent"),
        "units": ("unit", "conversion", "consistent"),
        "decomposition": ("decompose", "sub-problem", "subproblem"),
        "aggregation": ("aggregate", "risk aggregation"),
    },
}


@dataclass
class PackAudit:
    pack: str
    loaded: bool
    payload_type: str | None
    payload_keys: list[str]
    text_chars: int
    list_item_count: int
    dict_item_count: int
    capability_hits: dict[str, bool]
    capability_hit_count: int
    capability_total: int
    missing_capabilities: list[str]
    load_error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _payload(pack: Any) -> dict[str, Any]:
    to_dict = getattr(pack, "to_dict", None)

    if callable(to_dict):
        value = to_dict()
        if isinstance(value, dict):
            return value

    raw = getattr(pack, "__dict__", None)

    if isinstance(raw, dict):
        return dict(raw)

    return {"repr": repr(pack)}


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        return "\n".join(
            f"{key}\n{_flatten_text(item)}"
            for key, item in value.items()
        )

    if isinstance(value, (list, tuple, set)):
        return "\n".join(
            _flatten_text(item)
            for item in value
        )

    return str(value)


def _count_structure(value: Any) -> tuple[int, int]:
    list_items = 0
    dict_items = 0

    if isinstance(value, dict):
        dict_items += len(value)
        for item in value.values():
            a, b = _count_structure(item)
            list_items += a
            dict_items += b

    elif isinstance(value, (list, tuple, set)):
        list_items += len(value)
        for item in value:
            a, b = _count_structure(item)
            list_items += a
            dict_items += b

    return list_items, dict_items


def _signal_hit(text: str, alternatives: tuple[str, ...]) -> bool:
    lowered = f" {text.lower()} "
    return any(
        alternative.lower() in lowered
        for alternative in alternatives
    )


def audit_pack(name: str) -> PackAudit:
    try:
        loaded = load_skill_packs([name])

        if len(loaded) != 1:
            raise RuntimeError(
                f"expected exactly one pack, got {len(loaded)}"
            )

        pack = loaded[0]
        payload = _payload(pack)
        text = _flatten_text(payload)

        list_items, dict_items = _count_structure(payload)

        expected = CAPABILITY_SIGNALS.get(name, {})
        hits = {
            capability: _signal_hit(text, alternatives)
            for capability, alternatives in expected.items()
        }

        missing = [
            capability
            for capability, hit in hits.items()
            if not hit
        ]

        return PackAudit(
            pack=name,
            loaded=True,
            payload_type=type(pack).__name__,
            payload_keys=sorted(str(key) for key in payload.keys()),
            text_chars=len(text),
            list_item_count=list_items,
            dict_item_count=dict_items,
            capability_hits=hits,
            capability_hit_count=sum(hits.values()),
            capability_total=len(hits),
            missing_capabilities=missing,
            load_error=None,
        )

    except Exception as exc:
        return PackAudit(
            pack=name,
            loaded=False,
            payload_type=None,
            payload_keys=[],
            text_chars=0,
            list_item_count=0,
            dict_item_count=0,
            capability_hits={},
            capability_hit_count=0,
            capability_total=0,
            missing_capabilities=[],
            load_error=f"{type(exc).__name__}: {exc}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only audit of Agenthon T1 skill-pack depth."
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmark") / "skill_pack_audit",
    )

    args = parser.parse_args()

    rows = [
        audit_pack(name)
        for name in TARGET_PACKS
    ]

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_path = out_dir / "skill_pack_audit_rows.json"
    report_path = out_dir / "skill_pack_audit_report.json"

    rows_path.write_text(
        json.dumps(
            [row.to_dict() for row in rows],
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    missing_counter = Counter(
        capability
        for row in rows
        for capability in row.missing_capabilities
    )

    report = {
        "packs_audited": len(rows),
        "load_failures": sum(not row.loaded for row in rows),
        "packs_with_missing_capability_signals": sum(
            bool(row.missing_capabilities)
            for row in rows
        ),
        "missing_capability_signal_counts": dict(
            missing_counter.most_common()
        ),
        "packs": [row.to_dict() for row in rows],
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("AGENTHON T1 — SKILL-PACK DEPTH AUDIT")
    print("=" * 72)
    print("Packs audited:", report["packs_audited"])
    print("Load failures:", report["load_failures"])
    print(
        "Packs with missing capability signals:",
        report["packs_with_missing_capability_signals"],
    )

    print("\nPer-pack summary:")

    for row in rows:
        score = (
            f"{row.capability_hit_count}/{row.capability_total}"
            if row.capability_total
            else "n/a"
        )

        print(
            f"  {row.pack:18} "
            f"signals={score:5} "
            f"text_chars={row.text_chars:5} "
            f"list_items={row.list_item_count:3}"
        )

        if row.load_error:
            print(f"    ERROR: {row.load_error}")

        elif row.missing_capabilities:
            print(
                "    missing-review-signals:",
                ", ".join(row.missing_capabilities),
            )

    print("\nSaved:")
    print(f"  {rows_path}")
    print(f"  {report_path}")

    print(
        "\nNOTE: Missing capability signals are review flags, not hard failures."
    )


if __name__ == "__main__":
    main()