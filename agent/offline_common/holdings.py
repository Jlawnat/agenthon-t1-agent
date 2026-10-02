from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AmendmentResolution:
    accession: str
    action: str
    effective_accessions: tuple[str, ...]


def resolve_amendment_sequence(filings: Sequence[Mapping[str, object]]) -> list[AmendmentResolution]:
    """Resolve ordered initial, restatement, and new-holdings filing states."""
    effective: list[str] = []
    result: list[AmendmentResolution] = []
    for filing in filings:
        accession = str(filing["accession"])
        amended = str(filing.get("is_amendment", "")).strip().upper() in {"Y", "YES", "TRUE", "1"}
        amendment_type = str(filing.get("amendment_type", "")).strip().upper()
        if not amended:
            action = "replace_initial"
            effective = [accession]
        elif amendment_type == "RESTATEMENT":
            action = "replace_restatement"
            effective = [accession]
        elif amendment_type == "NEW HOLDINGS":
            action = "append_new_holdings"
            effective.append(accession)
        else:
            raise ValueError(f"unsupported amendment type: {amendment_type!r}")
        result.append(AmendmentResolution(accession, action, tuple(effective)))
    return result


def clean_long_holdings(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Filter and aggregate a normalized 13F-like long-share holdings table."""
    required = {"CUSIP", "SSHPRNAMTTYPE", "PUTCALL", "VALUE", "SSHPRNAMT"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("missing holdings columns: " + ", ".join(missing))
    work = frame.copy()
    for column in ("CUSIP", "SSHPRNAMTTYPE", "PUTCALL"):
        work[column] = work[column].fillna("").astype(str).str.strip().str.upper()
    for column in ("VALUE", "SSHPRNAMT"):
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0)

    stages = []
    masks = (
        work["CUSIP"].ne(""),
        work["SSHPRNAMTTYPE"].eq("SH"),
        work["PUTCALL"].eq(""),
        work["VALUE"].gt(0.0) & work["SSHPRNAMT"].gt(0.0),
    )
    current = work
    for mask in masks:
        aligned = mask.reindex(current.index)
        stages.append(int((~aligned).sum()))
        current = current.loc[aligned].copy()

    numeric = [column for column in ("VALUE", "SSHPRNAMT", "VOTING_AUTH_SOLE", "VOTING_AUTH_SHARED", "VOTING_AUTH_NONE") if column in current]
    aggregations: dict[str, object] = {column: "sum" for column in numeric}
    for column in ("NAMEOFISSUER", "TITLEOFCLASS"):
        if column in current:
            aggregations[column] = "first"
    grouped = current.groupby("CUSIP", as_index=False, sort=True).agg(aggregations)
    total = float(grouped["VALUE"].sum())
    grouped["weight"] = grouped["VALUE"] / total if total > 0.0 else 0.0
    audit = {
        "raw_rows": int(len(work)),
        "kept_rows": int(len(current)),
        "filtered_blank_cusip_rows": stages[0],
        "filtered_nonshare_rows": stages[1],
        "filtered_option_rows": stages[2],
        "filtered_nonpositive_rows": stages[3],
    }
    return grouped, audit


def concentration_metrics(weights: Sequence[float]) -> dict[str, float]:
    """Return HHI, effective name count, and top weight."""
    values = np.asarray(weights, dtype=float).ravel()
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("weights must be a non-empty finite non-negative vector")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("weights must have positive sum")
    normalized = values / total
    hhi = float(np.dot(normalized, normalized))
    return {"hhi": hhi, "effective_num_names": float(1.0 / hhi), "top_weight": float(normalized.max())}


def portfolio_turnover(previous: Mapping[str, float], current: Mapping[str, float]) -> float:
    """Compute one-way turnover over the union of security identifiers."""
    names = set(previous) | set(current)
    return float(0.5 * sum(abs(float(current.get(name, 0.0)) - float(previous.get(name, 0.0))) for name in names))


def weighted_overlap(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    """Compute the sum of minimum portfolio weights by common identifier."""
    return float(sum(min(float(first[name]), float(second[name])) for name in set(first) & set(second)))


def aggregate_crowding(holdings: pd.DataFrame, *, owner_column: str = "owner") -> pd.DataFrame:
    """Aggregate holder count, value, and holder-weight statistics by CUSIP."""
    required = {owner_column, "CUSIP", "VALUE", "weight"}
    if not required.issubset(holdings.columns):
        raise ValueError("holdings lacks owner, CUSIP, VALUE, or weight")
    return (
        holdings.groupby("CUSIP", as_index=False)
        .agg(
            holder_count=(owner_column, "nunique"),
            total_value=("VALUE", "sum"),
            avg_holder_weight=("weight", "mean"),
            max_holder_weight=("weight", "max"),
        )
        .sort_values(["holder_count", "total_value", "CUSIP"], ascending=[False, False, True], kind="stable")
        .reset_index(drop=True)
    )
