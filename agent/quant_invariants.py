from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import json
import re

import numpy as np
import pandas as pd

from agent.candidate_contract import VerificationEvidence
from agent.compiled_specification import CompiledSpecification


_DOMAIN_IDS = (
    "derivatives-pricing",
    "fixed-income",
    "credit",
    "factor-research",
    "backtesting",
    "risk-management",
    "microstructure",
    "fx",
    "nlp-on-finance",
)

_DOMAIN_ALIASES = {
    "derivatives": "derivatives-pricing",
    "derivative-pricing": "derivatives-pricing",
    "derivatives-pricing": "derivatives-pricing",
    "options": "derivatives-pricing",
    "options-pricing": "derivatives-pricing",
    "fixed-income": "fixed-income",
    "fixedincome": "fixed-income",
    "rates": "fixed-income",
    "credit": "credit",
    "credit-risk": "credit",
    "factor": "factor-research",
    "factors": "factor-research",
    "factor-research": "factor-research",
    "backtest": "backtesting",
    "backtesting": "backtesting",
    "risk": "risk-management",
    "risk-management": "risk-management",
    "market-risk": "risk-management",
    "microstructure": "microstructure",
    "market-microstructure": "microstructure",
    "fx": "fx",
    "foreign-exchange": "fx",
    "nlp": "nlp-on-finance",
    "financial-nlp": "nlp-on-finance",
    "nlp-finance": "nlp-on-finance",
    "nlp-on-finance": "nlp-on-finance",
    "cross-domain": "cross-domain",
    "crossdomain": "cross-domain",
}


@dataclass(frozen=True)
class QuantInvariantConfig:
    """Safe deterministic inputs for quantitative verification."""

    category: str | None = None
    review_packs: tuple[str, ...] = ()
    declared_invariants: tuple[str, ...] = ()
    units: dict[str, str] = field(default_factory=dict)
    conventions: tuple[str, ...] = ()
    absolute_tolerance: float = 1e-8
    relative_tolerance: float = 1e-6

    @classmethod
    def from_compiled_specification(
        cls,
        spec: CompiledSpecification,
    ) -> "QuantInvariantConfig":
        return cls(
            category=spec.category,
            review_packs=tuple(spec.review_packs),
            declared_invariants=tuple(spec.invariants),
            units=dict(spec.units),
            conventions=tuple(spec.conventions),
        )


@dataclass(frozen=True)
class _CheckResult:
    evidence: tuple[VerificationEvidence, ...] = ()
    skip_reason: str | None = None


@dataclass(frozen=True)
class _InvariantCheck:
    name: str
    evaluator: Callable[["_InvariantContext"], _CheckResult]
    review_packs: frozenset[str] = frozenset()
    categories: frozenset[str] = frozenset()
    always_active: bool = False

    def is_active(self, config: QuantInvariantConfig) -> bool:
        if self.always_active:
            return True

        declared = {
            _normalise_name(value)
            for value in config.declared_invariants
        }
        if _normalise_name(self.name) in declared:
            return True

        raw_packs = {
            _normalise_pack(value)
            for value in config.review_packs
        }
        if self.review_packs.intersection(raw_packs):
            return True

        category = _canonical_domain(config.category)
        check_categories = {
            _canonical_domain(value) or value
            for value in self.categories
        }

        if category in check_categories:
            return True

        declared_domains = {
            domain
            for value in config.review_packs
            if (domain := _canonical_domain(value)) in _DOMAIN_IDS
        }
        if check_categories.intersection(declared_domains):
            return True

        declared_domains = {
            domain
            for value in config.review_packs
            if (domain := _canonical_domain(value)) in _DOMAIN_IDS
        }

        if check_categories.intersection(declared_domains):
            return True

        return False


class _Columns:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._normalised: dict[str, str] = {}
        for column in frame.columns:
            self._normalised.setdefault(
                _normalise_name(str(column)),
                str(column),
            )

    def find(self, *aliases: str) -> str | None:
        for alias in aliases:
            actual = self._normalised.get(_normalise_name(alias))
            if actual is not None:
                return actual
        return None

    def matching(self, predicate: Callable[[str], bool]) -> list[str]:
        return [
            actual
            for normalised, actual in self._normalised.items()
            if predicate(normalised)
        ]

    def has(self, *aliases: str) -> bool:
        return self.find(*aliases) is not None


@dataclass(frozen=True)
class _InvariantContext:
    frame: pd.DataFrame
    columns: _Columns
    config: QuantInvariantConfig


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _normalise_pack(value: str) -> str:
    return _normalise_name(value).replace("_", "-")


def _canonical_domain(value: str | None) -> str | None:
    if value is None:
        return None
    key = _normalise_pack(str(value))
    return _DOMAIN_ALIASES.get(key, key if key in _DOMAIN_IDS else None)


def _evidence(
    *,
    name: str,
    passed: bool,
    message: str,
    details: dict[str, Any] | None = None,
    hard_failure: bool = True,
    confidence: float = 1.0,
    provenance: str = "deterministic_output_invariant",
) -> VerificationEvidence:
    if passed:
        status = "pass"
        severity = "info"
    elif hard_failure:
        status = "fail"
        severity = "hard_fail"
    else:
        status = "warning"
        severity = "warning"

    merged_details = {
        "provenance": provenance,
        "confidence": float(confidence),
        **(details or {}),
    }
    return VerificationEvidence(
        name=name,
        status=status,
        severity=severity,
        message=message,
        details=merged_details,
    )


def _numeric_series(
    frame: pd.DataFrame,
    column: str,
) -> tuple[pd.Series | None, str | None]:
    converted = pd.to_numeric(frame[column], errors="coerce")
    invalid = converted.isna() & frame[column].notna()
    if bool(invalid.any()):
        return None, f"Column is not consistently numeric: {column}"
    return converted.astype(float), None


def _numeric_inputs(
    context: _InvariantContext,
    mapping: dict[str, tuple[str, ...]],
) -> tuple[dict[str, pd.Series] | None, str | None]:
    values: dict[str, pd.Series] = {}
    for logical_name, aliases in mapping.items():
        column = context.columns.find(*aliases)
        if column is None:
            return None, f"required semantic column not found: {logical_name}"
        series, error = _numeric_series(context.frame, column)
        if error is not None or series is None:
            return None, error
        values[logical_name] = series
    return values, None


def _bad_row_details(mask: pd.Series | np.ndarray) -> dict[str, Any]:
    array = np.asarray(mask, dtype=bool)
    indices = np.flatnonzero(array)
    return {
        "violation_count": int(array.sum()),
        "sample_row_indices": [int(index) for index in indices[:10]],
    }


def _unit_for(config: QuantInvariantConfig, column: str) -> str:
    wanted = _normalise_name(column)
    for key, value in config.units.items():
        if _normalise_name(str(key)) == wanted:
            return str(value).strip().lower()
    return ""


def _is_percent_unit(unit: str) -> bool:
    return "%" in unit or "percent" in unit or unit in {"pct", "percentage"}


def _decimal_values(
    context: _InvariantContext,
    column: str,
    series: pd.Series,
) -> pd.Series:
    return series / 100.0 if _is_percent_unit(_unit_for(context.config, column)) else series


def _convention_text(config: QuantInvariantConfig) -> str:
    return " ".join(config.conventions).lower()


def _finite_numeric_values(context: _InvariantContext) -> _CheckResult:
    numeric = context.frame.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        return _CheckResult(skip_reason="no numeric output columns")

    violations: dict[str, dict[str, Any]] = {}
    for column in numeric.columns:
        values = numeric[column].to_numpy(dtype=float, copy=False)
        mask = ~np.isfinite(values)
        if bool(mask.any()):
            violations[str(column)] = _bad_row_details(mask)

    passed = not violations
    return _CheckResult(
        evidence=(
            _evidence(
                name="finite_numeric_values",
                passed=passed,
                message=(
                    "Numeric outputs are finite."
                    if passed
                    else "One or more numeric outputs contain NaN or infinity."
                ),
                details={
                    "checked_columns": [str(column) for column in numeric.columns],
                    "violations": violations,
                },
            ),
        )
    )


def _probability_bounds(context: _InvariantContext) -> _CheckResult:
    exact_names = {
        "pd",
        "probability",
        "default_probability",
        "survival_probability",
        "tail_probability",
        "confidence_level",
        "recovery_rate",
        "loss_given_default",
        "lgd",
    }
    columns = context.columns.matching(
        lambda name: (
            name in exact_names
            or name.endswith("_probability")
            or name.startswith("probability_")
            or name.startswith("prob_")
        )
    )
    if not columns:
        return _CheckResult(skip_reason="no probability-like columns")

    violations: dict[str, dict[str, Any]] = {}
    invalid_numeric: list[str] = []
    tolerance = context.config.absolute_tolerance

    for column in columns:
        series, error = _numeric_series(context.frame, column)
        if error is not None or series is None:
            invalid_numeric.append(column)
            continue
        unit = _unit_for(context.config, column)
        upper = 100.0 if _is_percent_unit(unit) else 1.0
        mask = (series < -tolerance) | (series > upper + tolerance)
        if bool(mask.any()):
            violations[column] = {
                **_bad_row_details(mask),
                "expected_bounds": [0.0, upper],
                "unit": unit or "decimal",
            }

    passed = not violations and not invalid_numeric
    return _CheckResult(
        evidence=(
            _evidence(
                name="probability_bounds",
                passed=passed,
                message=(
                    "Probability-like outputs satisfy their configured bounds."
                    if passed
                    else "Probability-like outputs violate their configured bounds."
                ),
                details={
                    "checked_columns": columns,
                    "invalid_numeric_columns": invalid_numeric,
                    "violations": violations,
                },
            ),
        )
    )


def _nonnegative_quantities(context: _InvariantContext) -> _CheckResult:
    exact_names = {
        "volatility",
        "vol",
        "variance",
        "standard_deviation",
        "std_dev",
        "discount_factor",
        "hazard_rate",
        "hazard_intensity",
        "loss_given_default",
        "lgd",
        "exposure_at_default",
        "ead",
        "turnover",
        "effective_spread",
        "market_impact",
        "transaction_cost",
        "transaction_costs",
        "cva",
        "expected_shortfall",
        "cvar",
    }
    columns = context.columns.matching(
        lambda name: (
            name in exact_names
            or name.endswith("_volatility")
            or name.endswith("_variance")
        )
    )
    if not columns:
        return _CheckResult(skip_reason="no nonnegative measure columns")

    violations: dict[str, dict[str, Any]] = {}
    invalid_numeric: list[str] = []
    tolerance = context.config.absolute_tolerance

    for column in columns:
        series, error = _numeric_series(context.frame, column)
        if error is not None or series is None:
            invalid_numeric.append(column)
            continue
        if _normalise_name(column) == "discount_factor":
            mask = series <= 0.0
        else:
            mask = series < -tolerance
        if bool(mask.any()):
            violations[column] = _bad_row_details(mask)

    passed = not violations and not invalid_numeric
    return _CheckResult(
        evidence=(
            _evidence(
                name="nonnegative_quantities",
                passed=passed,
                message=(
                    "Nonnegative finance measures are valid."
                    if passed
                    else "One or more nonnegative finance measures are invalid."
                ),
                details={
                    "checked_columns": columns,
                    "invalid_numeric_columns": invalid_numeric,
                    "violations": violations,
                },
            ),
        )
    )


def _time_causality(context: _InvariantContext) -> _CheckResult:
    comparisons: list[tuple[str, str, bool]] = []

    information = context.columns.find(
        "information_time", "information_timestamp", "available_time", "asof_time"
    )
    signal = context.columns.find("signal_time", "signal_timestamp")
    execution = context.columns.find(
        "execution_time", "execution_timestamp", "fill_time", "fill_timestamp"
    )
    if information is not None and signal is not None:
        comparisons.append((information, signal, False))
    if signal is not None and execution is not None:
        comparisons.append((signal, execution, True))

    additional_pairs = (
        (("signal_date", "signal_time"), ("return_start_date", "return_start"), True),
        (("event_date", "publication_date"), ("return_start_date", "return_start"), True),
        (("publication_date", "published_at"), ("trade_date", "execution_time"), False),
        (("data_time", "data_timestamp"), ("trade_time", "execution_time"), False),
    )
    for left_aliases, right_aliases, strict in additional_pairs:
        left = context.columns.find(*left_aliases)
        right = context.columns.find(*right_aliases)
        if left is None or right is None:
            continue
        comparison = (left, right, strict)
        if comparison not in comparisons:
            comparisons.append(comparison)

    if not comparisons:
        return _CheckResult(skip_reason="no supported causal timestamp pair is present")

    parse_failure = pd.Series(False, index=context.frame.index)
    ordering_failure = pd.Series(False, index=context.frame.index)
    checked_pairs: list[dict[str, Any]] = []

    for left, right, strict in comparisons:
        left_values = pd.to_datetime(context.frame[left], errors="coerce", utc=True)
        right_values = pd.to_datetime(context.frame[right], errors="coerce", utc=True)
        parse_failure |= left_values.isna() | right_values.isna()
        ordering_failure |= left_values >= right_values if strict else left_values > right_values
        checked_pairs.append(
            {
                "left": left,
                "right": right,
                "relationship": "<" if strict else "<=",
            }
        )

    failure = parse_failure | ordering_failure
    passed = not bool(failure.any())
    return _CheckResult(
        evidence=(
            _evidence(
                name="time_causality",
                passed=passed,
                message=(
                    "Timestamps satisfy causal ordering."
                    if passed
                    else "Timestamp causality is violated or timestamps are invalid."
                ),
                details={
                    **_bad_row_details(failure),
                    "checked_pairs": checked_pairs,
                    "parse_failure_count": int(parse_failure.sum()),
                    "ordering_failure_count": int(ordering_failure.sum()),
                },
            ),
        )
    )


def _nav_reconciliation(context: _InvariantContext) -> _CheckResult:
    values, error = _numeric_inputs(
        context,
        {
            "nav": ("nav", "net_asset_value", "portfolio_value"),
            "cash": ("cash", "cash_balance"),
            "market_value": ("market_value", "marked_value", "positions_market_value"),
        },
    )
    if values is None:
        return _CheckResult(skip_reason=error)

    expected = values["cash"] + values["market_value"]
    actual = values["nav"]
    close = np.isclose(
        actual,
        expected,
        atol=context.config.absolute_tolerance,
        rtol=context.config.relative_tolerance,
    )
    difference = actual - expected
    passed = bool(np.all(close))
    return _CheckResult(
        evidence=(
            _evidence(
                name="nav_reconciliation",
                passed=passed,
                message=(
                    "NAV reconciles to cash plus market value."
                    if passed
                    else "NAV does not reconcile to cash plus market value."
                ),
                details={
                    **_bad_row_details(~close),
                    "maximum_absolute_difference": (
                        float(np.nanmax(np.abs(difference))) if len(difference) else 0.0
                    ),
                },
            ),
        )
    )


def _pnl_reconciliation(context: _InvariantContext) -> _CheckResult:
    values, error = _numeric_inputs(
        context,
        {
            "total_pnl": ("total_pnl", "pnl"),
            "realized_pnl": ("realized_pnl", "realised_pnl"),
            "unrealized_pnl": ("unrealized_pnl", "unrealised_pnl"),
        },
    )
    if values is None:
        return _CheckResult(skip_reason=error)

    expected = values["realized_pnl"] + values["unrealized_pnl"]
    close = np.isclose(
        values["total_pnl"],
        expected,
        atol=context.config.absolute_tolerance,
        rtol=context.config.relative_tolerance,
    )
    passed = bool(np.all(close))
    return _CheckResult(
        evidence=(
            _evidence(
                name="pnl_reconciliation",
                passed=passed,
                message=(
                    "Total P&L reconciles to realized plus unrealized P&L."
                    if passed
                    else "Total P&L does not reconcile to realized plus unrealized P&L."
                ),
                details=_bad_row_details(~close),
            ),
        )
    )


# ---------------------------------------------------------------------------
# Domain pack: derivatives-pricing
# ---------------------------------------------------------------------------

def _option_greek_bounds(context: _InvariantContext) -> _CheckResult:
    aliases = {
        "call_delta": ("call_delta", "delta_call"),
        "put_delta": ("put_delta", "delta_put"),
        "gamma": ("gamma",),
        "vega": ("vega",),
    }
    present: dict[str, tuple[str, pd.Series]] = {}
    for logical, names in aliases.items():
        column = context.columns.find(*names)
        if column is None:
            continue
        series, error = _numeric_series(context.frame, column)
        if error is not None or series is None:
            return _CheckResult(
                evidence=(
                    _evidence(
                        name="option_greek_bounds",
                        passed=False,
                        message="Option Greek output is not consistently numeric.",
                        details={"column": column, "error": error},
                    ),
                )
            )
        present[logical] = (column, series)

    if not present:
        return _CheckResult(skip_reason="no supported option Greek columns")

    tol = context.config.absolute_tolerance
    violations: dict[str, dict[str, Any]] = {}
    for logical, (column, series) in present.items():
        if logical == "call_delta":
            mask = (series < -tol) | (series > 1.0 + tol)
            bounds = [0.0, 1.0]
        elif logical == "put_delta":
            mask = (series < -1.0 - tol) | (series > tol)
            bounds = [-1.0, 0.0]
        else:
            mask = series < -tol
            bounds = [0.0, None]
        if bool(mask.any()):
            violations[column] = {
                **_bad_row_details(mask),
                "expected_bounds": bounds,
            }

    passed = not violations
    return _CheckResult(
        evidence=(
            _evidence(
                name="option_greek_bounds",
                passed=passed,
                message=(
                    "Option Greeks satisfy no-arbitrage sign and delta bounds."
                    if passed
                    else "One or more option Greeks violate no-arbitrage bounds."
                ),
                details={
                    "checked_columns": [column for column, _ in present.values()],
                    "violations": violations,
                },
            ),
        )
    )


def _put_call_parity(context: _InvariantContext) -> _CheckResult:
    mapping = {
        "call": ("call_price", "call_value"),
        "put": ("put_price", "put_value"),
        "spot": ("spot", "spot_price", "underlying_price"),
        "strike": ("strike", "strike_price"),
        "rate": ("risk_free_rate", "risk_free", "rate"),
        "time": ("time_to_maturity", "time_to_expiry", "maturity_years", "expiry_years"),
    }
    values, error = _numeric_inputs(context, mapping)
    if values is None:
        return _CheckResult(skip_reason=error)

    rate_column = context.columns.find(*mapping["rate"])
    assert rate_column is not None
    rate = _decimal_values(context, rate_column, values["rate"])

    dividend_column = context.columns.find("dividend_yield", "continuous_dividend_yield", "q")
    if dividend_column is not None:
        dividend, dividend_error = _numeric_series(context.frame, dividend_column)
        if dividend_error is not None or dividend is None:
            return _CheckResult(skip_reason=dividend_error)
        dividend = _decimal_values(context, dividend_column, dividend)
    else:
        dividend = pd.Series(0.0, index=context.frame.index)

    expected = (
        values["spot"] * np.exp(-dividend * values["time"])
        - values["strike"] * np.exp(-rate * values["time"])
    )
    observed = values["call"] - values["put"]
    atol = max(context.config.absolute_tolerance, 1e-3)
    close = np.isclose(observed, expected, atol=atol, rtol=context.config.relative_tolerance)
    passed = bool(np.all(close))
    residual = observed - expected
    return _CheckResult(
        evidence=(
            _evidence(
                name="put_call_parity",
                passed=passed,
                message=(
                    "Call and put prices satisfy put-call parity."
                    if passed
                    else "Call and put prices violate put-call parity."
                ),
                details={
                    **_bad_row_details(~close),
                    "tolerance": atol,
                    "maximum_absolute_residual": (
                        float(np.nanmax(np.abs(residual))) if len(residual) else 0.0
                    ),
                },
            ),
        )
    )


def _option_price_bounds(context: _InvariantContext) -> _CheckResult:
    spot_col = context.columns.find("spot", "spot_price", "underlying_price")
    strike_col = context.columns.find("strike", "strike_price")
    rate_col = context.columns.find("risk_free_rate", "risk_free", "rate")
    time_col = context.columns.find(
        "time_to_maturity", "time_to_expiry", "maturity_years", "expiry_years"
    )
    call_col = context.columns.find("call_price", "call_value")
    put_col = context.columns.find("put_price", "put_value")
    if None in {spot_col, strike_col, rate_col, time_col} or (call_col is None and put_col is None):
        return _CheckResult(skip_reason="required option bound columns are not present")

    required = [spot_col, strike_col, rate_col, time_col]
    numeric: dict[str, pd.Series] = {}
    for column in required + [c for c in (call_col, put_col) if c is not None]:
        assert column is not None
        series, error = _numeric_series(context.frame, column)
        if error is not None or series is None:
            return _CheckResult(skip_reason=error)
        numeric[column] = series

    rate = _decimal_values(context, rate_col, numeric[rate_col])
    discounted_strike = numeric[strike_col] * np.exp(-rate * numeric[time_col])
    tol = max(context.config.absolute_tolerance, 1e-8)
    violations: dict[str, dict[str, Any]] = {}

    if call_col is not None:
        call = numeric[call_col]
        lower = np.maximum(0.0, numeric[spot_col] - discounted_strike)
        mask = (call < lower - tol) | (call > numeric[spot_col] + tol)
        if bool(mask.any()):
            violations[call_col] = _bad_row_details(mask)

    if put_col is not None:
        put = numeric[put_col]
        lower = np.maximum(0.0, discounted_strike - numeric[spot_col])
        mask = (put < lower - tol) | (put > discounted_strike + tol)
        if bool(mask.any()):
            violations[put_col] = _bad_row_details(mask)

    passed = not violations
    return _CheckResult(
        evidence=(
            _evidence(
                name="option_price_bounds",
                passed=passed,
                message=(
                    "Option prices satisfy intrinsic and upper no-arbitrage bounds."
                    if passed
                    else "One or more option prices violate no-arbitrage bounds."
                ),
                details={"violations": violations},
            ),
        )
    )


# ---------------------------------------------------------------------------
# Domain pack: fixed-income
# ---------------------------------------------------------------------------

def _discount_factor_monotonicity(context: _InvariantContext) -> _CheckResult:
    df_col = context.columns.find("discount_factor", "df")
    tenor_col = context.columns.find("tenor", "maturity", "maturity_years", "time_to_maturity", "t")
    if df_col is None or tenor_col is None:
        return _CheckResult(skip_reason="discount factor and tenor columns are required")

    discount, error = _numeric_series(context.frame, df_col)
    if error is not None or discount is None:
        return _CheckResult(skip_reason=error)
    tenor, error = _numeric_series(context.frame, tenor_col)
    if error is not None or tenor is None:
        return _CheckResult(skip_reason=error)

    ordered = pd.DataFrame({"tenor": tenor, "discount": discount}).sort_values("tenor")
    differences = ordered["discount"].diff().dropna()
    tol = context.config.absolute_tolerance
    violation_mask = differences > tol
    passed = not bool(violation_mask.any())

    # Decreasing discount factors are an official sanity check, but negative-rate
    # regimes can legitimately invert this relationship. Keep it a warning rather
    # than a hard rejection unless stronger task-specific evidence is available.
    return _CheckResult(
        evidence=(
            _evidence(
                name="discount_factor_monotonicity",
                passed=passed,
                hard_failure=False,
                confidence=0.80,
                message=(
                    "Discount factors are non-increasing with maturity."
                    if passed
                    else "Discount factors increase with maturity; review rate conventions."
                ),
                details={"violation_count": int(violation_mask.sum())},
            ),
        )
    )


def _dv01_duration_consistency(context: _InvariantContext) -> _CheckResult:
    values, error = _numeric_inputs(
        context,
        {
            "dv01": ("dv01",),
            "modified_duration": ("modified_duration", "mod_duration", "moddur"),
            "price": ("price", "bond_price", "dirty_price"),
        },
    )
    if values is None:
        return _CheckResult(skip_reason=error)

    expected = values["modified_duration"] * values["price"] * 0.0001
    close = np.isclose(
        values["dv01"],
        expected,
        atol=max(context.config.absolute_tolerance, 1e-8),
        rtol=max(context.config.relative_tolerance, 1e-3),
    )
    passed = bool(np.all(close))
    return _CheckResult(
        evidence=(
            _evidence(
                name="dv01_duration_consistency",
                passed=passed,
                message=(
                    "DV01 is consistent with modified duration and price."
                    if passed
                    else "DV01 is inconsistent with modified duration and price."
                ),
                details=_bad_row_details(~close),
            ),
        )
    )


# ---------------------------------------------------------------------------
# Domain pack: credit
# ---------------------------------------------------------------------------

def _survival_probability_monotonicity(context: _InvariantContext) -> _CheckResult:
    survival_col = context.columns.find("survival_probability", "survival_prob")
    tenor_col = context.columns.find("tenor", "maturity", "maturity_years", "time", "t")
    if survival_col is None or tenor_col is None:
        return _CheckResult(skip_reason="survival probability and tenor columns are required")

    survival, error = _numeric_series(context.frame, survival_col)
    if error is not None or survival is None:
        return _CheckResult(skip_reason=error)
    tenor, error = _numeric_series(context.frame, tenor_col)
    if error is not None or tenor is None:
        return _CheckResult(skip_reason=error)

    ordered = pd.DataFrame({"tenor": tenor, "survival": survival}).sort_values("tenor")
    diff = ordered["survival"].diff().dropna()
    mask = diff > context.config.absolute_tolerance
    passed = not bool(mask.any())
    return _CheckResult(
        evidence=(
            _evidence(
                name="survival_probability_monotonicity",
                passed=passed,
                message=(
                    "Survival probability is non-increasing with tenor."
                    if passed
                    else "Survival probability increases at one or more later tenors."
                ),
                details={"violation_count": int(mask.sum())},
            ),
        )
    )


# ---------------------------------------------------------------------------
# Domain pack: factor-research
# ---------------------------------------------------------------------------

def _information_coefficient_bounds(context: _InvariantContext) -> _CheckResult:
    columns = context.columns.matching(
        lambda name: name in {"ic", "information_coefficient"} or name.endswith("_ic")
    )
    if not columns:
        return _CheckResult(skip_reason="no information coefficient column")

    tol = context.config.absolute_tolerance
    violations: dict[str, dict[str, Any]] = {}
    for column in columns:
        series, error = _numeric_series(context.frame, column)
        if error is not None or series is None:
            return _CheckResult(skip_reason=error)
        mask = (series < -1.0 - tol) | (series > 1.0 + tol)
        if bool(mask.any()):
            violations[column] = _bad_row_details(mask)

    passed = not violations
    return _CheckResult(
        evidence=(
            _evidence(
                name="information_coefficient_bounds",
                passed=passed,
                message=(
                    "Information coefficients lie within [-1, 1]."
                    if passed
                    else "Information coefficient is outside [-1, 1]."
                ),
                details={"violations": violations},
            ),
        )
    )


def _dollar_neutrality(context: _InvariantContext) -> _CheckResult:
    weight_col = context.columns.find("weight", "portfolio_weight", "factor_weight")
    if weight_col is None:
        return _CheckResult(skip_reason="no portfolio weight column")
    weights, error = _numeric_series(context.frame, weight_col)
    if error is not None or weights is None:
        return _CheckResult(skip_reason=error)

    group_col = context.columns.find("date", "signal_date", "rebalance_date", "timestamp")
    if group_col is not None:
        grouped = pd.DataFrame({"group": context.frame[group_col], "weight": weights}).groupby(
            "group", dropna=False
        )["weight"].sum()
    else:
        grouped = pd.Series([float(weights.sum())])

    tolerance = max(context.config.absolute_tolerance, 1e-6)
    mask = np.abs(grouped.to_numpy(dtype=float)) > tolerance
    passed = not bool(mask.any())
    return _CheckResult(
        evidence=(
            _evidence(
                name="dollar_neutrality",
                passed=passed,
                message=(
                    "Portfolio weights are dollar-neutral."
                    if passed
                    else "Portfolio weights are not dollar-neutral."
                ),
                details={
                    "group_count": int(len(grouped)),
                    "violation_count": int(mask.sum()),
                    "maximum_absolute_net_weight": (
                        float(np.max(np.abs(grouped.to_numpy(dtype=float)))) if len(grouped) else 0.0
                    ),
                },
            ),
        )
    )


# ---------------------------------------------------------------------------
# Domain pack: backtesting
# ---------------------------------------------------------------------------

def _portfolio_compounding(context: _InvariantContext) -> _CheckResult:
    value_col = context.columns.find("portfolio_value", "equity_curve", "strategy_value")
    return_col = context.columns.find("portfolio_return", "strategy_return")
    if value_col is None or return_col is None:
        return _CheckResult(skip_reason="portfolio value and portfolio return columns are required")

    values, error = _numeric_series(context.frame, value_col)
    if error is not None or values is None:
        return _CheckResult(skip_reason=error)
    returns, error = _numeric_series(context.frame, return_col)
    if error is not None or returns is None:
        return _CheckResult(skip_reason=error)

    frame = pd.DataFrame({"value": values, "return": returns})
    time_col = context.columns.find("date", "timestamp", "trade_date")
    if time_col is not None:
        parsed = pd.to_datetime(context.frame[time_col], errors="coerce", utc=True)
        if bool(parsed.isna().any()):
            return _CheckResult(skip_reason=f"could not parse backtest time column: {time_col}")
        frame = frame.assign(_time=parsed).sort_values("_time")

    if len(frame) < 2:
        return _CheckResult(skip_reason="at least two backtest rows are required")

    expected = frame["value"].shift(1) * (1.0 + frame["return"])
    actual = frame["value"]
    close = np.isclose(
        actual.iloc[1:],
        expected.iloc[1:],
        atol=max(context.config.absolute_tolerance, 1e-8),
        rtol=max(context.config.relative_tolerance, 1e-6),
    )
    passed = bool(np.all(close))
    return _CheckResult(
        evidence=(
            _evidence(
                name="portfolio_compounding",
                passed=passed,
                message=(
                    "Portfolio values reconcile to compounded portfolio returns."
                    if passed
                    else "Portfolio values do not reconcile to compounded portfolio returns."
                ),
                details=_bad_row_details(~close),
            ),
        )
    )


# ---------------------------------------------------------------------------
# Domain pack: risk-management
# ---------------------------------------------------------------------------

def _risk_measure_ordering(context: _InvariantContext) -> _CheckResult:
    var_col = context.columns.find("var", "value_at_risk")
    es_col = context.columns.find("expected_shortfall", "es", "cvar")
    evidence_items: list[VerificationEvidence] = []
    found = False

    if var_col is not None and es_col is not None:
        var, error = _numeric_series(context.frame, var_col)
        if error is not None or var is None:
            return _CheckResult(skip_reason=error)
        es, error = _numeric_series(context.frame, es_col)
        if error is not None or es is None:
            return _CheckResult(skip_reason=error)
        found = True

        tol = context.config.absolute_tolerance
        if bool(((var >= -tol) & (es >= -tol)).all()):
            # Positive-loss convention: larger number means more risk.
            mask = es + tol < var
            convention = "positive_loss"
        elif bool(((var <= tol) & (es <= tol)).all()):
            # Signed-P&L convention: worse tail values are more negative.
            mask = es - tol > var
            convention = "signed_pnl"
        else:
            return _CheckResult(
                skip_reason="VaR/ES sign convention is mixed or ambiguous"
            )

        passed = not bool(mask.any())
        evidence_items.append(
            _evidence(
                name="expected_shortfall_vs_var",
                passed=passed,
                message=(
                    "Expected shortfall is at least as conservative as VaR."
                    if passed
                    else "Expected shortfall is less conservative than VaR."
                ),
                details={
                    **_bad_row_details(mask),
                    "inferred_convention": convention,
                },
            )
        )

    wide_var: list[tuple[float, str, pd.Series]] = []
    for column in context.frame.columns:
        normalised = _normalise_name(str(column))
        match = re.fullmatch(r"(?:var|value_at_risk)_?(\d+(?:_\d+)?)", normalised)
        if match is None:
            continue
        confidence = float(match.group(1).replace("_", "."))
        if confidence > 1.0:
            confidence /= 100.0
        series, error = _numeric_series(context.frame, str(column))
        if error is None and series is not None:
            wide_var.append((confidence, str(column), series))

    if len(wide_var) >= 2:
        found = True
        wide_var.sort(key=lambda item: item[0])
        all_values = pd.concat([item[2] for item in wide_var], ignore_index=True)
        tol = context.config.absolute_tolerance
        if bool((all_values >= -tol).all()):
            direction = "increasing"
        elif bool((all_values <= tol).all()):
            direction = "decreasing"
        else:
            direction = "ambiguous"

        if direction != "ambiguous":
            violation_count = 0
            checked_pairs: list[list[str]] = []
            for (_, left_name, left), (_, right_name, right) in zip(wide_var, wide_var[1:]):
                if direction == "increasing":
                    mask = right + tol < left
                else:
                    mask = right - tol > left
                violation_count += int(mask.sum())
                checked_pairs.append([left_name, right_name])
            passed = violation_count == 0
            evidence_items.append(
                _evidence(
                    name="var_confidence_monotonicity",
                    passed=passed,
                    message=(
                        "VaR becomes no less conservative as confidence increases."
                        if passed
                        else "VaR becomes less conservative at a higher confidence level."
                    ),
                    details={
                        "violation_count": violation_count,
                        "checked_pairs": checked_pairs,
                        "inferred_direction": direction,
                    },
                )
            )

    if not found:
        return _CheckResult(skip_reason="no comparable VaR/ES outputs are present")
    if not evidence_items:
        return _CheckResult(skip_reason="risk-measure sign convention is ambiguous")
    return _CheckResult(evidence=tuple(evidence_items))


# ---------------------------------------------------------------------------
# Domain pack: microstructure
# ---------------------------------------------------------------------------

def _bid_ask_ordering(context: _InvariantContext) -> _CheckResult:
    values, error = _numeric_inputs(
        context,
        {
            "bid": ("bid", "bid_price", "best_bid"),
            "ask": ("ask", "ask_price", "best_ask"),
        },
    )
    if values is None:
        return _CheckResult(skip_reason=error)
    mask = values["bid"] > values["ask"] + context.config.absolute_tolerance
    passed = not bool(mask.any())
    return _CheckResult(
        evidence=(
            _evidence(
                name="bid_ask_ordering",
                passed=passed,
                message=(
                    "Bid prices do not exceed ask prices."
                    if passed
                    else "One or more observations have bid above ask."
                ),
                details=_bad_row_details(mask),
            ),
        )
    )


def _vwap_range(context: _InvariantContext) -> _CheckResult:
    values, error = _numeric_inputs(
        context,
        {
            "vwap": ("vwap", "volume_weighted_average_price"),
            "low": ("low", "day_low", "low_price"),
            "high": ("high", "day_high", "high_price"),
        },
    )
    if values is None:
        return _CheckResult(skip_reason=error)
    tol = context.config.absolute_tolerance
    mask = (values["vwap"] < values["low"] - tol) | (values["vwap"] > values["high"] + tol)
    passed = not bool(mask.any())
    return _CheckResult(
        evidence=(
            _evidence(
                name="vwap_range",
                passed=passed,
                message=(
                    "VWAP lies within the observed low-high range."
                    if passed
                    else "VWAP lies outside the observed low-high range."
                ),
                details=_bad_row_details(mask),
            ),
        )
    )


def _execution_quantity_reconciliation(context: _InvariantContext) -> _CheckResult:
    executed_col = context.columns.find(
        "executed_quantity", "trade_quantity", "shares_sold", "scheduled_trade"
    )
    target_col = context.columns.find("target_quantity", "initial_position", "target_shares")
    if executed_col is None or target_col is None:
        return _CheckResult(skip_reason="executed and target quantity columns are required")
    executed, error = _numeric_series(context.frame, executed_col)
    if error is not None or executed is None:
        return _CheckResult(skip_reason=error)
    target, error = _numeric_series(context.frame, target_col)
    if error is not None or target is None:
        return _CheckResult(skip_reason=error)

    unique_target = target.dropna().unique()
    if len(unique_target) != 1:
        return _CheckResult(skip_reason="target quantity is not constant across the trajectory")
    expected = float(unique_target[0])
    actual = float(executed.sum())
    close = bool(
        np.isclose(
            actual,
            expected,
            atol=max(context.config.absolute_tolerance, 1e-8),
            rtol=context.config.relative_tolerance,
        )
    )
    return _CheckResult(
        evidence=(
            _evidence(
                name="execution_quantity_reconciliation",
                passed=close,
                message=(
                    "Executed quantity reconciles to the target quantity."
                    if close
                    else "Executed quantity does not reconcile to the target quantity."
                ),
                details={"executed_total": actual, "target_total": expected},
            ),
        )
    )


# ---------------------------------------------------------------------------
# Domain pack: fx
# ---------------------------------------------------------------------------

def _covered_interest_parity(context: _InvariantContext) -> _CheckResult:
    mapping = {
        "spot": ("spot", "spot_rate", "fx_spot"),
        "forward": ("forward", "forward_rate", "fx_forward"),
        "domestic_rate": ("domestic_rate", "r_domestic", "rd"),
        "foreign_rate": ("foreign_rate", "r_foreign", "rf"),
        "time": ("time_to_maturity", "maturity_years", "tenor_years", "t"),
    }
    values, error = _numeric_inputs(context, mapping)
    if values is None:
        return _CheckResult(skip_reason=error)

    domestic_col = context.columns.find(*mapping["domestic_rate"])
    foreign_col = context.columns.find(*mapping["foreign_rate"])
    assert domestic_col is not None and foreign_col is not None
    rd = _decimal_values(context, domestic_col, values["domestic_rate"])
    rf = _decimal_values(context, foreign_col, values["foreign_rate"])
    expected = values["spot"] * np.exp((rd - rf) * values["time"])
    close = np.isclose(
        values["forward"],
        expected,
        atol=max(context.config.absolute_tolerance, 1e-8),
        rtol=max(context.config.relative_tolerance, 1e-5),
    )
    passed = bool(np.all(close))
    residual = values["forward"] - expected
    return _CheckResult(
        evidence=(
            _evidence(
                name="covered_interest_parity",
                passed=passed,
                message=(
                    "FX forwards satisfy covered interest parity."
                    if passed
                    else "FX forwards violate covered interest parity."
                ),
                details={
                    **_bad_row_details(~close),
                    "maximum_absolute_residual": (
                        float(np.nanmax(np.abs(residual))) if len(residual) else 0.0
                    ),
                },
            ),
        )
    )


def _triangular_fx_consistency(context: _InvariantContext) -> _CheckResult:
    values, error = _numeric_inputs(
        context,
        {
            "eurusd": ("eurusd", "eur_usd"),
            "usdjpy": ("usdjpy", "usd_jpy"),
            "eurjpy": ("eurjpy", "eur_jpy"),
        },
    )
    if values is None:
        return _CheckResult(skip_reason=error)
    ratio = values["eurusd"] * values["usdjpy"] / values["eurjpy"]
    close = np.isclose(
        ratio,
        1.0,
        atol=max(context.config.absolute_tolerance, 1e-8),
        rtol=max(context.config.relative_tolerance, 1e-5),
    )
    passed = bool(np.all(close))
    return _CheckResult(
        evidence=(
            _evidence(
                name="triangular_fx_consistency",
                passed=passed,
                message=(
                    "FX cross rates are triangularly consistent."
                    if passed
                    else "FX cross rates contain a triangular inconsistency."
                ),
                details=_bad_row_details(~close),
            ),
        )
    )


# ---------------------------------------------------------------------------
# Domain pack: nlp-on-finance
# ---------------------------------------------------------------------------

def _sentiment_bounds(context: _InvariantContext) -> _CheckResult:
    columns = context.columns.matching(
        lambda name: name in {"sentiment", "sentiment_score", "finance_sentiment"}
        or name.endswith("_sentiment_score")
    )
    if not columns:
        return _CheckResult(skip_reason="no sentiment score columns")

    conventions = _convention_text(context.config)
    zero_one = any(
        marker in conventions
        for marker in ("sentiment [0,1]", "sentiment [0, 1]", "sentiment 0 to 1")
    )
    lower, upper = (0.0, 1.0) if zero_one else (-1.0, 1.0)
    tol = context.config.absolute_tolerance
    violations: dict[str, dict[str, Any]] = {}
    for column in columns:
        series, error = _numeric_series(context.frame, column)
        if error is not None or series is None:
            return _CheckResult(skip_reason=error)
        mask = (series < lower - tol) | (series > upper + tol)
        if bool(mask.any()):
            violations[column] = _bad_row_details(mask)

    passed = not violations
    return _CheckResult(
        evidence=(
            _evidence(
                name="sentiment_bounds",
                passed=passed,
                message=(
                    "Sentiment scores satisfy the declared range."
                    if passed
                    else "One or more sentiment scores are outside the declared range."
                ),
                details={"expected_bounds": [lower, upper], "violations": violations},
            ),
        )
    )


def _entity_count_integrality(context: _InvariantContext) -> _CheckResult:
    columns = context.columns.matching(
        lambda name: (
            name in {"entity_count", "named_entity_count", "risk_factor_count"}
            or name.endswith("_entity_count")
        )
    )
    if not columns:
        return _CheckResult(skip_reason="no named-entity count columns")

    tol = context.config.absolute_tolerance
    violations: dict[str, dict[str, Any]] = {}
    for column in columns:
        series, error = _numeric_series(context.frame, column)
        if error is not None or series is None:
            return _CheckResult(skip_reason=error)
        mask = (series < -tol) | (np.abs(series - np.round(series)) > tol)
        if bool(mask.any()):
            violations[column] = _bad_row_details(mask)

    passed = not violations
    return _CheckResult(
        evidence=(
            _evidence(
                name="entity_count_integrality",
                passed=passed,
                message=(
                    "Named-entity counts are nonnegative integers."
                    if passed
                    else "Named-entity counts must be nonnegative integers."
                ),
                details={"violations": violations},
            ),
        )
    )


# ---------------------------------------------------------------------------
# Cross-domain pack coverage
# ---------------------------------------------------------------------------

def _infer_domains(context: _InvariantContext) -> set[str]:
    inferred: set[str] = set()
    c = context.columns
    if c.has("call_price", "put_price", "call_delta", "put_delta", "gamma", "vega"):
        inferred.add("derivatives-pricing")
    if c.has("discount_factor", "dv01", "modified_duration"):
        inferred.add("fixed-income")
    if c.has("survival_probability", "hazard_rate", "cva"):
        inferred.add("credit")
    if c.has("information_coefficient", "ic", "factor_weight"):
        inferred.add("factor-research")
    if c.has("portfolio_return", "strategy_return", "equity_curve"):
        inferred.add("backtesting")
    if c.has("var", "value_at_risk", "expected_shortfall", "cvar") or c.matching(
        lambda name: name.startswith("var_")
    ):
        inferred.add("risk-management")
    if c.has("bid_price", "ask_price", "vwap", "effective_spread"):
        inferred.add("microstructure")
    if c.has("eurusd", "usd_jpy", "fx_forward") or (
        c.has("spot_rate")
        and c.has("forward_rate")
        and c.has("domestic_rate")
        and c.has("foreign_rate")
    ):
        inferred.add("fx")
    if c.has("sentiment_score", "entity_count", "risk_factor_count"):
        inferred.add("nlp-on-finance")
    return inferred


def _cross_domain_pack_coverage(context: _InvariantContext) -> _CheckResult:
    if _canonical_domain(context.config.category) != "cross-domain":
        return _CheckResult(skip_reason="category is not cross-domain")

    declared = {
        domain
        for value in context.config.review_packs
        if (domain := _canonical_domain(value)) in _DOMAIN_IDS
    }
    inferred = _infer_domains(context)
    active = sorted(declared | inferred)
    passed = len(active) >= 2
    return _CheckResult(
        evidence=(
            _evidence(
                name="cross_domain_pack_coverage",
                passed=passed,
                hard_failure=False,
                confidence=0.95,
                message=(
                    "Cross-domain output activates at least two domain packs."
                    if passed
                    else "Cross-domain output exposes fewer than two identifiable domain packs."
                ),
                details={
                    "declared_domain_packs": sorted(declared),
                    "inferred_domain_packs": sorted(inferred),
                    "active_domain_packs": active,
                },
            ),
        )
    )


_CHECKS: tuple[_InvariantCheck, ...] = (
    _InvariantCheck(
        name="finite_numeric_values",
        evaluator=_finite_numeric_values,
        always_active=True,
    ),
    _InvariantCheck(
        name="probability_bounds",
        evaluator=_probability_bounds,
        always_active=True,
    ),
    _InvariantCheck(
        name="nonnegative_quantities",
        evaluator=_nonnegative_quantities,
        always_active=True,
    ),
    _InvariantCheck(
        name="time_causality",
        evaluator=_time_causality,
        review_packs=frozenset({
            "data-causality", "factor-research", "backtesting", "microstructure", "nlp-finance"
        }),
        categories=frozenset({"factor-research", "backtesting", "microstructure", "nlp-on-finance"}),
    ),
    _InvariantCheck(
        name="nav_reconciliation",
        evaluator=_nav_reconciliation,
        review_packs=frozenset({"accounting", "backtesting", "microstructure"}),
        categories=frozenset({"backtesting", "microstructure"}),
    ),
    _InvariantCheck(
        name="pnl_reconciliation",
        evaluator=_pnl_reconciliation,
        review_packs=frozenset({"accounting", "backtesting", "microstructure"}),
        categories=frozenset({"backtesting", "microstructure"}),
    ),
    _InvariantCheck(
        name="option_greek_bounds",
        evaluator=_option_greek_bounds,
        categories=frozenset({"derivatives-pricing"}),
    ),
    _InvariantCheck(
        name="put_call_parity",
        evaluator=_put_call_parity,
        categories=frozenset({"derivatives-pricing"}),
    ),
    _InvariantCheck(
        name="option_price_bounds",
        evaluator=_option_price_bounds,
        categories=frozenset({"derivatives-pricing"}),
    ),
    _InvariantCheck(
        name="discount_factor_monotonicity",
        evaluator=_discount_factor_monotonicity,
        categories=frozenset({"fixed-income"}),
    ),
    _InvariantCheck(
        name="dv01_duration_consistency",
        evaluator=_dv01_duration_consistency,
        categories=frozenset({"fixed-income"}),
    ),
    _InvariantCheck(
        name="survival_probability_monotonicity",
        evaluator=_survival_probability_monotonicity,
        categories=frozenset({"credit"}),
    ),
    _InvariantCheck(
        name="information_coefficient_bounds",
        evaluator=_information_coefficient_bounds,
        categories=frozenset({"factor-research"}),
    ),
    _InvariantCheck(
        name="dollar_neutrality",
        evaluator=_dollar_neutrality,
        categories=frozenset({"factor-research"}),
    ),
    _InvariantCheck(
        name="portfolio_compounding",
        evaluator=_portfolio_compounding,
        categories=frozenset({"backtesting"}),
    ),
    _InvariantCheck(
        name="risk_measure_ordering",
        evaluator=_risk_measure_ordering,
        categories=frozenset({"risk-management"}),
    ),
    _InvariantCheck(
        name="bid_ask_ordering",
        evaluator=_bid_ask_ordering,
        categories=frozenset({"microstructure"}),
    ),
    _InvariantCheck(
        name="vwap_range",
        evaluator=_vwap_range,
        categories=frozenset({"microstructure"}),
    ),
    _InvariantCheck(
        name="execution_quantity_reconciliation",
        evaluator=_execution_quantity_reconciliation,
        categories=frozenset({"microstructure"}),
    ),
    _InvariantCheck(
        name="covered_interest_parity",
        evaluator=_covered_interest_parity,
        categories=frozenset({"fx"}),
    ),
    _InvariantCheck(
        name="triangular_fx_consistency",
        evaluator=_triangular_fx_consistency,
        categories=frozenset({"fx"}),
    ),
    _InvariantCheck(
        name="sentiment_bounds",
        evaluator=_sentiment_bounds,
        categories=frozenset({"nlp-on-finance"}),
    ),
    _InvariantCheck(
        name="entity_count_integrality",
        evaluator=_entity_count_integrality,
        categories=frozenset({"nlp-on-finance"}),
    ),
    _InvariantCheck(
        name="cross_domain_pack_coverage",
        evaluator=_cross_domain_pack_coverage,
        categories=frozenset({"cross-domain"}),
    ),
)


def invariant_registry_snapshot() -> tuple[dict[str, Any], ...]:
    """Deterministic, side-effect-free registry view for tests and telemetry."""
    return tuple(
        {
            "name": check.name,
            "review_packs": tuple(sorted(check.review_packs)),
            "categories": tuple(sorted(check.categories)),
            "always_active": check.always_active,
        }
        for check in _CHECKS
    )


def _json_to_frame(payload: Any) -> pd.DataFrame:
    if isinstance(payload, list):
        return pd.DataFrame(payload)

    if isinstance(payload, dict):
        values = list(payload.values())
        if values and all(isinstance(value, dict) for value in values):
            key_sets = [set(value) for value in values]
            if all(keys == key_sets[0] for keys in key_sets[1:]):
                frame = pd.DataFrame(payload)
                frame.insert(0, "key", frame.index.astype(str))
                return frame.reset_index(drop=True)
        if values and all(isinstance(value, list) for value in values):
            lengths = {len(value) for value in values}
            if len(lengths) == 1:
                return pd.DataFrame(payload)
        return pd.DataFrame([payload])

    return pd.DataFrame({"value": [payload]})


def _load_output(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8", errors="strict"))
        return _json_to_frame(payload)
    raise ValueError(f"Unsupported invariant-check format: {suffix}")


def evaluate_quant_invariants(
    *,
    output_path: Path,
    config: QuantInvariantConfig,
) -> list[VerificationEvidence]:
    """Run applicable checks without using benchmark-only infrastructure."""
    output_path = output_path.resolve()

    if not output_path.exists():
        return [
            _evidence(
                name="quant_invariants_skipped",
                passed=True,
                message="Quantitative checks skipped because output is absent.",
                details={"output_path": str(output_path), "skip_reason": "output absent"},
            )
        ]

    try:
        frame = _load_output(output_path)
    except Exception as exc:
        return [
            _evidence(
                name="quant_invariants_skipped",
                passed=True,
                message="Quantitative checks skipped because output is unreadable.",
                details={"error": str(exc), "skip_reason": "output unreadable"},
            )
        ]

    context = _InvariantContext(frame=frame, columns=_Columns(frame), config=config)
    evidence: list[VerificationEvidence] = []
    applied_checks: list[str] = []
    skipped_checks: dict[str, str] = {}

    for check in _CHECKS:
        if not check.is_active(config):
            continue
        try:
            result = check.evaluator(context)
        except Exception as exc:
            evidence.append(
                _evidence(
                    name=f"{check.name}_engine_error",
                    passed=False,
                    hard_failure=False,
                    confidence=0.0,
                    provenance="invariant_engine",
                    message=f"Invariant check {check.name} could not be completed.",
                    details={"error": str(exc)},
                )
            )
            skipped_checks[check.name] = "check raised an internal error"
            continue

        if result.evidence:
            evidence.extend(result.evidence)
            applied_checks.append(check.name)
        else:
            skipped_checks[check.name] = result.skip_reason or "not applicable"

    active_domains = sorted(
        {
            domain
            for value in (config.category, *config.review_packs)
            if (domain := _canonical_domain(value)) in _DOMAIN_IDS
        }
        | (_infer_domains(context) if _canonical_domain(config.category) == "cross-domain" else set())
    )

    evidence.append(
        _evidence(
            name="quant_invariant_coverage",
            passed=True,
            message="Quantitative invariant coverage was recorded.",
            details={
                "output_path": str(output_path),
                "category": config.category,
                "review_packs": list(config.review_packs),
                "active_domain_packs": active_domains,
                "applied_checks": applied_checks,
                "skipped_checks": skipped_checks,
                "declared_invariants": list(config.declared_invariants),
                "registry_order": [check.name for check in _CHECKS],
            },
        )
    )
    return evidence
