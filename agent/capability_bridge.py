from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Sequence

from agent.offline_router import (
    TaskFingerprint,
    inspect_task_fingerprint,
)
from agent.qf_primitives import PRIMITIVE_API_CATALOG


MAX_SELECTED_CAPABILITIES = 3
MIN_SEMANTIC_SCORE = 3
MIN_TOTAL_SCORE = 5
MAX_CAPABILITY_PROMPT_GROWTH_CHARS = 6000
CANDIDATE_LIBRARY_EXPORTS: dict[str, tuple[str, ...]] = {
    "execution.py": (
        "two_way_turnover",
        "one_way_turnover",
        "transaction_cost",
        "net_return_after_cost",
        "lag_positions",
        "rowwise_portfolio_returns",
    ),
    "ledger.py": (
        "TradeUpdate",
        "proportional_trade_cost",
        "apply_trade",
        "mark_to_market",
        "position_pnl",
    ),
    "microstructure.py": (
        "first_index_at_or_after",
        "half_open_window_indices",
        "participation_capped_quantity",
        "exact_synchronized_venue_rows",
    ),
    "volume_scheduling.py": (
        "normalize_profile",
        "mean_volume_profile",
        "median_volume_profile",
        "exponentially_weighted_volume_profile",
        "winsorized_mean_volume_profile",
        "profile_r_squared",
        "largest_remainder_allocation",
    ),
    "invariants.py": (
        "finite_numeric_mask",
        "probability_bounds_mask",
        "is_positive_semidefinite",
        "weight_sum_error",
        "causal_order_mask",
        "reconciliation_close",
    ),
    "panel.py": (
        "zscore_ddof0",
        "equal_count_bucket_labels",
        "spearman_rank_correlation",
        "residualize_against_controls",
        "beta_neutral_projection",
    ),
    "statistics.py": (
        "OlsFit",
        "GrsResult",
        "ols_with_inference",
        "optimal_newey_west_lag",
        "newey_west_covariance",
        "newey_west_mean_tstat",
        "durbin_watson",
        "variance_inflation_factors",
        "grs_joint_alpha_test",
        "rolling_ols_coefficient",
        "normalize_cross_section",
    ),
    "derivatives.py": (
        "TwoAssetCalibration",
        "calibrate_two_asset_gbm",
        "exchange_volatility",
        "margrabe_price",
        "kirk_spread_price",
        "monte_carlo_spread_prices",
    ),
    "event_study.py": (
        "EventStudySpec",
        "EventRecord",
        "prepare_event_study_returns",
        "build_event_records",
        "corrado_rank_statistics",
        "average_pairwise_residual_correlation",
        "kolari_pynnonen_statistics",
    ),
    "fixed_income.py": (
        "BootstrappedCurve",
        "discount_factor_from_zero",
        "interpolate_zero_rate",
        "bootstrap_par_curve",
        "reprice_par_bond",
    ),
    "portfolio.py": (
        "MomentumSpec",
        "clean_price_panel",
        "simple_returns_from_prices",
        "cross_sectional_momentum_path",
        "summarize_return_path",
    ),
    "risk.py": (
        "aligned_log_returns",
        "historical_var_es",
        "normal_var_es",
        "student_t_var_es",
        "exponential_weights",
        "weighted_quantile",
        "age_weighted_var_es",
        "ewma_covariance",
        "ewma_portfolio_volatility",
        "overlapping_horizon_returns",
        "expanding_historical_backtest",
    ),
    "volatility.py": (
        "OhlcVarianceEstimate",
        "validate_ohlc",
        "ohlc_variance_estimators",
        "annualized_volatility",
        "rolling_ohlc_estimators",
        "estimator_arrays",
        "efficiency_ratios",
    ),
}
CANDIDATE_LIBRARY_FILENAMES = (
    "__init__.py",
    *CANDIDATE_LIBRARY_EXPORTS,
)

_PLANNER_INSPECTION_KEYS = (
    "type",
    "rows",
    "sample_rows_used",
    "sampled",
    "columns",
    "dtypes",
    "missing",
    "delimiter",
    "keys",
    "truncated_keys",
    "length",
    "chars_at_least",
    "truncated",
    "member_count",
    "members_truncated",
    "extension_counts",
    "total_uncompressed_bytes",
    "total_compressed_bytes",
    "suffix",
    "error",
)


WeightedTerms = tuple[tuple[int, tuple[str, ...]], ...]
WeightedSchemas = tuple[tuple[frozenset[str], int], ...]


@dataclass(frozen=True)
class CapabilityDescriptor:
    """A general algorithm library that House-authored code may compose."""

    capability_id: str
    import_path: str
    summary: str
    api_signatures: tuple[str, ...]
    primitive_names: tuple[str, ...]
    semantic_terms: WeightedTerms
    csv_schemas: WeightedSchemas = ()
    json_schemas: WeightedSchemas = ()
    minimum_price_panels: int = 0

    def to_prompt_dict(self) -> dict[str, object]:
        return {
            "capability": self.capability_id,
            "import": self.import_path,
            "purpose": self.summary,
            "apis": list(self.api_signatures),
        }


@dataclass(frozen=True)
class RankedCapability:
    descriptor: CapabilityDescriptor
    semantic_score: int
    data_score: int
    matched_evidence: tuple[str, ...]

    @property
    def total_score(self) -> int:
        return self.semantic_score + self.data_score

    def to_prompt_dict(self) -> dict[str, object]:
        payload = self.descriptor.to_prompt_dict()
        payload["selection_evidence"] = list(self.matched_evidence)
        return payload


CAPABILITY_CATALOG: tuple[CapabilityDescriptor, ...] = (
    CapabilityDescriptor(
        capability_id="causal-portfolio-execution",
        import_path="offline_common.execution",
        summary=(
            "Portfolio weight turnover, linear transaction costs, "
            "causal position lagging, and row-wise weighted returns."
        ),
        api_signatures=(
            "two_way_turnover(previous, current) -> float",
            "one_way_turnover(previous, current) -> float",
            "transaction_cost(turnover, rate) -> float",
            "net_return_after_cost(gross_return, turnover, rate) -> float",
            "lag_positions(weights, *, periods=1, fill_value=0.0) -> ndarray",
            "rowwise_portfolio_returns(weights, asset_returns) -> ndarray",
        ),
        primitive_names=(),
        semantic_terms=(
            (
                6,
                (
                    "one-day execution lag",
                    "one day execution lag",
                    "execution lag",
                    "one-way turnover",
                    "one way turnover",
                ),
            ),
            (
                5,
                (
                    "transaction cost on rebalance",
                    "transaction costs on rebalance",
                    "sum of absolute weight changes",
                    "turnover cap",
                    "turnover-capped",
                ),
            ),
            (
                5,
                (
                    "transaction cost",
                    "transaction costs",
                ),
            ),
            (
                4,
                (
                    "portfolio turnover",
                    "rebalance turnover",
                    "weight changes",
                ),
            ),
        ),
    ),
    CapabilityDescriptor(
        capability_id="cash-position-ledger",
        import_path="offline_common.ledger",
        summary=(
            "Signed trade updates for positions and cash, proportional "
            "buy/sell costs, portfolio NAV from cash plus holdings, and "
            "position P&L reconciliation."
        ),
        api_signatures=(
            "proportional_trade_cost(quantity_change, price, *, buy_rate=0.0, sell_rate=None) -> float",
            "apply_trade(*, cash, quantity, quantity_change, price, buy_rate=0.0, sell_rate=None) -> TradeUpdate",
            "mark_to_market(cash, quantities, prices) -> float",
            "position_pnl(quantity, entry_price, exit_price, *, total_cost=0.0) -> float",
        ),
        primitive_names=(),
        semantic_terms=(
            (
                6,
                (
                    "terminal cash balance",
                    "cash balance after liquidation",
                    "portfolio value = cash",
                    "cash + shares",
                ),
            ),
            (
                6,
                (
                    "cash, equity, and daily p&l",
                    "cash equity and daily p&l",
                ),
            ),
            (
                5,
                (
                    "daily portfolio value",
                    "current_portfolio_value = cash",
                ),
            ),
        ),
    ),
    CapabilityDescriptor(
        capability_id="microstructure-temporal-alignment",
        import_path="offline_common.microstructure",
        summary=(
            "Sorted timestamp lookup, half-open event windows, "
            "participation-capped child sizing, and exact multi-venue "
            "snapshot synchronization."
        ),
        api_signatures=(
            "first_index_at_or_after(timestamps, target) -> int",
            "half_open_window_indices(timestamps, *, start, end) -> tuple[int, int]",
            "participation_capped_quantity(remaining_quantity, market_volume, participation_cap, max_child_quantity) -> float",
            "exact_synchronized_venue_rows(frame, *, group_columns, venue_column, required_venues) -> DataFrame",
        ),
        primitive_names=(),
        semantic_terms=(
            (
                6,
                (
                    "participation-capped",
                    "participation capped",
                    "first quote at or after",
                ),
            ),
            (
                6,
                (
                    "synchronized snapshot",
                    "synchronized snapshots",
                    "synchronised snapshot",
                    "synchronised snapshots",
                ),
            ),
            (
                5,
                (
                    "bucket volume",
                    "exact timestamp alignment",
                    "half-open time window",
                    "half open time window",
                ),
            ),
        ),
    ),
    CapabilityDescriptor(
        capability_id="intraday-volume-scheduling",
        import_path="offline_common.volume_scheduling",
        summary=(
            "Normalized intraday volume profiles, historical mean/median/"
            "exponentially weighted/winsorized profile estimation, profile "
            "R-squared, and deterministic integer allocation."
        ),
        api_signatures=(
            "normalize_profile(values) -> ndarray",
            "mean_volume_profile(history) -> ndarray",
            "median_volume_profile(history) -> ndarray",
            "exponentially_weighted_volume_profile(history, *, half_life) -> ndarray",
            "winsorized_mean_volume_profile(history, *, lower_percentile=5.0, upper_percentile=95.0) -> ndarray",
            "profile_r_squared(realized, predicted) -> float",
            "largest_remainder_allocation(quantity, weights) -> ndarray",
        ),
        primitive_names=(),
        semantic_terms=(
            (
                6,
                (
                    "intraday volume-share",
                    "intraday volume share",
                    "volume-share profile",
                    "volume share profile",
                ),
            ),
            (
                5,
                (
                    "final execution schedule",
                    "winsorized mean",
                ),
            ),
        ),
    ),
    CapabilityDescriptor(
        capability_id="quantitative-validation-invariants",
        import_path="offline_common.invariants",
        summary=(
            "Finite-number checks, probability bounds, positive-"
            "semidefinite matrix validation, portfolio-weight totals, "
            "causal timestamp ordering, and additive reconciliation."
        ),
        api_signatures=(
            "finite_numeric_mask(values) -> ndarray",
            "probability_bounds_mask(values, *, upper=1.0, tolerance=0.0) -> ndarray",
            "is_positive_semidefinite(matrix, *, tolerance=1e-10) -> bool",
            "weight_sum_error(weights, *, target=1.0) -> float",
            "causal_order_mask(earlier, later, *, strict=False) -> ndarray",
            "reconciliation_close(actual, components, *, atol=1e-8, rtol=1e-6) -> ndarray",
        ),
        primitive_names=(),
        semantic_terms=(
            (
                6,
                (
                    "positive semidefinite",
                    "positive semi-definite",
                    "psd covariance",
                    "psd correlation",
                    "causal ordering",
                    "causal timestamp",
                ),
            ),
            (
                5,
                (
                    "weights sum to one",
                    "weights sum to 1",
                    "probability bounds",
                    "reconciliation check",
                ),
            ),
            (
                4,
                (
                    "finite numeric",
                    "non-finite values",
                    "accounting reconciliation",
                ),
            ),
        ),
    ),
    CapabilityDescriptor(
        capability_id="cross-sectional-panel-operations",
        import_path="offline_common.panel",
        summary=(
            "Deterministic cross-sectional standardization, equal-count "
            "bucket assignment, rank correlation, control residualization, "
            "and beta-neutral weight projection."
        ),
        api_signatures=(
            "zscore_ddof0(values) -> ndarray",
            "equal_count_bucket_labels(n_observations, *, buckets) -> ndarray",
            "spearman_rank_correlation(first, second) -> float",
            "residualize_against_controls(values, controls, *, add_intercept=True) -> tuple[ndarray, ndarray]",
            "beta_neutral_projection(weights, beta) -> ndarray",
        ),
        primitive_names=(
            "ols_with_intercept",
        ),
        semantic_terms=(
            (
                6,
                (
                    "beta-neutral",
                    "beta neutral",
                    "dependent double sort",
                    "dependent double-sort",
                ),
            ),
            (
                5,
                (
                    "cross-sectional sort",
                    "cross sectional sort",
                    "residualize signal",
                    "residualized signal",
                    "residualise signal",
                    "residualised signal",
                ),
            ),
            (
                5,
                (
                    "rank correlation",
                    "spearman rank",
                    "spearman correlation",
                    "spearman ic",
                ),
            ),
            (
                4,
                (
                    "quantile portfolio",
                    "quintile portfolio",
                    "quintile sort",
                    "quintile sorts",
                ),
            ),
        ),
    ),
    CapabilityDescriptor(
        capability_id="regression-factor-statistics",
        import_path="offline_common.statistics",
        summary=(
            "OLS inference, Newey-West/HAC covariance, Durbin-Watson, "
            "variance-inflation factors, GRS joint-alpha testing, rolling "
            "OLS coefficients, and robust cross-sectional normalization."
        ),
        api_signatures=(
            "ols_with_inference(design, response) -> OlsFit",
            "optimal_newey_west_lag(n_observations) -> int",
            "newey_west_covariance(design, residuals, *, lag) -> ndarray",
            "newey_west_mean_tstat(values, *, lag) -> float",
            "durbin_watson(residuals) -> float",
            "variance_inflation_factors(factors) -> ndarray",
            "grs_joint_alpha_test(alphas, residual_matrix, factor_matrix) -> GrsResult",
            "rolling_ols_coefficient(design, response, *, window, coefficient_index) -> ndarray",
            "normalize_cross_section(series) -> Series",
        ),
        primitive_names=(
            "ols_with_intercept",
            "pca_from_observations",
        ),
        semantic_terms=(
            (
                6,
                (
                    "newey-west",
                    "newey west",
                    "hac covariance",
                    "durbin-watson",
                    "durbin watson",
                    "variance inflation factor",
                    "grs test",
                    "gibbons ross shanken",
                ),
            ),
            (
                5,
                (
                    "rolling ols",
                    "rolling regression",
                    "joint alpha test",
                ),
            ),
            (
                4,
                (
                    "factor regression",
                    "factor-model regression",
                    "fama-french regression",
                    "fama french regression",
                ),
            ),
        ),
    ),
    CapabilityDescriptor(
        capability_id="market-risk-statistics",
        import_path="offline_common.risk",
        summary=(
            "Return alignment, historical and parametric tail risk, "
            "exponential weighting, EWMA covariance, horizon aggregation, "
            "and expanding-window coverage diagnostics."
        ),
        api_signatures=(
            "aligned_log_returns(frame, *, date_column, symbol_column, close_column) -> tuple[DataFrame, DataFrame, dict[str, int]]",
            "historical_var_es(returns, *, alpha, notional) -> tuple[float, float]",
            "normal_var_es(returns, *, alpha, notional) -> tuple[float, float]",
            "student_t_var_es(returns, *, alpha, notional) -> tuple[float, float]",
            "age_weighted_var_es(returns, *, alpha, notional, decay) -> tuple[float, float]",
            "ewma_covariance(asset_returns, *, decay) -> ndarray",
            "ewma_portfolio_volatility(asset_returns, *, decay, weights) -> float",
            "overlapping_horizon_returns(returns, *, horizon) -> ndarray",
            "expanding_historical_backtest(returns, *, alpha, minimum_observations) -> tuple[int, int, float, float]",
        ),
        primitive_names=(
            "empirical_var_es_from_losses",
            "ewma_weights",
            "fit_garch11_zero_mean",
            "garch11_forecast_variance",
            "fit_gpd_exceedances",
            "evt_var_es_from_gpd",
            "hill_tail_index",
            "pca_from_covariance",
        ),
        semantic_terms=(
            (5, ("value-at-risk", "value at risk", "expected shortfall", "cvar")),
            (
                5,
                (
                    "tail index",
                    "hill estimator",
                    "generalized pareto",
                    "garch(1,1)",
                    "garch(1, 1)",
                ),
            ),
            (
                4,
                (
                    "historical var",
                    "tail risk",
                    "parametric var",
                    "var constraint",
                    "var-constrained",
                    "var and es",
                    "var & es",
                    "var/es",
                ),
            ),
            (3, ("ewma covariance", "risk backtest", "portfolio volatility")),
            (2, ("var",)),
        ),
        csv_schemas=(
            (frozenset({"date", "symbol", "close"}), 4),
        ),
    ),
    CapabilityDescriptor(
        capability_id="fixed-income-curves",
        import_path="offline_common.fixed_income",
        summary=(
            "Continuous zero-rate discounting, zero-rate interpolation, "
            "sequential par-curve bootstrapping, and calibration repricing."
        ),
        api_signatures=(
            "discount_factor_from_zero(zero_rate, maturity) -> float",
            "interpolate_zero_rate(maturity, *, known_maturities, known_zero_rates) -> float",
            "bootstrap_par_curve(maturities, par_rates, *, coupon_frequency) -> BootstrappedCurve",
            "reprice_par_bond(*, maturity, par_rate, coupon_frequency, curve) -> float",
        ),
        primitive_names=(
            "discount_cashflow",
            "discount_factor_from_continuous_zero_rate",
            "continuous_zero_rate_from_discount_factor",
            "log_linear_discount_factor",
            "bootstrap_annual_par_discount_factors",
        ),
        semantic_terms=(
            (5, ("curve bootstrap", "bootstrap a curve", "par curve")),
            (4, ("yield curve", "zero curve", "discount curve", "ois curve")),
            (3, ("discount factor", "zero rate", "forward rate", "par yield")),
        ),
        json_schemas=(
            (frozenset({"maturities", "par_rates"}), 4),
            (frozenset({"maturities", "rates"}), 3),
        ),
    ),
    CapabilityDescriptor(
        capability_id="two-asset-derivatives",
        import_path="offline_common.derivatives",
        summary=(
            "Aligned two-asset GBM calibration, exchange volatility, "
            "Margrabe and Kirk prices, and seeded correlated Monte Carlo."
        ),
        api_signatures=(
            "calibrate_two_asset_gbm(first, second, *, annualization=252) -> TwoAssetCalibration",
            "exchange_volatility(sigma1, sigma2, rho) -> float",
            "margrabe_price(*, S1, S2, sigma1, sigma2, rho, T, q1, q2) -> tuple[float, float]",
            "kirk_spread_price(*, S1, S2, K, sigma1, sigma2, rho, T, r, q1, q2) -> float",
            "monte_carlo_spread_prices(*, S1, S2, strikes, sigma1, sigma2, rho, T, r, q1, q2, n_paths, rng) -> list[tuple[float, float]]",
        ),
        primitive_names=(
            "black_scholes_price",
            "historical_log_return_calibration",
            "central_price_delta",
        ),
        semantic_terms=(
            (6, ("margrabe", "kirk")),
            (5, ("exchange option", "spread option")),
            (3, ("two-asset", "two asset", "correlated assets")),
        ),
        csv_schemas=(
            (frozenset({"date", "close"}), 2),
        ),
        minimum_price_panels=2,
    ),
    CapabilityDescriptor(
        capability_id="portfolio-time-series",
        import_path="offline_common.portfolio",
        summary=(
            "Causal panel cleaning, simple-return construction, "
            "cross-sectional momentum paths, and return-path summaries."
        ),
        api_signatures=(
            "MomentumSpec(lookback_start, skip_recent, valid_start, long_count, short_count, periods_per_year)",
            "clean_price_panel(frame, *, date_column='date') -> tuple[DataFrame, list[str]]",
            "simple_returns_from_prices(prices, *, asset_columns, date_column='date') -> DataFrame",
            "cross_sectional_momentum_path(returns, *, asset_columns, spec, date_column='date') -> DataFrame",
            "summarize_return_path(path, *, periods_per_year) -> dict[str, float]",
        ),
        primitive_names=(
            "sma_seeded_ema",
            "ewma_annualized_volatility",
            "log_return_performance",
            "normalize_nonnegative",
            "largest_remainder_allocate",
            "ols_with_intercept",
            "pca_from_observations",
        ),
        semantic_terms=(
            (5, ("cross-sectional momentum", "cross sectional momentum")),
            (4, ("long-short portfolio", "long short portfolio", "relative strength")),
            (3, ("portfolio backtest", "momentum strategy")),
        ),
    ),
    CapabilityDescriptor(
        capability_id="event-study-statistics",
        import_path="offline_common.event_study",
        summary=(
            "Trading-day event alignment, market-model abnormal returns, "
            "rank tests, residual dependence, and adjusted inference."
        ),
        api_signatures=(
            "EventStudySpec(estimation_start, estimation_end, event_start, event_end)",
            "prepare_event_study_returns(stock_prices, market_prices) -> tuple[DataFrame, list[str]]",
            "build_event_records(merged_returns, events, *, spec) -> list[EventRecord]",
            "corrado_rank_statistics(records) -> tuple[float, float]",
            "average_pairwise_residual_correlation(records, *, minimum_overlap=10) -> float",
            "kolari_pynnonen_statistics(records) -> tuple[float, float, float]",
        ),
        primitive_names=(
            "ols_with_intercept",
        ),
        semantic_terms=(
            (3, ("event study", "event-study")),
            (4, ("abnormal return", "abnormal returns", "caar")),
            (3, ("corrado", "kolari", "pynnönen", "pynnonen")),
        ),
        csv_schemas=(
            (frozenset({"ticker", "event_date"}), 5),
        ),
    ),
    CapabilityDescriptor(
        capability_id="ohlc-volatility",
        import_path="offline_common.volatility",
        summary=(
            "OHLC validation, close-to-close and range-based variance, "
            "rolling estimators, annualisation, and relative efficiency."
        ),
        api_signatures=(
            "validate_ohlc(frame) -> bool",
            "ohlc_variance_estimators(frame, *, prior_close=None) -> OhlcVarianceEstimate",
            "annualized_volatility(daily_variance, *, trading_days=252.0) -> float",
            "rolling_ohlc_estimators(frame, *, window) -> list[OhlcVarianceEstimate]",
            "estimator_arrays(estimates) -> dict[str, ndarray]",
            "efficiency_ratios(variance_series) -> dict[str, float]",
        ),
        primitive_names=(
            "historical_log_return_calibration",
            "ewma_annualized_volatility",
            "fit_garch11_zero_mean",
        ),
        semantic_terms=(
            (6, ("ohlc volatility", "range-based volatility", "range based volatility")),
            (5, ("parkinson", "garman-klass", "garman klass", "rogers-satchell", "yang-zhang")),
        ),
        csv_schemas=(
            (frozenset({"open", "high", "low", "close"}), 5),
        ),
    ),
)


def curated_candidate_module_source(
    source: str,
    *,
    filename: str,
) -> str:
    """Return the dependency closure of the approved computational API."""

    exports = CANDIDATE_LIBRARY_EXPORTS.get(filename)
    if exports is None:
        if filename == "__init__.py":
            return '"""Curated general finance computations."""\n'
        raise ValueError(f"Unsupported candidate capability module: {filename}")

    tree = ast.parse(source, filename=filename)
    definitions: dict[str, ast.stmt] = {}

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node

    missing = sorted(set(exports) - definitions.keys())
    if missing:
        raise ValueError(
            f"Candidate capability exports missing from {filename}: "
            + ", ".join(missing)
        )

    selected_names = set(exports)
    pending = list(exports)
    while pending:
        name = pending.pop()
        node = definitions[name]
        for reference in ast.walk(node):
            if not isinstance(reference, ast.Name):
                continue
            dependency = reference.id
            if dependency in definitions and dependency not in selected_names:
                selected_names.add(dependency)
                pending.append(dependency)

    selected_node_ids = {
        id(definitions[name])
        for name in selected_names
    }
    body: list[ast.stmt] = []
    for index, node in enumerate(tree.body):
        is_docstring = (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        if (
            is_docstring
            or isinstance(node, (ast.Import, ast.ImportFrom))
            or id(node) in selected_node_ids
        ):
            body.append(node)

    body.append(
        ast.Assign(
            targets=[ast.Name(id="__all__", ctx=ast.Store())],
            value=ast.Tuple(
                elts=[ast.Constant(value=name) for name in exports],
                ctx=ast.Load(),
            ),
        )
    )
    curated = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(curated)
    return ast.unparse(curated) + "\n"


def compact_planner_data_inspections(
    inspections: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Keep planning-relevant structure while omitting raw data previews."""

    compact: dict[str, dict[str, Any]] = {}
    for path, details in inspections.items():
        item = {
            key: details[key]
            for key in _PLANNER_INSPECTION_KEYS
            if key in details
        }
        members = details.get("members")
        if isinstance(members, list):
            item["members"] = [
                {
                    key: member[key]
                    for key in ("name", "size", "compressed_size")
                    if isinstance(member, dict) and key in member
                }
                for member in members[:30]
                if isinstance(member, dict)
            ]
        compact[str(path)] = item
    return compact


def _normalize(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(text).lower().replace("–", "-").replace("—", "-"),
    ).strip()


_NEGATED_PREFIX = re.compile(
    r"(?:"
    r"\b(?:do|does|did|must|should)\s+not\b|"
    r"\bnever\b|"
    r"\bwithout\b|"
    r"\bavoid(?:ed|ing)?\b|"
    r"\bexclude(?:d|s|ing)?\b|"
    r"\binstead\s+of\b|"
    r"\brather\s+than\b"
    r")[^.;!?]{0,80}$"
)
_COMPARISON_ONLY_PREFIX = re.compile(
    r"\b(?:compare|benchmark|contrast)\b"
    r"[^.;!?]{0,80}\b(?:against|to|with)\s*$"
)


def _phrase_occurrences(text: str, phrase: str) -> list[re.Match[str]]:
    pattern = r"\bvar\b" if phrase == "var" else re.escape(phrase)
    return list(re.finditer(pattern, text))


def _is_actionable_occurrence(
    text: str,
    phrase: str,
    occurrence: re.Match[str],
) -> bool:
    prefix = text[max(0, occurrence.start() - 100):occurrence.start()]
    if _NEGATED_PREFIX.search(prefix):
        return False
    if _COMPARISON_ONLY_PREFIX.search(prefix):
        return False

    suffix = text[occurrence.end():occurrence.end() + 100]
    repeated_exclusion = re.compile(
        r"^\s*but\s+(?:do|does|must|should)\s+not\s+"
        r"(?:calculate|compute|estimate|use|apply)\s+(?:the\s+)?"
        + re.escape(phrase)
    )
    if repeated_exclusion.search(suffix):
        return False
    return True


def _contains_actionable(text: str, phrase: str) -> bool:
    return any(
        _is_actionable_occurrence(text, phrase, occurrence)
        for occurrence in _phrase_occurrences(text, phrase)
    )


def _semantic_score(
    text: str,
    descriptor: CapabilityDescriptor,
) -> tuple[int, list[str]]:
    score = 0
    evidence: list[str] = []
    for weight, alternatives in descriptor.semantic_terms:
        matched = next(
            (
                phrase
                for phrase in alternatives
                if _contains_actionable(text, phrase)
            ),
            None,
        )
        if matched is not None:
            score += weight
            evidence.append(f"instruction mentions {matched}")
    return score, evidence


def _schema_score(
    fingerprint: TaskFingerprint,
    descriptor: CapabilityDescriptor,
) -> tuple[int, list[str]]:
    score = 0
    evidence: list[str] = []
    for required, weight in descriptor.csv_schemas:
        if any(required.issubset(columns) for columns in fingerprint.csv_columns):
            score += weight
            evidence.append(
                "tabular schema includes " + ", ".join(sorted(required))
            )
    for required, weight in descriptor.json_schemas:
        if any(required.issubset(keys) for keys in fingerprint.json_keys):
            score += weight
            evidence.append(
                "structured data includes " + ", ".join(sorted(required))
            )
    if descriptor.minimum_price_panels:
        panel_count = sum(
            1
            for columns in fingerprint.csv_columns
            if {"date", "close"}.issubset(columns)
        )
        if panel_count >= descriptor.minimum_price_panels:
            score += 3
            evidence.append("multiple dated price series are available")
    return score, evidence


def rank_capabilities(
    *,
    instruction: str,
    task_dir: Path,
    limit: int = MAX_SELECTED_CAPABILITIES,
) -> tuple[RankedCapability, ...]:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    if limit == 0:
        return ()

    text = _normalize(instruction)
    fingerprint = inspect_task_fingerprint(Path(task_dir))
    ranked: list[RankedCapability] = []

    for descriptor in CAPABILITY_CATALOG:
        semantic_score, semantic_evidence = _semantic_score(text, descriptor)
        data_score, data_evidence = _schema_score(fingerprint, descriptor)
        if semantic_score < MIN_SEMANTIC_SCORE:
            continue
        if semantic_score + data_score < MIN_TOTAL_SCORE:
            continue
        ranked.append(
            RankedCapability(
                descriptor=descriptor,
                semantic_score=semantic_score,
                data_score=data_score,
                matched_evidence=tuple(semantic_evidence + data_evidence),
            )
        )

    ranked.sort(
        key=lambda item: (
            -item.total_score,
            -item.semantic_score,
            -item.data_score,
            item.descriptor.capability_id,
        )
    )
    return tuple(ranked[:limit])


def relevant_primitive_catalog(
    capabilities: Sequence[RankedCapability],
) -> tuple[str, ...]:
    requested = {
        name
        for capability in capabilities
        for name in capability.descriptor.primitive_names
    }
    return tuple(
        entry
        for entry in PRIMITIVE_API_CATALOG
        if entry.split("(", 1)[0].strip() in requested
    )
