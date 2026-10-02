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
    "fx.py": (
        "covered_interest_forward",
        "invert_bid_ask",
        "synthetic_cross_bid_ask",
        "forward_points",
        "implied_quote_rate_from_forward",
        "continuous_basis_bps",
        "forward_from_continuous_basis",
        "log_linear_forward",
    ),
    "finite_difference.py": (
        "FiniteDifferenceOptionSpec",
        "FiniteDifferenceResult",
        "crank_nicolson_option",
        "with_grid_size",
        "richardson_second_order",
    ),
    "monte_carlo_greeks.py": (
        "MonteCarloEstimate",
        "mc_estimate",
        "gbm_paths_from_normals",
        "finite_difference_greeks",
        "pathwise_greeks",
        "likelihood_ratio_greeks",
    ),
    "cliquet.py": (
        "forward_start_atm_call_price",
        "cliquet_forward_start_prices",
    ),
    "lookback_options.py": (
        "floating_lookback_call",
        "floating_lookback_put",
        "fixed_strike_lookback_call",
    ),
    "cap_floor.py": (
        "CapFloorStripResult",
        "black_caplet",
        "black_floorlet",
        "black_cap_floor_strip",
    ),
    "variance_swap.py": (
        "trapezoidal_strike_widths",
        "variance_swap_fair_variance",
        "interpolate_at_forward",
        "variance_swap_pnl",
    ),
    "copula_fitting.py": (
        "gaussian_copula_loglik",
        "student_t_copula_loglik",
        "clayton_copula_loglik",
        "gumbel_copula_loglik",
        "kendall_tau_to_gaussian_rho",
        "fit_student_t_degrees_of_freedom",
        "empirical_tail_dependence",
    ),
    "credit_migration.py": (
        "TransitionHomogeneityResult",
        "pooled_transition_matrix",
        "cumulative_target_probabilities",
        "transition_homogeneity_test",
        "continuous_time_generator",
    ),
    "first_passage.py": (
        "upper_first_passage_cdf",
        "lower_first_passage_cdf",
        "expected_upper_first_passage_time",
        "expected_lower_first_passage_time",
    ),
    "merton_jump_diffusion.py": (
        "MertonJumpCalibration",
        "merton_jump_negative_log_likelihood",
        "calibrate_merton_jump_diffusion",
        "merton_call_price",
        "merton_total_volatility",
    ),
    "ou_jump.py": (
        "OuAr1Fit",
        "JumpResidualFit",
        "OuJumpMoments",
        "fit_ou_ar1",
        "ou_diffusion_volatility",
        "fit_residual_jumps",
        "ou_jump_conditional_moments",
        "ou_jump_stationary_moments",
        "lognormal_moments",
    ),
    "regime_black_litterman.py": (
        "GaussianHmm2StateResult",
        "RegimeCovarianceResult",
        "BlackLittermanResult",
        "fit_two_state_gaussian_hmm",
        "hard_regime_covariances",
        "black_litterman_posterior",
    ),
    "factor_portfolio.py": (
        "FactorNeutralPortfolio",
        "factor_exposures",
        "minimum_norm_factor_neutral_hedge",
        "annualized_sharpe_ratio",
        "residual_variance_r_squared",
    ),
    "intraday_variation.py": (
        "realized_variance",
        "bandi_russell_noise_variance",
        "additive_noise_corrected_variance",
        "bipower_variation",
        "periodic_sampling_mask",
        "annualized_volatility_from_variance",
    ),
    "asian_options.py": (
        "monitoring_times",
        "geometric_asian_call",
        "arithmetic_moments",
        "levy_asian_call",
        "curran_asian_call",
        "monte_carlo_asian",
    ),
    "rate_curves.py": (
        "simple_forward_rate",
        "forward_discount_factor",
        "fixed_leg_annuity",
        "floating_leg_pv_per_unit",
        "par_swap_rate",
    ),
    "bonds.py": (
        "YieldDurationResult",
        "discounted_cashflow_price",
        "yield_and_durations",
        "z_spread_from_continuous_curve",
        "parallel_duration_convexity",
        "symmetric_key_rate_durations",
        "dv01_from_duration",
    ),
    "schedules.py": (
        "parse_iso_date",
        "add_months",
        "year_fraction",
        "generate_schedule",
        "is_business_day",
        "add_business_days",
        "following_business_day",
        "add_months_following",
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
    "earnings.py": (
        "EarningsSurprise",
        "earnings_surprise",
        "aggregate_earnings_surprises",
    ),
    "sec_filings.py": (
        "html_to_text",
        "extract_sec_event_date",
        "extract_sec_item_numbers",
        "money_to_millions",
        "revenue_guidance_midpoint_millions",
        "classify_executive_departure",
        "extract_debt_principal_millions",
        "event_alpha_score",
    ),
    "holdings.py": (
        "AmendmentResolution",
        "resolve_amendment_sequence",
        "clean_long_holdings",
        "concentration_metrics",
        "portfolio_turnover",
        "weighted_overlap",
        "aggregate_crowding",
    ),
    "ownership.py": (
        "Form4Transaction",
        "parse_price_range",
        "normalize_ownership_entity",
        "parse_form4_transactions",
        "sale_pressure_ratios",
    ),
    "xbrl.py": (
        "XbrlFact",
        "parse_xbrl_facts",
        "load_xbrl_zip_facts",
        "facts_by_concept",
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
        capability_id="earnings-surprise-statistics",
        import_path="offline_common.earnings",
        summary=(
            "Raw and percentage earnings surprises, standardized unexpected "
            "earnings, year-over-year growth, classification, and aggregation."
        ),
        api_signatures=(
            "EarningsSurprise(surprise_dollars, surprise_pct, sue, yoy_growth, classification)",
            "earnings_surprise(actual_eps, consensus_estimate, std_estimate, prior_year_eps, *, threshold_pct=2.0) -> EarningsSurprise",
            "aggregate_earnings_surprises(observations) -> dict[str, float]",
        ),
        primitive_names=(),
        semantic_terms=(
            (5, ("earnings surprise",)),
            (4, ("standardized unexpected earnings", "sue calculator")),
        ),
        json_schemas=((frozenset({"companies"}), 3),),
    ),
    CapabilityDescriptor(
        capability_id="sec-filing-event-extraction",
        import_path="offline_common.sec_filings",
        summary=(
            "Visible-text conversion and structured extraction of SEC event "
            "dates, item numbers, guidance, executive departures, and debt amounts."
        ),
        api_signatures=(
            "html_to_text(source) -> str",
            "extract_sec_event_date(text) -> str",
            "extract_sec_item_numbers(text) -> tuple[str, ...]",
            "money_to_millions(value) -> float",
            "revenue_guidance_midpoint_millions(text) -> float",
            "classify_executive_departure(text) -> str",
            "extract_debt_principal_millions(text) -> float",
            "event_alpha_score(base_score, severity_multiplier, liquidity_multiplier) -> float",
        ),
        primitive_names=(),
        semantic_terms=(
            (6, ("8-k event", "8 k event", "sec 8-k", "sec 8 k")),
            (4, ("guidance raise", "guidance cut", "executive departure", "debt financing")),
            (3, ("event alpha", "filing signals")),
        ),
    ),
    CapabilityDescriptor(
        capability_id="amendment-aware-holdings",
        import_path="offline_common.holdings",
        summary=(
            "Amendment-state resolution, long-share cleaning, concentration, "
            "turnover, weighted overlap, and security crowding."
        ),
        api_signatures=(
            "AmendmentResolution(accession, action, effective_accessions)",
            "resolve_amendment_sequence(filings) -> list[AmendmentResolution]",
            "clean_long_holdings(frame) -> tuple[DataFrame, dict[str, int]]",
            "concentration_metrics(weights) -> dict[str, float]",
            "portfolio_turnover(previous, current) -> float",
            "weighted_overlap(first, second) -> float",
            "aggregate_crowding(holdings, *, owner_column='owner') -> DataFrame",
        ),
        primitive_names=(),
        semantic_terms=(
            (6, ("13f amendment", "13f amendment-aware", "13f amendment aware")),
            (4, ("effective holdings", "filing reconstruction")),
            (3, ("portfolio crowding", "weighted overlap")),
        ),
        csv_schemas=((frozenset({"accession_number", "cusip", "sshprnamt", "value"}), 3),),
    ),
    CapabilityDescriptor(
        capability_id="form4-ownership-sales",
        import_path="offline_common.ownership",
        summary=(
            "Form 4 non-derivative transaction parsing, ownership-bucket "
            "normalization, reported price bands, and liquidity-scaled sale pressure."
        ),
        api_signatures=(
            "Form4Transaction(transaction_date, security_title, transaction_code, shares, price_per_share, shares_following, ownership_form, ownership_nature, footnote_ids)",
            "parse_price_range(text) -> tuple[float, float, float] | None",
            "normalize_ownership_entity(ownership_form, nature, footnote_texts=()) -> str",
            "parse_form4_transactions(xml_source, *, transaction_codes=('S',)) -> list[Form4Transaction]",
            "sale_pressure_ratios(shares_sold, gross_proceeds, inventory_before, adv20_shares, close_price) -> dict[str, float]",
        ),
        primitive_names=(),
        semantic_terms=(
            (6, ("form 4", "form4")),
            (4, ("non-derivative sale", "non derivative sale", "insider sale")),
            (3, ("sale pressure", "ownership bucket")),
        ),
    ),
    CapabilityDescriptor(
        capability_id="xbrl-fact-parsing",
        import_path="offline_common.xbrl",
        summary=(
            "Inline and instance XBRL fact parsing from documents or filing archives, "
            "including scale, sign, context, unit, and concept selection."
        ),
        api_signatures=(
            "XbrlFact(concept, value, context_ref, unit_ref, decimals)",
            "parse_xbrl_facts(source) -> list[XbrlFact]",
            "load_xbrl_zip_facts(path) -> list[XbrlFact]",
            "facts_by_concept(facts, concept) -> list[XbrlFact]",
        ),
        primitive_names=(),
        semantic_terms=(
            (6, ("xbrl", "inline xbrl")),
            (4, ("10-k report", "10 k report", "sec 10-k", "sec 10 k")),
            (3, ("fundamental metric extraction", "financial report extraction")),
        ),
    ),
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
        capability_id="financial-schedules-day-counts",
        import_path="offline_common.schedules",
        summary=(
            "Month-safe date rolling, standard ACT and 30/360 year fractions, "
            "backward payment schedules, and deterministic business-day adjustment."
        ),
        api_signatures=(
            "parse_iso_date(value) -> date",
            "add_months(value, months) -> date",
            "year_fraction(start, end, *, convention) -> float",
            "generate_schedule(effective, maturity, frequency_months) -> list[date]",
            "is_business_day(value, *, holidays=()) -> bool",
            "add_business_days(start, count, *, holidays=()) -> date",
            "following_business_day(value, *, holidays=()) -> date",
            "add_months_following(value, months, *, holidays=()) -> date",
        ),
        primitive_names=(),
        semantic_terms=(
            (
                6,
                (
                    "generated backward from maturity",
                    "coupon schedules",
                    "coupon schedule",
                ),
            ),
            (
                6,
                (
                    "following business-day adjustment",
                    "following business day adjustment",
                    "spot lags",
                ),
            ),
            (
                5,
                (
                    "30/360 us",
                    "30/360 day count",
                    "settlement dates",
                ),
            ),
        ),
    ),
    CapabilityDescriptor(
        capability_id="bond-risk-analytics",
        import_path="offline_common.bonds",
        summary=(
            "Discounted bond cash-flow valuation, periodic-compounding yield "
            "and duration analytics, constant z-spread solving, effective "
            "duration/convexity, symmetric key-rate duration, and DV01."
        ),
        api_signatures=(
            "discounted_cashflow_price(times, cash_flows, zero_rates) -> float",
            "yield_and_durations(*, price, times, cash_flows, frequency) -> YieldDurationResult",
            "z_spread_from_continuous_curve(*, price, times, cash_flows, zero_rates, lower_bound=-0.05, upper_bound=0.50) -> float",
            "parallel_duration_convexity(*, base_price, price_up, price_down, bump) -> tuple[float, float]",
            "symmetric_key_rate_durations(*, base_value, values_up, values_down, bump) -> ndarray",
            "dv01_from_duration(price, duration) -> float",
        ),
        primitive_names=(),
        semantic_terms=(
            (
                6,
                (
                    "bond immunization",
                    "key-rate duration",
                    "key rate duration",
                ),
            ),
            (
                5,
                (
                    "macaulay duration",
                    "modified duration",
                    "z-spread",
                    "z spread",
                ),
            ),
        ),
    ),
    CapabilityDescriptor(
        capability_id="fx-forward-cross-currency",
        import_path="offline_common.fx",
        summary=(
            "Covered-interest-parity FX forwards, bid/ask inversion and "
            "synthetic crosses, forward points, implied quote-currency "
            "rates, continuous FX basis, and log-linear forward interpolation."
        ),
        api_signatures=(
            "covered_interest_forward(spot, *, base_rate, quote_rate, base_year_fraction, quote_year_fraction) -> float",
            "invert_bid_ask(bid, ask) -> tuple[float, float]",
            "synthetic_cross_bid_ask(*, base_usd_bid, base_usd_ask, quote_usd_bid, quote_usd_ask) -> tuple[float, float, float]",
            "forward_points(spot, forward, *, pip_multiplier=10000.0) -> float",
            "implied_quote_rate_from_forward(spot, forward, *, base_rate, base_year_fraction, quote_year_fraction) -> float",
            "continuous_basis_bps(quoted_forward, cip_forward, year_fraction) -> float",
            "forward_from_continuous_basis(cip_forward, basis_bps, year_fraction) -> float",
            "log_linear_forward(spot, node_times, node_forwards, target_time) -> float",
        ),
        primitive_names=(),
        semantic_terms=(
            (
                6,
                (
                    "covered interest rate parity",
                    "synthetic cross rates",
                    "synthetic cross rate",
                ),
            ),
            (
                6,
                (
                    "implied foreign yields",
                    "implied foreign yield",
                ),
            ),
            (
                6,
                (
                    "cross-currency basis",
                    "cross currency basis",
                    "broken-date fx forwards",
                    "broken date fx forwards",
                ),
            ),
            (
                5,
                (
                    "implied basis",
                    "forward points",
                ),
            ),
        ),
    ),
    CapabilityDescriptor(
        capability_id="dual-curve-rate-mechanics",
        import_path="offline_common.rate_curves",
        summary=(
            "Projection-curve forward rates, forward discount factors, "
            "fixed/floating leg present values, and generic dual-curve "
            "par-swap-rate calculation."
        ),
        api_signatures=(
            "simple_forward_rate(start_discount_factor, end_discount_factor, accrual) -> float",
            "forward_discount_factor(anchor_discount_factor, target_discount_factor) -> float",
            "fixed_leg_annuity(accruals, discount_factors) -> float",
            "floating_leg_pv_per_unit(forward_rates, accruals, discount_factors) -> float",
            "par_swap_rate(*, fixed_accruals, fixed_discount_factors, floating_forward_rates, floating_accruals, floating_discount_factors) -> float",
        ),
        primitive_names=(),
        semantic_terms=(
            (
                6,
                (
                    "dual-curve framework",
                    "dual curve framework",
                ),
            ),
            (
                5,
                (
                    "projection curve",
                    "projection curves",
                ),
            ),
            (
                6,
                (
                    "forward discount factors",
                    "forward discount factor",
                ),
            ),
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
        capability_id="finite-difference-options",
        import_path="offline_common.finite_difference",
        summary=(
            "Crank-Nicolson option valuation with European or American "
            "exercise, PSOR obstacle handling, discrete cash-dividend jumps, "
            "grid resizing, and second-order Richardson extrapolation."
        ),
        api_signatures=(
            "FiniteDifferenceOptionSpec(...)",
            "crank_nicolson_option(spec, *, option_type, exercise_type, dividends=None, return_grid=False, return_boundary=False) -> FiniteDifferenceResult",
            "with_grid_size(spec, *, stock_steps, time_steps) -> FiniteDifferenceOptionSpec",
            "richardson_second_order(fine, coarse) -> float",
        ),
        primitive_names=(
            "black_scholes_price",
            "black_scholes_greeks",
        ),
        semantic_terms=(
            (
                6,
                (
                    "crank-nicolson",
                    "crank nicolson",
                ),
            ),
            (
                6,
                (
                    "psor",
                    "projected successive over-relaxation",
                    "projected successive over relaxation",
                ),
            ),
            (
                5,
                (
                    "richardson extrapolation",
                    "richardson",
                ),
            ),
        ),
    ),
    CapabilityDescriptor(
        capability_id="monte-carlo-greeks",
        import_path="offline_common.monte_carlo_greeks",
        summary=(
            "Monte Carlo estimation from supplied normal draws, GBM path "
            "generation, common-random-number finite-difference Greeks, "
            "pathwise estimators, likelihood-ratio estimators, and "
            "Monte Carlo standard errors."
        ),
        api_signatures=(
            "mc_estimate(samples) -> MonteCarloEstimate",
            "gbm_paths_from_normals(normals, *, spot, rate, dividend_yield, volatility, maturity) -> ndarray",
            "finite_difference_greeks(normals, **kwargs) -> dict[str, MonteCarloEstimate]",
            "pathwise_greeks(normals, *, spot, strike, rate, dividend_yield, volatility, maturity, option_type, asian) -> dict[str, MonteCarloEstimate | None]",
            "likelihood_ratio_greeks(normals, *, spot, strike, rate, dividend_yield, volatility, maturity, option_type, asian) -> dict[str, MonteCarloEstimate | None]",
        ),
        primitive_names=(
            "black_scholes_price",
            "black_scholes_greeks",
        ),
        semantic_terms=(
            (
                6,
                (
                    "monte carlo greeks",
                    "monte-carlo greeks",
                ),
            ),
            (
                6,
                (
                    "likelihood ratio",
                    "likelihood-ratio",
                ),
            ),
            (
                5,
                (
                    "pathwise estimator",
                    "pathwise estimators",
                    "pathwise",
                ),
            ),
        ),
    ),
    CapabilityDescriptor(
        capability_id="intraday-realized-variation",
        import_path="offline_common.intraday_variation",
        summary=(
            "Intraday realized-variance utilities including "
            "Bandi-Russell microstructure-noise estimation, "
            "additive noise correction, bipower variation, "
            "periodic sampling masks, and annualized volatility."
        ),
        api_signatures=(
            "realized_variance(log_returns) -> float",
            "bandi_russell_noise_variance(log_returns) -> float",
            "additive_noise_corrected_variance(realized_variance_value, *, n_returns, noise_variance) -> float",
            "bipower_variation(log_returns) -> float",
            "periodic_sampling_mask(offsets, *, frequency) -> ndarray",
            "annualized_volatility_from_variance(variance, *, periods_per_year, clip_negative=True) -> float",
        ),
        primitive_names=(),
        semantic_terms=(
            (
                6,
                (
                    "bandi-russell",
                    "bandi russell",
                ),
            ),
            (
                6,
                (
                    "bipower variation",
                ),
            ),
            (
                6,
                (
                    "volatility signature",
                ),
            ),
            (
                6,
                (
                    "microstructure-noise correction",
                    "microstructure noise correction",
                ),
            ),
        ),
    ),
    CapabilityDescriptor(
        capability_id="pca-factor-neutral-portfolio",
        import_path="offline_common.factor_portfolio",
        summary=(
            "Factor exposure measurement and minimum-norm "
            "factor-neutral portfolio hedging from PCA component "
            "loadings, plus Sharpe and residual-variance diagnostics."
        ),
        api_signatures=(
            "factor_exposures(components, weights) -> ndarray",
            "minimum_norm_factor_neutral_hedge(components, target_weights, *, dollar_neutral_hedge=True) -> FactorNeutralPortfolio",
            "annualized_sharpe_ratio(returns, *, periods_per_year) -> float",
            "residual_variance_r_squared(residual_returns, benchmark_returns, *, ddof=0) -> float",
        ),
        primitive_names=(),
        semantic_terms=(
            (
                6,
                (
                    "pca factor portfolio construction",
                    "pca factor portfolio",
                ),
            ),
            (
                6,
                (
                    "factor-neutral portfolio",
                    "factor neutral portfolio",
                ),
            ),
            (
                6,
                (
                    "target_factor_exposures",
                ),
            ),
        ),
    ),
    CapabilityDescriptor(
        capability_id="regime-black-litterman",
        import_path="offline_common.regime_black_litterman",
        summary=(
            "Two-state Gaussian Hidden Markov Model fitting with Baum-Welch, "
            "hard regime-conditional covariance estimation, and "
            "Black-Litterman posterior portfolio construction."
        ),
        api_signatures=(
            "fit_two_state_gaussian_hmm(observations, *, initial_probabilities, transition_matrix, max_iterations=100, tolerance=1e-8) -> GaussianHmm2StateResult",
            "hard_regime_covariances(returns, posterior_probabilities, *, bull_state, threshold=0.5, minimum_observations=11) -> RegimeCovarianceResult",
            "black_litterman_posterior(covariance, market_weights, *, risk_aversion, tau, view_matrix, view_returns, view_covariance, normalize_gross=True) -> BlackLittermanResult",
        ),
        primitive_names=(),
        semantic_terms=(
            (
                6,
                (
                    "black-litterman",
                    "black litterman",
                ),
            ),
            (
                6,
                (
                    "baum-welch",
                    "baum welch",
                ),
            ),
            (
                6,
                (
                    "regime-aware black-litterman",
                    "regime aware black-litterman",
                ),
            ),
        ),
    ),
    CapabilityDescriptor(
        capability_id="ou-jump-mean-reversion",
        import_path="offline_common.ou_jump",
        summary=(
            "Ornstein-Uhlenbeck mean-reversion with compound Poisson jumps: "
            "AR(1) calibration, continuous diffusion volatility, residual "
            "jump estimation, conditional and stationary moments, and "
            "lognormal level moments for log-state models."
        ),
        api_signatures=(
            "fit_ou_ar1(levels, *, dt) -> OuAr1Fit",
            "ou_diffusion_volatility(residual_std, *, kappa, slope) -> float",
            "fit_residual_jumps(residuals, *, dt, threshold_std=3.0) -> JumpResidualFit",
            "ou_jump_conditional_moments(*, initial_level, horizon, kappa, theta, diffusion_volatility, jump_intensity=0.0, jump_mean=0.0, jump_volatility=0.0) -> OuJumpMoments",
            "ou_jump_stationary_moments(*, kappa, theta, diffusion_volatility, jump_intensity=0.0, jump_mean=0.0, jump_volatility=0.0) -> OuJumpMoments",
            "lognormal_moments(normal_mean, normal_variance) -> tuple[float, float]",
        ),
        primitive_names=(),
        semantic_terms=(
            (
                6,
                (
                    "ou process with jumps",
                    "ou process with poisson jumps",
                ),
            ),
            (
                6,
                (
                    "geometric mean-reverting jump-diffusion",
                    "geometric mean reverting jump-diffusion",
                ),
            ),
        ),
    ),
    CapabilityDescriptor(
        capability_id="merton-jump-diffusion",
        import_path="offline_common.merton_jump_diffusion",
        summary=(
            "Merton jump-diffusion likelihood calibration and European "
            "call pricing via the Poisson-mixture series, including total "
            "diffusion-plus-jump volatility."
        ),
        api_signatures=(
            "merton_jump_negative_log_likelihood(log_returns, *, dt, drift, diffusion_volatility, jump_intensity, jump_mean, jump_volatility, max_jumps=15) -> float",
            "calibrate_merton_jump_diffusion(log_returns, *, dt, drift, bounds, restarts=8, seed=0, max_jumps=15) -> MertonJumpCalibration",
            "merton_call_price(*, spot, strike, rate, maturity, diffusion_volatility, jump_intensity, jump_mean, jump_volatility, n_terms=50) -> float",
            "merton_total_volatility(diffusion_volatility, jump_intensity, jump_mean, jump_volatility) -> float",
        ),
        primitive_names=(
            "black_scholes_price",
            "implied_volatility_black_scholes",
        ),
        semantic_terms=(
            (
                6,
                (
                    "merton jump-diffusion",
                    "merton jump diffusion",
                ),
            ),
            (
                6,
                (
                    "merton series formula",
                    "poisson-weighted mixture",
                    "poisson weighted mixture",
                ),
            ),
        ),
    ),
    CapabilityDescriptor(
        capability_id="brownian-first-passage",
        import_path="offline_common.first_passage",
        summary=(
            "First-passage probabilities and expected hitting times for "
            "drifted Brownian motion, with upper and lower barriers."
        ),
        api_signatures=(
            "upper_first_passage_cdf(drift, volatility, time, barrier) -> float",
            "lower_first_passage_cdf(drift, volatility, time, barrier) -> float",
            "expected_upper_first_passage_time(drift, barrier) -> float",
            "expected_lower_first_passage_time(drift, barrier) -> float",
        ),
        primitive_names=(
            "brownian_running_max_hit_probability",
            "brownian_running_min_hit_probability",
            "brownian_joint_terminal_max_cdf",
            "historical_log_return_calibration",
        ),
        semantic_terms=(
            (
                6,
                (
                    "first passage time",
                    "first-passage time",
                ),
            ),
        ),
    ),
    CapabilityDescriptor(
        capability_id="credit-migration-transition-matrix",
        import_path="offline_common.credit_migration",
        summary=(
            "Credit-state transition analysis with pooled cohort matrices, "
            "multi-horizon target-state probabilities, chi-square transition "
            "homogeneity testing, and repaired continuous-time generators."
        ),
        api_signatures=(
            "pooled_transition_matrix(cohort_counts, *, absorbing_index=None) -> tuple[np.ndarray, np.ndarray]",
            "cumulative_target_probabilities(transition_matrix, *, target_index, horizons, source_indices=None) -> dict[int, np.ndarray]",
            "transition_homogeneity_test(cohort_counts, *, origin_index, min_destination_total=5.0, alpha=0.05) -> TransitionHomogeneityResult",
            "continuous_time_generator(transition_matrix, *, absorbing_index=None) -> np.ndarray",
        ),
        primitive_names=(),
        semantic_terms=(
            (
                6,
                (
                    "credit rating migration",
                    "rating migration matrix",
                ),
            ),
            (
                6,
                (
                    "cumulative default probabilities",
                    "cumulative default probability",
                ),
            ),
            (
                6,
                (
                    "continuous-time generator matrix",
                    "continuous time generator matrix",
                ),
            ),
        ),
    ),
    CapabilityDescriptor(
        capability_id="copula-dependence-fitting",
        import_path="offline_common.copula_fitting",
        summary=(
            "Likelihood evaluation for Gaussian, Student-t, Clayton and "
            "Gumbel copulas, Kendall-tau dependence conversion, Student-t "
            "degrees-of-freedom fitting, and empirical tail dependence."
        ),
        api_signatures=(
            "gaussian_copula_loglik(u1, u2, rho) -> float",
            "student_t_copula_loglik(u1, u2, rho, degrees_of_freedom) -> float",
            "clayton_copula_loglik(u1, u2, theta) -> float",
            "gumbel_copula_loglik(u1, u2, theta) -> float",
            "kendall_tau_to_gaussian_rho(kendall_tau) -> float",
            "fit_student_t_degrees_of_freedom(u1, u2, *, rho, lower=2.1, upper=100.0) -> float",
            "empirical_tail_dependence(u1, u2, *, quantile=0.95) -> tuple[float, float]",
        ),
        primitive_names=(
            "pseudo_observations",
        ),
        semantic_terms=(
            (
                6,
                (
                    "copula fitting",
                    "fit copulas",
                ),
            ),
            (
                6,
                (
                    "pseudo-observations",
                    "pseudo observations",
                ),
            ),
        ),
    ),
    CapabilityDescriptor(
        capability_id="variance-swap-replication",
        import_path="offline_common.variance_swap",
        summary=(
            "Variance-swap static replication using OTM option prices on "
            "non-uniform strike grids, trapezoidal strike weights, "
            "forward interpolation, and long variance-swap P&L."
        ),
        api_signatures=(
            "trapezoidal_strike_widths(strikes) -> np.ndarray",
            "variance_swap_fair_variance(strikes, otm_option_prices, *, risk_free_rate, maturity) -> float",
            "interpolate_at_forward(lower_strike, upper_strike, lower_value, upper_value, forward_price) -> float",
            "variance_swap_pnl(realized_volatility, fair_variance, *, variance_notional=1.0) -> float",
        ),
        primitive_names=(
            "implied_volatility_black_scholes",
        ),
        semantic_terms=(
            (
                6,
                (
                    "variance swap",
                    "variance-swap",
                ),
            ),
            (
                6,
                (
                    "log-contract replication",
                    "log contract replication",
                ),
            ),
            (
                5,
                (
                    "trapezoidal strike",
                    "non-uniform strike grid",
                    "non uniform strike grid",
                ),
            ),
        ),
    ),
    CapabilityDescriptor(
        capability_id="interest-rate-cap-floor",
        import_path="offline_common.cap_floor",
        summary=(
            "Black-model caplet and floorlet pricing, immediate-fixing "
            "intrinsic handling, cap/floor strip aggregation, and "
            "cap-floor parity against the corresponding swap value."
        ),
        api_signatures=(
            "black_caplet(*, forward_rate, strike, volatility, fixing_time, discount_factor, accrual, notional=1.0) -> float",
            "black_floorlet(*, forward_rate, strike, volatility, fixing_time, discount_factor, accrual, notional=1.0) -> float",
            "black_cap_floor_strip(forward_rates, discount_factors, *, strike, volatility, accrual, notional=1.0) -> CapFloorStripResult",
        ),
        primitive_names=(),
        semantic_terms=(
            (
                6,
                (
                    "interest rate cap and floor",
                    "interest-rate cap and floor",
                ),
            ),
            (
                6,
                (
                    "black's model",
                    "blacks model",
                ),
            ),
            (
                6,
                (
                    "floorlet",
                    "floorlets",
                ),
            ),
        ),
    ),
    CapabilityDescriptor(
        capability_id="lookback-option-pricing",
        import_path="offline_common.lookback_options",
        summary=(
            "Closed-form pricing for floating-strike lookback calls and puts "
            "and fixed-strike lookback calls using running extrema."
        ),
        api_signatures=(
            "floating_lookback_call(*, spot, running_min, maturity, risk_free_rate, dividend_yield, volatility) -> float",
            "floating_lookback_put(*, spot, running_max, maturity, risk_free_rate, dividend_yield, volatility) -> float",
            "fixed_strike_lookback_call(*, spot, running_max, strike, maturity, risk_free_rate, dividend_yield, volatility) -> float",
        ),
        primitive_names=(
            "black_scholes_price",
        ),
        semantic_terms=(
            (
                6,
                (
                    "lookback option",
                    "lookback options",
                ),
            ),
            (
                6,
                (
                    "floating-strike lookback",
                    "floating strike lookback",
                ),
            ),
            (
                6,
                (
                    "fixed-strike lookback",
                    "fixed strike lookback",
                ),
            ),
        ),
    ),
    CapabilityDescriptor(
        capability_id="asian-option-approximations",
        import_path="offline_common.asian_options",
        summary=(
            "Discrete-monitoring Asian-option calculations including "
            "geometric-average pricing, arithmetic-average moments, "
            "Levy and Curran approximations, and Monte Carlo estimates."
        ),
        api_signatures=(
            "monitoring_times(T, n_monitoring) -> np.ndarray",
            "geometric_asian_call(*, S0, K, T, r, sigma, n_monitoring) -> float",
            "arithmetic_moments(*, S0, T, r, sigma, n_monitoring) -> tuple[float, float]",
            "levy_asian_call(*, S0, K, T, r, sigma, n_monitoring) -> float",
            "curran_asian_call(*, S0, K, T, r, sigma, n_monitoring) -> float",
            "monte_carlo_asian(*, S0, strikes, T, r, sigma, n_monitoring, n_paths, rng) -> list[tuple[float, float, float]]",
        ),
        primitive_names=(),
        semantic_terms=(
            (
                6,
                (
                    "levy asian",
                    "levy approximation",
                ),
            ),
            (
                6,
                (
                    "curran asian",
                    "curran approximation",
                ),
            ),
            (
                5,
                (
                    "geometric asian",
                    "arithmetic asian",
                ),
            ),
        ),
    ),
    CapabilityDescriptor(
        capability_id="cliquet-forward-start-pricing",
        import_path="offline_common.cliquet",
        summary=(
            "Forward-start at-the-money call valuation and decomposition "
            "of a cliquet into reset-period forward-start option pieces."
        ),
        api_signatures=(
            "forward_start_atm_call_price(*, spot, rate, dividend_yield, volatility, start, end) -> float",
            "cliquet_forward_start_prices(*, spot, rate, dividend_yield, volatility, maturity, resets) -> np.ndarray",
        ),
        primitive_names=(),
        semantic_terms=(
            (
                6,
                (
                    "cliquet",
                ),
            ),
            (
                6,
                (
                    "forward-start option",
                    "forward start option",
                    "forward-start options",
                    "forward start options",
                ),
            ),
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
