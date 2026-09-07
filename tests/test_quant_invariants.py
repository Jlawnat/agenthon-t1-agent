from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import numpy as np
import pandas as pd

from agent.quant_invariants import (
    QuantInvariantConfig,
    _CHECKS,
    evaluate_quant_invariants,
    invariant_registry_snapshot,
)

def _by_name(evidence):
    return {item.name: item for item in evidence}


def _evaluate_csv(frame: pd.DataFrame, config: QuantInvariantConfig):
    with TemporaryDirectory() as directory:
        output_path = Path(directory) / "result.csv"
        frame.to_csv(output_path, index=False)
        return evaluate_quant_invariants(output_path=output_path, config=config)


def _evaluate_parquet(testcase, frame: pd.DataFrame, config: QuantInvariantConfig):
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        try:
            import fastparquet  # noqa: F401
        except ImportError:
            testcase.skipTest("Parquet engine is not installed in this test environment")
    with TemporaryDirectory() as directory:
        output_path = Path(directory) / "result.parquet"
        frame.to_parquet(output_path, index=False)
        return evaluate_quant_invariants(output_path=output_path, config=config)


class QuantInvariantFoundationTests(unittest.TestCase):
    def test_decimal_probability_violation_is_hard_failure(self) -> None:
        evidence = _evaluate_csv(
            pd.DataFrame({"default_probability": [0.10, 1.20]}),
            QuantInvariantConfig(category="credit"),
        )
        result = _by_name(evidence)["probability_bounds"]
        self.assertEqual(result.status, "fail")
        self.assertEqual(result.severity, "hard_fail")
        self.assertEqual(
            result.details["violations"]["default_probability"]["violation_count"],
            1,
        )

    def test_percentage_probability_unit_is_respected(self) -> None:
        evidence = _evaluate_csv(
            pd.DataFrame({"default_probability": [10.0, 85.0]}),
            QuantInvariantConfig(
                category="credit",
                units={"default_probability": "percent"},
            ),
        )
        self.assertEqual(_by_name(evidence)["probability_bounds"].status, "pass")

    def test_negative_finance_measure_is_hard_failure(self) -> None:
        evidence = _evaluate_csv(
            pd.DataFrame({"volatility": [0.20, -0.05]}),
            QuantInvariantConfig(category="risk-management"),
        )
        result = _by_name(evidence)["nonnegative_quantities"]
        self.assertEqual(result.status, "fail")
        self.assertEqual(result.severity, "hard_fail")

    def test_nonfinite_numeric_output_is_hard_failure(self) -> None:
        evidence = _evaluate_csv(
            pd.DataFrame({"price": [100.0, np.inf]}),
            QuantInvariantConfig(category="fixed-income"),
        )
        result = _by_name(evidence)["finite_numeric_values"]
        self.assertEqual(result.status, "fail")
        self.assertEqual(result.severity, "hard_fail")

    def test_same_time_execution_is_hard_failure(self) -> None:
        evidence = _evaluate_csv(
            pd.DataFrame(
                {
                    "information_time": ["2026-01-01T09:00:00Z"],
                    "signal_time": ["2026-01-01T09:01:00Z"],
                    "execution_time": ["2026-01-01T09:01:00Z"],
                }
            ),
            QuantInvariantConfig(
                category="backtesting",
                review_packs=("data-causality",),
            ),
        )
        result = _by_name(evidence)["time_causality"]
        self.assertEqual(result.status, "fail")
        self.assertEqual(result.severity, "hard_fail")
        self.assertEqual(result.details["ordering_failure_count"], 1)

    def test_accounting_identities_are_recomputed(self) -> None:
        evidence = _evaluate_csv(
            pd.DataFrame(
                {
                    "nav": [150.0],
                    "cash": [50.0],
                    "market_value": [100.0],
                    "total_pnl": [12.0],
                    "realized_pnl": [5.0],
                    "unrealized_pnl": [6.0],
                }
            ),
            QuantInvariantConfig(category="backtesting", review_packs=("accounting",)),
        )
        results = _by_name(evidence)
        self.assertEqual(results["nav_reconciliation"].status, "pass")
        self.assertEqual(results["pnl_reconciliation"].status, "fail")
        self.assertEqual(results["pnl_reconciliation"].severity, "hard_fail")

    def test_nested_json_output_is_supported(self) -> None:
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "results.json"
            output_path.write_text(
                json.dumps(
                    {
                        "zero_rates": {"1": 0.04, "2": 0.045},
                        "discount_factor": {"1": 0.96, "2": 0.91},
                    }
                ),
                encoding="utf-8",
            )
            evidence = evaluate_quant_invariants(
                output_path=output_path,
                config=QuantInvariantConfig(category="fixed-income"),
            )
        self.assertEqual(_by_name(evidence)["nonnegative_quantities"].status, "pass")

    def test_registry_is_deterministic_unique_and_has_all_domain_packs(self) -> None:
        first = invariant_registry_snapshot()
        second = invariant_registry_snapshot()
        self.assertEqual(first, second)
        names = [item["name"] for item in first]
        self.assertEqual(len(names), len(set(names)))
        categories = {category for item in first for category in item["categories"]}
        expected = {
            "derivatives-pricing",
            "fixed-income",
            "credit",
            "factor-research",
            "backtesting",
            "risk-management",
            "microstructure",
            "fx",
            "nlp-on-finance",
            "cross-domain",
        }
        self.assertTrue(expected.issubset(categories))

    def test_evidence_records_provenance_and_confidence(self) -> None:
        evidence = _evaluate_csv(
            pd.DataFrame({"default_probability": [0.2]}),
            QuantInvariantConfig(category="credit"),
        )
        result = _by_name(evidence)["probability_bounds"]
        self.assertEqual(result.details["provenance"], "deterministic_output_invariant")
        self.assertEqual(result.details["confidence"], 1.0)

    def test_missing_domain_inputs_skip_with_explicit_reason(self) -> None:
        evidence = _evaluate_csv(
            pd.DataFrame({"unrelated_metric": [1.0]}),
            QuantInvariantConfig(category="derivatives-pricing"),
        )
        skipped = _by_name(evidence)["quant_invariant_coverage"].details["skipped_checks"]
        self.assertIn("put_call_parity", skipped)
        self.assertTrue(skipped["put_call_parity"])

    def test_declared_invariant_can_activate_check(self) -> None:
        evidence = _evaluate_csv(
            pd.DataFrame({"bid_price": [101.0], "ask_price": [100.0]}),
            QuantInvariantConfig(
                category="credit",
                declared_invariants=("bid_ask_ordering",),
            ),
        )
        self.assertEqual(_by_name(evidence)["bid_ask_ordering"].status, "fail")


class QuantInvariantDomainPackTests(unittest.TestCase):
    def test_derivatives_pack_recomputes_put_call_parity(self) -> None:
        spot = 100.0
        strike = 100.0
        rate = 0.05
        time = 1.0
        put = 5.0
        call = put + spot - strike * np.exp(-rate * time)
        evidence = _evaluate_csv(
            pd.DataFrame(
                {
                    "spot": [spot],
                    "strike": [strike],
                    "risk_free_rate": [rate],
                    "time_to_maturity": [time],
                    "call_price": [call],
                    "put_price": [put],
                    "call_delta": [0.60],
                    "put_delta": [-0.40],
                    "gamma": [0.02],
                    "vega": [20.0],
                }
            ),
            QuantInvariantConfig(category="derivatives-pricing"),
        )
        results = _by_name(evidence)
        self.assertEqual(results["put_call_parity"].status, "pass")
        self.assertEqual(results["option_greek_bounds"].status, "pass")
        self.assertEqual(results["option_price_bounds"].status, "pass")

    def test_derivatives_parity_violation_is_hard_failure(self) -> None:
        evidence = _evaluate_csv(
            pd.DataFrame(
                {
                    "spot": [100.0],
                    "strike": [100.0],
                    "risk_free_rate": [0.05],
                    "time_to_maturity": [1.0],
                    "call_price": [20.0],
                    "put_price": [5.0],
                }
            ),
            QuantInvariantConfig(category="derivatives-pricing"),
        )
        result = _by_name(evidence)["put_call_parity"]
        self.assertEqual(result.status, "fail")
        self.assertEqual(result.severity, "hard_fail")

    def test_fixed_income_dv01_identity_is_recomputed(self) -> None:
        evidence = _evaluate_csv(
            pd.DataFrame(
                {
                    "price": [100.0],
                    "modified_duration": [5.0],
                    "dv01": [0.10],
                    "tenor": [1.0],
                    "discount_factor": [0.96],
                }
            ),
            QuantInvariantConfig(category="fixed-income"),
        )
        result = _by_name(evidence)["dv01_duration_consistency"]
        self.assertEqual(result.status, "fail")
        self.assertEqual(result.severity, "hard_fail")

    def test_parquet_round_trip_when_engine_available(self) -> None:
        evidence = _evaluate_parquet(
            self,
            pd.DataFrame({"tenor": [1.0, 2.0], "discount_factor": [0.96, 0.91]}),
            QuantInvariantConfig(category="fixed-income"),
        )
        self.assertEqual(_by_name(evidence)["nonnegative_quantities"].status, "pass")

    def test_fixed_income_monotonicity_is_warning_not_hard_failure(self) -> None:
        evidence = _evaluate_csv(
            pd.DataFrame({"tenor": [1.0, 2.0], "discount_factor": [0.95, 0.97]}),
            QuantInvariantConfig(category="fixed-income"),
        )
        result = _by_name(evidence)["discount_factor_monotonicity"]
        self.assertEqual(result.status, "warning")
        self.assertEqual(result.severity, "warning")

    def test_credit_survival_curve_must_not_increase(self) -> None:
        evidence = _evaluate_csv(
            pd.DataFrame(
                {
                    "tenor": [1.0, 3.0, 5.0],
                    "survival_probability": [0.98, 0.94, 0.96],
                }
            ),
            QuantInvariantConfig(category="credit"),
        )
        self.assertEqual(
            _by_name(evidence)["survival_probability_monotonicity"].status,
            "fail",
        )

    def test_factor_pack_checks_ic_and_dollar_neutrality(self) -> None:
        evidence = _evaluate_csv(
            pd.DataFrame(
                {
                    "date": ["2026-01-01", "2026-01-01"],
                    "weight": [0.6, -0.4],
                    "information_coefficient": [0.2, 1.2],
                }
            ),
            QuantInvariantConfig(category="factor-research"),
        )
        results = _by_name(evidence)
        self.assertEqual(results["information_coefficient_bounds"].status, "fail")
        self.assertEqual(results["dollar_neutrality"].status, "fail")

    def test_backtesting_pack_recomputes_compounding(self) -> None:
        evidence = _evaluate_csv(
            pd.DataFrame(
                {
                    "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
                    "portfolio_value": [100.0, 110.0, 108.9],
                    "portfolio_return": [0.0, 0.10, -0.01],
                }
            ),
            QuantInvariantConfig(category="backtesting"),
        )
        self.assertEqual(_by_name(evidence)["portfolio_compounding"].status, "pass")

    def test_risk_pack_enforces_positive_loss_ordering(self) -> None:
        evidence = _evaluate_csv(
            pd.DataFrame(
                {
                    "var": [10.0],
                    "expected_shortfall": [8.0],
                    "var_95": [9.0],
                    "var_99": [7.0],
                }
            ),
            QuantInvariantConfig(category="risk-management"),
        )
        results = _by_name(evidence)
        self.assertEqual(results["expected_shortfall_vs_var"].status, "fail")
        self.assertEqual(results["var_confidence_monotonicity"].status, "fail")

    def test_risk_pack_supports_signed_pnl_convention(self) -> None:
        evidence = _evaluate_csv(
            pd.DataFrame(
                {
                    "var": [-10.0],
                    "expected_shortfall": [-12.0],
                    "var_95": [-9.0],
                    "var_99": [-11.0],
                }
            ),
            QuantInvariantConfig(category="risk-management"),
        )
        results = _by_name(evidence)
        self.assertEqual(results["expected_shortfall_vs_var"].status, "pass")
        self.assertEqual(results["var_confidence_monotonicity"].status, "pass")

    def test_microstructure_pack_checks_book_and_vwap(self) -> None:
        evidence = _evaluate_csv(
            pd.DataFrame(
                {
                    "bid_price": [101.0],
                    "ask_price": [100.0],
                    "low": [95.0],
                    "high": [105.0],
                    "vwap": [106.0],
                }
            ),
            QuantInvariantConfig(category="microstructure"),
        )
        results = _by_name(evidence)
        self.assertEqual(results["bid_ask_ordering"].status, "fail")
        self.assertEqual(results["vwap_range"].status, "fail")

    def test_microstructure_execution_quantity_reconciles(self) -> None:
        evidence = _evaluate_csv(
            pd.DataFrame(
                {
                    "executed_quantity": [40.0, 60.0],
                    "target_quantity": [100.0, 100.0],
                }
            ),
            QuantInvariantConfig(category="microstructure"),
        )
        self.assertEqual(
            _by_name(evidence)["execution_quantity_reconciliation"].status,
            "pass",
        )

    def test_fx_pack_checks_cip_and_triangle(self) -> None:
        spot = 1.10
        rd = 0.05
        rf = 0.03
        time = 0.5
        forward = spot * np.exp((rd - rf) * time)
        evidence = _evaluate_csv(
            pd.DataFrame(
                {
                    "spot_rate": [spot],
                    "forward_rate": [forward],
                    "domestic_rate": [rd],
                    "foreign_rate": [rf],
                    "time_to_maturity": [time],
                    "eurusd": [1.10],
                    "usdjpy": [150.0],
                    "eurjpy": [165.0],
                }
            ),
            QuantInvariantConfig(category="fx"),
        )
        results = _by_name(evidence)
        self.assertEqual(results["covered_interest_parity"].status, "pass")
        self.assertEqual(results["triangular_fx_consistency"].status, "pass")

    def test_fx_pack_respects_percentage_rate_units(self) -> None:
        spot = 1.10
        forward = spot * np.exp((0.05 - 0.03) * 0.5)
        evidence = _evaluate_csv(
            pd.DataFrame(
                {
                    "spot_rate": [spot],
                    "forward_rate": [forward],
                    "domestic_rate": [5.0],
                    "foreign_rate": [3.0],
                    "time_to_maturity": [0.5],
                }
            ),
            QuantInvariantConfig(
                category="fx",
                units={"domestic_rate": "percent", "foreign_rate": "percent"},
            ),
        )
        self.assertEqual(_by_name(evidence)["covered_interest_parity"].status, "pass")

    def test_nlp_pack_checks_sentiment_and_entity_counts(self) -> None:
        evidence = _evaluate_csv(
            pd.DataFrame({"sentiment_score": [1.3], "entity_count": [2.5]}),
            QuantInvariantConfig(category="nlp-on-finance"),
        )
        results = _by_name(evidence)
        self.assertEqual(results["sentiment_bounds"].status, "fail")
        self.assertEqual(results["entity_count_integrality"].status, "fail")

    def test_cross_domain_uses_union_of_domain_packs(self) -> None:
        evidence = _evaluate_csv(
            pd.DataFrame(
                {
                    "survival_probability": [0.98, 0.95],
                    "tenor": [1.0, 2.0],
                    "spot_rate": [1.10, 1.10],
                    "forward_rate": [1.11105518, 1.11105518],
                    "domestic_rate": [0.05, 0.05],
                    "foreign_rate": [0.03, 0.03],
                    "time_to_maturity": [0.5, 0.5],
                }
            ),
            QuantInvariantConfig(category="cross-domain", review_packs=("credit", "fx")),
        )
        results = _by_name(evidence)
        self.assertIn("survival_probability_monotonicity", results)
        self.assertIn("covered_interest_parity", results)
        self.assertEqual(results["cross_domain_pack_coverage"].status, "pass")
        self.assertEqual(
            results["quant_invariant_coverage"].details["active_domain_packs"],
            ["credit", "fx"],
        )

    def test_cross_domain_undercoverage_is_warning_not_hard_failure(self) -> None:
        evidence = _evaluate_csv(
            pd.DataFrame({"sentiment_score": [0.2]}),
            QuantInvariantConfig(category="cross-domain"),
        )
        result = _by_name(evidence)["cross_domain_pack_coverage"]
        self.assertEqual(result.status, "warning")
        self.assertEqual(result.severity, "warning")

class CrossDomainInvariantActivationTests(
    unittest.TestCase
):
    def test_cross_domain_activates_only_constituent_domains(
        self,
    ) -> None:
        config = QuantInvariantConfig(
            category="cross-domain",
            review_packs=(
                "software",
                "data-causality",
                "numerical",
                "accounting",
                "cross-domain",
                "derivatives",
                "fixed-income",
            ),
        )

        active = {
            check.name
            for check in _CHECKS
            if check.is_active(config)
        }

        self.assertIn(
            "put_call_parity",
            active,
        )
        self.assertIn(
            "option_price_bounds",
            active,
        )
        self.assertIn(
            "discount_factor_monotonicity",
            active,
        )
        self.assertIn(
            "dv01_duration_consistency",
            active,
        )

        unrelated = {
            "survival_probability_monotonicity",
            "information_coefficient_bounds",
            "dollar_neutrality",
            "portfolio_compounding",
            "risk_measure_ordering",
            "bid_ask_ordering",
            "vwap_range",
            "execution_quantity_reconciliation",
            "covered_interest_parity",
            "triangular_fx_consistency",
            "sentiment_bounds",
            "entity_count_integrality",
        }

        self.assertTrue(
            unrelated.isdisjoint(active),
            msg=(
                "Unexpected unrelated invariant activation: "
                f"{sorted(unrelated & active)}"
            ),
        )

        self.assertIn(
            "cross_domain_pack_coverage",
            active,
        )
if __name__ == "__main__":
    unittest.main()
