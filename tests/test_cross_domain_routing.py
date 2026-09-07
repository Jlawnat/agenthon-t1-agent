import unittest

from agent.spec_compiler import infer_review_packs
from agent.specification import TaskSpecification


BASE_CROSS_DOMAIN_PACKS = [
    "software",
    "data-causality",
    "numerical",
    "accounting",
    "cross-domain",
]


def _spec(text: str) -> TaskSpecification:
    return TaskSpecification(
        task_id="cross-domain-test",
        category="cross-domain",
        instruction_text=text,
        required_output_paths=[],
        candidate_schema_fields=[],
        required_output_columns=[],
        runtime_seconds=60,
        network_mode="restricted",
        raw_card={},
        required_output_dtypes={},
    )


class CrossDomainRoutingTests(unittest.TestCase):
    def test_infers_relevant_constituent_domains(self) -> None:
        spec = _spec(
            """
            Price a down-and-out barrier option analytically.
            Compute Greeks and a parametric VaR decomposition
            for the option portfolio.
            """
        )

        self.assertEqual(
            infer_review_packs(spec),
            BASE_CROSS_DOMAIN_PACKS
            + [
                "derivatives",
                "risk-management",
            ],
        )

    def test_infers_multiple_distinct_domains(self) -> None:
        spec = _spec(
            """
            Decompose a structured note into a zero-coupon bond
            and barrier option. Compute DV01, option value,
            VaR and expected shortfall.
            """
        )

        self.assertEqual(
            infer_review_packs(spec),
            BASE_CROSS_DOMAIN_PACKS
            + [
                "derivatives",
                "fixed-income",
                "risk-management",
            ],
        )

    def test_does_not_route_incidental_finance_words(self) -> None:
        spec = _spec(
            """
            Use NumPy default population standard deviation.
            All P&L fields are expressed in currency units.
            Report each Greek factor contribution.
            This diagnostic is not a tail stress test.
            """
        )

        self.assertEqual(
            infer_review_packs(spec),
            BASE_CROSS_DOMAIN_PACKS,
        )

    def test_credit_background_reference_does_not_activate_credit(self) -> None:
        spec = _spec(
            """
            First-passage-time formulas are foundational for
            barrier option pricing, credit risk modeling,
            and drawdown analysis.

            Compute first-passage probabilities for an
            up-barrier and a down-barrier.
            """
        )

        packs = infer_review_packs(spec)

        self.assertIn(
            "derivatives",
            packs,
        )
        self.assertNotIn(
            "credit",
            packs,
        )

    def test_specific_credit_semantics_activate_credit(self) -> None:
        spec = _spec(
            """
            Compute CDS survival probabilities from hazard rates,
            recovery rates, and loss-given-default assumptions.
            """
        )

        self.assertEqual(
            infer_review_packs(spec),
            BASE_CROSS_DOMAIN_PACKS
            + [
                "credit",
            ],
        )


if __name__ == "__main__":
    unittest.main()
