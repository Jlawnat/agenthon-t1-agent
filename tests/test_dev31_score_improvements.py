from __future__ import annotations

from types import SimpleNamespace
import unittest

from agent.candidate_prompt import (
    build_precision_guidance,
)
from agent.orchestrator import (
    _resolve_execution_timeout,
)
from agent.spec_compiler import (
    compile_specification,
    infer_difficulty,
    resolve_difficulty,
)
from agent.spec_enrichment import (
    SpecificationEnrichment,
)
from agent.spec_merge import (
    merge_specification,
)
from agent.specification import (
    TaskSpecification,
)


def _historical_var_spec() -> TaskSpecification:
    return TaskSpecification(
        task_id="historical-var-test",
        category="risk-management",
        instruction_text=(
            "Historical Portfolio Value-at-Risk with Data Cleaning. "
            "Produce results.json and solution.json for a portfolio "
            "simulation after cleaning dirty daily returns."
        ),
        required_output_paths=[
            "results.json",
            "solution.json",
        ],
        candidate_schema_fields=[],
        required_output_columns=[],
        required_output_dtypes={},
        runtime_seconds=1800.0,
        network_mode="restricted",
        raw_card={
            "metadata": {
                "difficulty": "easy",
            }
        },
    )


class Dev31ScoreImprovementTests(
    unittest.TestCase
):
    def test_historical_var_guidance_preserves_cleaning_semantics(
        self,
    ) -> None:
        notes = build_precision_guidance(
            "Historical Portfolio Value-at-Risk with Data Cleaning "
            "from dirty daily returns and a supplied trading calendar."
        )
        joined = " ".join(
            notes
        ).lower()

        self.assertIn(
            "ordered state transitions",
            joined,
        )
        self.assertIn(
            "last occurrence",
            joined,
        )
        self.assertIn(
            "nan-safe",
            joined,
        )
        self.assertIn(
            "percentile",
            joined,
        )
        self.assertIn(
            "reconcile",
            joined,
        )

    def test_card_difficulty_overrides_text_heuristic(
        self,
    ) -> None:
        spec = _historical_var_spec()

        self.assertNotEqual(
            infer_difficulty(spec),
            "easy",
        )

        self.assertEqual(
            resolve_difficulty(spec),
            "easy",
        )

        compiled = compile_specification(
            spec
        )

        self.assertEqual(
            compiled.difficulty,
            "easy",
        )

    def test_model_enrichment_cannot_override_deterministic_difficulty(
        self,
    ) -> None:
        spec = _historical_var_spec()
        compiled = compile_specification(
            spec
        )

        enrichment = SpecificationEnrichment(
            deliverables=[],
            required_columns=[],
            required_row_rules=[],
            ordering_rules=[],
            units={},
            conventions=[],
            edge_cases=[],
            invariants=[],
            assumptions=[],
            unresolved_questions=[],
            difficulty="hard",
            reasoning_summary="model guess",
        )

        merged = merge_specification(
            deterministic_spec=spec,
            compiled_spec=compiled,
            enrichment=enrichment,
        )

        self.assertEqual(
            merged.difficulty,
            "easy",
        )

    def test_adaptive_execution_timeout_scales_with_card_budget(
        self,
    ) -> None:
        medium = SimpleNamespace(
            card_timeout_seconds=1800.0,
        )
        hard = SimpleNamespace(
            card_timeout_seconds=2400.0,
        )

        self.assertEqual(
            _resolve_execution_timeout(
                requested_seconds=120.0,
                run_context=medium,
                candidate_count=2,
            ),
            180.0,
        )

        self.assertEqual(
            _resolve_execution_timeout(
                requested_seconds=120.0,
                run_context=hard,
                candidate_count=3,
            ),
            160.0,
        )

    def test_adaptive_execution_timeout_has_cap_but_respects_explicit_larger_request(
        self,
    ) -> None:
        context = SimpleNamespace(
            card_timeout_seconds=5400.0,
        )

        self.assertEqual(
            _resolve_execution_timeout(
                requested_seconds=120.0,
                run_context=context,
                candidate_count=1,
            ),
            240.0,
        )

        self.assertEqual(
            _resolve_execution_timeout(
                requested_seconds=300.0,
                run_context=context,
                candidate_count=1,
            ),
            300.0,
        )


if __name__ == "__main__":
    unittest.main()
