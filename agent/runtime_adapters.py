from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from agent.candidate_production import (
    GeneratedCandidate,
    GeneratorAdapter,
)
from agent.candidate_prompt import (
    build_candidate_prompt,
    build_precision_guidance,
)
from agent.compiled_specification import CompiledSpecification
from agent.front_half import FrontHalfDependencies
from agent.model_client import ModelClient, ModelResponse
from agent.planner import build_planner_prompt
from agent.planner_validator import parse_planner_output
from agent.planning import build_task_plan
from agent.run_context import RunContext
from agent.repair_pipeline import (
    RepairedCandidate,
    RepairAdapter,
    RepairRequest,
)
from agent.skill_packs import load_skill_packs
from agent.spec_enrichment import build_specification_prompt
from agent.spec_merge import merge_specification
from agent.spec_response_validator import parse_spec_enrichment
from agent.specification import TaskSpecification
from agent.task_snapshot import TaskSnapshot


@dataclass(frozen=True)
class _StrategyEnvelope:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True)
class RuntimePlan:
    """
    Production plan payload shared by the generator adapter.

    It preserves the existing deterministic task plan, optional
    model planner output, active skill packs, and safe data
    inspections without changing the Phase 4 orchestrator API.
    """

    planning_specification: CompiledSpecification
    task_plan: Any
    planner_output: Any
    skill_packs: tuple[Any, ...]
    data_inspections: dict[str, dict[str, Any]]
    strategies: tuple[_StrategyEnvelope, ...]


def _response_tokens(
    response: ModelResponse,
) -> int:
    usage = response.raw.get("usage")

    if not isinstance(usage, dict):
        return 0

    for key in (
        "output_tokens",
        "completion_tokens",
        "total_tokens",
    ):
        value = usage.get(key)

        if isinstance(value, bool):
            continue

        if isinstance(value, (int, float)):
            return max(0, int(value))

    return 0


def _object_to_dict(
    value: Any,
) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return dict(value)

    to_dict = getattr(value, "to_dict", None)

    if callable(to_dict):
        result = to_dict()

        if isinstance(result, dict):
            return dict(result)

    return None


def _find_strategy_payloads(
    planner_output: Any,
) -> list[dict[str, Any]]:
    """
    Accept the current planner schema without coupling the
    competition entrypoint to one attribute spelling.
    """

    root = _object_to_dict(planner_output)

    if root is None:
        return []

    preferred_keys = (
        "candidate_strategies",
        "strategies",
        "candidates",
    )

    for key in preferred_keys:
        value = root.get(key)

        if not isinstance(value, list):
            continue

        payloads = [
            item
            for item in value
            if isinstance(item, dict)
        ]

        if payloads:
            return [
                dict(item)
                for item in payloads
            ]

    queue: list[Any] = [root]

    while queue:
        current = queue.pop(0)

        if isinstance(current, dict):
            for key, value in current.items():
                if (
                    key in preferred_keys
                    and isinstance(value, list)
                ):
                    payloads = [
                        item
                        for item in value
                        if isinstance(item, dict)
                    ]

                    if payloads:
                        return [
                            dict(item)
                            for item in payloads
                        ]

                queue.extend(current.values())

        elif isinstance(current, list):
            queue.extend(current)

    return []


def _fallback_strategy_payload(
    candidate_id: int,
) -> dict[str, Any]:
    variants = (
        {
            "approach_name": "contract_first",
            "objective": (
                "Implement the simplest correct solution that "
                "satisfies the exact output contract."
            ),
            "verification_steps": [
                "re-read required filenames and schemas",
                "verify identifiers and row coverage",
                "check core financial invariants",
            ],
        },
        {
            "approach_name": "independent_recomputation",
            "objective": (
                "Use an independently checkable quantitative "
                "formulation where practical."
            ),
            "verification_steps": [
                "compute the result using the chosen primary method",
                "independently reconcile high-value identities",
                "check units, signs, timing, and tolerances",
            ],
        },
        {
            "approach_name": "robust_edge_case_first",
            "objective": (
                "Prioritise numerical stability, edge cases, "
                "and defensive input handling."
            ),
            "verification_steps": [
                "test finite-value and near-zero cases",
                "check ordering and time causality",
                "verify output serialization exactly",
            ],
        },
    )

    base = dict(
        variants[
            (candidate_id - 1)
            % len(variants)
        ]
    )

    base["candidate_id"] = candidate_id

    return base


def _strategy_for_candidate(
    plan: RuntimePlan,
    candidate_id: int,
    candidate_seed: int,
) -> _StrategyEnvelope:
    if (
        candidate_id > 0
        and candidate_id <= len(plan.strategies)
    ):
        payload = (
            plan.strategies[
                candidate_id - 1
            ].to_dict()
        )
    else:
        payload = _fallback_strategy_payload(
            candidate_id
        )

    payload["candidate_id"] = candidate_id
    payload["candidate_seed"] = candidate_seed

    return _StrategyEnvelope(
        payload=payload
    )


def _approach_name(
    strategy: _StrategyEnvelope,
    candidate_id: int,
) -> str:
    payload = strategy.to_dict()

    for key in (
        "approach_name",
        "name",
        "strategy_name",
        "method",
    ):
        value = payload.get(key)

        if isinstance(value, str):
            value = value.strip()

            if value:
                return value[:120]

    return f"candidate-{candidate_id}"


def _build_repair_prompt(
    request: RepairRequest,
) -> str:
    payload = {
        "instruction": (
            request.specification
            .instruction_text
        ),
        "compiled_specification": (
            request.compiled_specification
            .to_dict()
        ),
        "candidate_id": request.candidate_id,
        "candidate_seed": request.candidate_seed,
        "repair_brief": request.brief.to_dict(),
        "precision_guidance": build_precision_guidance(
            request.specification.instruction_text
        ),
        "source_code": request.source_code,
    }

    context = json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

    return f"""
You are repairing one candidate solution for a quantitative-finance coding task.

Return ONLY complete valid Python source code.
Do not use Markdown.
Do not use code fences.
Do not include explanations before or after the code.

Security and contract requirements:
- Use only the task contract and the sanitized repair evidence supplied below.
- Do not infer or request hidden checker logic, canaries, oracle values, reference outputs, reward files, or grader secrets.
- Do not access the internet or use network clients.
- Do not create child processes, invoke shells, or call external commands.
- Do not install packages at runtime.
- Inspect environment variables only for INPUT_DIR, OUTPUT_DIR, QFBENCH_SEED, PYTHONHASHSEED, and ordinary deterministic runtime settings.
- Resolve task input with Path(os.environ.get("INPUT_DIR", "/input")).
- Resolve task output with Path(os.environ.get("OUTPUT_DIR", "/output")).
- All staged task data is under INPUT_DIR / "data".
- If the benchmark instruction mentions legacy input locations such as /app/params.json, /app/data/file.csv, or /input/environment/data/file.csv, map the corresponding task-data file into INPUT_DIR / "data" using the supplied task/data context. Do not access those legacy absolute paths literally.
- If the benchmark instruction names /app/output/<path> or /output/<path>, write that relative path under the resolved OUTPUT_DIR. Do not hardcode /app/output or /output when OUTPUT_DIR is supplied.
- Do not modify input files.
- Write only the required deliverables; do not create reward.json, reward.txt, pytest_report.json, checker files, canaries, or diagnostic side artifacts.
- Preserve required filenames, nested paths, schemas, units, ordering, and financial conventions.
- Keep deterministic behavior and use the supplied candidate seed if randomness is required.

Repair requirements:
- Fix the root cause identified by the repair brief.
- Preserve correct parts of the original solution.
- Do not hardcode expected benchmark answers.
- Recheck syntax, imports, output paths, schema, dtypes, identifiers, financial invariants, and numerical edge cases before returning code.

REPAIR CONTEXT:

{context}
""".strip()


def build_runtime_components(
    model_client: ModelClient,
) -> tuple[
    FrontHalfDependencies,
    GeneratorAdapter,
    RepairAdapter,
]:
    """
    Construct the real competition-facing adapters used by
    the Phase 4 orchestrator.
    """

    if not model_client.is_available():
        raise RuntimeError(
            "MODEL_ENDPOINT is required for the production runtime."
        )

    shared_run_context: RunContext | None = None

    def _model_timeout(
        run_context: RunContext,
    ) -> float:
        return run_context.bounded_timeout(
            model_client.timeout_seconds
        )

    def snapshot_adapter(
        task_dir,
    ):
        return TaskSnapshot.build(
            task_dir
        )

    def plan_adapter(
        snapshot,
        specification: TaskSpecification,
        compiled_specification: CompiledSpecification,
        run_context: RunContext,
    ) -> RuntimePlan:
        nonlocal shared_run_context

        shared_run_context = run_context

        data_inspections = (
            snapshot.inspect_data()
        )

        specification_prompt = (
            build_specification_prompt(
                spec=specification,
                compiled_spec=(
                    compiled_specification
                ),
            )
        )

        run_context.budget.reserve_model_call()

        enrichment_response = (
            model_client.complete(
                specification_prompt,
                temperature=0.0,
                timeout_seconds=_model_timeout(
                    run_context
                ),
            )
        )

        enrichment_tokens = _response_tokens(
            enrichment_response
        )

        if enrichment_tokens:
            run_context.budget.record_tokens(
                enrichment_tokens
            )

        try:
            enrichment = (
                parse_spec_enrichment(
                    enrichment_response.text
                )
            )

            planning_specification = (
                merge_specification(
                    deterministic_spec=(
                        specification
                    ),
                    compiled_spec=(
                        compiled_specification
                    ),
                    enrichment=enrichment,
                )
            )

        except (ValueError, TypeError, KeyError):
            # Model enrichment is optional. A malformed structured
            # response must not terminate the competition unit.
            # Fall back to the deterministic compiled specification.
            planning_specification = (
                compiled_specification
            )

        task_plan = build_task_plan(
            planning_specification
        )

        skill_packs = tuple(
            load_skill_packs(
                task_plan.review_packs
            )
        )

        planner_prompt = (
            build_planner_prompt(
                spec=planning_specification,
                task_plan=task_plan,
                skill_packs=list(
                    skill_packs
                ),
                instruction_text=(
                    specification
                    .instruction_text
                ),
            )
        )

        run_context.budget.reserve_model_call()

        planner_response = (
            model_client.complete(
                planner_prompt,
                temperature=0.2,
                timeout_seconds=_model_timeout(
                    run_context
                ),
            )
        )

        planner_tokens = _response_tokens(
            planner_response
        )

        if planner_tokens:
            run_context.budget.record_tokens(
                planner_tokens
            )

        try:
            planner_output = (
                parse_planner_output(
                    planner_response.text,
                    expected_candidates=(
                        task_plan.candidate_count
                    ),
                )
            )

        except (ValueError, TypeError, KeyError):
            # Planner structure is advisory. If the model returns an
            # invalid structured plan, continue with the deterministic
            # fallback strategies defined below instead of crashing.
            planner_output = {}

        strategy_payloads = (
            _find_strategy_payloads(
                planner_output
            )
        )

        strategies = tuple(
            _StrategyEnvelope(
                payload=dict(payload)
            )
            for payload in strategy_payloads
        )

        return RuntimePlan(
            planning_specification=(
                planning_specification
            ),
            task_plan=task_plan,
            planner_output=planner_output,
            skill_packs=skill_packs,
            data_inspections=(
                data_inspections
            ),
            strategies=strategies,
        )

    def generate(
        request,
    ) -> GeneratedCandidate:
        if not isinstance(
            request.plan,
            RuntimePlan,
        ):
            raise TypeError(
                "Production generator expected RuntimePlan."
            )

        strategy = _strategy_for_candidate(
            request.plan,
            request.candidate_id,
            request.candidate_seed,
        )

        prompt = build_candidate_prompt(
            instruction_text=(
                request.specification
                .instruction_text
            ),
            spec=(
                request.plan
                .planning_specification
            ),
            strategy=strategy,
            skill_packs=list(
                request.plan.skill_packs
            ),
            data_inspections=(
                request.plan
                .data_inspections
            ),
        )

        if shared_run_context is None:
            raise RuntimeError(
                "RunContext is unavailable for model generation."
            )

        response = model_client.complete(
            prompt,
            temperature=0.0,
            timeout_seconds=_model_timeout(
                shared_run_context
            ),
        )

        return GeneratedCandidate(
            code=response.text.strip(),
            approach_name=_approach_name(
                strategy,
                request.candidate_id,
            ),
            strategy=strategy.to_dict(),
            tokens_used=_response_tokens(
                response
            ),
        )

    def repair(
        request: RepairRequest,
    ) -> RepairedCandidate:
        prompt = _build_repair_prompt(
            request
        )

        if shared_run_context is None:
            raise RuntimeError(
                "RunContext is unavailable for model repair."
            )

        response = model_client.complete(
            prompt,
            temperature=0.0,
            timeout_seconds=_model_timeout(
                shared_run_context
            ),
        )

        return RepairedCandidate(
            code=response.text.strip(),
            tokens_used=_response_tokens(
                response
            ),
        )

    return (
        FrontHalfDependencies(
            snapshot=snapshot_adapter,
            plan=plan_adapter,
        ),
        GeneratorAdapter(
            generate=generate,
            uses_model_budget=True,
            name="house-model-generator",
        ),
        RepairAdapter(
            repair=repair,
            uses_model_budget=True,
            name="house-model-repairer",
        ),
    )