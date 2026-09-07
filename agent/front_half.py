from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any, Callable, Mapping

from agent.compiled_specification import (
    CompiledSpecification,
)
from agent.orchestrator_state import (
    OrchestratorState,
    PipelineStage,
)
from agent.run_context import RunContext
from agent.spec_compiler import (
    compile_specification,
)
from agent.specification import (
    TaskSpecification,
    load_task_specification,
)


SnapshotAdapter = Callable[
    [Path],
    Any,
]

# Phase 5 keeps legacy three-argument plan adapters working while
# allowing the production adapter to receive the shared RunContext.
PlanAdapter = Callable[..., Any]


@dataclass(frozen=True)
class FrontHalfDependencies:
    """
    Thin adapters around the existing Phase 1–2
    snapshot and planning implementations.

    They do not replace those components.
    """

    snapshot: SnapshotAdapter
    plan: PlanAdapter


@dataclass(frozen=True)
class FrontHalfResult:
    snapshot: Any

    specification: TaskSpecification

    compiled_specification: (
        CompiledSpecification
    )

    plan: Any

    run_context: RunContext


def _fail_active_stage(
    *,
    state: OrchestratorState,
    stage: PipelineStage,
    exc: Exception,
) -> None:
    state.fail_stage(
        stage,
        reason=(
            f"{type(exc).__name__}: {exc}"
        ),
    )


def _invoke_plan_adapter(
    *,
    adapter: PlanAdapter,
    snapshot: Any,
    specification: TaskSpecification,
    compiled_specification: CompiledSpecification,
    run_context: RunContext,
) -> Any:
    """
    Call the plan adapter without breaking the existing Phase 4
    three-argument contract.

    Production Phase 5 adapters may opt into a fourth RunContext
    argument so model-assisted planning consumes the same global
    budget used by generation and repair.
    """

    try:
        signature = inspect.signature(
            adapter
        )
    except (
        TypeError,
        ValueError,
    ):
        # Preserve the original three-argument behavior for
        # callables whose signatures cannot be inspected.
        return adapter(
            snapshot,
            specification,
            compiled_specification,
        )

    parameters = list(
        signature.parameters.values()
    )

    if any(
        parameter.kind
        == inspect.Parameter.VAR_POSITIONAL
        for parameter in parameters
    ):
        return adapter(
            snapshot,
            specification,
            compiled_specification,
            run_context,
        )

    run_context_parameter = (
        signature.parameters.get(
            "run_context"
        )
    )

    if (
        run_context_parameter is not None
        and run_context_parameter.kind
        == inspect.Parameter.KEYWORD_ONLY
    ):
        return adapter(
            snapshot,
            specification,
            compiled_specification,
            run_context=run_context,
        )

    positional_parameters = [
        parameter
        for parameter in parameters
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    ]

    if len(positional_parameters) >= 4:
        return adapter(
            snapshot,
            specification,
            compiled_specification,
            run_context,
        )

    return adapter(
        snapshot,
        specification,
        compiled_specification,
    )


def run_deterministic_front_half(
    *,
    task_dir: Path,
    state: OrchestratorState,
    dependencies: FrontHalfDependencies,
    environ: Mapping[str, str] | None = None,
) -> FrontHalfResult:
    """
    Execute exactly:

    snapshot
        ->
    specification
        ->
    compile
        ->
    plan

    This module performs no model/provider calls itself.
    A production planning adapter may use the supplied shared
    RunContext so any model-assisted planning remains globally
    budgeted.
    """

    task_dir = task_dir.resolve()

    # -------------------------------------------------
    # 1. SNAPSHOT
    # -------------------------------------------------

    stage = PipelineStage.SNAPSHOT

    state.start_stage(stage)

    try:
        snapshot = dependencies.snapshot(
            task_dir
        )

    except Exception as exc:
        _fail_active_stage(
            state=state,
            stage=stage,
            exc=exc,
        )
        raise

    state.complete_stage(
        stage,
        detail="task snapshot created",
    )

    # -------------------------------------------------
    # 2. SPECIFICATION
    # -------------------------------------------------

    stage = PipelineStage.SPECIFICATION

    state.start_stage(stage)

    try:
        specification = (
            load_task_specification(
                task_dir
            )
        )

        state.task_id = (
            specification.task_id
        )

        run_context = (
            RunContext.from_task_specification(
                specification,
                environ=environ,
            )
        )

    except Exception as exc:
        _fail_active_stage(
            state=state,
            stage=stage,
            exc=exc,
        )
        raise

    state.complete_stage(
        stage,
        detail="task specification loaded",
    )

    # -------------------------------------------------
    # 3. COMPILE
    # -------------------------------------------------

    stage = PipelineStage.COMPILE

    state.start_stage(stage)

    try:
        compiled_specification = (
            compile_specification(
                specification
            )
        )

    except Exception as exc:
        _fail_active_stage(
            state=state,
            stage=stage,
            exc=exc,
        )
        raise

    state.complete_stage(
        stage,
        detail="specification compiled",
    )

    # -------------------------------------------------
    # 4. PLAN
    # -------------------------------------------------

    stage = PipelineStage.PLAN

    state.start_stage(stage)

    try:
        plan = _invoke_plan_adapter(
            adapter=dependencies.plan,
            snapshot=snapshot,
            specification=specification,
            compiled_specification=(
                compiled_specification
            ),
            run_context=run_context,
        )

    except Exception as exc:
        _fail_active_stage(
            state=state,
            stage=stage,
            exc=exc,
        )
        raise

    state.complete_stage(
        stage,
        detail="task plan created",
    )

    return FrontHalfResult(
        snapshot=snapshot,
        specification=specification,
        compiled_specification=(
            compiled_specification
        ),
        plan=plan,
        run_context=run_context,
    )
