from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PipelineStage(str, Enum):
    SNAPSHOT = "snapshot"
    SPECIFICATION = "specification"
    COMPILE = "compile"
    PLAN = "plan"
    GENERATE = "generate"
    VALIDATE_CODE = "validate_code"
    ISOLATED_EXECUTE = "isolated_execute"
    COLLECT_EVIDENCE = "collect_evidence"
    EVALUATE = "structural_quant_evaluate"
    TARGETED_REPAIR = "targeted_repair"
    SELECT = "select"
    CLEAN_ROOM_AUDIT = "clean_room_audit"
    PUBLISH = "publish"


PIPELINE_ORDER: tuple[PipelineStage, ...] = (
    PipelineStage.SNAPSHOT,
    PipelineStage.SPECIFICATION,
    PipelineStage.COMPILE,
    PipelineStage.PLAN,
    PipelineStage.GENERATE,
    PipelineStage.VALIDATE_CODE,
    PipelineStage.ISOLATED_EXECUTE,
    PipelineStage.COLLECT_EVIDENCE,
    PipelineStage.EVALUATE,
    PipelineStage.TARGETED_REPAIR,
    PipelineStage.SELECT,
    PipelineStage.CLEAN_ROOM_AUDIT,
    PipelineStage.PUBLISH,
)


class RunStatus(str, Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class InvalidStateTransition(RuntimeError):
    pass


@dataclass(frozen=True)
class StageEvent:
    stage: PipelineStage
    status: str
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass
class OrchestratorState:
    run_id: str
    task_id: str | None = None
    status: RunStatus = RunStatus.NOT_STARTED
    current_stage: PipelineStage | None = None
    completed_stages: list[PipelineStage] = field(
        default_factory=list
    )
    history: list[StageEvent] = field(
        default_factory=list
    )
    failure_reason: str | None = None

    def expected_stage(self) -> PipelineStage | None:
        if len(self.completed_stages) >= len(
            PIPELINE_ORDER
        ):
            return None

        return PIPELINE_ORDER[
            len(self.completed_stages)
        ]

    def start_stage(
        self,
        stage: PipelineStage,
    ) -> None:
        if self.status in {
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
        }:
            raise InvalidStateTransition(
                f"Cannot start {stage.value}: "
                f"run is already {self.status.value}."
            )

        if self.current_stage is not None:
            raise InvalidStateTransition(
                f"Cannot start {stage.value}: "
                f"{self.current_stage.value} "
                "is still active."
            )

        expected = self.expected_stage()

        if stage is not expected:
            expected_name = (
                expected.value
                if expected is not None
                else "<none>"
            )

            raise InvalidStateTransition(
                "Invalid stage transition: "
                f"expected {expected_name}, "
                f"got {stage.value}."
            )

        self.status = RunStatus.RUNNING
        self.current_stage = stage

        self.history.append(
            StageEvent(
                stage=stage,
                status="started",
            )
        )

    def complete_stage(
        self,
        stage: PipelineStage,
        *,
        detail: str | None = None,
    ) -> None:
        if self.current_stage is not stage:
            active = (
                self.current_stage.value
                if self.current_stage is not None
                else "<none>"
            )

            raise InvalidStateTransition(
                f"Cannot complete {stage.value}: "
                f"active stage is {active}."
            )

        self.completed_stages.append(stage)
        self.current_stage = None

        self.history.append(
            StageEvent(
                stage=stage,
                status="completed",
                detail=detail,
            )
        )

        if len(self.completed_stages) == len(
            PIPELINE_ORDER
        ):
            self.status = RunStatus.SUCCEEDED

    def fail_stage(
        self,
        stage: PipelineStage,
        *,
        reason: str,
    ) -> None:
        if self.current_stage is not stage:
            active = (
                self.current_stage.value
                if self.current_stage is not None
                else "<none>"
            )

            raise InvalidStateTransition(
                f"Cannot fail {stage.value}: "
                f"active stage is {active}."
            )

        self.status = RunStatus.FAILED
        self.failure_reason = reason
        self.current_stage = None

        self.history.append(
            StageEvent(
                stage=stage,
                status="failed",
                detail=reason,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "status": self.status.value,
            "current_stage": (
                self.current_stage.value
                if self.current_stage is not None
                else None
            ),
            "completed_stages": [
                stage.value
                for stage in self.completed_stages
            ],
            "history": [
                event.to_dict()
                for event in self.history
            ],
            "failure_reason": self.failure_reason,
        }