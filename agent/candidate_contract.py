from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class VerificationEvidence:
    name: str
    status: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateAttempt:
    attempt_number: int

    solver_path: str | None = None

    return_code: int | None = None
    timed_out: bool = False
    runtime_seconds: float | None = None

    stdout: str = ""
    stderr: str = ""

    output_files: list[str] = field(default_factory=list)

    structural_evidence: list[VerificationEvidence] = field(
        default_factory=list
    )

    quant_evidence: list[VerificationEvidence] = field(
        default_factory=list
    )

    repair_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateRecord:
    candidate_id: int
    approach_name: str

    strategy: dict[str, Any]

    attempts: list[CandidateAttempt] = field(
        default_factory=list
    )

    current_status: str = "pending"

    hard_failures: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    score: float | None = None

    selected: bool = False

    def add_attempt(
        self,
        attempt: CandidateAttempt,
    ) -> None:
        self.attempts.append(attempt)

    def latest_attempt(
        self,
    ) -> CandidateAttempt | None:
        if not self.attempts:
            return None

        return self.attempts[-1]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)