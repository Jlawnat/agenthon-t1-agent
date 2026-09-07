from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any

from agent.compiled_specification import CompiledSpecification
from agent.planning import TaskPlan
from agent.skill_packs import SkillPack


@dataclass
class CandidateStrategy:
    candidate_id: int
    approach_name: str
    implementation_steps: list[str]
    verification_steps: list[str]
    numerical_method: str | None
    risks: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlannerOutput:
    task_summary: str
    shared_requirements: list[str]
    candidate_strategies: list[CandidateStrategy]
    final_checks: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_planner_prompt(
    spec: CompiledSpecification,
    task_plan: TaskPlan,
    skill_packs: list[SkillPack],
    instruction_text: str,
) -> str:

    payload = {
        "instruction": instruction_text,
        "compiled_specification": spec.to_dict(),
        "task_plan": task_plan.to_dict(),
        "active_skill_packs": [
            pack.to_dict()
            for pack in skill_packs
        ],
    }

    context = json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

    return f"""
You are the planning component of a quantitative-finance coding agent.

Your job is to design an implementation plan.

Do NOT write code yet.

You must produce candidate strategies that satisfy the task specification
and the quantitative-finance verification requirements.

Important rules:

- Respect all explicit output paths and schema requirements.
- Do not invent additional deliverables.
- Use the requested number of candidate strategies.
- Candidates should be meaningfully different when independent methods
  are requested.
- Prefer numerically stable approaches.
- Include verification steps before accepting a candidate.
- Include financial invariants where relevant.
- Include edge-case handling.
- For backtesting or time-series work, enforce causality.
- For trading systems, include accounting reconciliation.
- For numerical tasks, include perturbation and convergence checks
  where required.
- Do not rely on hidden tests or benchmark checker files.
- Do not generate implementation code.

Return ONLY valid JSON in this exact structure:

{{
  "task_summary": "",
  "shared_requirements": [],
  "candidate_strategies": [
    {{
      "candidate_id": 1,
      "approach_name": "",
      "implementation_steps": [],
      "verification_steps": [],
      "numerical_method": null,
      "risks": []
    }}
  ],
  "final_checks": []
}}

The number of candidate strategies MUST equal:

{task_plan.candidate_count}

TASK CONTEXT:

{context}
""".strip()