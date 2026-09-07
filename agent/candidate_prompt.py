from __future__ import annotations

import json
from typing import Any

from agent.compiled_specification import CompiledSpecification
from agent.planner import CandidateStrategy
from agent.skill_packs import SkillPack


def build_candidate_prompt(
    *,
    instruction_text: str,
    spec: CompiledSpecification,
    strategy: CandidateStrategy,
    skill_packs: list[SkillPack],
    data_inspections: dict[str, dict[str, Any]],
) -> str:
    payload = {
        "instruction": instruction_text,
        "compiled_specification": spec.to_dict(),
        "candidate_strategy": strategy.to_dict(),
        "active_skill_packs": [
            pack.to_dict()
            for pack in skill_packs
        ],
        "data_inspections": data_inspections,
    }

    context = json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

    return f"""
You are implementing one candidate solution for a quantitative-finance coding task.

Your output will be saved directly as solver.py and executed.

Return ONLY valid Python source code.
Do not use Markdown.
Do not use code fences.
Do not include explanations before or after the code.

Requirements:
- Follow the task instruction exactly.
- Respect all required output filenames and schemas.
- Do not invent extra deliverables.
- Use only information provided in this context.
- Do not rely on hidden tests, checker files, canaries, or benchmark internals.
- The program must run non-interactively.
- Use robust numerical methods.
- Handle edge cases relevant to the active skill packs.
- Preserve units and financial conventions exactly.
- Ensure deterministic behavior unless randomness is required.
- If randomness is required, use an explicit seed.
- Do not modify input files.
- Avoid unnecessary dependencies.
- Fail clearly if required inputs are missing.
- Keep runtime appropriate for the task.
- Implement the supplied candidate strategy rather than silently switching to another approach.

Runtime path contract:
- Resolve the input root with:
  Path(os.environ.get("INPUT_DIR", "/input"))
- Resolve the output root with:
  Path(os.environ.get("OUTPUT_DIR", "/output"))
- A task path under /output/... or /app/output/... means the same relative deliverable under that resolved output root.
- Do not hardcode /output or /app/output when OUTPUT_DIR is supplied.
- All task data copied into this isolated candidate workspace is under INPUT_DIR/data/.
- If the benchmark instruction mentions a legacy input path such as /app/params.json, /app/data/file.csv, or /input/environment/data/file.csv, resolve the corresponding task-data file under INPUT_DIR/data/ using the supplied data inspections; do not access those legacy absolute paths literally.
- Write only required deliverables under the resolved output root.
- Temporary files, if truly needed, must use the normal temporary directory supplied by the runtime.

Security:
- Do not access the network.
- Do not launch child processes or shell commands.
- Do not install packages at runtime.
- Do not inspect environment variables other than INPUT_DIR, OUTPUT_DIR, QFBENCH_SEED, and ordinary deterministic runtime settings.
- Do not read files outside the supplied input root except normal installed Python modules and libraries.

Before finishing the code, internally verify:
- syntax and imports,
- input and output roots,
- output schema,
- row and identifier consistency,
- financial invariants,
- numerical edge cases,
- required units,
- candidate-specific verification steps.

TASK CONTEXT:

{context}

""".strip()
