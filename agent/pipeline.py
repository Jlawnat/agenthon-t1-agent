from __future__ import annotations

from pathlib import Path

from agent.task_snapshot import TaskSnapshot
from agent.workspace import Workspace
from agent.snapshot_serializer import (
    build_snapshot_payload,
    save_snapshot_json,
)
from agent.run_logger import create_run_dir, save_json
from agent.specification import load_task_specification
from agent.spec_serializer import save_specification
from agent.spec_compiler import compile_specification
from agent.spec_enrichment import build_specification_prompt
from agent.model_client import ModelClient
from agent.spec_response_validator import parse_spec_enrichment
from agent.spec_merge import merge_specification
from agent.planning import build_task_plan
from agent.skill_packs import load_skill_packs
from agent.planner import build_planner_prompt
from agent.planner_validator import parse_planner_output


class QuantAgentPipeline:
    def __init__(
        self,
        *,
        task_dir: Path,
        out_dir: Path,
    ) -> None:
        self.task_dir = task_dir
        self.out_dir = out_dir

    def run(self) -> None:
        workspace = Workspace.create(
            task_dir=self.task_dir,
            out_dir=self.out_dir,
        )

        try:
            print(
                f"[QuantAgent] task_dir = "
                f"{workspace.task_dir}"
            )
            print(
                f"[QuantAgent] out_dir  = "
                f"{workspace.out_dir}"
            )
            print(
                f"[QuantAgent] work_dir = "
                f"{workspace.work_dir}"
            )

            # --------------------------------------------------
            # 1. Discover task files
            # --------------------------------------------------
            task_files = workspace.discover_task_files()

            print(
                f"[QuantAgent] discovered "
                f"{len(task_files)} task files"
            )

            # --------------------------------------------------
            # 2. Build task snapshot
            # --------------------------------------------------
            snapshot = TaskSnapshot.build(
                workspace.task_dir
            )

            data_inspections = snapshot.inspect_data()

            snapshot_payload = build_snapshot_payload(
                snapshot=snapshot,
                data_inspections=data_inspections,
            )

            # --------------------------------------------------
            # 3. Deterministic specification
            # --------------------------------------------------
            spec = load_task_specification(
                workspace.task_dir
            )

            compiled_spec = compile_specification(
                spec
            )

            # --------------------------------------------------
            # 4. Model-assisted specification
            # --------------------------------------------------
            spec_prompt = build_specification_prompt(
                spec=spec,
                compiled_spec=compiled_spec,
            )

            model_client = ModelClient()

            enriched_spec = compiled_spec
            model_enrichment = None

            if model_client.is_available():
                print(
                    "[QuantAgent] model endpoint detected"
                )

                response = model_client.complete(
                    spec_prompt,
                    temperature=0.0,
                )

                model_enrichment = (
                    parse_spec_enrichment(
                        response.text
                    )
                )

                enriched_spec = merge_specification(
                    deterministic_spec=spec,
                    compiled_spec=compiled_spec,
                    enrichment=model_enrichment,
                )

                print(
                    "[QuantAgent] specification "
                    "enrichment completed"
                )

            else:
                print(
                    "[QuantAgent] MODEL_ENDPOINT "
                    "not set; using deterministic "
                    "specification only"
                )

            # --------------------------------------------------
            # 5. Planning
            # --------------------------------------------------
            task_plan = build_task_plan(
                enriched_spec
            )

            active_skill_packs = load_skill_packs(
                task_plan.review_packs
            )

            planner_prompt = build_planner_prompt(
                spec=enriched_spec,
                task_plan=task_plan,
                skill_packs=active_skill_packs,
                instruction_text=(
                    spec.instruction_text
                ),
            )

            planner_output = None

            if model_client.is_available():
                planner_response = (
                    model_client.complete(
                        planner_prompt,
                        temperature=0.2,
                    )
                )

                planner_output = (
                    parse_planner_output(
                        planner_response.text,
                        expected_candidates=(
                            task_plan.candidate_count
                        ),
                    )
                )

                print(
                    "[QuantAgent] planner completed"
                )

            # --------------------------------------------------
            # 6. Persistent run directory
            # --------------------------------------------------
            runs_root = (
                Path.home()
                / "agenthon-t1-agent"
                / "runs"
            )

            run_dir = create_run_dir(
                base_dir=runs_root,
                task_id=spec.task_id,
            )

            print(
                f"[QuantAgent] persistent run dir = "
                f"{run_dir}"
            )

            # --------------------------------------------------
            # 7. Save development artifacts
            # --------------------------------------------------
            save_json(
                snapshot_payload,
                run_dir / "task_snapshot.json",
            )

            save_json(
                spec.to_dict(),
                run_dir
                / "task_specification.json",
            )

            save_json(
                compiled_spec.to_dict(),
                run_dir
                / "compiled_specification.json",
            )

            save_json(
                enriched_spec.to_dict(),
                run_dir
                / "enriched_specification.json",
            )

            save_json(
                task_plan.to_dict(),
                run_dir / "task_plan.json",
            )

            save_json(
                {
                    "active_skill_packs": [
                        pack.to_dict()
                        for pack
                        in active_skill_packs
                    ]
                },
                run_dir
                / "active_skill_packs.json",
            )

            if model_enrichment is not None:
                save_json(
                    model_enrichment.to_dict(),
                    run_dir
                    / "model_spec_enrichment.json",
                )

            if planner_output is not None:
                save_json(
                    planner_output.to_dict(),
                    run_dir
                    / "planner_output.json",
                )

            specification_prompt_path = (
                run_dir
                / "specification_prompt.txt"
            )

            specification_prompt_path.write_text(
                spec_prompt,
                encoding="utf-8",
            )

            planner_prompt_path = (
                run_dir
                / "planner_prompt.txt"
            )

            planner_prompt_path.write_text(
                planner_prompt,
                encoding="utf-8",
            )

            # --------------------------------------------------
            # 8. Temporary development artifacts
            # --------------------------------------------------
            snapshot_path = (
                workspace.work_dir
                / "task_snapshot.json"
            )

            save_snapshot_json(
                payload=snapshot_payload,
                destination=snapshot_path,
            )

            spec_path = (
                workspace.work_dir
                / "task_specification.json"
            )

            save_specification(
                spec=spec,
                destination=spec_path,
            )

            # --------------------------------------------------
            # 9. Compact console summary
            # --------------------------------------------------
            print(
                f"[QuantAgent] plan: "
                f"difficulty={task_plan.difficulty}, "
                f"candidates="
                f"{task_plan.candidate_count}, "
                f"packs="
                f"{task_plan.review_packs}"
            )

        finally:
            workspace.cleanup()