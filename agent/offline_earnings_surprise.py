from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


def _find_earnings_json(task_dir: Path) -> Path:
    for path in sorted(task_dir.rglob("*.json")):
        if "checks" in path.parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("companies"), list):
            companies = payload["companies"]
            if companies and isinstance(companies[0], dict) and "quarters" in companies[0]:
                return path
    raise RuntimeError("Could not discover an earnings-consensus JSON input.")


def _solve(task_dir: Path, out_dir: Path) -> None:
    data = json.loads(_find_earnings_json(task_dir).read_text(encoding="utf-8"))
    output = {}

    for company in data["companies"]:
        ticker = str(company["ticker"])
        quarters_out = []

        for q in company["quarters"]:
            actual = float(q["actual_eps"])
            estimate = float(q["consensus_estimate"])
            std = float(q["std_estimate"])
            prior = float(q["prior_year_eps"])
            surprise_dollars = actual - estimate
            surprise_pct = surprise_dollars / abs(estimate) * 100.0 if estimate != 0 else 0.0
            sue = surprise_dollars / std if std != 0 else 0.0
            yoy = (actual - prior) / abs(prior) * 100.0 if prior != 0 else 0.0

            if surprise_pct > 2.0:
                classification = "beat"
            elif surprise_pct < -2.0:
                classification = "miss"
            else:
                classification = "meet"

            quarters_out.append(
                {
                    "period": q["period"],
                    "surprise_dollars": round(surprise_dollars, 4),
                    "surprise_pct": round(surprise_pct, 4),
                    "sue": round(sue, 4),
                    "yoy_growth": round(yoy, 4),
                    "classification": classification,
                }
            )

        n = len(quarters_out)
        output[ticker] = {
            "quarters": quarters_out,
            "summary": {
                "avg_surprise_pct": round(sum(row["surprise_pct"] for row in quarters_out) / n, 4),
                "avg_sue": round(sum(row["sue"] for row in quarters_out) / n, 4),
                "beat_rate": round(sum(row["classification"] == "beat" for row in quarters_out) / n, 4),
                "avg_yoy_growth": round(sum(row["yoy_growth"] for row in quarters_out) / n, 4),
            },
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class EarningsSurpriseSkill:
    name: str = "earnings-surprise-domain"

    def matches(self, *, instruction: str, task_dir: Path) -> bool:
        lowered = instruction.lower()
        semantic = "earnings surprise" in lowered and ("sue" in lowered or "standardized unexpected earnings" in lowered)
        file_hint = any(p.name == "earnings.json" for p in task_dir.rglob("*.json"))
        return bool(semantic or file_hint)

    def solve(self, *, instruction: str, task_dir: Path, out_dir: Path, seed: int) -> None:
        del instruction, seed
        _solve(task_dir, out_dir)
