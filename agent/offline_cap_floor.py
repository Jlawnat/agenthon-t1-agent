from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from scipy.stats import norm


def _find_params(task_dir: Path) -> Path:
    for path in sorted(task_dir.rglob("*.json")):
        if "checks" in path.parts:
            continue
        try:
            p = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(p, dict) and {"forward_rates", "vol", "strike", "notional", "period_length", "discount_factors"}.issubset(p):
            return path
    raise RuntimeError("Could not discover cap/floor parameters.")


def _solve(task_dir: Path, out_dir: Path) -> None:
    p = json.loads(_find_params(task_dir).read_text(encoding="utf-8"))
    fwds = p["forward_rates"]
    vol = float(p["vol"])
    strike = float(p["strike"])
    notional = float(p["notional"])
    tau = float(p["period_length"])
    dfs = p["discount_factors"]

    caplets, floorlets = [], []
    for i, (fwd, df) in enumerate(zip(fwds, dfs)):
        fwd, df = float(fwd), float(df)
        t_fix = i * tau
        if t_fix == 0:
            cap = notional * tau * df * max(fwd - strike, 0.0)
            floor = notional * tau * df * max(strike - fwd, 0.0)
        else:
            d1 = (math.log(fwd / strike) + 0.5 * vol * vol * t_fix) / (vol * math.sqrt(t_fix))
            d2 = d1 - vol * math.sqrt(t_fix)
            cap = notional * tau * df * (fwd * norm.cdf(d1) - strike * norm.cdf(d2))
            floor = notional * tau * df * (strike * norm.cdf(-d2) - fwd * norm.cdf(-d1))
        caplets.append(round(cap, 4))
        floorlets.append(round(floor, 4))

    cap_price = sum(caplets)
    floor_price = sum(floorlets)
    swap = notional * tau * sum(float(dfs[i]) * (float(fwds[i]) - strike) for i in range(len(fwds)))
    parity = cap_price - floor_price - swap

    result = {
        "cap_price": round(cap_price, 2),
        "floor_price": round(floor_price, 2),
        "caplets": caplets,
        "floorlets": floorlets,
        "swap_value": round(swap, 2),
        "put_call_parity_error": round(parity, 6),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class InterestRateCapFloorSkill:
    name: str = "interest-rate-cap-floor-domain"

    def matches(self, *, instruction: str, task_dir: Path) -> bool:
        lowered = instruction.lower()
        return ("caplet" in lowered and "floorlet" in lowered and "black" in lowered) or (
            "interest rate cap" in lowered and "floor" in lowered
        )

    def solve(self, *, instruction: str, task_dir: Path, out_dir: Path, seed: int) -> None:
        del instruction, seed
        _solve(task_dir, out_dir)
