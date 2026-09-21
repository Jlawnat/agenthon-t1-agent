from __future__ import annotations

from agent.offline_stable_residual import StableResidualSkill


def test_stable_residual_skill_name() -> None:
    assert StableResidualSkill().name == "stable-residual-domain"
