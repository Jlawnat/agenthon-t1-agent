from __future__ import annotations

from agent.offline_residual_momentum import ResidualMomentumSkill


def test_residual_momentum_skill_name() -> None:
    assert ResidualMomentumSkill().name == "residual-momentum-domain"
