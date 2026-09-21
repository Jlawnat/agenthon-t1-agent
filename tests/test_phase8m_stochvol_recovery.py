from __future__ import annotations

from agent.offline_stochvol_implied_surface import StochVolImpliedSurfaceSkill


def test_stochvol_implied_surface_skill_name() -> None:
    assert (
        StochVolImpliedSurfaceSkill().name
        == "stochvol-implied-surface-domain"
    )
