from __future__ import annotations

from agent.offline_merton_jump_diffusion import MertonJumpDiffusionSkill


def test_merton_skill_name() -> None:
    assert MertonJumpDiffusionSkill().name == "merton-jump-diffusion-domain"
