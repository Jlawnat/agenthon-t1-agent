from __future__ import annotations

from agent.offline_dupire_local_vol import DupireLocalVolSkill


def test_dupire_skill_name() -> None:
    assert DupireLocalVolSkill().name == "dupire-local-vol-domain"
