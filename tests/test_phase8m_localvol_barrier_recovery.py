from __future__ import annotations

from agent.offline_localvol_barrier import LocalVolBarrierSkill


def test_localvol_barrier_skill_name() -> None:
    assert LocalVolBarrierSkill().name == "localvol-barrier-domain"
