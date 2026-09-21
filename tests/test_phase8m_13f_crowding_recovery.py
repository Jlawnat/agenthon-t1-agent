from __future__ import annotations

from agent.offline_13f_crowding import AmendmentAware13FCrowdingSkill


def test_13f_crowding_skill_name() -> None:
    assert (
        AmendmentAware13FCrowdingSkill().name
        == "13f-amendment-aware-crowding-domain"
    )
