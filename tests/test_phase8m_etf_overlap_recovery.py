from __future__ import annotations

from agent.offline_etf_overlap_redemption import (
    EtfOverlapRedemptionPressureSkill,
)


def test_etf_overlap_skill_name() -> None:
    assert (
        EtfOverlapRedemptionPressureSkill().name
        == "etf-overlap-redemption-pressure-domain"
    )
