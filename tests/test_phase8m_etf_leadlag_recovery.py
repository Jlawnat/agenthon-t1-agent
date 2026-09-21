from __future__ import annotations

from agent.offline_etf_cross_asset_lead_lag import (
    EtfCrossAssetLeadLagSkill,
)


def test_etf_cross_asset_lead_lag_skill_name() -> None:
    assert (
        EtfCrossAssetLeadLagSkill().name
        == "etf-cross-asset-lead-lag-domain"
    )
