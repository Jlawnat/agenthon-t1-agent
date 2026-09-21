from __future__ import annotations

from agent.offline_cme_hdd import CmeHddOptionPricingSkill


def test_cme_hdd_skill_name() -> None:
    assert CmeHddOptionPricingSkill().name == "cme-hdd-option-pricing-domain"
