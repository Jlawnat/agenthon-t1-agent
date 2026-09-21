from __future__ import annotations

from agent.offline_regime_cta_vol_target import RegimeCtaVolTargetSkill


def test_regime_cta_matcher() -> None:
    instruction = '''
    Regime-Aware CTA Strategy Analysis.
    Compute composite volatility and regime_frac_high.
    Report dynamic_sharpe_ratio and avg_garch_persistence.
    '''
    assert RegimeCtaVolTargetSkill().matches(
        instruction=instruction,
        task_dir=None,  # type: ignore[arg-type]
    )
