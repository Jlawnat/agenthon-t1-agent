from __future__ import annotations

from agent.offline_bl_regime_hmm import BlackLittermanRegimeHmmSkill
from agent.offline_sec_10k import Sec10KLongExtractionSkill


def test_bl_hmm_matcher() -> None:
    instruction = """
    Black-Litterman with Hidden Markov Model.
    Fit with Baum-Welch and report regime_prob_bull,
    bl_weight_asset1 and cov_bull_diag_asset1.
    """
    assert BlackLittermanRegimeHmmSkill().matches(
        instruction=instruction,
        task_dir=None,  # type: ignore[arg-type]
    )


def test_sec_10k_matcher() -> None:
    instruction = """
    SEC 10-K extraction.
    DocumentType__inA
    Corporate_Affairs_Officier
    Walmart_U.S._Supercenters_Total_Square_Feet
    Shareholders'_Equity_Restricted_stock_and_performance-based_restricted_stock_units
    """
    assert Sec10KLongExtractionSkill().matches(
        instruction=instruction,
        task_dir=None,  # type: ignore[arg-type]
    )
