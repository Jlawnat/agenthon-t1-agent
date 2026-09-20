from __future__ import annotations

from pathlib import Path

from agent.offline_copula_fitting import CopulaEquityFittingSkill
from agent.offline_ewma_risk_decomp import EwmaPortfolioRiskDecompositionSkill
from agent.offline_hull_white_swaption import HullWhiteSwaptionSkill
from agent.offline_ipca_latent_factors import IpcaLatentFactorsSkill
from agent.offline_digital_barrier_options import DigitalBarrierOptionsSkill
from agent.offline_regime_riskparity import RegimeRiskParityCvarSkill
from agent.offline_runtime import _expected_output_files, _candidate_output_complete


def test_phase8f_matchers_cover_all_six_domains(tmp_path: Path) -> None:
    cases = [
        (
            CopulaEquityFittingSkill(),
            "Fit Gaussian, Student-t, Clayton and Gumbel copulas to equity pairs using pseudo-observations, Kendall tau, AIC and tail dependence.",
        ),
        (
            EwmaPortfolioRiskDecompositionSkill(),
            "Compute RiskMetrics EWMA covariance, parametric VaR and Euler risk decomposition with marginal risk.",
        ),
        (
            HullWhiteSwaptionSkill(),
            "Calibrate a one-factor Hull-White model and price European and Bermudan swaptions.",
        ),
        (
            IpcaLatentFactorsSkill(),
            "Estimate IPCA latent factors using characteristics and alternating least squares.",
        ),
        (
            DigitalBarrierOptionsSkill(),
            "Price digital cash-or-nothing, gap options and continuously monitored barrier digitals.",
        ),
        (
            RegimeRiskParityCvarSkill(),
            "Use absorption ratio regimes for inverse-volatility risk parity and historical CVaR.",
        ),
    ]
    for skill, instruction in cases:
        assert skill.matches(instruction=instruction, task_dir=tmp_path)


def test_output_contract_includes_png_and_rejects_incomplete_candidate(tmp_path: Path) -> None:
    instruction = """
    ## Output Files
    - `calibration.json`
    - `yield_curve.csv`
    - `calibration_fit.png`
    """
    assert _expected_output_files(instruction) == {
        "calibration.json",
        "yield_curve.csv",
        "calibration_fit.png",
    }

    (tmp_path / "calibration.json").write_text("{}", encoding="utf-8")
    (tmp_path / "yield_curve.csv").write_text("x\n", encoding="utf-8")
    assert not _candidate_output_complete(tmp_path, instruction)

    (tmp_path / "calibration_fit.png").write_bytes(b"png")
    assert _candidate_output_complete(tmp_path, instruction)
