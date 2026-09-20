from pathlib import Path
import numpy as np

from agent.offline_credit_migration import CreditMigrationMatrixSkill
from agent.offline_yield_curve_dynamics import YieldCurvePcaDynamicsSkill, _nelson_siegel
from agent.offline_fx_forward import FxForwardCrossRateSkill, _day_count_fraction
from agent.offline_smith_tail import TailIndexEstimationSkill
from agent.offline_earnings_surprise import EarningsSurpriseSkill
from agent.offline_implied_vol_approximations import ImpliedVolApproximationSkill, _bs_price, _newton


def test_phase8d_matchers_cover_six_domains():
    samples = [
        (CreditMigrationMatrixSkill(), "Credit rating migration transition matrix with cumulative default probabilities, Markov tests and generator matrix."),
        (YieldCurvePcaDynamicsSkill(), "Analyze yield curve PCA principal components and forward rates."),
        (FxForwardCrossRateSkill(), "FX forward cross rate triangulation and portfolio valuation."),
        (TailIndexEstimationSkill(), "Smith tail index and Hill GPD estimation."),
        (EarningsSurpriseSkill(), "Calculate earnings surprise and SUE standardized unexpected earnings."),
        (ImpliedVolApproximationSkill(), "Compare implied volatility approximation methods with Newton inversion."),
    ]
    for skill, instruction in samples:
        assert skill.matches(instruction=instruction, task_dir=Path("."))


def test_day_count_and_ns_are_numerically_sane():
    assert np.isclose(_day_count_fraction(92, "ACT/360"), 92 / 360)
    y = _nelson_siegel(np.array([1.0, 5.0]), 4.0, -1.0, 1.0)
    assert np.all(np.isfinite(y))


def test_newton_recovers_generated_black_scholes_vol():
    market = _bs_price(500.0, 0.5, 0.2)
    sigma, converged = _newton(market, 500.0, 0.5)
    assert converged
    assert np.isclose(sigma, 0.2, atol=1e-8)
