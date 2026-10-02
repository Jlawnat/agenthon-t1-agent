from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class HestonFactor:
    kappa: float
    theta: float
    volatility_of_variance: float
    initial_variance: float
    correlation: float
    variance_risk_premium: float = 0.0


def two_factor_heston_characteristic(
    frequencies: np.ndarray,
    maturity: float,
    spot: float,
    rate: float,
    dividend_yield: float,
    factors: Sequence[HestonFactor],
    probability_index: int,
) -> np.ndarray:
    """Characteristic integrand numerator for a multi-factor affine Heston model."""
    eta = np.asarray(frequencies, dtype=float)
    shifted = 1j - eta if probability_index == 1 else -eta
    lambda_term = 1j * shifted - shifted**2
    exponent = 1j * eta * (math.log(spot) + (rate - dividend_yield) * maturity)
    for factor in factors:
        beta = factor.kappa + factor.variance_risk_premium
        theta = beta + 1j * shifted * factor.correlation * factor.volatility_of_variance
        omega = np.sqrt(theta**2 - lambda_term * factor.volatility_of_variance**2)
        ratio = (theta + omega) / (theta - omega)
        growth = np.exp(omega * maturity)
        exponent += factor.kappa * factor.theta / factor.volatility_of_variance**2 * ((theta + omega) * maturity - 2.0 * np.log((1.0 - ratio * growth) / (1.0 - ratio)))
        loading = (theta + omega) / factor.volatility_of_variance**2 * (1.0 - growth) / (1.0 - ratio * growth)
        exponent += loading * factor.initial_variance
    return np.exp(exponent)


def two_factor_heston_call(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    factors: Sequence[HestonFactor],
    *,
    integration_upper: float = 100.0,
    quadrature_nodes: int = 256,
) -> float:
    """Price a European call by Gauss-Legendre inversion of two Heston factors."""
    if len(tuple(factors)) != 2 or quadrature_nodes < 16:
        raise ValueError("exactly two factors and at least 16 nodes are required")
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_nodes)
    eta = 0.5 * (nodes + 1.0) * integration_upper
    scaled_weights = 0.5 * integration_upper * weights
    probabilities = []
    for index in (1, 2):
        characteristic = two_factor_heston_characteristic(eta, maturity, spot, rate, dividend_yield, factors, index)
        integrand = np.real(characteristic * np.exp(-1j * eta * math.log(strike)) / (1j * eta))
        probabilities.append(float(0.5 + np.sum(scaled_weights * integrand) / math.pi))
    return float(max(spot * math.exp(-dividend_yield * maturity) * probabilities[0] - strike * math.exp(-rate * maturity) * probabilities[1], 0.0))


def put_from_call_parity(call_price: float, spot: float, strike: float, rate: float, dividend_yield: float, maturity: float) -> float:
    """Convert a European call to a put by continuous-carry parity."""
    return float(call_price - spot * math.exp(-dividend_yield * maturity) + strike * math.exp(-rate * maturity))
