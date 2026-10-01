from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence
import numpy as np

@dataclass(frozen=True)
class MonteCarloEstimate:
    value: float
    se: float

def _finite(values: Sequence[float], *, minimum: int=2) -> np.ndarray:
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < minimum:
        raise RuntimeError(f'Expected at least {minimum} finite observations.')
    return x

def mc_estimate(samples: Sequence[float]) -> MonteCarloEstimate:
    x = _finite(samples, minimum=2)
    return MonteCarloEstimate(value=float(np.mean(x)), se=float(np.std(x, ddof=1) / math.sqrt(x.size)))

def gbm_paths_from_normals(normals: np.ndarray, *, spot: float, rate: float, dividend_yield: float, volatility: float, maturity: float) -> np.ndarray:
    z = np.asarray(normals, dtype=float)
    if z.ndim != 2 or z.shape[0] < 2 or z.shape[1] < 1:
        raise RuntimeError('GBM paths require an N x M normal matrix.')
    if spot <= 0.0 or volatility < 0.0 or maturity <= 0.0:
        raise RuntimeError('Invalid GBM parameters.')
    steps = z.shape[1]
    dt = maturity / float(steps)
    increments = (rate - dividend_yield - 0.5 * volatility * volatility) * dt + volatility * math.sqrt(dt) * z
    return spot * np.exp(np.cumsum(increments, axis=1))

def _discounted_payoff_samples(normals: np.ndarray, *, spot: float, strike: float, rate: float, dividend_yield: float, volatility: float, maturity: float, option_type: str, asian: bool) -> tuple[np.ndarray, np.ndarray]:
    paths = gbm_paths_from_normals(normals, spot=spot, rate=rate, dividend_yield=dividend_yield, volatility=volatility, maturity=maturity)
    underlying = np.mean(paths, axis=1) if asian else paths[:, -1]
    if option_type == 'call':
        payoff = np.maximum(underlying - strike, 0.0)
    elif option_type == 'put':
        payoff = np.maximum(strike - underlying, 0.0)
    else:
        raise RuntimeError("option_type must be 'call' or 'put'.")
    return (math.exp(-rate * maturity) * payoff, paths)

def _fd_greek_samples(normals: np.ndarray, *, spot: float, strike: float, rate: float, dividend_yield: float, volatility: float, maturity: float, option_type: str, asian: bool, greek: str) -> np.ndarray:

    def samples(*, s: float=spot, r: float=rate, sigma: float=volatility, t: float=maturity) -> np.ndarray:
        return _discounted_payoff_samples(normals, spot=s, strike=strike, rate=r, dividend_yield=dividend_yield, volatility=sigma, maturity=t, option_type=option_type, asian=asian)[0]
    if greek == 'delta':
        h = 0.5
        return (samples(s=spot + h) - samples(s=spot - h)) / (2.0 * h)
    if greek == 'gamma':
        h = 0.5
        return (samples(s=spot + h) - 2.0 * samples() + samples(s=spot - h)) / (h * h)
    if greek == 'vega':
        h = 0.01
        return (samples(sigma=volatility + h) - samples(sigma=volatility - h)) / (2.0 * h)
    if greek == 'theta':
        h = 1.0 / 252.0
        return -(samples(t=maturity - h) - samples()) / h
    if greek == 'rho':
        h = 0.001
        return (samples(r=rate + h) - samples(r=rate - h)) / (2.0 * h)
    raise RuntimeError(f'Unsupported finite-difference Greek {greek!r}.')

def finite_difference_greeks(normals: np.ndarray, **kwargs) -> dict[str, MonteCarloEstimate]:
    return {greek: mc_estimate(_fd_greek_samples(normals, greek=greek, **kwargs)) for greek in ('delta', 'gamma', 'vega', 'theta', 'rho')}

def pathwise_greeks(normals: np.ndarray, *, spot: float, strike: float, rate: float, dividend_yield: float, volatility: float, maturity: float, option_type: str, asian: bool) -> dict[str, MonteCarloEstimate | None]:
    z = np.asarray(normals, dtype=float)
    _, paths = _discounted_payoff_samples(z, spot=spot, strike=strike, rate=rate, dividend_yield=dividend_yield, volatility=volatility, maturity=maturity, option_type=option_type, asian=asian)
    disc = math.exp(-rate * maturity)
    sign = 1.0 if option_type == 'call' else -1.0
    if asian:
        average = np.mean(paths, axis=1)
        itm = average > strike if option_type == 'call' else average < strike
        delta_samples = disc * sign * itm * average / spot
        steps = z.shape[1]
        dt = maturity / float(steps)
        times = dt * np.arange(1, steps + 1, dtype=float)
        brownian = math.sqrt(dt) * np.cumsum(z, axis=1)
        dpaths_dsigma = paths * (brownian - volatility * times[None, :])
        daverage_dsigma = np.mean(dpaths_dsigma, axis=1)
        vega_samples = disc * sign * itm * daverage_dsigma
        return {'delta': mc_estimate(delta_samples), 'gamma': None, 'vega': mc_estimate(vega_samples), 'theta': None, 'rho': None}
    terminal = paths[:, -1]
    itm = terminal > strike if option_type == 'call' else terminal < strike
    z_terminal = np.sum(z, axis=1) / math.sqrt(z.shape[1])
    raw_payoff = np.maximum(terminal - strike, 0.0) if option_type == 'call' else np.maximum(strike - terminal, 0.0)
    delta_samples = disc * sign * itm * terminal / spot
    dterminal_dsigma = terminal * (math.sqrt(maturity) * z_terminal - volatility * maturity)
    vega_samples = disc * sign * itm * dterminal_dsigma
    dterminal_dt = terminal * (rate - dividend_yield - 0.5 * volatility * volatility + volatility * z_terminal / (2.0 * math.sqrt(maturity)))
    theta_samples = disc * (-rate * raw_payoff + sign * itm * dterminal_dt)
    rho_samples = disc * (-maturity * raw_payoff + sign * itm * terminal * maturity)
    return {'delta': mc_estimate(delta_samples), 'gamma': None, 'vega': mc_estimate(vega_samples), 'theta': mc_estimate(theta_samples), 'rho': mc_estimate(rho_samples)}

def likelihood_ratio_greeks(normals: np.ndarray, *, spot: float, strike: float, rate: float, dividend_yield: float, volatility: float, maturity: float, option_type: str, asian: bool) -> dict[str, MonteCarloEstimate | None]:
    z = np.asarray(normals, dtype=float)
    discounted, _ = _discounted_payoff_samples(z, spot=spot, strike=strike, rate=rate, dividend_yield=dividend_yield, volatility=volatility, maturity=maturity, option_type=option_type, asian=asian)
    if asian:
        dt = maturity / float(z.shape[1])
        z_first = z[:, 0]
        score_delta = z_first / (spot * volatility * math.sqrt(dt))
        score_gamma = (z_first * z_first - 1.0 - z_first * volatility * math.sqrt(dt)) / (spot * volatility * math.sqrt(dt)) ** 2
        score_vega = np.sum((z * z - 1.0) / volatility - z * math.sqrt(dt), axis=1)
        score_rho = math.sqrt(dt) * np.sum(z, axis=1) / volatility - maturity
    else:
        z_terminal = np.sum(z, axis=1) / math.sqrt(z.shape[1])
        score_delta = z_terminal / (spot * volatility * math.sqrt(maturity))
        score_gamma = (z_terminal * z_terminal - 1.0 - z_terminal * volatility * math.sqrt(maturity)) / (spot * volatility * math.sqrt(maturity)) ** 2
        score_vega = (z_terminal * z_terminal - 1.0) / volatility - z_terminal * math.sqrt(maturity)
        score_rho = z_terminal * math.sqrt(maturity) / volatility - maturity
    return {'delta': mc_estimate(discounted * score_delta), 'gamma': mc_estimate(discounted * score_gamma), 'vega': mc_estimate(discounted * score_vega), 'theta': None, 'rho': mc_estimate(discounted * score_rho)}
__all__ = ('MonteCarloEstimate', 'mc_estimate', 'gbm_paths_from_normals', 'finite_difference_greeks', 'pathwise_greeks', 'likelihood_ratio_greeks')
