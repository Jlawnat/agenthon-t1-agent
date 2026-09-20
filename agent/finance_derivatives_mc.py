from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from agent.finance_schema import TaskDataCatalog


@dataclass(frozen=True)
class HistoricalVolatilityCalibration:
    n_returns: int
    return_mean: float
    return_std: float
    return_skewness: float
    return_excess_kurtosis: float
    annualized_vol: float
    spot: float


@dataclass(frozen=True)
class MonteCarloEstimate:
    value: float
    se: float


def _finite(values: Sequence[float], *, minimum: int = 2) -> np.ndarray:
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < minimum:
        raise RuntimeError(f"Expected at least {minimum} finite observations.")
    return x


def historical_log_return_calibration(
    closes: Sequence[float],
    *,
    annualization: float = 252.0,
) -> HistoricalVolatilityCalibration:
    prices = _finite(closes, minimum=3)
    if np.any(prices <= 0.0):
        raise RuntimeError("Close prices must be positive.")
    returns = np.log(prices[1:] / prices[:-1])
    std = float(np.std(returns, ddof=1))
    return HistoricalVolatilityCalibration(
        n_returns=int(returns.size),
        return_mean=float(np.mean(returns)),
        return_std=std,
        return_skewness=float(skew(returns)),
        return_excess_kurtosis=float(kurtosis(returns, fisher=True)),
        annualized_vol=float(std * math.sqrt(annualization)),
        spot=float(prices[-1]),
    )


def black_scholes_price(
    *,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
    option_type: str,
) -> float:
    if spot <= 0.0 or strike <= 0.0:
        raise RuntimeError("Black-Scholes requires positive spot and strike.")
    if maturity <= 0.0:
        return float(
            max(spot - strike, 0.0)
            if option_type == "call"
            else max(strike - spot, 0.0)
        )
    if volatility <= 0.0:
        pv_forward = (
            spot * math.exp(-dividend_yield * maturity)
            - strike * math.exp(-rate * maturity)
        )
        return float(
            max(pv_forward, 0.0)
            if option_type == "call"
            else max(-pv_forward, 0.0)
        )

    root_t = math.sqrt(maturity)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility) * maturity
    ) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    disc_q = math.exp(-dividend_yield * maturity)
    disc_r = math.exp(-rate * maturity)

    if option_type == "call":
        return float(
            spot * disc_q * norm.cdf(d1)
            - strike * disc_r * norm.cdf(d2)
        )
    if option_type == "put":
        return float(
            strike * disc_r * norm.cdf(-d2)
            - spot * disc_q * norm.cdf(-d1)
        )
    raise RuntimeError("option_type must be 'call' or 'put'.")


def black_scholes_greeks(
    *,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
    option_type: str,
) -> dict[str, float]:
    if min(spot, strike, volatility, maturity) <= 0.0:
        raise RuntimeError("Analytical Greeks require positive S,K,sigma,T.")
    root_t = math.sqrt(maturity)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility) * maturity
    ) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    disc_q = math.exp(-dividend_yield * maturity)
    disc_r = math.exp(-rate * maturity)
    pdf = norm.pdf(d1)

    gamma = disc_q * pdf / (spot * volatility * root_t)
    vega = spot * disc_q * pdf * root_t

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (
            spot * disc_q * pdf * volatility / (2.0 * root_t)
            - dividend_yield * spot * disc_q * norm.cdf(d1)
            + rate * strike * disc_r * norm.cdf(d2)
        )
        rho = strike * maturity * disc_r * norm.cdf(d2)
    elif option_type == "put":
        delta = disc_q * (norm.cdf(d1) - 1.0)
        theta = (
            spot * disc_q * pdf * volatility / (2.0 * root_t)
            + dividend_yield * spot * disc_q * norm.cdf(-d1)
            - rate * strike * disc_r * norm.cdf(-d2)
        )
        rho = -strike * maturity * disc_r * norm.cdf(-d2)
    else:
        raise RuntimeError("option_type must be 'call' or 'put'.")

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }


def forward_start_atm_call_price(
    *,
    spot: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    start: float,
    end: float,
) -> float:
    tau = float(end - start)
    if tau <= 0.0:
        return 0.0
    if volatility <= 0.0:
        unit = max(
            math.exp(-dividend_yield * tau)
            - math.exp(-rate * tau),
            0.0,
        )
        return float(spot * math.exp(-dividend_yield * start) * unit)

    root_tau = math.sqrt(tau)
    d1 = (
        (rate - dividend_yield + 0.5 * volatility * volatility) * tau
    ) / (volatility * root_tau)
    d2 = d1 - volatility * root_tau
    unit = (
        math.exp(-dividend_yield * tau) * norm.cdf(d1)
        - math.exp(-rate * tau) * norm.cdf(d2)
    )
    return float(spot * math.exp(-dividend_yield * start) * unit)


def cliquet_forward_start_prices(
    *,
    spot: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
    resets: int,
) -> np.ndarray:
    if maturity <= 0.0 or resets <= 0:
        raise RuntimeError("Cliquet maturity and reset count must be positive.")
    dt = maturity / float(resets)
    return np.asarray(
        [
            forward_start_atm_call_price(
                spot=spot,
                rate=rate,
                dividend_yield=dividend_yield,
                volatility=volatility,
                start=i * dt,
                end=(i + 1) * dt,
            )
            for i in range(resets)
        ],
        dtype=float,
    )


def mc_estimate(samples: Sequence[float]) -> MonteCarloEstimate:
    x = _finite(samples, minimum=2)
    return MonteCarloEstimate(
        value=float(np.mean(x)),
        se=float(np.std(x, ddof=1) / math.sqrt(x.size)),
    )


def gbm_paths_from_normals(
    normals: np.ndarray,
    *,
    spot: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
) -> np.ndarray:
    z = np.asarray(normals, dtype=float)
    if z.ndim != 2 or z.shape[0] < 2 or z.shape[1] < 1:
        raise RuntimeError("GBM paths require an N x M normal matrix.")
    if spot <= 0.0 or volatility < 0.0 or maturity <= 0.0:
        raise RuntimeError("Invalid GBM parameters.")
    steps = z.shape[1]
    dt = maturity / float(steps)
    increments = (
        (rate - dividend_yield - 0.5 * volatility * volatility) * dt
        + volatility * math.sqrt(dt) * z
    )
    return spot * np.exp(np.cumsum(increments, axis=1))


def _discounted_payoff_samples(
    normals: np.ndarray,
    *,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
    option_type: str,
    asian: bool,
) -> tuple[np.ndarray, np.ndarray]:
    paths = gbm_paths_from_normals(
        normals,
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        maturity=maturity,
    )
    underlying = np.mean(paths, axis=1) if asian else paths[:, -1]
    if option_type == "call":
        payoff = np.maximum(underlying - strike, 0.0)
    elif option_type == "put":
        payoff = np.maximum(strike - underlying, 0.0)
    else:
        raise RuntimeError("option_type must be 'call' or 'put'.")
    return math.exp(-rate * maturity) * payoff, paths


def _fd_greek_samples(
    normals: np.ndarray,
    *,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
    option_type: str,
    asian: bool,
    greek: str,
) -> np.ndarray:
    def samples(
        *,
        s: float = spot,
        r: float = rate,
        sigma: float = volatility,
        t: float = maturity,
    ) -> np.ndarray:
        return _discounted_payoff_samples(
            normals,
            spot=s,
            strike=strike,
            rate=r,
            dividend_yield=dividend_yield,
            volatility=sigma,
            maturity=t,
            option_type=option_type,
            asian=asian,
        )[0]

    if greek == "delta":
        h = 0.5
        return (samples(s=spot + h) - samples(s=spot - h)) / (2.0 * h)
    if greek == "gamma":
        h = 0.5
        return (
            samples(s=spot + h)
            - 2.0 * samples()
            + samples(s=spot - h)
        ) / (h * h)
    if greek == "vega":
        h = 0.01
        return (
            samples(sigma=volatility + h)
            - samples(sigma=volatility - h)
        ) / (2.0 * h)
    if greek == "theta":
        h = 1.0 / 252.0
        return -(samples(t=maturity - h) - samples()) / h
    if greek == "rho":
        h = 0.001
        return (samples(r=rate + h) - samples(r=rate - h)) / (2.0 * h)
    raise RuntimeError(f"Unsupported finite-difference Greek {greek!r}.")


def finite_difference_greeks(
    normals: np.ndarray,
    **kwargs,
) -> dict[str, MonteCarloEstimate]:
    return {
        greek: mc_estimate(
            _fd_greek_samples(normals, greek=greek, **kwargs)
        )
        for greek in ("delta", "gamma", "vega", "theta", "rho")
    }


def pathwise_greeks(
    normals: np.ndarray,
    *,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
    option_type: str,
    asian: bool,
) -> dict[str, MonteCarloEstimate | None]:
    z = np.asarray(normals, dtype=float)
    _, paths = _discounted_payoff_samples(
        z,
        spot=spot,
        strike=strike,
        rate=rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        maturity=maturity,
        option_type=option_type,
        asian=asian,
    )
    disc = math.exp(-rate * maturity)
    sign = 1.0 if option_type == "call" else -1.0

    if asian:
        average = np.mean(paths, axis=1)
        itm = average > strike if option_type == "call" else average < strike
        delta_samples = disc * sign * itm * average / spot

        steps = z.shape[1]
        dt = maturity / float(steps)
        times = dt * np.arange(1, steps + 1, dtype=float)
        brownian = math.sqrt(dt) * np.cumsum(z, axis=1)
        dpaths_dsigma = paths * (
            brownian - volatility * times[None, :]
        )
        daverage_dsigma = np.mean(dpaths_dsigma, axis=1)
        vega_samples = disc * sign * itm * daverage_dsigma

        return {
            "delta": mc_estimate(delta_samples),
            "gamma": None,
            "vega": mc_estimate(vega_samples),
            "theta": None,
            "rho": None,
        }

    terminal = paths[:, -1]
    itm = terminal > strike if option_type == "call" else terminal < strike
    z_terminal = np.sum(z, axis=1) / math.sqrt(z.shape[1])
    raw_payoff = (
        np.maximum(terminal - strike, 0.0)
        if option_type == "call"
        else np.maximum(strike - terminal, 0.0)
    )

    delta_samples = disc * sign * itm * terminal / spot
    dterminal_dsigma = terminal * (
        math.sqrt(maturity) * z_terminal - volatility * maturity
    )
    vega_samples = disc * sign * itm * dterminal_dsigma
    dterminal_dt = terminal * (
        rate
        - dividend_yield
        - 0.5 * volatility * volatility
        + volatility * z_terminal / (2.0 * math.sqrt(maturity))
    )
    theta_samples = disc * (
        -rate * raw_payoff + sign * itm * dterminal_dt
    )
    rho_samples = disc * (
        -maturity * raw_payoff + sign * itm * terminal * maturity
    )

    return {
        "delta": mc_estimate(delta_samples),
        "gamma": None,
        "vega": mc_estimate(vega_samples),
        "theta": mc_estimate(theta_samples),
        "rho": mc_estimate(rho_samples),
    }


def likelihood_ratio_greeks(
    normals: np.ndarray,
    *,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
    option_type: str,
    asian: bool,
) -> dict[str, MonteCarloEstimate | None]:
    z = np.asarray(normals, dtype=float)
    discounted, _ = _discounted_payoff_samples(
        z,
        spot=spot,
        strike=strike,
        rate=rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        maturity=maturity,
        option_type=option_type,
        asian=asian,
    )

    if asian:
        dt = maturity / float(z.shape[1])
        z_first = z[:, 0]
        score_delta = z_first / (
            spot * volatility * math.sqrt(dt)
        )
        score_gamma = (
            z_first * z_first
            - 1.0
            - z_first * volatility * math.sqrt(dt)
        ) / (spot * volatility * math.sqrt(dt)) ** 2
        score_vega = np.sum(
            (z * z - 1.0) / volatility - z * math.sqrt(dt),
            axis=1,
        )
        score_rho = (
            math.sqrt(dt) * np.sum(z, axis=1) / volatility
            - maturity
        )
    else:
        z_terminal = np.sum(z, axis=1) / math.sqrt(z.shape[1])
        score_delta = z_terminal / (
            spot * volatility * math.sqrt(maturity)
        )
        score_gamma = (
            z_terminal * z_terminal
            - 1.0
            - z_terminal * volatility * math.sqrt(maturity)
        ) / (spot * volatility * math.sqrt(maturity)) ** 2
        score_vega = (
            (z_terminal * z_terminal - 1.0) / volatility
            - z_terminal * math.sqrt(maturity)
        )
        score_rho = (
            z_terminal * math.sqrt(maturity) / volatility
            - maturity
        )

    return {
        "delta": mc_estimate(discounted * score_delta),
        "gamma": mc_estimate(discounted * score_gamma),
        "vega": mc_estimate(discounted * score_vega),
        "theta": None,
        "rho": mc_estimate(discounted * score_rho),
    }


def _payload(
    estimate: MonteCarloEstimate | None,
) -> dict[str, float | None]:
    return (
        {"value": None, "se": None}
        if estimate is None
        else {"value": float(estimate.value), "se": float(estimate.se)}
    )


def run_cliquet_workflow(task_dir: Path, out_dir: Path) -> None:
    catalog = TaskDataCatalog.discover(task_dir)
    csv_path = catalog.csv_with_columns({"date", "close"})
    frame = pd.read_csv(csv_path)
    columns = {str(c).strip().lower(): c for c in frame.columns}
    closes = pd.to_numeric(
        frame[columns["close"]],
        errors="coerce",
    ).dropna().to_numpy(dtype=float)
    calibration = historical_log_return_calibration(closes)

    rate = 0.05
    dividend_yield = 0.013
    rows = []
    details = []
    for maturity in (1.0, 2.0, 3.0):
        european = black_scholes_price(
            spot=calibration.spot,
            strike=calibration.spot,
            rate=rate,
            dividend_yield=dividend_yield,
            volatility=calibration.annualized_vol,
            maturity=maturity,
            option_type="call",
        )
        for resets in (4, 12):
            dt = maturity / resets
            values = cliquet_forward_start_prices(
                spot=calibration.spot,
                rate=rate,
                dividend_yield=dividend_yield,
                volatility=calibration.annualized_vol,
                maturity=maturity,
                resets=resets,
            )
            cliquet = float(np.sum(values))
            rows.append({
                "T": maturity,
                "N": resets,
                "period_length": dt,
                "cliquet_price": cliquet,
                "european_price": european,
                "ratio_cliquet_over_european": cliquet / european,
            })
            for i, value in enumerate(values):
                details.append({
                    "T": maturity,
                    "N": resets,
                    "period_index": i,
                    "t_start": i * dt,
                    "t_end": (i + 1) * dt,
                    "forward_start_price": float(value),
                })

    prices = pd.DataFrame(rows).sort_values(["T", "N"], kind="stable")
    detail_frame = pd.DataFrame(details).sort_values(
        ["T", "N", "period_index"],
        kind="stable",
    )
    pivot = prices.pivot(index="T", columns="N", values="cliquet_price")

    calibration_payload = {
        "n_returns": calibration.n_returns,
        "return_mean": calibration.return_mean,
        "return_std": calibration.return_std,
        "return_skewness": calibration.return_skewness,
        "return_excess_kurtosis": calibration.return_excess_kurtosis,
        "annualized_vol": calibration.annualized_vol,
        "S0": calibration.spot,
    }
    summary = {
        "n_configurations": int(len(prices)),
        "total_forward_starts": int(len(detail_frame)),
        "max_cliquet_price": float(prices["cliquet_price"].max()),
        "min_cliquet_price": float(prices["cliquet_price"].min()),
        "mean_ratio_cliquet_over_european": float(
            prices["ratio_cliquet_over_european"].mean()
        ),
        "cliquet_increases_with_N": bool(
            np.all(
                pivot[12].to_numpy(dtype=float)
                > pivot[4].to_numpy(dtype=float)
            )
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration.json").write_text(
        json.dumps(calibration_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    prices.to_csv(out_dir / "cliquet_prices.csv", index=False)
    detail_frame.to_csv(out_dir / "forward_start_details.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def run_mc_greek_surface_workflow(out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spot = 100.0
    strike = 100.0
    rate = 0.05
    dividend_yield = 0.02
    volatility = 0.30
    maturity = 1.0
    n_paths = 500_000
    monitoring = 12

    rng = np.random.RandomState(42)
    normals = rng.standard_normal((n_paths, monitoring))

    option_specs = {
        "european_call": ("call", False),
        "european_put": ("put", False),
        "asian_call": ("call", True),
        "asian_put": ("put", True),
    }

    prices_payload: dict[str, dict[str, float | None]] = {}
    greeks_payload = {}
    cache = {}

    for name, (option_type, asian) in option_specs.items():
        payoff, _ = _discounted_payoff_samples(
            normals,
            spot=spot,
            strike=strike,
            rate=rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            maturity=maturity,
            option_type=option_type,
            asian=asian,
        )
        price_est = mc_estimate(payoff)
        bs_price = (
            black_scholes_price(
                spot=spot,
                strike=strike,
                rate=rate,
                dividend_yield=dividend_yield,
                volatility=volatility,
                maturity=maturity,
                option_type=option_type,
            )
            if not asian
            else None
        )
        prices_payload[name] = {
            "mc_price": price_est.value,
            "mc_se": price_est.se,
            "bs_price": bs_price,
        }

        kwargs = dict(
            spot=spot,
            strike=strike,
            rate=rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            maturity=maturity,
            option_type=option_type,
            asian=asian,
        )
        fd = finite_difference_greeks(normals, **kwargs)
        pw = pathwise_greeks(normals, **kwargs)
        lr = likelihood_ratio_greeks(normals, **kwargs)
        cache[name] = {"fd": fd, "pathwise": pw, "lr": lr}

        greeks_payload[name] = {}
        for greek in ("delta", "gamma", "vega", "theta", "rho"):
            greeks_payload[name][greek] = {
                "fd": _payload(fd[greek]),
                "pathwise": _payload(pw[greek]),
                "lr": _payload(lr[greek]),
            }

    analytic_call = black_scholes_greeks(
        spot=spot,
        strike=strike,
        rate=rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        maturity=maturity,
        option_type="call",
    )
    analytic_put = black_scholes_greeks(
        spot=spot,
        strike=strike,
        rate=rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        maturity=maturity,
        option_type="put",
    )

    bs_errors = {}
    for short, name, analytic in (
        ("call", "european_call", analytic_call),
        ("put", "european_put", analytic_put),
    ):
        for greek in ("delta", "gamma", "vega", "theta", "rho"):
            bs_errors[f"{greek}_{short}"] = float(
                cache[name]["fd"][greek].value - analytic[greek]
            )

    call_fd = cache["european_call"]["fd"]
    put_fd = cache["european_put"]["fd"]
    parity = {
        "delta_diff": float(call_fd["delta"].value - put_fd["delta"].value),
        "gamma_diff": float(call_fd["gamma"].value - put_fd["gamma"].value),
        "rho_diff": float(call_fd["rho"].value - put_fd["rho"].value),
    }

    call_discounted, call_paths = _discounted_payoff_samples(
        normals,
        spot=spot,
        strike=strike,
        rate=rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        maturity=maturity,
        option_type="call",
        asian=False,
    )
    terminal = call_paths[:, -1]
    z_terminal = np.sum(normals, axis=1) / math.sqrt(monitoring)
    disc = math.exp(-rate * maturity)
    pw_delta_samples = disc * (terminal > strike) * terminal / spot
    lr_delta_samples = (
        call_discounted
        * z_terminal
        / (spot * volatility * math.sqrt(maturity))
    )
    pw_var = float(np.var(pw_delta_samples, ddof=1))
    lr_var = float(np.var(lr_delta_samples, ddof=1))
    variance_ratio = {
        "pw_variance": pw_var,
        "lr_variance": lr_var,
        "pw_se": float(math.sqrt(pw_var / n_paths)),
        "lr_se": float(math.sqrt(lr_var / n_paths)),
    }

    asian_vs_european = {
        "price_call": bool(
            prices_payload["asian_call"]["mc_price"]
            < prices_payload["european_call"]["mc_price"]
        ),
        "price_put": bool(
            prices_payload["asian_put"]["mc_price"]
            < prices_payload["european_put"]["mc_price"]
        ),
        "delta_call": bool(
            cache["asian_call"]["fd"]["delta"].value
            < cache["european_call"]["fd"]["delta"].value
        ),
    }
    consistency = {
        "bs_errors": bs_errors,
        "put_call_parity": parity,
        "variance_ratio": variance_ratio,
        "asian_vs_european": asian_vs_european,
    }

    spot_grid = np.linspace(70.0, 130.0, 13)
    vol_grid = np.linspace(0.10, 0.50, 9)
    delta_rows = []
    vega_rows = []
    call_delta_matrix = np.empty((13, 9), dtype=float)

    for i, s0 in enumerate(spot_grid):
        for j, sigma in enumerate(vol_grid):
            call = black_scholes_greeks(
                spot=float(s0),
                strike=strike,
                rate=rate,
                dividend_yield=dividend_yield,
                volatility=float(sigma),
                maturity=maturity,
                option_type="call",
            )
            put = black_scholes_greeks(
                spot=float(s0),
                strike=strike,
                rate=rate,
                dividend_yield=dividend_yield,
                volatility=float(sigma),
                maturity=maturity,
                option_type="put",
            )
            call_delta_matrix[i, j] = call["delta"]
            delta_rows.append({
                "S0": float(s0),
                "sigma": float(sigma),
                "delta_call_eu": call["delta"],
                "delta_put_eu": put["delta"],
            })
            vega_rows.append({
                "S0": float(s0),
                "sigma": float(sigma),
                "vega_call_eu": call["vega"],
                "vega_put_eu": put["vega"],
            })

    convergence_rows = []
    for n in (10_000, 50_000, 100_000, 200_000, 500_000):
        subset = normals[:n]
        kwargs = dict(
            spot=spot,
            strike=strike,
            rate=rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            maturity=maturity,
            option_type="call",
            asian=False,
        )
        estimates = {
            "fd": finite_difference_greeks(subset, **kwargs)["delta"],
            "pathwise": pathwise_greeks(subset, **kwargs)["delta"],
            "lr": likelihood_ratio_greeks(subset, **kwargs)["delta"],
        }
        for method, estimate in estimates.items():
            convergence_rows.append({
                "N": n,
                "method": method,
                "greek": "delta",
                "value": estimate.value,
                "se": estimate.se,
            })

    delta_surface = pd.DataFrame(delta_rows)
    vega_surface = pd.DataFrame(vega_rows)
    convergence = pd.DataFrame(convergence_rows)

    parity_ok = (
        abs(parity["delta_diff"] - math.exp(-dividend_yield * maturity)) <= 0.005
        and abs(parity["gamma_diff"]) <= 0.001
        and abs(
            parity["rho_diff"]
            - strike * maturity * math.exp(-rate * maturity)
        ) <= 1.0
    )
    bs_ok = all(
        abs(cache[name]["fd"][greek].value - analytic[greek])
        <= max(0.05 * abs(analytic[greek]), 0.01)
        for name, analytic in (
            ("european_call", analytic_call),
            ("european_put", analytic_put),
        )
        for greek in ("delta", "gamma", "vega", "theta", "rho")
    )
    variance_ok = pw_var < lr_var
    surface_ok = (
        delta_surface["delta_call_eu"].between(0.0, 1.0).all()
        and delta_surface["delta_put_eu"].between(-1.0, 0.0).all()
        and (
            vega_surface[["vega_call_eu", "vega_put_eu"]] > 0.0
        ).all().all()
        and all(
            np.all(np.diff(call_delta_matrix[:, j]) >= -1e-12)
            for j in range(call_delta_matrix.shape[1])
        )
    )
    conv_500k = convergence[convergence["N"] == 500_000]
    convergence_ok = bool(
        np.all(
            np.abs(
                conv_500k["value"].to_numpy(dtype=float)
                - analytic_call["delta"]
            )
            <= 0.01
        )
    )
    prices_ok = bool(
        abs(
            prices_payload["european_call"]["mc_price"]
            - prices_payload["european_call"]["bs_price"]
        )
        <= 4.0 * prices_payload["european_call"]["mc_se"]
        and abs(
            prices_payload["european_put"]["mc_price"]
            - prices_payload["european_put"]["bs_price"]
        )
        <= 4.0 * prices_payload["european_put"]["mc_se"]
    )
    summary = {
        "prices_match_bs": prices_ok,
        "greeks_match_bs": bool(bs_ok),
        "put_call_parity_passed": bool(parity_ok),
        "pathwise_variance_lt_lr": bool(variance_ok),
        "surface_checks_passed": bool(surface_ok),
        "convergence_passed": bool(convergence_ok),
        "all_checks_passed": bool(
            prices_ok
            and bs_ok
            and parity_ok
            and variance_ok
            and surface_ok
            and convergence_ok
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("prices.json", prices_payload),
        ("greeks.json", greeks_payload),
        ("consistency.json", consistency),
        ("summary.json", summary),
    ):
        (out_dir / name).write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    delta_surface.to_csv(out_dir / "delta_surface.csv", index=False)
    vega_surface.to_csv(out_dir / "vega_surface.csv", index=False)
    convergence.to_csv(out_dir / "convergence.csv", index=False)

    s_mesh, v_mesh = np.meshgrid(spot_grid, vol_grid, indexing="ij")
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(s_mesh, v_mesh, call_delta_matrix)
    ax.set_xlabel("S0")
    ax.set_ylabel("sigma")
    ax.set_zlabel("call delta")
    ax.set_title("European Call Delta Surface")
    fig.tight_layout()
    fig.savefig(out_dir / "delta_surface.png", dpi=120)
    plt.close(fig)
