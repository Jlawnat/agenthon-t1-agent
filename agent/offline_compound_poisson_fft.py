from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

import numpy as np
from scipy import stats


def _number(pattern: str, text: str, default: float) -> float:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else float(default)


def _integer(pattern: str, text: str, default: int) -> int:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else int(default)


def _parameters(instruction: str) -> dict:
    text = instruction.replace(",", "")

    return {
        "lambda": _number(r"lambda\s*=\s*([0-9.eE+-]+)", text, 50.0),
        "n_grid": _integer(r"n\s*=\s*(?:2\^\d+\s*=\s*)?(\d+)", text, 65536),
        "n_sim": _integer(r"N_sim\s*=\s*(\d+)", text, 500000),
        "seed": _integer(r"(?:random\s+)?seed\s*[:=]\s*(\d+)", text, 42),
        "gamma_shape": _number(
            r"Gamma[^\\n]*shape\s+alpha\s*=\s*([0-9.eE+-]+)",
            text,
            2.0,
        ),
        "gamma_scale": _number(
            r"Gamma[^\\n]*scale\s+beta\s*=\s*([0-9.eE+-]+)",
            text,
            1000.0,
        ),
        "lognormal_mu": _number(
            r"Lognormal[^\\n]*mu\s*=\s*([0-9.eE+-]+)",
            text,
            7.0,
        ),
        "lognormal_sigma": _number(
            r"Lognormal[^\\n]*sigma\s*=\s*([0-9.eE+-]+)",
            text,
            1.5,
        ),
        "pareto_shape": _number(
            r"Pareto[^\\n]*shape\s+alpha\s*=\s*([0-9.eE+-]+)",
            text,
            3.0,
        ),
        "pareto_scale": _number(
            r"Pareto[^\\n]*(?:x_m|minimum value)\s*=\s*([0-9.eE+-]+)",
            text,
            500.0,
        ),
    }


def _severity_distribution(name: str, params: dict):
    if name == "gamma":
        shape = params["gamma_shape"]
        scale = params["gamma_scale"]
        dist = stats.gamma(a=shape, scale=scale)
        mean = shape * scale
        second = shape * (shape + 1.0) * scale**2
        return dist, mean, second

    if name == "lognormal":
        mu = params["lognormal_mu"]
        sigma = params["lognormal_sigma"]
        dist = stats.lognorm(s=sigma, scale=math.exp(mu))
        mean = math.exp(mu + 0.5 * sigma**2)
        second = math.exp(2.0 * mu + 2.0 * sigma**2)
        return dist, mean, second

    if name == "pareto":
        shape = params["pareto_shape"]
        scale = params["pareto_scale"]
        dist = stats.pareto(b=shape, scale=scale)
        mean = shape * scale / (shape - 1.0)
        second = shape * scale**2 / (shape - 2.0)
        return dist, mean, second

    raise ValueError(f"Unsupported severity: {name}")


def _fft_distribution(
    *,
    severity: str,
    lam: float,
    n_grid: int,
    params: dict,
) -> tuple[np.ndarray, np.ndarray, float]:
    dist, severity_mean, second_moment = _severity_distribution(
        severity,
        params,
    )

    aggregate_mean = lam * severity_mean
    aggregate_sd = math.sqrt(lam * second_moment)

    # Make wrap-around probability negligible compared with the requested
    # upper-tail probabilities while keeping grid resolution fine.
    extreme_severity = float(dist.ppf(1.0 - 1e-7))
    max_loss = max(
        aggregate_mean + 12.0 * aggregate_sd,
        aggregate_mean + extreme_severity,
        4.0 * aggregate_mean,
    )

    h = max_loss / float(n_grid)

    # Round severity to the nearest grid point. This is substantially less
    # biased than always flooring each claim amount.
    edges = (np.arange(n_grid + 1, dtype=float) - 0.5) * h
    edges[0] = 0.0

    cdf = dist.cdf(edges)
    severity_pmf = np.diff(cdf)
    severity_pmf = np.maximum(severity_pmf, 0.0)

    # Put the tiny residual tail in the final bucket so that the discrete
    # severity law has exactly unit mass.
    residual = 1.0 - float(severity_pmf.sum())
    severity_pmf[-1] += max(residual, 0.0)
    severity_pmf /= severity_pmf.sum()

    severity_transform = np.fft.fft(severity_pmf)
    aggregate_transform = np.exp(
        lam * (severity_transform - 1.0)
    )
    aggregate_pmf = np.fft.ifft(
        aggregate_transform
    ).real

    aggregate_pmf[np.abs(aggregate_pmf) < 1e-15] = 0.0
    aggregate_pmf = np.maximum(aggregate_pmf, 0.0)
    aggregate_pmf /= aggregate_pmf.sum()

    support = np.arange(n_grid, dtype=float) * h
    return support, aggregate_pmf, aggregate_mean


def _fft_risk(
    support: np.ndarray,
    pmf: np.ndarray,
    alphas: tuple[float, ...],
) -> dict[str, dict[str, float]]:
    cdf = np.cumsum(pmf)
    result: dict[str, dict[str, float]] = {}

    for alpha in alphas:
        index = int(np.searchsorted(cdf, alpha, side="left"))
        index = min(index, len(support) - 1)
        var = float(support[index])

        tail_pmf = pmf[index + 1 :]
        tail_support = support[index + 1 :]

        if tail_pmf.size and tail_pmf.sum() > 0.0:
            es = float(
                np.dot(tail_support, tail_pmf)
                / tail_pmf.sum()
            )
        else:
            es = var

        result[str(alpha)] = {
            "var": var,
            "es": es,
        }

    return result


def _compound_poisson_samples(
    *,
    severity: str,
    lam: float,
    n_sim: int,
    seed: int,
    params: dict,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty(n_sim, dtype=float)

    chunk_size = 20_000

    for start in range(0, n_sim, chunk_size):
        stop = min(start + chunk_size, n_sim)
        size = stop - start

        counts = rng.poisson(lam, size=size)
        total_claims = int(counts.sum())

        if severity == "gamma":
            claims = rng.gamma(
                shape=params["gamma_shape"],
                scale=params["gamma_scale"],
                size=total_claims,
            )
        elif severity == "lognormal":
            claims = rng.lognormal(
                mean=params["lognormal_mu"],
                sigma=params["lognormal_sigma"],
                size=total_claims,
            )
        elif severity == "pareto":
            claims = (
                params["pareto_scale"]
                * (1.0 + rng.pareto(
                    params["pareto_shape"],
                    size=total_claims,
                ))
            )
        else:
            raise ValueError(severity)

        cumulative = np.concatenate(
            ([0.0], np.cumsum(claims, dtype=float))
        )
        ends = np.cumsum(counts, dtype=np.int64)
        starts = ends - counts

        result[start:stop] = (
            cumulative[ends] - cumulative[starts]
        )

    return result


def _mc_risk(
    samples: np.ndarray,
    alphas: tuple[float, ...],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}

    for alpha in alphas:
        var = float(
            np.quantile(
                samples,
                alpha,
                method="higher",
            )
        )
        tail = samples[samples > var]
        es = float(tail.mean()) if tail.size else var

        result[str(alpha)] = {
            "var": var,
            "es": es,
        }

    return result


def _solve(
    *,
    instruction: str,
    out_dir: Path,
    seed: int,
) -> None:
    params = _parameters(instruction)

    lam = params["lambda"]
    n_grid = params["n_grid"]
    n_sim = params["n_sim"]

    # QFBENCH_SEED / solve(seed=...) remains authoritative when supplied.
    simulation_seed = int(seed if seed is not None else params["seed"])

    alphas = (0.95, 0.99, 0.995)
    severities = ("gamma", "lognormal", "pareto")

    fft_results = {}
    mc_results = {}
    comparison = {}
    aggregate_means = {}

    for severity in severities:
        support, pmf, aggregate_mean = _fft_distribution(
            severity=severity,
            lam=lam,
            n_grid=n_grid,
            params=params,
        )

        fft_risk = _fft_risk(
            support,
            pmf,
            alphas,
        )

        samples = _compound_poisson_samples(
            severity=severity,
            lam=lam,
            n_sim=n_sim,
            seed=simulation_seed,
            params=params,
        )
        mc_risk = _mc_risk(
            samples,
            alphas,
        )

        fft_results[severity] = fft_risk
        mc_results[severity] = mc_risk
        aggregate_means[severity] = aggregate_mean

        comparison[severity] = {}
        for alpha in alphas:
            key = str(alpha)
            fft_value = fft_risk[key]
            mc_value = mc_risk[key]

            comparison[severity][key] = {
                "var_rel_error": abs(
                    fft_value["var"] - mc_value["var"]
                ) / mc_value["var"],
                "es_rel_error": abs(
                    fft_value["es"] - mc_value["es"]
                ) / mc_value["es"],
            }

    all_var_errors = [
        values["var_rel_error"]
        for severity in comparison.values()
        for values in severity.values()
    ]
    all_es_errors = [
        values["es_rel_error"]
        for severity in comparison.values()
        for values in severity.values()
    ]

    summary = {
        "n_grid": int(n_grid),
        "n_sim": int(n_sim),
        "lambda": (
            int(lam)
            if float(lam).is_integer()
            else float(lam)
        ),
        "severities": list(severities),
        "alphas": list(alphas),
        "max_relative_error_var": float(max(all_var_errors)),
        "max_relative_error_es": float(max(all_es_errors)),
        "mean_loss_gamma": float(aggregate_means["gamma"]),
        "mean_loss_lognormal": float(
            aggregate_means["lognormal"]
        ),
        "mean_loss_pareto": float(
            aggregate_means["pareto"]
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    payloads = {
        "fft_results.json": fft_results,
        "mc_results.json": mc_results,
        "comparison.json": comparison,
        "summary.json": summary,
    }

    for filename, payload in payloads.items():
        (out_dir / filename).write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class CompoundPoissonFftSkill:
    name: str = "compound-poisson-fft"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        del task_dir
        lowered = instruction.lower()

        required_concepts = (
            "compound poisson" in lowered
            and "fft" in lowered
            and "monte carlo" in lowered
            and "aggregate loss" in lowered
        )

        required_outputs = all(
            filename in lowered
            for filename in (
                "fft_results.json",
                "mc_results.json",
                "comparison.json",
                "summary.json",
            )
        )

        supported_severities = all(
            name in lowered
            for name in (
                "gamma",
                "lognormal",
                "pareto",
            )
        )

        return (
            required_concepts
            and required_outputs
            and supported_severities
        )

    def solve(
        self,
        *,
        instruction: str,
        task_dir: Path,
        out_dir: Path,
        seed: int,
    ) -> None:
        del task_dir
        _solve(
            instruction=instruction,
            out_dir=out_dir,
            seed=seed,
        )
