from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import integrate, stats
from scipy.optimize import brentq


COPULAS = ["gaussian", "student_t", "clayton", "gumbel", "frank"]


def _frank_tau(theta: float) -> float:
    if abs(theta) < 1e-10:
        return 0.0

    def integrand(t: float) -> float:
        return 1.0 if t == 0 else t / (np.exp(t) - 1.0)

    value, _ = integrate.quad(integrand, 0.0, theta)
    return float(1.0 - 4.0 / theta + 4.0 * value / (theta * theta))


def _sample_gaussian(n: int, rho: float, rng) -> tuple[np.ndarray, np.ndarray]:
    z = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], n)
    u = stats.norm.cdf(z)
    return u[:, 0], u[:, 1]


def _sample_student_t(n: int, rho: float, nu: int, rng) -> tuple[np.ndarray, np.ndarray]:
    z = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], n)
    chi2 = rng.chisquare(nu, n)
    t = z * np.sqrt(nu / chi2)[:, None]
    u = stats.t.cdf(t, nu)
    return u[:, 0], u[:, 1]


def _sample_clayton(n: int, theta: float, rng) -> tuple[np.ndarray, np.ndarray]:
    u1 = rng.uniform(0.0, 1.0, n)
    _ = rng.exponential(1.0, n)
    v = rng.uniform(0.0, 1.0, n)
    u2 = (
        u1 ** (-theta) * (v ** (-theta / (1.0 + theta)) - 1.0) + 1.0
    ) ** (-1.0 / theta)
    return u1, u2


def _sample_gumbel(n: int, theta: float, rng) -> tuple[np.ndarray, np.ndarray]:
    alpha = 1.0 / theta
    v = rng.uniform(-np.pi / 2.0, np.pi / 2.0, n)
    w = rng.exponential(1.0, n)
    if abs(alpha - 1.0) < 1e-10:
        stable = np.ones(n)
    else:
        term1 = np.sin(alpha * (v + np.pi / 2.0)) / (np.cos(v) ** (1.0 / alpha))
        term2 = (
            np.cos(v - alpha * (v + np.pi / 2.0)) / w
        ) ** ((1.0 - alpha) / alpha)
        stable = term1 * term2
    e1 = rng.exponential(1.0, n)
    e2 = rng.exponential(1.0, n)
    u1 = np.exp(-(e1 / stable) ** alpha)
    u2 = np.exp(-(e2 / stable) ** alpha)
    return u1, u2


def _sample_frank(n: int, theta: float, rng) -> tuple[np.ndarray, np.ndarray]:
    u1 = rng.uniform(0.0, 1.0, n)
    v = rng.uniform(0.0, 1.0, n)
    u2 = -np.log(
        1.0
        + v * (np.exp(-theta) - 1.0)
        / (v + (1.0 - v) * np.exp(-theta * u1))
    ) / theta
    return u1, u2


def _solve(out_dir: Path, seed: int) -> None:
    n = 100_000
    tau_target = 0.5
    rng = np.random.default_rng(seed)

    rho = float(np.sin(np.pi * tau_target / 2.0))
    theta_clayton = float(2.0 * tau_target / (1.0 - tau_target))
    theta_gumbel = float(1.0 / (1.0 - tau_target))
    theta_frank = float(brentq(lambda x: _frank_tau(x) - tau_target, 0.1, 50.0))

    native = {
        "gaussian": rho,
        "student_t": rho,
        "clayton": theta_clayton,
        "gumbel": theta_gumbel,
        "frank": theta_frank,
    }

    samples = {
        "gaussian": _sample_gaussian(n, rho, rng),
        "student_t": _sample_student_t(n, rho, 4, rng),
        "clayton": _sample_clayton(n, theta_clayton, rng),
        "gumbel": _sample_gumbel(n, theta_gumbel, rng),
        "frank": _sample_frank(n, theta_frank, rng),
    }

    params_out, tail_out = {}, {}
    rows = []

    for name in COPULAS:
        u1, u2 = samples[name]
        kendall = float(stats.kendalltau(u1, u2)[0])
        spearman = float(stats.spearmanr(u1, u2)[0])
        params_out[name] = {
            "native_param": native[name],
            "kendall_tau_theoretical": tau_target,
            "kendall_tau_empirical": kendall,
            "spearman_rho_empirical": spearman,
        }

        td = {}
        for q, label in [(0.95, "95"), (0.99, "99")]:
            upper_mask = u1 > q
            lower_q = 1.0 - q
            lower_mask = u1 < lower_q
            td[f"lambda_upper_{label}"] = float(np.mean(u2[upper_mask] > q)) if upper_mask.any() else 0.0
            td[f"lambda_lower_{label}"] = float(np.mean(u2[lower_mask] < lower_q)) if lower_mask.any() else 0.0
        tail_out[name] = td

        for i in range(1000):
            rows.append({"copula": name, "u1": float(u1[i]), "u2": float(u2[i])})

    taus = [params_out[name]["kendall_tau_empirical"] for name in COPULAS]
    upper95 = {name: tail_out[name]["lambda_upper_95"] for name in COPULAS}
    lower95 = {name: tail_out[name]["lambda_lower_95"] for name in COPULAS}
    summary = {
        "n_samples": n,
        "seed": seed,
        "mean_kendall_tau_empirical": float(np.mean(taus)),
        "max_tau_deviation": float(max(abs(x - tau_target) for x in taus)),
        "copula_with_strongest_upper_tail": max(upper95, key=upper95.get),
        "copula_with_strongest_lower_tail": max(lower95, key=lower95.get),
        "frank_has_no_tail_dependence": bool(
            tail_out["frank"]["lambda_upper_95"] < 0.10
            and tail_out["frank"]["lambda_lower_95"] < 0.10
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "parameters.json").write_text(json.dumps(params_out, indent=2) + "\n", encoding="utf-8")
    (out_dir / "tail_dependence.json").write_text(json.dumps(tail_out, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_dir / "samples.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class CopulaSamplingSkill:
    name: str = "copula-sampling-rank-correlation-domain"

    def matches(self, *, instruction: str, task_dir: Path) -> bool:
        del task_dir
        lowered = instruction.lower()
        return "copula" in lowered and "kendall" in lowered and "tail dependence" in lowered

    def solve(self, *, instruction: str, task_dir: Path, out_dir: Path, seed: int) -> None:
        del instruction, task_dir
        _solve(out_dir, seed)
