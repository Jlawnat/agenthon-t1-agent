
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import integrate, optimize, stats


_DISTRIBUTIONS = (
    "normal",
    "student_t",
    "contaminated_normal",
)

_METHODS = (
    "historical",
    "parametric_normal",
    "parametric_t",
    "kernel_density",
)

_ALPHAS = (
    0.95,
    0.99,
)


def _normal_true(alpha: float) -> tuple[float, float]:
    q = float(stats.norm.ppf(alpha))
    es = float(stats.norm.pdf(q) / (1.0 - alpha))
    return q, es


def _student_t_true(
    alpha: float,
    *,
    df: float,
) -> tuple[float, float]:
    q = float(stats.t.ppf(alpha, df))
    density = float(stats.t.pdf(q, df))
    es = float(
        (df + q * q)
        / (df - 1.0)
        * density
        / (1.0 - alpha)
    )
    return q, es


def _contaminated_cdf(x: float) -> float:
    return float(
        0.95 * stats.norm.cdf(x, loc=0.0, scale=1.0)
        + 0.05 * stats.norm.cdf(x, loc=0.0, scale=3.0)
    )


def _contaminated_true(
    alpha: float,
) -> tuple[float, float]:
    q = float(
        optimize.brentq(
            lambda x: _contaminated_cdf(x) - alpha,
            -20.0,
            20.0,
            xtol=1e-14,
            rtol=1e-14,
        )
    )

    tail_first_moment = float(
        0.95 * stats.norm.pdf(q)
        + 0.05 * 3.0 * stats.norm.pdf(q / 3.0)
    )

    es = float(
        tail_first_moment
        / (1.0 - alpha)
    )

    return q, es


def analytical_true_values() -> dict[str, dict[str, dict[str, float]]]:
    result = {}

    for distribution in _DISTRIBUTIONS:
        result[distribution] = {}

        for alpha in _ALPHAS:
            if distribution == "normal":
                var, es = _normal_true(alpha)
            elif distribution == "student_t":
                var, es = _student_t_true(
                    alpha,
                    df=5.0,
                )
            else:
                var, es = _contaminated_true(
                    alpha
                )

            result[
                distribution
            ][
                f"{alpha:.2f}"
            ] = {
                "var": float(var),
                "es": float(es),
            }

    return result


def generate_loss_samples(
    *,
    n_samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)

    normal = rng.normal(
        0.0,
        1.0,
        size=n_samples,
    )

    student_t = rng.standard_t(
        5,
        size=n_samples,
    )

    components = rng.uniform(
        0.0,
        1.0,
        size=n_samples,
    )

    core = components < 0.95

    contaminated = np.empty(
        n_samples,
        dtype=float,
    )

    contaminated[core] = rng.normal(
        0.0,
        1.0,
        size=int(core.sum()),
    )

    contaminated[~core] = rng.normal(
        0.0,
        3.0,
        size=int((~core).sum()),
    )

    return {
        "normal": normal,
        "student_t": student_t,
        "contaminated_normal": contaminated,
    }


def historical_estimate(
    sample: np.ndarray,
    *,
    alpha: float,
) -> tuple[float, float]:
    q = float(
        np.quantile(
            sample,
            alpha,
        )
    )
    tail = sample[
        sample >= q
    ]
    return q, float(
        np.mean(tail)
    )


def parametric_normal_estimate(
    sample: np.ndarray,
    *,
    alpha: float,
) -> tuple[float, float]:
    mu = float(
        np.mean(sample)
    )
    sigma = float(
        np.std(
            sample,
            ddof=0,
        )
    )
    z = float(
        stats.norm.ppf(alpha)
    )

    q = mu + sigma * z
    es = (
        mu
        + sigma
        * stats.norm.pdf(z)
        / (1.0 - alpha)
    )

    return float(q), float(es)


def parametric_t_estimate(
    sample: np.ndarray,
    *,
    alpha: float,
) -> tuple[float, float]:
    df, loc, scale = stats.t.fit(
        sample
    )

    q = float(
        stats.t.ppf(
            alpha,
            df,
            loc=loc,
            scale=scale,
        )
    )

    tail_first_moment = float(
        integrate.quad(
            lambda x: (
                x
                * stats.t.pdf(
                    x,
                    df,
                    loc=loc,
                    scale=scale,
                )
            ),
            q,
            np.inf,
            epsabs=1e-10,
            epsrel=1e-9,
            limit=200,
        )[0]
    )

    es = tail_first_moment / (
        1.0 - alpha
    )

    return float(q), float(es)


def _kde_cdf(
    kde: stats.gaussian_kde,
    x: float,
) -> float:
    return float(
        kde.integrate_box_1d(
            -np.inf,
            x,
        )
    )


def kernel_density_estimate(
    sample: np.ndarray,
    *,
    alpha: float,
) -> tuple[float, float]:
    kde = stats.gaussian_kde(
        sample,
        bw_method="silverman",
    )

    sample_std = float(
        np.std(
            sample,
            ddof=1,
        )
    )

    lower = float(
        np.min(sample)
        - 8.0 * sample_std
    )

    upper = float(
        np.max(sample)
        + 8.0 * sample_std
    )

    q = float(
        optimize.brentq(
            lambda x: (
                _kde_cdf(kde, x)
                - alpha
            ),
            lower,
            upper,
            xtol=1e-10,
            rtol=1e-10,
        )
    )

    tail_upper = max(
        upper,
        q + 12.0 * sample_std,
    )

    tail_first_moment = float(
        integrate.quad(
            lambda x: (
                x
                * float(
                    kde(
                        np.asarray(
                            [x],
                            dtype=float,
                        )
                    )[0]
                )
            ),
            q,
            tail_upper,
            epsabs=1e-8,
            epsrel=1e-7,
            limit=200,
        )[0]
    )

    tail_probability = float(
        1.0
        - _kde_cdf(
            kde,
            q,
        )
    )

    es = (
        tail_first_moment
        / tail_probability
    )

    return float(q), float(es)


def estimate_all_methods(
    samples: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []

    for distribution in _DISTRIBUTIONS:
        sample = np.asarray(
            samples[distribution],
            dtype=float,
        )

        for method in _METHODS:
            for alpha in _ALPHAS:
                if method == "historical":
                    var, es = historical_estimate(
                        sample,
                        alpha=alpha,
                    )
                elif method == "parametric_normal":
                    var, es = (
                        parametric_normal_estimate(
                            sample,
                            alpha=alpha,
                        )
                    )
                elif method == "parametric_t":
                    var, es = (
                        parametric_t_estimate(
                            sample,
                            alpha=alpha,
                        )
                    )
                else:
                    var, es = (
                        kernel_density_estimate(
                            sample,
                            alpha=alpha,
                        )
                    )

                rows.append(
                    {
                        "distribution": distribution,
                        "method": method,
                        "alpha": float(alpha),
                        "var_estimate": float(var),
                        "es_estimate": float(es),
                    }
                )

    return pd.DataFrame(
        rows,
        columns=[
            "distribution",
            "method",
            "alpha",
            "var_estimate",
            "es_estimate",
        ],
    )


def comparison_from_estimates(
    estimates: pd.DataFrame,
    true_values: dict[str, dict[str, dict[str, float]]],
) -> dict[str, object]:
    method_rmse = {}

    for method in _METHODS:
        subset = estimates[
            estimates["method"]
            == method
        ]

        var_errors = []
        es_errors = []

        for row in subset.itertuples(
            index=False
        ):
            truth = true_values[
                str(row.distribution)
            ][
                f"{float(row.alpha):.2f}"
            ]

            var_errors.append(
                (
                    float(row.var_estimate)
                    - float(truth["var"])
                )
                ** 2
            )

            es_errors.append(
                (
                    float(row.es_estimate)
                    - float(truth["es"])
                )
                ** 2
            )

        method_rmse[method] = {
            "var_rmse": float(
                math.sqrt(
                    float(
                        np.mean(var_errors)
                    )
                )
            ),
            "es_rmse": float(
                math.sqrt(
                    float(
                        np.mean(es_errors)
                    )
                )
            ),
        }

    best_per_distribution = {}

    for distribution in _DISTRIBUTIONS:
        scores = []

        for method in _METHODS:
            subset = estimates[
                (
                    estimates["distribution"]
                    == distribution
                )
                & (
                    estimates["method"]
                    == method
                )
            ]

            squared = []

            for row in subset.itertuples(
                index=False
            ):
                truth = true_values[
                    distribution
                ][
                    f"{float(row.alpha):.2f}"
                ]

                squared.append(
                    (
                        float(row.var_estimate)
                        - float(truth["var"])
                    )
                    ** 2
                )
                squared.append(
                    (
                        float(row.es_estimate)
                        - float(truth["es"])
                    )
                    ** 2
                )

            scores.append(
                (
                    float(
                        math.sqrt(
                            float(
                                np.mean(squared)
                            )
                        )
                    ),
                    method,
                )
            )

        scores.sort(
            key=lambda item: (
                item[0],
                _METHODS.index(item[1]),
            )
        )

        best_per_distribution[
            distribution
        ] = scores[0][1]

    return {
        "method_rmse": method_rmse,
        "best_method_per_distribution": (
            best_per_distribution
        ),
    }


@dataclass(frozen=True)
class DistributionRiskSkill:
    name: str = "distribution-risk-domain"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        del task_dir

        lowered = instruction.lower()

        return (
            (
                "value-at-risk" in lowered
                or "value at risk" in lowered
                or "var and es" in lowered
            )
            and "expected shortfall" in lowered
            and (
                "simulated loss" in lowered
                or "data generation" in lowered
                or "contaminated normal" in lowered
            )
            and (
                "kernel density" in lowered
                or "gaussian kde" in lowered
            )
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

        lowered = instruction.lower()

        n_samples = 10000

        match = re.search(
            r"\bn\s*=\s*([\d,]+)",
            lowered,
        )

        if match:
            n_samples = int(
                match.group(1).replace(
                    ",",
                    "",
                )
            )

        effective_seed = (
            42
            if (
                "seed = 42" in lowered
                or "seed=42" in lowered
            )
            else int(seed)
        )

        true_values = (
            analytical_true_values()
        )

        samples = generate_loss_samples(
            n_samples=n_samples,
            seed=effective_seed,
        )

        estimates = estimate_all_methods(
            samples
        )

        comparison = (
            comparison_from_estimates(
                estimates,
                true_values,
            )
        )

        overall_best_method = min(
            _METHODS,
            key=lambda method: (
                float(
                    comparison[
                        "method_rmse"
                    ][
                        method
                    ][
                        "var_rmse"
                    ]
                )
                + float(
                    comparison[
                        "method_rmse"
                    ][
                        method
                    ][
                        "es_rmse"
                    ]
                ),
                _METHODS.index(method),
            ),
        )

        summary = {
            "n_samples": int(n_samples),
            "seed": int(effective_seed),
            "alpha_levels": [0.95, 0.99],
            "distributions": list(
                _DISTRIBUTIONS
            ),
            "methods": list(
                _METHODS
            ),
            "overall_best_method": (
                overall_best_method
            ),
        }

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            out_dir
            / "true_values.json"
        ).write_text(
            json.dumps(
                true_values,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        estimates.to_csv(
            out_dir
            / "estimates.csv",
            index=False,
        )

        (
            out_dir
            / "comparison.json"
        ).write_text(
            json.dumps(
                comparison,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        (
            out_dir
            / "summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
