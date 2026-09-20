from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def _find_credit_data_dir(task_dir: Path) -> Path:
    required = {
        "portfolio.csv",
        "sector_correlations.csv",
        "config.json",
    }

    directories = sorted(
        {
            path.parent
            for path in task_dir.rglob("*")
            if path.is_file() and "checks" not in path.parts
        },
        key=lambda path: str(path),
    )

    for directory in directories:
        names = {
            path.name
            for path in directory.iterdir()
            if path.is_file()
        }

        if required.issubset(names):
            return directory

    raise RuntimeError(
        "Could not discover credit-portfolio input files."
    )


def _load_inputs(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    portfolio = pd.read_csv(
        data_dir / "portfolio.csv"
    )

    correlation = pd.read_csv(
        data_dir / "sector_correlations.csv",
        index_col=0,
    )

    config = json.loads(
        (
            data_dir / "config.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    required_portfolio = {
        "obligor_id",
        "issuer",
        "pd_1y",
        "exposure_usd",
        "recovery_rate",
        "recovery_stressed",
        "factor_sector",
        "basel_correlation",
    }

    missing = (
        required_portfolio
        - set(portfolio.columns)
    )

    if missing:
        raise RuntimeError(
            "Credit portfolio is missing columns: "
            + ", ".join(sorted(missing))
        )

    correlation = correlation.astype(
        float
    )

    if (
        correlation.shape[0]
        != correlation.shape[1]
    ):
        raise RuntimeError(
            "Sector correlation matrix must be square."
        )

    if list(
        correlation.index
    ) != list(
        correlation.columns
    ):
        correlation = correlation.reindex(
            index=correlation.columns,
            columns=correlation.columns,
        )

    matrix = correlation.to_numpy(
        dtype=float
    )

    if not np.allclose(
        matrix,
        matrix.T,
        atol=1e-10,
    ):
        raise RuntimeError(
            "Sector correlation matrix must be symmetric."
        )

    eigenvalues = np.linalg.eigvalsh(
        matrix
    )

    if float(
        eigenvalues.min()
    ) < -1e-8:
        raise RuntimeError(
            "Sector correlation matrix must be positive semi-definite."
        )

    return (
        portfolio,
        correlation,
        config,
    )


def _prepare_arrays(
    portfolio: pd.DataFrame,
    correlation: pd.DataFrame,
    *,
    stressed: bool,
    wrong_way_sectors: set[str],
    correlation_boost: float,
) -> dict[str, np.ndarray | list[str]]:
    sector_names = list(
        correlation.columns
    )

    sector_lookup = {
        sector: index
        for index, sector in enumerate(
            sector_names
        )
    }

    portfolio_sectors = (
        portfolio[
            "factor_sector"
        ]
        .astype(str)
        .to_numpy()
    )

    unknown = sorted(
        set(portfolio_sectors)
        - set(sector_names)
    )

    if unknown:
        raise RuntimeError(
            "Portfolio contains sectors missing from correlation matrix: "
            + ", ".join(unknown)
        )

    pd_values = (
        pd.to_numeric(
            portfolio["pd_1y"],
            errors="raise",
        )
        .astype(float)
        .to_numpy()
    )

    exposure = (
        pd.to_numeric(
            portfolio["exposure_usd"],
            errors="raise",
        )
        .astype(float)
        .to_numpy()
    )

    recovery_normal = (
        pd.to_numeric(
            portfolio["recovery_rate"],
            errors="raise",
        )
        .astype(float)
        .to_numpy()
    )

    recovery_stressed = (
        pd.to_numeric(
            portfolio["recovery_stressed"],
            errors="raise",
        )
        .astype(float)
        .to_numpy()
    )

    rho = (
        pd.to_numeric(
            portfolio["basel_correlation"],
            errors="raise",
        )
        .astype(float)
        .to_numpy()
    )

    stress_mask = np.asarray(
        [
            sector in wrong_way_sectors
            for sector in portfolio_sectors
        ],
        dtype=bool,
    )

    if stressed:
        rho = rho.copy()
        rho[
            stress_mask
        ] = np.minimum(
            rho[
                stress_mask
            ]
            + float(
                correlation_boost
            ),
            0.999,
        )

        recovery = np.where(
            stress_mask,
            recovery_stressed,
            recovery_normal,
        )
    else:
        recovery = recovery_normal

    rho = np.clip(
        rho,
        0.0,
        0.999,
    )

    sector_index = np.asarray(
        [
            sector_lookup[
                sector
            ]
            for sector in portfolio_sectors
        ],
        dtype=int,
    )

    loss_given_default = (
        exposure
        * (
            1.0
            - recovery
        )
    )

    return {
        "sector_names": sector_names,
        "sector_index": sector_index,
        "pd": pd_values,
        "exposure": exposure,
        "recovery": recovery,
        "rho": rho,
        "lgd_amount": loss_given_default,
        "sector": portfolio_sectors,
    }


def _simulate_losses(
    *,
    arrays: dict,
    correlation_matrix: np.ndarray,
    n_simulations: int,
    seed: int,
    copula: str,
    t_df: int,
    chunk_size: int = 2000,
    tail_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    sector_index = np.asarray(
        arrays[
            "sector_index"
        ],
        dtype=int,
    )

    pd_values = np.asarray(
        arrays[
            "pd"
        ],
        dtype=float,
    )

    rho = np.asarray(
        arrays[
            "rho"
        ],
        dtype=float,
    )

    lgd_amount = np.asarray(
        arrays[
            "lgd_amount"
        ],
        dtype=float,
    )

    n_obligors = len(
        pd_values
    )

    if copula == "gaussian":
        thresholds = stats.norm.ppf(
            pd_values
        )
    elif copula == "student_t":
        thresholds = stats.t.ppf(
            pd_values,
            df=t_df,
        )
    else:
        raise RuntimeError(
            f"Unsupported copula: {copula}"
        )

    thresholds = np.where(
        pd_values <= 0.0,
        -np.inf,
        thresholds,
    )

    sqrt_rho = np.sqrt(
        rho
    )

    sqrt_idio = np.sqrt(
        1.0
        - rho
    )

    chol = np.linalg.cholesky(
        correlation_matrix
    )

    rng = np.random.RandomState(
        int(seed)
    )

    losses = np.empty(
        n_simulations,
        dtype=float,
    )

    contribution_sum = (
        np.zeros(
            n_obligors,
            dtype=float,
        )
        if tail_threshold is not None
        else None
    )

    tail_count = 0
    cursor = 0

    while cursor < n_simulations:
        size = min(
            int(chunk_size),
            n_simulations
            - cursor,
        )

        independent_factors = (
            rng.standard_normal(
                (
                    size,
                    correlation_matrix.shape[
                        0
                    ],
                )
            )
        )

        factors = (
            independent_factors
            @ chol.T
        )

        idiosyncratic = (
            rng.standard_normal(
                (
                    size,
                    n_obligors,
                )
            )
        )

        asset = (
            factors[
                :,
                sector_index,
            ]
            * sqrt_rho[
                None,
                :,
            ]
            + idiosyncratic
            * sqrt_idio[
                None,
                :,
            ]
        )

        if copula == "student_t":
            mixing = (
                rng.chisquare(
                    int(t_df),
                    size=size,
                )
                / float(
                    t_df
                )
            )

            asset = (
                asset
                / np.sqrt(
                    mixing
                )[
                    :,
                    None,
                ]
            )

        defaults = (
            asset
            < thresholds[
                None,
                :,
            ]
        )

        chunk_losses = (
            defaults
            @ lgd_amount
        )

        losses[
            cursor:
            cursor + size
        ] = chunk_losses

        if (
            tail_threshold is not None
            and contribution_sum
            is not None
        ):
            tail = (
                chunk_losses
                >= float(
                    tail_threshold
                )
            )

            selected = int(
                np.sum(
                    tail
                )
            )

            if selected:
                contribution_sum += (
                    defaults[
                        tail
                    ].sum(
                        axis=0,
                        dtype=float,
                    )
                    * lgd_amount
                )

                tail_count += selected

        cursor += size

    return (
        losses,
        contribution_sum,
        tail_count,
    )


def _var_cvar(
    losses: np.ndarray,
    confidence_levels: list[float],
) -> dict[
    float,
    tuple[
        float,
        float,
    ]
]:
    result = {}

    for alpha in confidence_levels:
        var = float(
            np.quantile(
                losses,
                float(
                    alpha
                ),
            )
        )

        tail = losses[
            losses
            >= var
        ]

        cvar = float(
            np.mean(
                tail
            )
        )

        result[
            float(
                alpha
            )
        ] = (
            var,
            cvar,
        )

    return result


def _loss_statistics(
    losses: np.ndarray,
    analytical_el: float,
) -> dict[str, float]:
    return {
        "mean_loss": float(
            np.mean(
                losses
            )
        ),
        "std_loss": float(
            np.std(
                losses,
                ddof=1,
            )
        ),
        "skewness": float(
            stats.skew(
                losses,
                bias=False,
            )
        ),
        "kurtosis": float(
            stats.kurtosis(
                losses,
                fisher=True,
                bias=False,
            )
        ),
        "max_loss": float(
            np.max(
                losses
            )
        ),
        "pct_zero_loss": float(
            100.0
            * np.mean(
                losses
                == 0.0
            )
        ),
        "expected_loss_analytical": float(
            analytical_el
        ),
    }


def _vasicek_and_gordy(
    *,
    pd_values: np.ndarray,
    exposure: np.ndarray,
    recovery: np.ndarray,
    rho: np.ndarray,
    confidence_levels: list[float],
) -> dict[
    float,
    dict[
        str,
        float,
    ]
]:
    total_exposure = float(
        np.sum(
            exposure
        )
    )

    weights = (
        exposure
        / total_exposure
    )

    lgd = (
        1.0
        - recovery
    )

    valid = (
        pd_values
        > 0.0
    )

    result = {}

    for alpha in confidence_levels:
        z = float(
            stats.norm.ppf(
                float(
                    alpha
                )
            )
        )

        pd_valid = pd_values[
            valid
        ]
        rho_valid = rho[
            valid
        ]
        weights_valid = weights[
            valid
        ]
        lgd_valid = lgd[
            valid
        ]

        h = (
            stats.norm.ppf(
                pd_valid
            )
            + np.sqrt(
                rho_valid
            )
            * z
        ) / np.sqrt(
            1.0
            - rho_valid
        )

        conditional_pd = (
            stats.norm.cdf(
                h
            )
        )

        vasicek = float(
            total_exposure
            * np.sum(
                weights_valid
                * lgd_valid
                * conditional_pd
            )
        )

        conditional_variance = float(
            np.sum(
                weights_valid
                * weights_valid
                * lgd_valid
                * lgd_valid
                * conditional_pd
                * (
                    1.0
                    - conditional_pd
                )
            )
        )

        dh_dalpha = (
            np.sqrt(
                rho_valid
            )
            / (
                np.sqrt(
                    1.0
                    - rho_valid
                )
                * stats.norm.pdf(
                    z
                )
            )
        )

        sensitivity = float(
            np.sum(
                weights_valid
                * lgd_valid
                * stats.norm.pdf(
                    h
                )
                * dh_dalpha
            )
        )

        if sensitivity <= 0.0:
            granularity_adjustment = 0.0
        else:
            granularity_adjustment = float(
                total_exposure
                * conditional_variance
                / (
                    2.0
                    * sensitivity
                )
            )

        result[
            float(
                alpha
            )
        ] = {
            "vasicek_var": (
                vasicek
            ),
            "granularity_adjustment": (
                granularity_adjustment
            ),
            "gordy_var": float(
                vasicek
                + granularity_adjustment
            ),
        }

    return result


def _round_money(
    value: float,
) -> float:
    return round(
        float(
            value
        ),
        2,
    )


@dataclass(frozen=True)
class CreditPortfolioRiskSkill:
    name: str = "credit-portfolio-risk-domain"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        del task_dir

        lowered = (
            instruction.lower()
        )

        return (
            "credit" in lowered
            and "portfolio" in lowered
            and (
                "copula" in lowered
                or "default correlation" in lowered
            )
            and (
                "cvar" in lowered
                or "conditional var" in lowered
                or "expected shortfall" in lowered
            )
            and (
                "obligor" in lowered
                or "recovery" in lowered
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
        del instruction, seed

        data_dir = _find_credit_data_dir(
            task_dir
        )

        (
            portfolio,
            correlation,
            config,
        ) = _load_inputs(
            data_dir
        )

        n_simulations = int(
            config[
                "n_simulations"
            ]
        )

        base_seed = int(
            config[
                "seed"
            ]
        )

        confidence_levels = [
            float(
                value
            )
            for value
            in config[
                "confidence_levels"
            ]
        ]

        t_df = int(
            config[
                "t_copula_df"
            ]
        )

        wrong_way_sectors = set(
            str(
                value
            )
            for value
            in config.get(
                "wrong_way_sectors",
                [],
            )
        )

        correlation_boost = float(
            config.get(
                "wrong_way_correlation_boost",
                0.0,
            )
        )

        matrix = correlation.to_numpy(
            dtype=float
        )

        base_arrays = _prepare_arrays(
            portfolio,
            correlation,
            stressed=False,
            wrong_way_sectors=(
                wrong_way_sectors
            ),
            correlation_boost=(
                correlation_boost
            ),
        )

        stressed_arrays = (
            _prepare_arrays(
                portfolio,
                correlation,
                stressed=True,
                wrong_way_sectors=(
                    wrong_way_sectors
                ),
                correlation_boost=(
                    correlation_boost
                ),
            )
        )

        pd_values = np.asarray(
            base_arrays[
                "pd"
            ],
            dtype=float,
        )

        exposure = np.asarray(
            base_arrays[
                "exposure"
            ],
            dtype=float,
        )

        recovery = np.asarray(
            base_arrays[
                "recovery"
            ],
            dtype=float,
        )

        rho = np.asarray(
            base_arrays[
                "rho"
            ],
            dtype=float,
        )

        normal_lgd_amount = np.asarray(
            base_arrays[
                "lgd_amount"
            ],
            dtype=float,
        )

        total_exposure = float(
            np.sum(
                exposure
            )
        )

        analytical_el = float(
            np.sum(
                pd_values
                * normal_lgd_amount
            )
        )

        gaussian_losses, _, _ = (
            _simulate_losses(
                arrays=base_arrays,
                correlation_matrix=(
                    matrix
                ),
                n_simulations=(
                    n_simulations
                ),
                seed=base_seed,
                copula="gaussian",
                t_df=t_df,
            )
        )

        student_losses, _, _ = (
            _simulate_losses(
                arrays=base_arrays,
                correlation_matrix=(
                    matrix
                ),
                n_simulations=(
                    n_simulations
                ),
                seed=(
                    base_seed
                    + 1000
                ),
                copula="student_t",
                t_df=t_df,
            )
        )

        wrong_way_losses, _, _ = (
            _simulate_losses(
                arrays=stressed_arrays,
                correlation_matrix=(
                    matrix
                ),
                n_simulations=(
                    n_simulations
                ),
                seed=base_seed,
                copula="gaussian",
                t_df=t_df,
            )
        )

        gaussian_risk = _var_cvar(
            gaussian_losses,
            confidence_levels,
        )

        student_risk = _var_cvar(
            student_losses,
            confidence_levels,
        )

        wrong_way_risk = _var_cvar(
            wrong_way_losses,
            confidence_levels,
        )

        gaussian_var_99 = (
            gaussian_risk[
                0.99
            ][
                0
            ]
        )

        gaussian_cvar_99 = (
            gaussian_risk[
                0.99
            ][
                1
            ]
        )

        _, contribution_sum, tail_count = (
            _simulate_losses(
                arrays=base_arrays,
                correlation_matrix=(
                    matrix
                ),
                n_simulations=(
                    n_simulations
                ),
                seed=base_seed,
                copula="gaussian",
                t_df=t_df,
                tail_threshold=(
                    gaussian_var_99
                ),
            )
        )

        if (
            contribution_sum
            is None
            or tail_count <= 0
        ):
            raise RuntimeError(
                "Could not compute 99% marginal CVaR contributions."
            )

        marginal = (
            contribution_sum
            / float(
                tail_count
            )
        )

        marginal_sum = float(
            np.sum(
                marginal
            )
        )

        exposure_weights = (
            exposure
            / total_exposure
        )

        marginal_frame = pd.DataFrame(
            {
                "obligor_id": (
                    portfolio[
                        "obligor_id"
                    ].astype(
                        int
                    )
                ),
                "issuer": (
                    portfolio[
                        "issuer"
                    ].astype(
                        str
                    )
                ),
                "sector": (
                    portfolio[
                        "factor_sector"
                    ].astype(
                        str
                    )
                ),
                "exposure_usd": (
                    exposure
                ),
                "exposure_weight": (
                    exposure_weights
                ),
                "marginal_cvar_99": (
                    marginal
                ),
                "pct_contribution": (
                    100.0
                    * marginal
                    / gaussian_cvar_99
                ),
            }
        ).sort_values(
            [
                "marginal_cvar_99",
                "obligor_id",
            ],
            ascending=[
                False,
                True,
            ],
            kind="stable",
        ).reset_index(
            drop=True
        )

        normal_benchmarks = (
            _vasicek_and_gordy(
                pd_values=pd_values,
                exposure=exposure,
                recovery=recovery,
                rho=rho,
                confidence_levels=(
                    confidence_levels
                ),
            )
        )

        var_cvar_rows = []

        for copula, risk in (
            (
                "gaussian",
                gaussian_risk,
            ),
            (
                "student_t",
                student_risk,
            ),
        ):
            for alpha in confidence_levels:
                var, cvar = risk[
                    alpha
                ]

                var_cvar_rows.append(
                    {
                        "copula": copula,
                        "confidence_level": (
                            alpha
                        ),
                        "VaR": _round_money(
                            var
                        ),
                        "CVaR": _round_money(
                            cvar
                        ),
                    }
                )

        wrong_way_rows = []

        for scenario, risk in (
            (
                "base",
                gaussian_risk,
            ),
            (
                "wrong_way",
                wrong_way_risk,
            ),
        ):
            for alpha in confidence_levels:
                var, cvar = risk[
                    alpha
                ]

                wrong_way_rows.append(
                    {
                        "scenario": (
                            scenario
                        ),
                        "confidence_level": (
                            alpha
                        ),
                        "VaR": _round_money(
                            var
                        ),
                        "CVaR": _round_money(
                            cvar
                        ),
                    }
                )

        loss_rows = []

        for copula, losses in (
            (
                "gaussian",
                gaussian_losses,
            ),
            (
                "student_t",
                student_losses,
            ),
        ):
            statistics = _loss_statistics(
                losses,
                analytical_el,
            )

            loss_rows.append(
                {
                    "copula": copula,
                    "mean_loss": _round_money(
                        statistics[
                            "mean_loss"
                        ]
                    ),
                    "std_loss": _round_money(
                        statistics[
                            "std_loss"
                        ]
                    ),
                    "skewness": round(
                        statistics[
                            "skewness"
                        ],
                        4,
                    ),
                    "kurtosis": round(
                        statistics[
                            "kurtosis"
                        ],
                        4,
                    ),
                    "max_loss": _round_money(
                        statistics[
                            "max_loss"
                        ]
                    ),
                    "pct_zero_loss": round(
                        statistics[
                            "pct_zero_loss"
                        ],
                        4,
                    ),
                    "expected_loss_analytical": (
                        _round_money(
                            analytical_el
                        )
                    ),
                }
            )

        gaussian_mean = float(
            np.mean(
                gaussian_losses
            )
        )

        analytical_rows = []

        granularity_rows = []

        for alpha in confidence_levels:
            vasicek = (
                normal_benchmarks[
                    alpha
                ][
                    "vasicek_var"
                ]
            )

            ga = (
                normal_benchmarks[
                    alpha
                ][
                    "granularity_adjustment"
                ]
            )

            gordy = (
                normal_benchmarks[
                    alpha
                ][
                    "gordy_var"
                ]
            )

            mc_var = (
                gaussian_risk[
                    alpha
                ][
                    0
                ]
            )

            analytical_rows.append(
                {
                    "confidence_level": (
                        alpha
                    ),
                    "mc_var_gaussian": (
                        _round_money(
                            mc_var
                        )
                    ),
                    "vasicek_var": (
                        _round_money(
                            vasicek
                        )
                    ),
                    "analytical_el": (
                        _round_money(
                            analytical_el
                        )
                    ),
                    "mc_mean_loss": (
                        _round_money(
                            gaussian_mean
                        )
                    ),
                    "el_ratio": round(
                        gaussian_mean
                        / analytical_el,
                        6,
                    ),
                }
            )

            granularity_rows.append(
                {
                    "confidence_level": (
                        alpha
                    ),
                    "vasicek_var": (
                        _round_money(
                            vasicek
                        )
                    ),
                    "granularity_adjustment": (
                        _round_money(
                            ga
                        )
                    ),
                    "gordy_var": (
                        _round_money(
                            gordy
                        )
                    ),
                    "mc_var_gaussian": (
                        _round_money(
                            mc_var
                        )
                    ),
                    "gordy_to_mc_ratio": round(
                        gordy
                        / mc_var,
                        6,
                    ),
                }
            )

        sector_weights_series = (
            pd.Series(
                exposure_weights,
                index=(
                    portfolio[
                        "factor_sector"
                    ].astype(
                        str
                    )
                ),
            )
            .groupby(
                level=0
            )
            .sum()
            .sort_index()
        )

        sector_hhi = float(
            np.sum(
                sector_weights_series.to_numpy(
                    dtype=float
                )
                ** 2
            )
        )

        obligor_hhi = float(
            np.sum(
                exposure_weights
                ** 2
            )
        )

        sector_rows = []

        marginal_by_id = (
            marginal_frame.set_index(
                "obligor_id"
            )[
                "marginal_cvar_99"
            ]
        )

        working = portfolio[
            [
                "obligor_id",
                "factor_sector",
                "exposure_usd",
            ]
        ].copy()

        working[
            "marginal_cvar_99"
        ] = (
            working[
                "obligor_id"
            ]
            .map(
                marginal_by_id
            )
            .astype(
                float
            )
        )

        for sector, group in working.groupby(
            "factor_sector",
            sort=True,
        ):
            sector_exposure = float(
                group[
                    "exposure_usd"
                ].sum()
            )

            sector_weight = (
                sector_exposure
                / total_exposure
            )

            sector_marginal = float(
                group[
                    "marginal_cvar_99"
                ].sum()
            )

            pct_total = (
                100.0
                * sector_marginal
                / gaussian_cvar_99
            )

            risk_weight_ratio = (
                (
                    pct_total
                    / 100.0
                )
                / sector_weight
                if sector_weight > 0.0
                else 0.0
            )

            sector_rows.append(
                {
                    "sector": str(
                        sector
                    ),
                    "n_obligors": int(
                        len(
                            group
                        )
                    ),
                    "total_exposure": (
                        _round_money(
                            sector_exposure
                        )
                    ),
                    "exposure_weight": round(
                        sector_weight,
                        6,
                    ),
                    "marginal_cvar_99": (
                        _round_money(
                            sector_marginal
                        )
                    ),
                    "pct_of_total_cvar": round(
                        pct_total,
                        4,
                    ),
                    "risk_weight_ratio": round(
                        risk_weight_ratio,
                        6,
                    ),
                }
            )

        top_exposure_indices = np.argsort(
            -exposure_weights
        )[
            :5
        ]

        top_5_exposures = [
            {
                "obligor_id": int(
                    portfolio.iloc[
                        index
                    ][
                        "obligor_id"
                    ]
                ),
                "issuer": str(
                    portfolio.iloc[
                        index
                    ][
                        "issuer"
                    ]
                ),
                "weight": float(
                    exposure_weights[
                        index
                    ]
                ),
            }
            for index in top_exposure_indices
        ]

        top_5_marginal = [
            {
                "obligor_id": int(
                    row[
                        "obligor_id"
                    ]
                ),
                "issuer": str(
                    row[
                        "issuer"
                    ]
                ),
                "marginal_cvar_99": float(
                    row[
                        "marginal_cvar_99"
                    ]
                ),
            }
            for _, row in marginal_frame.head(
                5
            ).iterrows()
        ]

        concentration = {
            "obligor_hhi": (
                obligor_hhi
            ),
            "sector_hhi": (
                sector_hhi
            ),
            "top_5_exposures": (
                top_5_exposures
            ),
            "top_5_marginal_cvar": (
                top_5_marginal
            ),
            "sector_weights": {
                str(
                    sector
                ): float(
                    weight
                )
                for sector, weight
                in sector_weights_series.items()
            },
        }

        student_var_99 = (
            student_risk[
                0.99
            ][
                0
            ]
        )

        student_cvar_99 = (
            student_risk[
                0.99
            ][
                1
            ]
        )

        wrong_var_99 = (
            wrong_way_risk[
                0.99
            ][
                0
            ]
        )

        wrong_cvar_99 = (
            wrong_way_risk[
                0.99
            ][
                1
            ]
        )

        vasicek_99 = (
            normal_benchmarks[
                0.99
            ][
                "vasicek_var"
            ]
        )

        gordy_99 = (
            normal_benchmarks[
                0.99
            ][
                "gordy_var"
            ]
        )

        ga_99 = (
            normal_benchmarks[
                0.99
            ][
                "granularity_adjustment"
            ]
        )

        mean_portfolio_pd = float(
            np.sum(
                exposure_weights
                * pd_values
            )
        )

        summary = {
            "n_obligors": int(
                len(
                    portfolio
                )
            ),
            "total_exposure_usd": (
                total_exposure
            ),
            "n_simulations": (
                n_simulations
            ),
            "gaussian_var_99": (
                _round_money(
                    gaussian_var_99
                )
            ),
            "gaussian_cvar_99": (
                _round_money(
                    gaussian_cvar_99
                )
            ),
            "student_t_var_99": (
                _round_money(
                    student_var_99
                )
            ),
            "student_t_cvar_99": (
                _round_money(
                    student_cvar_99
                )
            ),
            "wrong_way_var_99": (
                _round_money(
                    wrong_var_99
                )
            ),
            "wrong_way_cvar_99": (
                _round_money(
                    wrong_cvar_99
                )
            ),
            "var_ratio_t_vs_gaussian_99": (
                student_var_99
                / gaussian_var_99
            ),
            "wrong_way_cvar_increase_pct": (
                (
                    wrong_cvar_99
                    - gaussian_cvar_99
                )
                / gaussian_cvar_99
                * 100.0
            ),
            "obligor_hhi": (
                obligor_hhi
            ),
            "max_exposure_weight": float(
                np.max(
                    exposure_weights
                )
            ),
            "mean_portfolio_pd": (
                mean_portfolio_pd
            ),
            "analytical_el": (
                analytical_el
            ),
            "mc_el_ratio": (
                gaussian_mean
                / analytical_el
            ),
            "vasicek_var_99": (
                vasicek_99
            ),
            "vasicek_to_mc_ratio_99": (
                vasicek_99
                / gaussian_var_99
            ),
            "marginal_cvar_sum_check": (
                marginal_sum
                / gaussian_cvar_99
            ),
            "gordy_var_99": (
                gordy_99
            ),
            "granularity_adjustment_99": (
                ga_99
            ),
        }

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame(
            var_cvar_rows,
            columns=[
                "copula",
                "confidence_level",
                "VaR",
                "CVaR",
            ],
        ).to_csv(
            out_dir
            / "var_cvar_results.csv",
            index=False,
        )

        pd.DataFrame(
            wrong_way_rows,
            columns=[
                "scenario",
                "confidence_level",
                "VaR",
                "CVaR",
            ],
        ).to_csv(
            out_dir
            / "wrong_way_results.csv",
            index=False,
        )

        marginal_output = (
            marginal_frame.copy()
        )

        marginal_output[
            "exposure_usd"
        ] = marginal_output[
            "exposure_usd"
        ].round(
            2
        )

        marginal_output[
            "exposure_weight"
        ] = marginal_output[
            "exposure_weight"
        ].round(
            6
        )

        marginal_output[
            "marginal_cvar_99"
        ] = marginal_output[
            "marginal_cvar_99"
        ].round(
            2
        )

        marginal_output[
            "pct_contribution"
        ] = marginal_output[
            "pct_contribution"
        ].round(
            4
        )

        marginal_output.to_csv(
            out_dir
            / "marginal_contributions.csv",
            index=False,
        )

        pd.DataFrame(
            loss_rows
        ).to_csv(
            out_dir
            / "loss_statistics.csv",
            index=False,
        )

        (
            out_dir
            / "concentration_risk.json"
        ).write_text(
            json.dumps(
                concentration,
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

        pd.DataFrame(
            analytical_rows
        ).to_csv(
            out_dir
            / "analytical_benchmark.csv",
            index=False,
        )

        pd.DataFrame(
            sector_rows
        ).to_csv(
            out_dir
            / "sector_risk_decomposition.csv",
            index=False,
        )

        pd.DataFrame(
            granularity_rows
        ).to_csv(
            out_dir
            / "granularity_adjustment.csv",
            index=False,
        )
