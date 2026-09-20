from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats


def _find_evt_data_dir(
    task_dir: Path,
) -> tuple[Path, Path]:
    directories = sorted(
        {
            path.parent
            for path in task_dir.rglob("*")
            if path.is_file()
            and "checks" not in path.parts
        },
        key=lambda path: str(path),
    )

    for directory in directories:
        params_path = (
            directory
            / "params.json"
        )

        if not params_path.is_file():
            continue

        try:
            params = json.loads(
                params_path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            continue

        required_params = {
            "threshold_quantile",
            "confidence_levels",
            "backtest_window",
            "backtest_level",
            "garch_evt_threshold_quantile",
        }

        if not required_params.issubset(
            params
        ):
            continue

        for csv_path in sorted(
            directory.glob(
                "*.csv"
            )
        ):
            try:
                frame = pd.read_csv(
                    csv_path,
                    nrows=5,
                )
            except Exception:
                continue

            lower = {
                str(
                    column
                ).lower()
                for column
                in frame.columns
            }

            if "close" in lower:
                return (
                    directory,
                    csv_path,
                )

    raise RuntimeError(
        "Could not discover EVT tail-risk price data and parameters."
    )


def _lower_quantile(
    values: np.ndarray,
    q: float,
) -> float:
    return float(
        np.quantile(
            np.asarray(
                values,
                dtype=float,
            ),
            float(
                q
            ),
            method="lower",
        )
    )


def _pot_fit(
    losses: np.ndarray,
    threshold_quantile: float,
) -> dict[
    str,
    float | int | np.ndarray,
]:
    values = np.asarray(
        losses,
        dtype=float,
    )

    threshold = _lower_quantile(
        values,
        threshold_quantile,
    )

    exceedances = values[
        values
        > threshold
    ]

    excesses = (
        exceedances
        - threshold
    )

    if len(
        excesses
    ) < 3:
        raise RuntimeError(
            "Too few POT exceedances for a GPD fit."
        )

    shape, _location, scale = (
        stats.genpareto.fit(
            excesses,
            floc=0.0,
        )
    )

    if not (
        np.isfinite(
            shape
        )
        and np.isfinite(
            scale
        )
        and scale > 0.0
    ):
        raise RuntimeError(
            "Invalid GPD fit."
        )

    return {
        "threshold": float(
            threshold
        ),
        "num_exceedances": int(
            len(
                excesses
            )
        ),
        "mean_excess": float(
            np.mean(
                excesses
            )
        ),
        "shape": float(
            shape
        ),
        "scale": float(
            scale
        ),
        "excesses": (
            excesses
        ),
    }


def _evt_var_es(
    *,
    probability: float,
    threshold: float,
    shape: float,
    scale: float,
    n_total: int,
    n_exceedances: int,
) -> tuple[
    float,
    float,
]:
    p = float(
        probability
    )

    tail_probability = (
        float(
            n_total
        )
        / float(
            n_exceedances
        )
        * (
            1.0
            - p
        )
    )

    xi = float(
        shape
    )

    sigma = float(
        scale
    )

    u = float(
        threshold
    )

    if tail_probability <= 0.0:
        raise RuntimeError(
            "Invalid POT tail probability."
        )

    if abs(
        xi
    ) < 1e-10:
        var = (
            u
            - sigma
            * math.log(
                tail_probability
            )
        )
    else:
        var = (
            u
            + sigma
            / xi
            * (
                tail_probability
                ** (
                    -xi
                )
                - 1.0
            )
        )

    if xi >= 1.0:
        raise RuntimeError(
            "GPD expected shortfall is infinite for xi >= 1."
        )

    es = (
        var
        / (
            1.0
            - xi
        )
        + (
            sigma
            - xi
            * u
        )
        / (
            1.0
            - xi
        )
    )

    return (
        float(
            var
        ),
        float(
            es
        ),
    )


def _historical_var_es(
    losses: np.ndarray,
    probability: float,
) -> tuple[
    float,
    float,
]:
    var = _lower_quantile(
        losses,
        probability,
    )

    tail = np.asarray(
        losses,
        dtype=float,
    )

    tail = tail[
        tail
        >= var
    ]

    return (
        float(
            var
        ),
        float(
            np.mean(
                tail
            )
        ),
    )


def _normal_var_es(
    losses: np.ndarray,
    probability: float,
) -> tuple[
    float,
    float,
]:
    values = np.asarray(
        losses,
        dtype=float,
    )

    mean = float(
        np.mean(
            values
        )
    )

    std = float(
        np.std(
            values,
            ddof=1,
        )
    )

    z = float(
        stats.norm.ppf(
            probability
        )
    )

    var = (
        mean
        + std
        * z
    )

    es = (
        mean
        + std
        * stats.norm.pdf(
            z
        )
        / (
            1.0
            - float(
                probability
            )
        )
    )

    return (
        float(
            var
        ),
        float(
            es
        ),
    )


def _hill_xi(
    losses: np.ndarray,
) -> float:
    values = np.sort(
        np.asarray(
            losses,
            dtype=float,
        )
    )[
        ::-1
    ]

    n = len(
        values
    )

    k = int(
        math.floor(
            math.sqrt(
                n
            )
        )
    )

    if (
        k < 1
        or k >= n
        or values[
            k
        ]
        <= 0.0
    ):
        raise RuntimeError(
            "Hill estimator requires positive upper-tail observations."
        )

    top = values[
        :k
    ]

    if np.any(
        top <= 0.0
    ):
        raise RuntimeError(
            "Hill estimator upper-tail observations must be positive."
        )

    return float(
        np.mean(
            np.log(
                top
            )
        )
        - math.log(
            float(
                values[
                    k
                ]
            )
        )
    )


def _fit_garch(
    log_returns: np.ndarray,
):
    # Import lazily so the rest of the offline runtime remains importable in
    # development environments that do not install the optional arch package.
    from arch import arch_model

    percent_returns = (
        np.asarray(
            log_returns,
            dtype=float,
        )
        * 100.0
    )

    model = arch_model(
        percent_returns,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist="Normal",
        rescale=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )

        fitted = model.fit(
            update_freq=0,
            disp="off",
            show_warning=False,
        )

    return fitted


def _garch_evt_fit(
    log_returns: np.ndarray,
    threshold_quantile: float,
) -> dict[
    str,
    object,
]:
    returns = np.asarray(
        log_returns,
        dtype=float,
    )

    fitted = _fit_garch(
        returns
    )

    conditional_vol = (
        np.asarray(
            fitted.conditional_volatility,
            dtype=float,
        )
        / 100.0
    )

    valid = (
        np.isfinite(
            conditional_vol
        )
        & (
            conditional_vol
            > 0.0
        )
        & np.isfinite(
            returns
        )
    )

    standardised_losses = (
        -returns[
            valid
        ]
        / conditional_vol[
            valid
        ]
    )

    pot = _pot_fit(
        standardised_losses,
        threshold_quantile,
    )

    params = fitted.params

    return {
        "fitted": fitted,
        "conditional_vol": (
            conditional_vol
        ),
        "standardised_losses": (
            standardised_losses
        ),
        "pot": pot,
        "omega": float(
            params[
                "omega"
            ]
            / 1e4
        ),
        "alpha": float(
            params[
                "alpha[1]"
            ]
        ),
        "beta": float(
            params[
                "beta[1]"
            ]
        ),
        "last_vol": float(
            conditional_vol[
                np.flatnonzero(
                    np.isfinite(
                        conditional_vol
                    )
                )[
                    -1
                ]
            ]
        ),
    }


def _garch_evt_var_es(
    *,
    fitted_data: dict[
        str,
        object,
    ],
    probability: float,
    volatility_scale: float,
) -> tuple[
    float,
    float,
]:
    pot = fitted_data[
        "pot"
    ]

    standardised_losses = np.asarray(
        fitted_data[
            "standardised_losses"
        ],
        dtype=float,
    )

    var_z, es_z = _evt_var_es(
        probability=probability,
        threshold=float(
            pot[
                "threshold"
            ]
        ),
        shape=float(
            pot[
                "shape"
            ]
        ),
        scale=float(
            pot[
                "scale"
            ]
        ),
        n_total=int(
            len(
                standardised_losses
            )
        ),
        n_exceedances=int(
            pot[
                "num_exceedances"
            ]
        ),
    )

    return (
        float(
            volatility_scale
            * var_z
        ),
        float(
            volatility_scale
            * es_z
        ),
    )


def _one_step_garch_vol(
    fitted,
) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )

        forecast = fitted.forecast(
            horizon=1,
            reindex=False,
        )

    variance = float(
        np.asarray(
            forecast.variance,
            dtype=float,
        )[
            -1,
            0,
        ]
    )

    return float(
        math.sqrt(
            max(
                variance,
                0.0,
            )
        )
        / 100.0
    )


def _safe_xlogp(
    count: int,
    probability: float,
) -> float:
    if count == 0:
        return 0.0

    p = float(
        probability
    )

    if p <= 0.0:
        return float(
            "-inf"
        )

    return float(
        count
        * math.log(
            p
        )
    )


def _bernoulli_loglik(
    *,
    successes: int,
    total: int,
    probability: float,
) -> float:
    failures = int(
        total
        - successes
    )

    return float(
        _safe_xlogp(
            successes,
            probability,
        )
        + _safe_xlogp(
            failures,
            1.0
            - float(
                probability
            ),
        )
    )


def _coverage_tests(
    violations: np.ndarray,
    expected_violation_rate: float,
) -> dict[
    str,
    float | int,
]:
    v = np.asarray(
        violations,
        dtype=int,
    )

    total = int(
        len(
            v
        )
    )

    x = int(
        np.sum(
            v
        )
    )

    pi_hat = float(
        x
        / total
        if total
        else 0.0
    )

    ll_null = _bernoulli_loglik(
        successes=x,
        total=total,
        probability=float(
            expected_violation_rate
        ),
    )

    ll_alt = _bernoulli_loglik(
        successes=x,
        total=total,
        probability=(
            pi_hat
        ),
    )

    kupiec_lr = float(
        max(
            0.0,
            -2.0
            * (
                ll_null
                - ll_alt
            ),
        )
    )

    if total < 2:
        n00 = n01 = n10 = n11 = 0
    else:
        previous = v[
            :-1
        ]

        current = v[
            1:
        ]

        n00 = int(
            np.sum(
                (
                    previous
                    == 0
                )
                & (
                    current
                    == 0
                )
            )
        )

        n01 = int(
            np.sum(
                (
                    previous
                    == 0
                )
                & (
                    current
                    == 1
                )
            )
        )

        n10 = int(
            np.sum(
                (
                    previous
                    == 1
                )
                & (
                    current
                    == 0
                )
            )
        )

        n11 = int(
            np.sum(
                (
                    previous
                    == 1
                )
                & (
                    current
                    == 1
                )
            )
        )

    transitions = (
        n00
        + n01
        + n10
        + n11
    )

    pi = float(
        (
            n01
            + n11
        )
        / transitions
        if transitions
        else 0.0
    )

    denom0 = (
        n00
        + n01
    )

    denom1 = (
        n10
        + n11
    )

    pi01 = float(
        n01
        / denom0
        if denom0
        else 0.0
    )

    pi11 = float(
        n11
        / denom1
        if denom1
        else 0.0
    )

    ll_ind_null = float(
        _safe_xlogp(
            n01 + n11,
            pi,
        )
        + _safe_xlogp(
            n00 + n10,
            1.0 - pi,
        )
    )

    ll_ind_alt = float(
        _safe_xlogp(
            n01,
            pi01,
        )
        + _safe_xlogp(
            n00,
            1.0 - pi01,
        )
        + _safe_xlogp(
            n11,
            pi11,
        )
        + _safe_xlogp(
            n10,
            1.0 - pi11,
        )
    )

    christoffersen_lr = float(
        max(
            0.0,
            -2.0
            * (
                ll_ind_null
                - ll_ind_alt
            ),
        )
    )

    cc_lr = float(
        kupiec_lr
        + christoffersen_lr
    )

    return {
        "violations": x,
        "kupiec_lr": (
            kupiec_lr
        ),
        "kupiec_pvalue": float(
            stats.chi2.sf(
                kupiec_lr,
                1,
            )
        ),
        "christoffersen_ind_lr": (
            christoffersen_lr
        ),
        "christoffersen_ind_pvalue": float(
            stats.chi2.sf(
                christoffersen_lr,
                1,
            )
        ),
        "cc_lr": cc_lr,
        "cc_pvalue": float(
            stats.chi2.sf(
                cc_lr,
                2,
            )
        ),
    }


def _level_key(
    probability: float,
) -> str:
    text = (
        f"{float(probability):.10f}"
        .rstrip(
            "0"
        )
        .rstrip(
            "."
        )
    )

    if text.startswith(
        "0."
    ):
        return text[
            2:
        ]

    return text.replace(
        ".",
        "",
    )


def _rolling_backtest(
    *,
    log_returns: np.ndarray,
    losses: np.ndarray,
    window: int,
    probability: float,
    threshold_quantile: float,
    garch_threshold_quantile: float,
) -> dict[
    str,
    np.ndarray,
]:
    returns = np.asarray(
        log_returns,
        dtype=float,
    )

    loss_values = np.asarray(
        losses,
        dtype=float,
    )

    methods = (
        "historical",
        "normal",
        "evt",
        "garch_evt",
    )

    violations = {
        method: []
        for method
        in methods
    }

    for index in range(
        int(
            window
        ),
        len(
            loss_values
        ),
    ):
        window_losses = loss_values[
            index
            - window:
            index
        ]

        window_returns = returns[
            index
            - window:
            index
        ]

        actual = float(
            loss_values[
                index
            ]
        )

        historical_var = (
            _lower_quantile(
                window_losses,
                probability,
            )
        )

        normal_var, _ = (
            _normal_var_es(
                window_losses,
                probability,
            )
        )

        pot = _pot_fit(
            window_losses,
            threshold_quantile,
        )

        evt_var, _ = (
            _evt_var_es(
                probability=probability,
                threshold=float(
                    pot[
                        "threshold"
                    ]
                ),
                shape=float(
                    pot[
                        "shape"
                    ]
                ),
                scale=float(
                    pot[
                        "scale"
                    ]
                ),
                n_total=int(
                    len(
                        window_losses
                    )
                ),
                n_exceedances=int(
                    pot[
                        "num_exceedances"
                    ]
                ),
            )
        )

        garch_data = (
            _garch_evt_fit(
                window_returns,
                garch_threshold_quantile,
            )
        )

        next_vol = _one_step_garch_vol(
            garch_data[
                "fitted"
            ]
        )

        garch_var, _ = (
            _garch_evt_var_es(
                fitted_data=(
                    garch_data
                ),
                probability=(
                    probability
                ),
                volatility_scale=(
                    next_vol
                ),
            )
        )

        forecasts = {
            "historical": (
                historical_var
            ),
            "normal": (
                normal_var
            ),
            "evt": (
                evt_var
            ),
            "garch_evt": (
                garch_var
            ),
        }

        for method in methods:
            violations[
                method
            ].append(
                int(
                    actual
                    > forecasts[
                        method
                    ]
                )
            )

    return {
        method: np.asarray(
            values,
            dtype=int,
        )
        for method, values
        in violations.items()
    }


@dataclass(frozen=True)
class EvtPotRiskSkill:
    """POT-GPD and GARCH-EVT VaR/ES with rolling coverage diagnostics."""

    name: str = "evt-pot-risk-domain"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        del task_dir

        normalized = (
            instruction.lower()
            .replace(
                "-",
                " ",
            )
            .replace(
                "_",
                " ",
            )
        )

        return (
            (
                "evt" in normalized
                or "extreme value" in normalized
            )
            and (
                "pot" in normalized
                or "peaks over threshold" in normalized
                or "gpd" in normalized
            )
            and "var" in normalized
            and (
                "expected shortfall" in normalized
                or " es " in (
                    " "
                    + normalized
                    + " "
                )
            )
            and (
                "garch" in normalized
                or "backtest" in normalized
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

        (
            data_dir,
            price_path,
        ) = _find_evt_data_dir(
            task_dir
        )

        params = json.loads(
            (
                data_dir
                / "params.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        prices = pd.read_csv(
            price_path
        )

        by_lower = {
            str(
                column
            ).lower(): str(
                column
            )
            for column
            in prices.columns
        }

        close_column = by_lower.get(
            "close"
        )

        if close_column is None:
            raise RuntimeError(
                "EVT price data requires a close column."
            )

        close = (
            pd.to_numeric(
                prices[
                    close_column
                ],
                errors="coerce",
            )
            .dropna()
            .astype(
                float
            )
            .to_numpy()
        )

        if (
            len(
                close
            )
            < 3
            or np.any(
                close <= 0.0
            )
        ):
            raise RuntimeError(
                "EVT price series must contain positive closes."
            )

        log_returns = np.diff(
            np.log(
                close
            )
        )

        losses = (
            -log_returns
        )

        threshold_quantile = float(
            params[
                "threshold_quantile"
            ]
        )

        levels = [
            float(
                value
            )
            for value
            in params[
                "confidence_levels"
            ]
        ]

        backtest_window = int(
            params[
                "backtest_window"
            ]
        )

        backtest_level = float(
            params[
                "backtest_level"
            ]
        )

        garch_threshold_quantile = float(
            params[
                "garch_evt_threshold_quantile"
            ]
        )

        full_pot = _pot_fit(
            losses,
            threshold_quantile,
        )

        full_garch = (
            _garch_evt_fit(
                log_returns,
                garch_threshold_quantile,
            )
        )

        results: dict[
            str,
            float | int,
        ] = {}

        for level in levels:
            level_key = (
                _level_key(
                    level
                )
            )

            hist_var, hist_es = (
                _historical_var_es(
                    losses,
                    level,
                )
            )

            normal_var, normal_es = (
                _normal_var_es(
                    losses,
                    level,
                )
            )

            evt_var, evt_es = (
                _evt_var_es(
                    probability=level,
                    threshold=float(
                        full_pot[
                            "threshold"
                        ]
                    ),
                    shape=float(
                        full_pot[
                            "shape"
                        ]
                    ),
                    scale=float(
                        full_pot[
                            "scale"
                        ]
                    ),
                    n_total=int(
                        len(
                            losses
                        )
                    ),
                    n_exceedances=int(
                        full_pot[
                            "num_exceedances"
                        ]
                    ),
                )
            )

            garch_var, garch_es = (
                _garch_evt_var_es(
                    fitted_data=(
                        full_garch
                    ),
                    probability=level,
                    volatility_scale=float(
                        full_garch[
                            "last_vol"
                        ]
                    ),
                )
            )

            for method, var, es in (
                (
                    "historical",
                    hist_var,
                    hist_es,
                ),
                (
                    "normal",
                    normal_var,
                    normal_es,
                ),
                (
                    "evt",
                    evt_var,
                    evt_es,
                ),
                (
                    "garch_evt",
                    garch_var,
                    garch_es,
                ),
            ):
                results[
                    f"var_{level_key}_{method}"
                ] = float(
                    var
                )

                results[
                    f"es_{level_key}_{method}"
                ] = float(
                    es
                )

        rolling = _rolling_backtest(
            log_returns=(
                log_returns
            ),
            losses=losses,
            window=backtest_window,
            probability=backtest_level,
            threshold_quantile=(
                threshold_quantile
            ),
            garch_threshold_quantile=(
                garch_threshold_quantile
            ),
        )

        results[
            "backtest_num_days"
        ] = int(
            len(
                losses
            )
            - backtest_window
        )

        expected_violation_rate = (
            1.0
            - backtest_level
        )

        for method, indicators in (
            rolling.items()
        ):
            tests = _coverage_tests(
                indicators,
                expected_violation_rate,
            )

            for name, value in (
                tests.items()
            ):
                results[
                    f"backtest_{method}_{name}"
                ] = value

        solution = {
            "intermediates": {
                "num_returns": {
                    "value": int(
                        len(
                            log_returns
                        )
                    )
                },
                "mean_log_return": {
                    "value": float(
                        np.mean(
                            log_returns
                        )
                    )
                },
                "std_log_return": {
                    "value": float(
                        np.std(
                            log_returns,
                            ddof=1,
                        )
                    )
                },
                "threshold_value": {
                    "value": float(
                        full_pot[
                            "threshold"
                        ]
                    )
                },
                "num_exceedances": {
                    "value": int(
                        full_pot[
                            "num_exceedances"
                        ]
                    )
                },
                "mean_excess": {
                    "value": float(
                        full_pot[
                            "mean_excess"
                        ]
                    )
                },
                "gpd_shape_xi": {
                    "value": float(
                        full_pot[
                            "shape"
                        ]
                    )
                },
                "gpd_scale_sigma": {
                    "value": float(
                        full_pot[
                            "scale"
                        ]
                    )
                },
                "hill_xi": {
                    "value": float(
                        _hill_xi(
                            losses
                        )
                    )
                },
                "garch_omega": {
                    "value": float(
                        full_garch[
                            "omega"
                        ]
                    )
                },
                "garch_alpha": {
                    "value": float(
                        full_garch[
                            "alpha"
                        ]
                    )
                },
                "garch_beta": {
                    "value": float(
                        full_garch[
                            "beta"
                        ]
                    )
                },
                "garch_persistence": {
                    "value": float(
                        full_garch[
                            "alpha"
                        ]
                        + full_garch[
                            "beta"
                        ]
                    )
                },
                "garch_last_vol": {
                    "value": float(
                        full_garch[
                            "last_vol"
                        ]
                    )
                },
            }
        }

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            out_dir
            / "results.json"
        ).write_text(
            json.dumps(
                results,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        (
            out_dir
            / "solution.json"
        ).write_text(
            json.dumps(
                solution,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
