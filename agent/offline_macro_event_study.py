from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd


_PANEL_COLUMNS = [
    "meeting_date",
    "event_date",
    "prior_count",
    "token_count",
    "hawkish_count",
    "dovish_count",
    "tone_score",
    "prior_tone_mean",
    "tone_surprise",
    "dgs2_event",
    "dgs10_event",
    "dgs2_change_0_2_bps",
    "curve_change_0_2_bps",
    "dgs2_change_m1_p1_bps",
    "tfidf_novelty",
    "pre_curve_level_bps",
    "pre_curve_z_63d",
    "pre_dgs2_momentum_5d_bps",
    "ar1_phi_252d",
    "expected_dgs2_change_0_2_bps",
    "abnormal_dgs2_change_0_2_bps",
]

_NUMERIC_PANEL_COLUMNS = [
    column
    for column in _PANEL_COLUMNS
    if column
    not in {
        "meeting_date",
        "event_date",
    }
]


def normalize_tokens(
    text: str,
) -> list[str]:
    normalized = re.sub(
        r"[^a-z]",
        " ",
        str(text).lower(),
    )

    return normalized.split()


def _find_data_dir(
    task_dir: Path,
) -> Path:
    required = {
        "fomc_statements.csv",
        "tone_lexicon.json",
        "treasury_yields.csv",
    }

    candidate_dirs = sorted(
        {
            path.parent
            for path
            in task_dir.rglob("*")
            if (
                path.is_file()
                and "checks"
                not in path.parts
            )
        },
        key=lambda value: str(
            value
        ),
    )

    for directory in candidate_dirs:
        names = {
            path.name
            for path
            in directory.iterdir()
            if path.is_file()
        }

        if required.issubset(
            names
        ):
            return directory

    raise RuntimeError(
        "Could not discover FOMC event-study input files."
    )


def _build_text_features(
    statements: pd.DataFrame,
    lexicon: dict[str, object],
) -> pd.DataFrame:
    data = (
        statements.copy()
        .sort_values(
            "meeting_date",
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    token_lists = [
        normalize_tokens(
            text
        )
        for text
        in data[
            "statement_text"
        ].astype(
            str
        )
    ]

    hawkish = {
        token
        for term
        in lexicon[
            "hawkish"
        ]
        for token
        in normalize_tokens(
            str(
                term
            )
        )
    }

    dovish = {
        token
        for term
        in lexicon[
            "dovish"
        ]
        for token
        in normalize_tokens(
            str(
                term
            )
        )
    }

    token_counts = np.asarray(
        [
            len(
                tokens
            )
            for tokens
            in token_lists
        ],
        dtype=int,
    )

    if (
        token_counts
        <= 0
    ).any():
        raise RuntimeError(
            "Every statement must contain at least one normalized token."
        )

    hawkish_counts = np.asarray(
        [
            sum(
                token
                in hawkish
                for token
                in tokens
            )
            for tokens
            in token_lists
        ],
        dtype=int,
    )

    dovish_counts = np.asarray(
        [
            sum(
                token
                in dovish
                for token
                in tokens
            )
            for tokens
            in token_lists
        ],
        dtype=int,
    )

    tone_scores = (
        1000.0
        * (
            hawkish_counts
            - dovish_counts
        )
        / token_counts
    )

    document_frequency: dict[
        str,
        int,
    ] = {}

    for tokens in token_lists:
        for token in set(
            tokens
        ):
            document_frequency[
                token
            ] = (
                document_frequency.get(
                    token,
                    0,
                )
                + 1
            )

    vocabulary = sorted(
        token
        for token, frequency
        in document_frequency.items()
        if frequency >= 2
    )

    index = {
        token: position
        for position, token
        in enumerate(
            vocabulary
        )
    }

    n_documents = len(
        token_lists
    )

    idf = np.asarray(
        [
            math.log(
                (
                    1.0
                    + n_documents
                )
                / (
                    1.0
                    + document_frequency[
                        token
                    ]
                )
            )
            + 1.0
            for token
            in vocabulary
        ],
        dtype=float,
    )

    vectors = np.zeros(
        (
            n_documents,
            len(
                vocabulary
            ),
        ),
        dtype=float,
    )

    for row_index, tokens in enumerate(
        token_lists
    ):
        counts: dict[
            str,
            int,
        ] = {}

        for token in tokens:
            if token in index:
                counts[
                    token
                ] = (
                    counts.get(
                        token,
                        0,
                    )
                    + 1
                )

        denominator = float(
            len(
                tokens
            )
        )

        for token, count in counts.items():
            position = index[
                token
            ]

            vectors[
                row_index,
                position
            ] = (
                count
                / denominator
                * idf[
                    position
                ]
            )

    novelties = np.zeros(
        n_documents,
        dtype=float,
    )

    for row_index in range(
        1,
        n_documents,
    ):
        start = max(
            0,
            row_index
            - 4,
        )

        reference = np.mean(
            vectors[
                start:row_index
            ],
            axis=0,
        )

        current = vectors[
            row_index
        ]

        denominator = float(
            np.linalg.norm(
                current
            )
            * np.linalg.norm(
                reference
            )
        )

        similarity = (
            float(
                np.dot(
                    current,
                    reference,
                )
                / denominator
            )
            if denominator > 0.0
            else 0.0
        )

        novelties[
            row_index
        ] = (
            1.0
            - similarity
        )

    prior_counts = np.zeros(
        n_documents,
        dtype=int,
    )

    prior_means = np.zeros(
        n_documents,
        dtype=float,
    )

    tone_surprises = np.zeros(
        n_documents,
        dtype=float,
    )

    for row_index in range(
        n_documents
    ):
        start = max(
            0,
            row_index
            - 4,
        )

        previous = tone_scores[
            start:row_index
        ]

        prior_counts[
            row_index
        ] = len(
            previous
        )

        prior_mean = (
            float(
                np.mean(
                    previous
                )
            )
            if len(
                previous
            )
            else 0.0
        )

        prior_means[
            row_index
        ] = prior_mean

        tone_surprises[
            row_index
        ] = (
            tone_scores[
                row_index
            ]
            - prior_mean
        )

    data[
        "prior_count"
    ] = prior_counts

    data[
        "token_count"
    ] = token_counts

    data[
        "hawkish_count"
    ] = hawkish_counts

    data[
        "dovish_count"
    ] = dovish_counts

    data[
        "tone_score"
    ] = tone_scores

    data[
        "prior_tone_mean"
    ] = prior_means

    data[
        "tone_surprise"
    ] = tone_surprises

    data[
        "tfidf_novelty"
    ] = novelties

    return data


def _prepare_yields(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    by_lower = {
        str(column).lower(): str(column)
        for column
        in frame.columns
    }

    required = {
        "date",
        "dgs2",
        "dgs10",
    }

    if not required.issubset(
        by_lower
    ):
        raise RuntimeError(
            "Treasury data requires date, dgs2 and dgs10."
        )

    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                frame[
                    by_lower[
                        "date"
                    ]
                ],
                errors="coerce",
            ),
            "dgs2": pd.to_numeric(
                frame[
                    by_lower[
                        "dgs2"
                    ]
                ],
                errors="coerce",
            ),
            "dgs10": pd.to_numeric(
                frame[
                    by_lower[
                        "dgs10"
                    ]
                ],
                errors="coerce",
            ),
        }
    )

    data = (
        data.dropna(
            subset=[
                "date",
                "dgs2",
                "dgs10",
            ]
        )
        .sort_values(
            "date",
            kind="stable",
        )
        .drop_duplicates(
            subset=[
                "date"
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        data
    ) < 10:
        raise RuntimeError(
            "Treasury data has too few valid observations."
        )

    return data


def _ols(
    y: np.ndarray,
    X: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    y = np.asarray(
        y,
        dtype=float,
    )

    X = np.asarray(
        X,
        dtype=float,
    )

    beta = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )[0]

    fitted = (
        X
        @ beta
    )

    residuals = (
        y
        - fitted
    )

    sse = float(
        residuals
        @ residuals
    )

    return (
        beta,
        fitted,
        residuals,
        sse,
    )


def _newey_west_covariance(
    X: np.ndarray,
    residuals: np.ndarray,
    *,
    lag: int,
) -> np.ndarray:
    X = np.asarray(
        X,
        dtype=float,
    )

    residuals = np.asarray(
        residuals,
        dtype=float,
    )

    n, p = X.shape

    S = np.zeros(
        (
            p,
            p,
        ),
        dtype=float,
    )

    for row_index in range(
        n
    ):
        x = X[
            row_index
        ][
            :,
            None
        ]

        S += (
            residuals[
                row_index
            ]
            ** 2
            * (
                x
                @ x.T
            )
        )

    for offset in range(
        1,
        lag + 1,
    ):
        weight = (
            1.0
            - offset
            / (
                lag
                + 1.0
            )
        )

        for row_index in range(
            offset,
            n,
        ):
            x_now = X[
                row_index
            ][
                :,
                None
            ]

            x_previous = X[
                row_index
                - offset
            ][
                :,
                None
            ]

            cross = (
                x_now
                @ x_previous.T
                + x_previous
                @ x_now.T
            )

            S += (
                weight
                * residuals[
                    row_index
                ]
                * residuals[
                    row_index
                    - offset
                ]
                * cross
            )

    xtx_inverse = np.linalg.inv(
        X.T
        @ X
    )

    return (
        xtx_inverse
        @ S
        @ xtx_inverse
    )


def _event_state_features(
    meeting_dates: pd.Series,
    yields: pd.DataFrame,
) -> list[
    dict[str, object]
]:
    dates = yields[
        "date"
    ].to_numpy(
        dtype="datetime64[ns]"
    )

    dgs2 = yields[
        "dgs2"
    ].to_numpy(
        dtype=float
    )

    dgs10 = yields[
        "dgs10"
    ].to_numpy(
        dtype=float
    )

    curve = (
        dgs10
        - dgs2
    ) * 100.0

    delta = np.full(
        len(
            yields
        ),
        np.nan,
        dtype=float,
    )

    delta[
        1:
    ] = np.diff(
        dgs2
    ) * 100.0

    rows = []

    for meeting_value in meeting_dates:
        meeting = np.datetime64(
            pd.Timestamp(
                meeting_value
            ),
            "ns",
        )

        event_index = int(
            np.searchsorted(
                dates,
                meeting,
                side="left",
            )
        )

        if (
            event_index < 6
            or event_index + 2
            >= len(
                yields
            )
        ):
            raise RuntimeError(
                "Treasury history does not cover the required event windows."
            )

        event_date = pd.Timestamp(
            dates[
                event_index
            ]
        )

        pre_curve = float(
            curve[
                event_index
                - 1
            ]
        )

        z_start = max(
            0,
            event_index
            - 63,
        )

        z_window = curve[
            z_start:event_index
        ]

        if len(
            z_window
        ) < 2:
            pre_curve_z = 0.0
        else:
            z_std = float(
                np.std(
                    z_window,
                    ddof=1,
                )
            )

            pre_curve_z = (
                float(
                    (
                        pre_curve
                        - float(
                            np.mean(
                                z_window
                            )
                        )
                    )
                    / z_std
                )
                if z_std > 0.0
                else 0.0
            )

        momentum = float(
            (
                dgs2[
                    event_index
                    - 1
                ]
                - dgs2[
                    event_index
                    - 6
                ]
            )
            * 100.0
        )

        ar_start = max(
            2,
            event_index
            - 252,
        )

        ar_indices = np.arange(
            ar_start,
            event_index,
            dtype=int,
        )

        if len(
            ar_indices
        ) < 2:
            raise RuntimeError(
                "Insufficient data to estimate the pre-event AR(1)."
            )

        ar_y = delta[
            ar_indices
        ]

        ar_x = delta[
            ar_indices
            - 1
        ]

        ar_design = np.column_stack(
            [
                np.ones(
                    len(
                        ar_indices
                    )
                ),
                ar_x,
            ]
        )

        ar_beta = np.linalg.lstsq(
            ar_design,
            ar_y,
            rcond=None,
        )[0]

        ar_intercept = float(
            ar_beta[
                0
            ]
        )

        ar_phi = float(
            ar_beta[
                1
            ]
        )

        event_delta = float(
            delta[
                event_index
            ]
        )

        forecast_1 = (
            ar_intercept
            + ar_phi
            * event_delta
        )

        forecast_2 = (
            ar_intercept
            + ar_phi
            * forecast_1
        )

        expected_change = float(
            forecast_1
            + forecast_2
        )

        actual_change = float(
            (
                dgs2[
                    event_index
                    + 2
                ]
                - dgs2[
                    event_index
                ]
            )
            * 100.0
        )

        curve_change = float(
            curve[
                event_index
                + 2
            ]
            - curve[
                event_index
            ]
        )

        m1_p1_change = float(
            (
                dgs2[
                    event_index
                    + 1
                ]
                - dgs2[
                    event_index
                    - 1
                ]
            )
            * 100.0
        )

        rows.append(
            {
                "event_date": (
                    event_date.strftime(
                        "%Y-%m-%d"
                    )
                ),
                "dgs2_event": float(
                    dgs2[
                        event_index
                    ]
                ),
                "dgs10_event": float(
                    dgs10[
                        event_index
                    ]
                ),
                "dgs2_change_0_2_bps": (
                    actual_change
                ),
                "curve_change_0_2_bps": (
                    curve_change
                ),
                "dgs2_change_m1_p1_bps": (
                    m1_p1_change
                ),
                "pre_curve_level_bps": (
                    pre_curve
                ),
                "pre_curve_z_63d": (
                    pre_curve_z
                ),
                "pre_dgs2_momentum_5d_bps": (
                    momentum
                ),
                "ar1_phi_252d": (
                    ar_phi
                ),
                "expected_dgs2_change_0_2_bps": (
                    expected_change
                ),
                "abnormal_dgs2_change_0_2_bps": float(
                    actual_change
                    - expected_change
                ),
            }
        )

    return rows


def _simple_regression(
    regression_data: pd.DataFrame,
) -> dict[str, float]:
    y = regression_data[
        "dgs2_change_0_2_bps"
    ].to_numpy(
        dtype=float
    )

    x = regression_data[
        "tone_surprise"
    ].to_numpy(
        dtype=float
    )

    X = np.column_stack(
        [
            np.ones(
                len(
                    regression_data
                )
            ),
            x,
        ]
    )

    beta, fitted, residuals, sse = _ols(
        y,
        X,
    )

    del fitted

    n = len(
        y
    )

    sigma2 = (
        sse
        / (
            n
            - 2
        )
    )

    covariance = (
        sigma2
        * np.linalg.inv(
            X.T
            @ X
        )
    )

    standard_error_beta = math.sqrt(
        float(
            covariance[
                1,
                1
            ]
        )
    )

    t_stat = float(
        beta[
            1
        ]
        / standard_error_beta
    )

    hac_covariance = (
        _newey_west_covariance(
            X,
            residuals,
            lag=2,
        )
    )

    hac_standard_error = (
        math.sqrt(
            float(
                hac_covariance[
                    1,
                    1
                ]
            )
        )
    )

    t_stat_hac = float(
        beta[
            1
        ]
        / hac_standard_error
    )

    if sse > 0.0:
        durbin_watson = float(
            np.sum(
                np.diff(
                    residuals
                )
                ** 2
            )
            / sse
        )
    else:
        durbin_watson = 0.0

    centered = (
        y
        - float(
            np.mean(
                y
            )
        )
    )

    sst = float(
        centered
        @ centered
    )

    r_squared = (
        float(
            1.0
            - sse
            / sst
        )
        if sst > 0.0
        else 1.0
    )

    return {
        "intercept": float(
            beta[
                0
            ]
        ),
        "beta_tone_surprise": float(
            beta[
                1
            ]
        ),
        "t_stat_beta": (
            t_stat
        ),
        "t_stat_beta_nw_lag2": (
            t_stat_hac
        ),
        "durbin_watson": (
            durbin_watson
        ),
        "r_squared": (
            r_squared
        ),
    }


def _state_design(
    regression_data: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    tone = regression_data[
        "tone_surprise"
    ].to_numpy(
        dtype=float
    )

    curve = regression_data[
        "pre_curve_z_63d"
    ].to_numpy(
        dtype=float
    )

    momentum = regression_data[
        "pre_dgs2_momentum_5d_bps"
    ].to_numpy(
        dtype=float
    )

    novelty = regression_data[
        "tfidf_novelty"
    ].to_numpy(
        dtype=float
    )

    X = np.column_stack(
        [
            np.ones(
                len(
                    regression_data
                )
            ),
            tone,
            curve,
            momentum,
            novelty,
            tone
            * curve,
            tone
            * novelty,
        ]
    )

    y = regression_data[
        "abnormal_dgs2_change_0_2_bps"
    ].to_numpy(
        dtype=float
    )

    return (
        X,
        y,
    )


def _wild_sign_p_value(
    X: np.ndarray,
    fitted: np.ndarray,
    residuals: np.ndarray,
    full_beta_tone: float,
) -> float:
    n = len(
        residuals
    )

    count = 0
    total = 0

    for tail_signs in itertools.product(
        (
            -1.0,
            1.0,
        ),
        repeat=(
            n
            - 1
        ),
    ):
        signs = np.asarray(
            (
                1.0,
                *tail_signs,
            ),
            dtype=float,
        )

        y_star = (
            fitted
            + signs
            * residuals
        )

        beta_star = np.linalg.lstsq(
            X,
            y_star,
            rcond=None,
        )[0]

        if (
            abs(
                float(
                    beta_star[
                        1
                    ]
                )
            )
            >= abs(
                full_beta_tone
            )
        ):
            count += 1

        total += 1

    return float(
        count
        / total
    )


def _state_regression_outputs(
    regression_data: pd.DataFrame,
) -> tuple[
    dict[str, float],
    pd.DataFrame,
    pd.DataFrame,
]:
    X, y = _state_design(
        regression_data
    )

    beta, fitted, residuals, sse = _ols(
        y,
        X,
    )

    n, p = X.shape

    hac_covariance = (
        _newey_west_covariance(
            X,
            residuals,
            lag=2,
        )
    )

    tone_hac_t = float(
        beta[
            1
        ]
        / math.sqrt(
            float(
                hac_covariance[
                    1,
                    1
                ]
            )
        )
    )

    tone_novelty_hac_t = float(
        beta[
            6
        ]
        / math.sqrt(
            float(
                hac_covariance[
                    6,
                    6
                ]
            )
        )
    )

    centered = (
        y
        - float(
            np.mean(
                y
            )
        )
    )

    sst = float(
        centered
        @ centered
    )

    r_squared = (
        float(
            1.0
            - sse
            / sst
        )
        if sst > 0.0
        else 1.0
    )

    wild_p = _wild_sign_p_value(
        X,
        fitted,
        residuals,
        float(
            beta[
                1
            ]
        ),
    )

    state_summary = {
        "intercept": float(
            beta[
                0
            ]
        ),
        "beta_tone_surprise": float(
            beta[
                1
            ]
        ),
        "beta_pre_curve_z_63d": float(
            beta[
                2
            ]
        ),
        "beta_pre_dgs2_momentum_5d_bps": float(
            beta[
                3
            ]
        ),
        "beta_tfidf_novelty": float(
            beta[
                4
            ]
        ),
        "beta_tone_curve_interaction": float(
            beta[
                5
            ]
        ),
        "beta_tone_novelty_interaction": float(
            beta[
                6
            ]
        ),
        "t_stat_tone_nw_lag2": (
            tone_hac_t
        ),
        "t_stat_tone_novelty_interaction_nw_lag2": (
            tone_novelty_hac_t
        ),
        "wild_sign_p_value_tone": (
            wild_p
        ),
        "r_squared": (
            r_squared
        ),
    }

    influence_rows = []

    full_beta_tone = float(
        beta[
            1
        ]
    )

    meeting_dates = regression_data[
        "meeting_date"
    ].astype(
        str
    ).tolist()

    for excluded_index in range(
        n
    ):
        keep = np.ones(
            n,
            dtype=bool,
        )

        keep[
            excluded_index
        ] = False

        leave_beta = np.linalg.lstsq(
            X[
                keep
            ],
            y[
                keep
            ],
            rcond=None,
        )[0]

        beta_tone = float(
            leave_beta[
                1
            ]
        )

        delta = (
            beta_tone
            - full_beta_tone
        )

        influence_rows.append(
            {
                "excluded_meeting_date": meeting_dates[
                    excluded_index
                ],
                "beta_tone_leave_one_out": (
                    beta_tone
                ),
                "delta_beta_tone": float(
                    delta
                ),
                "abs_delta_beta_tone": float(
                    abs(
                        delta
                    )
                ),
            }
        )

    influence = pd.DataFrame(
        influence_rows
    ).sort_values(
        [
            "abs_delta_beta_tone",
            "excluded_meeting_date",
        ],
        ascending=[
            False,
            True,
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    xtx_inverse = np.linalg.inv(
        X.T
        @ X
    )

    leverage = np.einsum(
        "ij,jk,ik->i",
        X,
        xtx_inverse,
        X,
    )

    mse = (
        sse
        / (
            n
            - p
        )
    )

    cooks = (
        residuals
        * residuals
        / (
            p
            * mse
        )
        * leverage
        / (
            (
                1.0
                - leverage
            )
            ** 2
        )
    )

    deleted_mse = (
        (
            sse
            - (
                residuals
                * residuals
                / (
                    1.0
                    - leverage
                )
            )
        )
        / (
            n
            - p
            - 1
        )
    )

    externally_studentized = (
        residuals
        / np.sqrt(
            deleted_mse
            * (
                1.0
                - leverage
            )
        )
    )

    diagnostics = pd.DataFrame(
        {
            "meeting_date": (
                meeting_dates
            ),
            "leverage": (
                leverage
            ),
            "externally_studentized_resid": (
                externally_studentized
            ),
            "cooks_distance": (
                cooks
            ),
        }
    ).sort_values(
        [
            "cooks_distance",
            "meeting_date",
        ],
        ascending=[
            False,
            True,
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    return (
        state_summary,
        influence,
        diagnostics,
    )


@dataclass(frozen=True)
class MacroTextEventStudySkill:
    name: str = "macro-text-event-study-domain"

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
            "fomc" in lowered
            and (
                "tone" in lowered
                or "lexicon" in lowered
            )
            and (
                "event study" in lowered
                or "event-study" in lowered
            )
            and (
                "treasury" in lowered
                or "yield" in lowered
            )
            and (
                "tf-idf" in lowered
                or "tfidf" in lowered
            )
            and (
                "newey-west" in lowered
                or "newey west" in lowered
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

        data_dir = _find_data_dir(
            task_dir
        )

        statements = pd.read_csv(
            data_dir
            / "fomc_statements.csv"
        )

        statements[
            "meeting_date"
        ] = pd.to_datetime(
            statements[
                "meeting_date"
            ],
            errors="raise",
        )

        lexicon = json.loads(
            (
                data_dir
                / "tone_lexicon.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        text_features = (
            _build_text_features(
                statements,
                lexicon,
            )
        )

        treasury = _prepare_yields(
            pd.read_csv(
                data_dir
                / "treasury_yields.csv"
            )
        )

        event_features = (
            _event_state_features(
                text_features[
                    "meeting_date"
                ],
                treasury,
            )
        )

        panel_rows = []

        for row_index, feature in enumerate(
            event_features
        ):
            source = text_features.iloc[
                row_index
            ]

            panel_rows.append(
                {
                    "meeting_date": source[
                        "meeting_date"
                    ].strftime(
                        "%Y-%m-%d"
                    ),
                    "event_date": feature[
                        "event_date"
                    ],
                    "prior_count": int(
                        source[
                            "prior_count"
                        ]
                    ),
                    "token_count": int(
                        source[
                            "token_count"
                        ]
                    ),
                    "hawkish_count": int(
                        source[
                            "hawkish_count"
                        ]
                    ),
                    "dovish_count": int(
                        source[
                            "dovish_count"
                        ]
                    ),
                    "tone_score": float(
                        source[
                            "tone_score"
                        ]
                    ),
                    "prior_tone_mean": float(
                        source[
                            "prior_tone_mean"
                        ]
                    ),
                    "tone_surprise": float(
                        source[
                            "tone_surprise"
                        ]
                    ),
                    "dgs2_event": feature[
                        "dgs2_event"
                    ],
                    "dgs10_event": feature[
                        "dgs10_event"
                    ],
                    "dgs2_change_0_2_bps": feature[
                        "dgs2_change_0_2_bps"
                    ],
                    "curve_change_0_2_bps": feature[
                        "curve_change_0_2_bps"
                    ],
                    "dgs2_change_m1_p1_bps": feature[
                        "dgs2_change_m1_p1_bps"
                    ],
                    "tfidf_novelty": float(
                        source[
                            "tfidf_novelty"
                        ]
                    ),
                    "pre_curve_level_bps": feature[
                        "pre_curve_level_bps"
                    ],
                    "pre_curve_z_63d": feature[
                        "pre_curve_z_63d"
                    ],
                    "pre_dgs2_momentum_5d_bps": feature[
                        "pre_dgs2_momentum_5d_bps"
                    ],
                    "ar1_phi_252d": feature[
                        "ar1_phi_252d"
                    ],
                    "expected_dgs2_change_0_2_bps": feature[
                        "expected_dgs2_change_0_2_bps"
                    ],
                    "abnormal_dgs2_change_0_2_bps": feature[
                        "abnormal_dgs2_change_0_2_bps"
                    ],
                }
            )

        panel = pd.DataFrame(
            panel_rows,
            columns=(
                _PANEL_COLUMNS
            ),
        ).sort_values(
            "meeting_date",
            kind="stable",
        ).reset_index(
            drop=True
        )

        panel[
            _NUMERIC_PANEL_COLUMNS
        ] = panel[
            _NUMERIC_PANEL_COLUMNS
        ].round(
            6
        )

        regression_data = (
            panel.loc[
                panel[
                    "prior_count"
                ]
                == 4
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        simple = (
            _simple_regression(
                regression_data
            )
        )

        (
            state_adjusted,
            influence,
            diagnostics,
        ) = (
            _state_regression_outputs(
                regression_data
            )
        )

        tone_max_index = int(
            panel[
                "tone_surprise"
            ].idxmax()
        )

        novelty_max_index = int(
            panel[
                "tfidf_novelty"
            ].idxmax()
        )

        influence_top = (
            influence.iloc[
                0
            ]
        )

        diagnostics_top = (
            diagnostics.iloc[
                0
            ]
        )

        state_p = 7
        state_n = len(
            regression_data
        )

        results = {
            "meta": {
                "num_events": int(
                    len(
                        panel
                    )
                ),
                "regression_events": int(
                    len(
                        regression_data
                    )
                ),
                "first_event_date": str(
                    panel[
                        "event_date"
                    ].iloc[
                        0
                    ]
                ),
                "last_event_date": str(
                    panel[
                        "event_date"
                    ].iloc[
                        -1
                    ]
                ),
            },
            "tone": {
                "mean_score": float(
                    panel[
                        "tone_score"
                    ].mean()
                ),
                "mean_surprise": float(
                    panel[
                        "tone_surprise"
                    ].mean()
                ),
                "max_surprise_date": str(
                    panel.loc[
                        tone_max_index,
                        "meeting_date",
                    ]
                ),
                "max_surprise": float(
                    panel.loc[
                        tone_max_index,
                        "tone_surprise",
                    ]
                ),
            },
            "novelty": {
                "mean_tfidf_novelty": float(
                    panel[
                        "tfidf_novelty"
                    ].mean()
                ),
                "max_tfidf_novelty_date": str(
                    panel.loc[
                        novelty_max_index,
                        "meeting_date",
                    ]
                ),
                "max_tfidf_novelty": float(
                    panel.loc[
                        novelty_max_index,
                        "tfidf_novelty",
                    ]
                ),
            },
            "regression": (
                simple
            ),
            "state_adjusted": (
                state_adjusted
            ),
            "influence": {
                "max_influence_date": str(
                    influence_top[
                        "excluded_meeting_date"
                    ]
                ),
                "max_abs_delta_beta_tone": float(
                    influence_top[
                        "abs_delta_beta_tone"
                    ]
                ),
                "mean_abs_delta_beta_tone": float(
                    influence[
                        "abs_delta_beta_tone"
                    ].mean()
                ),
            },
            "diagnostics": {
                "max_cooks_date": str(
                    diagnostics_top[
                        "meeting_date"
                    ]
                ),
                "max_cooks_distance": float(
                    diagnostics_top[
                        "cooks_distance"
                    ]
                ),
                "max_abs_studentized_resid": float(
                    diagnostics[
                        "externally_studentized_resid"
                    ].abs().max()
                ),
                "high_leverage_count": int(
                    (
                        diagnostics[
                            "leverage"
                        ]
                        > (
                            2.0
                            * state_p
                            / state_n
                        )
                    ).sum()
                ),
            },
        }

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        panel.to_csv(
            out_dir
            / "event_panel.csv",
            index=False,
            float_format="%.6f",
        )

        influence.to_csv(
            out_dir
            / "influence.csv",
            index=False,
        )

        diagnostics.to_csv(
            out_dir
            / "diagnostics.csv",
            index=False,
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
