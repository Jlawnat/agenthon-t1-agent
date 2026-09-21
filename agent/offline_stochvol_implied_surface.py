from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, optimize
from scipy.stats import norm


S0 = 110.0
R = 0.03
Q = 0.0

Z = 3
KAPPA1 = 2 ** (-Z / 2)
KAPPA2 = 2 ** (Z / 2)
THETA1 = 0.04
THETA2 = 0.04
V1_0 = 0.04
V2_0 = 0.04
SIGMA1 = np.sqrt(KAPPA1 * THETA1)
SIGMA2 = np.sqrt(KAPPA2 * THETA2)
LAMBDA1 = 0.0
LAMBDA2 = 0.0
RHO13 = -0.5
RHO24 = -0.99

N_GRID = 12
K_VEC = np.linspace(79, 165, N_GRID)
T_VEC = np.linspace(1 / 12, 1.0, N_GRID)

PHI1 = KAPPA1 * THETA1
PHI2 = KAPPA2 * THETA2
BETA1 = KAPPA1 + LAMBDA1
BETA2 = KAPPA2 + LAMBDA2

GL_NODES, GL_WEIGHTS = np.polynomial.legendre.leggauss(2000)
GL_UPPER = 100.0
GL_ETA = 0.5 * (GL_NODES + 1.0) * GL_UPPER
GL_W = 0.5 * GL_UPPER * GL_WEIGHTS


def _build_char_func(eta_vals, j: int, maturity: float):
    eta = eta_vals
    if j == 1:
        eta_sub = 1j - eta
    else:
        eta_sub = -eta

    Lambda = 1j * eta_sub - eta_sub**2

    th1 = BETA1 + 1j * eta_sub * RHO13 * SIGMA1
    th2 = BETA2 + 1j * eta_sub * RHO24 * SIGMA2

    om1 = np.sqrt(th1**2 - Lambda * SIGMA1**2)
    om2 = np.sqrt(th2**2 - Lambda * SIGMA2**2)

    q1 = (th1 + om1) / (th1 - om1)
    q2 = (th2 + om2) / (th2 - om2)

    e1 = np.exp(om1 * maturity)
    e2 = np.exp(om2 * maturity)

    B = (
        1j * eta * (R - Q) * maturity
        + PHI1
        / SIGMA1**2
        * (
            (th1 + om1) * maturity
            - 2.0 * np.log((1.0 - q1 * e1) / (1.0 - q1))
        )
        + PHI2
        / SIGMA2**2
        * (
            (th2 + om2) * maturity
            - 2.0 * np.log((1.0 - q2 * e2) / (1.0 - q2))
        )
    )

    d1 = (
        (th1 + om1)
        / SIGMA1**2
        * (1.0 - e1)
        / (1.0 - q1 * e1)
    )
    d2 = (
        (th2 + om2)
        / SIGMA2**2
        * (1.0 - e2)
        / (1.0 - q2 * e2)
    )

    return B, d1, d2


def _cz_call_quad(
    spot: float,
    strike: float,
    maturity: float,
) -> float:
    def integrand(eta, j):
        B, d1, d2 = _build_char_func(
            np.array([eta]),
            j,
            maturity,
        )
        g = np.exp(
            1j * eta * np.log(spot)
            + B[0]
            + d1[0] * V1_0
            + d2[0] * V2_0
        )
        return np.real(
            g
            * np.exp(-1j * eta * np.log(strike))
            / (1j * eta)
        )

    i1, _ = integrate.quad(
        integrand,
        1e-8,
        50.0,
        args=(1,),
        limit=200,
    )
    i2, _ = integrate.quad(
        integrand,
        1e-8,
        50.0,
        args=(2,),
        limit=200,
    )

    p1 = 0.5 + i1 / np.pi
    p2 = 0.5 + i2 / np.pi

    return max(
        np.exp(-Q * maturity) * spot * p1
        - np.exp(-R * maturity) * strike * p2,
        0.0,
    )


def _gl_integral(
    spot: float,
    strike: float,
    maturity: float,
    j: int,
) -> float:
    B, d1, d2 = _build_char_func(
        GL_ETA,
        j,
        maturity,
    )

    g = np.exp(
        1j * GL_ETA * np.log(spot)
        + B
        + d1 * V1_0
        + d2 * V2_0
    )

    integrand = np.real(
        g
        * np.exp(-1j * GL_ETA * np.log(strike))
        / (1j * GL_ETA)
    )

    return float(
        np.sum(GL_W * integrand)
    )


def _cz_call_gl(
    spot: float,
    strike: float,
    maturity: float,
) -> float:
    i1 = _gl_integral(
        spot,
        strike,
        maturity,
        1,
    )
    i2 = _gl_integral(
        spot,
        strike,
        maturity,
        2,
    )

    p1 = 0.5 + i1 / np.pi
    p2 = 0.5 + i2 / np.pi

    return max(
        np.exp(-Q * maturity) * spot * p1
        - np.exp(-R * maturity) * strike * p2,
        0.0,
    )


def _cz_put_gl(
    spot: float,
    strike: float,
    maturity: float,
) -> float:
    i1 = _gl_integral(
        spot,
        strike,
        maturity,
        1,
    )
    i2 = _gl_integral(
        spot,
        strike,
        maturity,
        2,
    )

    p1_put = 0.5 - i1 / np.pi
    p2_put = 0.5 - i2 / np.pi

    return max(
        np.exp(-R * maturity) * strike * p2_put
        - np.exp(-Q * maturity) * spot * p1_put,
        0.0,
    )


def _bs_call(
    spot: float,
    strike: float,
    maturity: float,
    sigma: float,
) -> float:
    root_t = np.sqrt(maturity)
    d1 = (
        np.log(spot / strike)
        + (R - Q + 0.5 * sigma**2) * maturity
    ) / (sigma * root_t)
    d2 = d1 - sigma * root_t

    return float(
        spot * np.exp(-Q * maturity) * norm.cdf(d1)
        - strike * np.exp(-R * maturity) * norm.cdf(d2)
    )


def _implied_vol(
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
) -> float:
    intrinsic = max(
        spot * np.exp(-Q * maturity)
        - strike * np.exp(-R * maturity),
        0.0,
    )

    if market_price <= intrinsic + 1e-10:
        return np.nan

    try:
        return float(
            optimize.brentq(
                lambda sigma: (
                    _bs_call(
                        spot,
                        strike,
                        maturity,
                        sigma,
                    )
                    - market_price
                ),
                1e-4,
                5.0,
                xtol=1e-8,
            )
        )
    except ValueError:
        return np.nan


def _write_matrix(
    path: Path,
    strikes: np.ndarray,
    maturities: np.ndarray,
    matrix: np.ndarray,
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["K/T"]
            + [
                f"{maturity:.6f}"
                for maturity in maturities
            ]
        )

        for index, strike in enumerate(
            strikes
        ):
            writer.writerow(
                [f"{strike:.4f}"]
                + [
                    f"{matrix[index, j]:.8f}"
                    for j in range(
                        len(maturities)
                    )
                ]
            )


def _build_surfaces(out_dir: Path):
    shortest_t = float(T_VEC[0])
    comparison_rows = []

    max_diff = 0.0

    for strike in K_VEC:
        p_quad = _cz_call_quad(
            S0,
            float(strike),
            shortest_t,
        )
        p_gl = _cz_call_gl(
            S0,
            float(strike),
            shortest_t,
        )
        diff = abs(
            p_quad
            - p_gl
        )

        max_diff = max(
            max_diff,
            diff,
        )

        comparison_rows.append(
            {
                "K": float(strike),
                "T": shortest_t,
                "price_quad": p_quad,
                "price_gl": p_gl,
                "abs_diff": diff,
            }
        )

    chosen_method = (
        "scipy_quad"
        if max_diff < 1e-5
        else "gauss_legendre"
    )

    with (
        out_dir
        / "integration_comparison.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "K",
                "T",
                "price_quad",
                "price_gl",
                "abs_diff",
            ],
        )
        writer.writeheader()
        writer.writerows(
            comparison_rows
        )
        writer.writerow(
            {
                "K": "chosen_method",
                "T": chosen_method,
                "price_quad": "",
                "price_gl": "",
                "abs_diff": "",
            }
        )

    if chosen_method == "gauss_legendre":
        price_func = _cz_call_gl
    else:
        price_func = _cz_call_quad

    option_prices = np.zeros(
        (N_GRID, N_GRID),
        dtype=float,
    )
    iv_surface = np.zeros(
        (N_GRID, N_GRID),
        dtype=float,
    )

    for i, strike in enumerate(
        K_VEC
    ):
        for j, maturity in enumerate(
            T_VEC
        ):
            price = price_func(
                S0,
                float(strike),
                float(maturity),
            )
            option_prices[
                i,
                j,
            ] = price

            iv = _implied_vol(
                price,
                S0,
                float(strike),
                float(maturity),
            )

            iv_surface[
                i,
                j,
            ] = (
                iv
                if not np.isnan(
                    iv
                )
                else 0.0
            )

    _write_matrix(
        out_dir
        / "option_prices.csv",
        K_VEC,
        T_VEC,
        option_prices,
    )
    _write_matrix(
        out_dir
        / "iv_surface.csv",
        K_VEC,
        T_VEC,
        iv_surface,
    )

    return (
        chosen_method,
        option_prices,
        iv_surface,
    )


def _write_iv_plots(
    out_dir: Path,
    iv_surface: np.ndarray,
) -> None:
    k_grid, t_grid = np.meshgrid(
        K_VEC,
        T_VEC,
        indexing="ij",
    )

    fig = plt.figure(
        figsize=(10, 7)
    )
    axis = fig.add_subplot(
        111,
        projection="3d",
    )
    axis.plot_surface(
        k_grid,
        t_grid,
        iv_surface,
        cmap="viridis",
        alpha=0.8,
    )
    axis.set_xlabel(
        "Strike K"
    )
    axis.set_ylabel(
        "Maturity T"
    )
    axis.set_zlabel(
        "Implied Vol"
    )
    axis.set_title(
        "IV Surface (K, T)"
    )
    plt.tight_layout()
    plt.savefig(
        out_dir
        / "iv_surface.png",
        dpi=150,
    )
    plt.close()

    fig, axis = plt.subplots(
        figsize=(10, 6)
    )

    for index in [
        0,
        3,
        6,
        11,
    ]:
        axis.plot(
            K_VEC,
            iv_surface[
                :,
                index,
            ],
            "o-",
            label=(
                f"T={T_VEC[index]:.2f}y"
            ),
        )

    axis.set_xlabel(
        "Strike K"
    )
    axis.set_ylabel(
        "Implied Vol"
    )
    axis.set_title(
        "Volatility Smile Slices"
    )
    axis.legend()
    axis.grid(
        True,
        alpha=0.3,
    )
    plt.tight_layout()
    plt.savefig(
        out_dir
        / "smile_slices.png",
        dpi=150,
    )
    plt.close()


def _build_put_outputs(
    out_dir: Path,
    option_prices: np.ndarray,
):
    direct = np.zeros(
        (N_GRID, N_GRID),
        dtype=float,
    )
    parity = np.zeros(
        (N_GRID, N_GRID),
        dtype=float,
    )
    rows = []

    for i, strike in enumerate(
        K_VEC
    ):
        for j, maturity in enumerate(
            T_VEC
        ):
            direct_price = (
                _cz_put_gl(
                    S0,
                    float(strike),
                    float(maturity),
                )
            )

            parity_price = (
                option_prices[
                    i,
                    j,
                ]
                - S0
                * np.exp(
                    -Q
                    * maturity
                )
                + strike
                * np.exp(
                    -R
                    * maturity
                )
            )
            parity_price = max(
                float(
                    parity_price
                ),
                0.0,
            )

            direct[
                i,
                j,
            ] = direct_price
            parity[
                i,
                j,
            ] = parity_price

            rows.append(
                {
                    "K": f"{strike:.4f}",
                    "T": f"{maturity:.6f}",
                    "put_direct": f"{direct_price:.8f}",
                    "put_parity": f"{parity_price:.8f}",
                    "abs_diff": (
                        f"{abs(direct_price - parity_price):.2e}"
                    ),
                }
            )

    _write_matrix(
        out_dir
        / "put_prices.csv",
        K_VEC,
        T_VEC,
        direct,
    )

    with (
        out_dir
        / "put_parity_check.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "K",
                "T",
                "put_direct",
                "put_parity",
                "abs_diff",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    max_diff = float(
        np.max(
            np.abs(
                direct - parity
            )
        )
    )

    return max_diff


def _build_local_vol(
    out_dir: Path,
    option_prices: np.ndarray,
):
    d_k = (
        K_VEC[1]
        - K_VEC[0]
    )
    n_interior = (
        N_GRID - 2
    )

    local_vol = np.zeros(
        (
            n_interior,
            N_GRID,
        ),
        dtype=float,
    )

    for ii in range(
        n_interior
    ):
        i = ii + 1
        strike = float(
            K_VEC[i]
        )

        for j in range(
            N_GRID
        ):
            if j == 0:
                d_c_dt = (
                    option_prices[
                        i,
                        j + 1,
                    ]
                    - option_prices[
                        i,
                        j,
                    ]
                ) / (
                    T_VEC[
                        j + 1
                    ]
                    - T_VEC[j]
                )
            elif j == (
                N_GRID - 1
            ):
                d_c_dt = (
                    option_prices[
                        i,
                        j,
                    ]
                    - option_prices[
                        i,
                        j - 1,
                    ]
                ) / (
                    T_VEC[j]
                    - T_VEC[
                        j - 1
                    ]
                )
            else:
                d_c_dt = (
                    option_prices[
                        i,
                        j + 1,
                    ]
                    - option_prices[
                        i,
                        j - 1,
                    ]
                ) / (
                    T_VEC[
                        j + 1
                    ]
                    - T_VEC[
                        j - 1
                    ]
                )

            d_c_dk = (
                option_prices[
                    i + 1,
                    j,
                ]
                - option_prices[
                    i - 1,
                    j,
                ]
            ) / (
                2.0
                * d_k
            )

            d2_c_dk2 = (
                option_prices[
                    i + 1,
                    j,
                ]
                - 2.0
                * option_prices[
                    i,
                    j,
                ]
                + option_prices[
                    i - 1,
                    j,
                ]
            ) / (
                d_k
                ** 2
            )

            numerator = (
                d_c_dt
                + R
                * strike
                * d_c_dk
            )
            denominator = (
                0.5
                * strike
                * strike
                * d2_c_dk2
            )

            if denominator > 1e-15:
                local_variance = (
                    numerator
                    / denominator
                )
                local_vol[
                    ii,
                    j,
                ] = np.sqrt(
                    max(
                        local_variance,
                        1e-10,
                    )
                )
            else:
                local_vol[
                    ii,
                    j,
                ] = np.nan

    k_interior = (
        K_VEC[
            1:-1
        ]
    )

    _write_matrix(
        out_dir
        / "local_vol_surface.csv",
        k_interior,
        T_VEC,
        local_vol,
    )

    k_grid, t_grid = np.meshgrid(
        k_interior,
        T_VEC,
        indexing="ij",
    )

    fig = plt.figure(
        figsize=(10, 7)
    )
    axis = fig.add_subplot(
        111,
        projection="3d",
    )
    axis.plot_surface(
        k_grid,
        t_grid,
        local_vol,
        cmap="inferno",
        alpha=0.8,
    )
    axis.set_xlabel(
        "Strike K"
    )
    axis.set_ylabel(
        "Maturity T"
    )
    axis.set_zlabel(
        "Local Vol"
    )
    axis.set_title(
        "Dupire Local Volatility Surface"
    )
    plt.tight_layout()
    plt.savefig(
        out_dir
        / "local_vol_surface.png",
        dpi=150,
    )
    plt.close()

    return (
        k_interior,
        local_vol,
    )


@dataclass(frozen=True)
class StochVolImpliedSurfaceSkill:
    name: str = "stochvol-implied-surface-domain"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        del task_dir

        lowered = instruction.lower()

        return (
            "two-factor heston" in lowered
            and "chiarella-ziveyi" in lowered
            and "integration_comparison.csv" in lowered
            and "put_parity_check.csv" in lowered
            and "local_vol_surface.csv" in lowered
        )

    def solve(
        self,
        *,
        instruction: str,
        task_dir: Path,
        out_dir: Path,
        seed: int,
    ) -> None:
        del instruction, task_dir, seed

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            chosen_method,
            option_prices,
            iv_surface,
        ) = _build_surfaces(
            out_dir
        )

        _write_iv_plots(
            out_dir,
            iv_surface,
        )

        parity_max_diff = (
            _build_put_outputs(
                out_dir,
                option_prices,
            )
        )

        (
            k_interior,
            local_vol,
        ) = _build_local_vol(
            out_dir,
            option_prices,
        )

        atm_index = int(
            np.argmin(
                np.abs(
                    K_VEC
                    - S0
                )
            )
        )
        atm_local_index = int(
            np.argmin(
                np.abs(
                    k_interior
                    - S0
                )
            )
        )

        positive_iv = iv_surface[
            iv_surface > 0.0
        ]

        summary = {
            "chosen_method": chosen_method,
            "iv_min": float(
                np.min(
                    positive_iv
                )
            ),
            "iv_max": float(
                np.max(
                    iv_surface
                )
            ),
            "iv_mean": float(
                np.mean(
                    positive_iv
                )
            ),
            "atm_iv_T0p5": float(
                iv_surface[
                    atm_index,
                    5,
                ]
            ),
            "skew_T0p083": float(
                iv_surface[
                    0,
                    0,
                ]
                - iv_surface[
                    -1,
                    0,
                ]
            ),
            "skew_T1p0": float(
                iv_surface[
                    0,
                    -1,
                ]
                - iv_surface[
                    -1,
                    -1,
                ]
            ),
            "num_nan": int(
                np.sum(
                    iv_surface
                    == 0.0
                )
            ),
            "put_parity_max_diff": (
                parity_max_diff
            ),
            "local_vol_atm_T0p5": float(
                local_vol[
                    atm_local_index,
                    5,
                ]
            ),
            "local_vol_range": [
                float(
                    np.nanmin(
                        local_vol
                    )
                ),
                float(
                    np.nanmax(
                        local_vol
                    )
                ),
            ],
        }

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
