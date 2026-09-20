from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist


def _discover(task_dir: Path) -> tuple[Path, Path, Path]:
    for directory in sorted(
        {p.parent for p in task_dir.rglob("*") if p.is_file() and "checks" not in p.parts},
        key=str,
    ):
        ret = directory / "returns.csv"
        char = directory / "characteristics.csv"
        params = directory / "params.json"
        if not (ret.is_file() and char.is_file() and params.is_file()):
            continue
        try:
            p = json.loads(params.read_text(encoding="utf-8"))
        except Exception:
            continue
        if {"n_factors", "characteristic_names", "als_max_iter", "als_tol", "ridge"}.issubset(p):
            return ret, char, params
    raise RuntimeError("Could not discover IPCA returns/characteristics inputs.")


def _solve(task_dir: Path, out_dir: Path) -> None:
    ret_path, char_path, params_path = _discover(task_dir)
    df_ret = pd.read_csv(ret_path)
    df_char = pd.read_csv(char_path)
    p = json.loads(params_path.read_text(encoding="utf-8"))

    K = int(p["n_factors"])
    names = list(p["characteristic_names"])
    L = len(names)
    max_iter = int(p["als_max_iter"])
    tol = float(p["als_tol"])
    ridge = float(p["ridge"])
    init_seed = int(p.get("init_seed", 42))
    annualization = float(p["annualization_factor"])

    dates = sorted(df_ret["date"].unique())
    tickers = sorted(df_ret["ticker"].unique())
    T = len(dates)
    N = len(tickers)

    ret_wide = (
        df_ret.pivot(index="date", columns="ticker", values="excess_return")
        .loc[dates, tickers]
        .to_numpy(dtype=float)
    )
    char_wide = np.zeros((T, N, L), dtype=float)
    for ci, name in enumerate(names):
        char_wide[:, :, ci] = (
            df_char.pivot(index="date", columns="ticker", values=name)
            .loc[dates, tickers]
            .to_numpy(dtype=float)
        )

    nan_count = int(np.isnan(ret_wide).sum())
    ret_wide = np.nan_to_num(ret_wide, nan=0.0)
    char_wide = np.nan_to_num(char_wide, nan=0.0)

    R = ret_wide[1:]
    Z = char_wide[:-1]
    T_eff = T - 1

    rng = np.random.default_rng(init_seed)
    F = rng.standard_normal((T_eff, K))
    for k in range(K):
        norm_k = np.linalg.norm(F[:, k])
        if norm_k > 0:
            F[:, k] /= norm_k

    prev_obj = np.inf
    converged = False
    Gamma = np.zeros((L, K))

    for it in range(max_iter):
        A = np.zeros((K * L, K * L))
        b = np.zeros(K * L)

        for t in range(T_eff):
            design = np.einsum("il,k->ilk", Z[t], F[t]).reshape(N, L * K)
            A += design.T @ design
            b += design.T @ R[t]

        A += ridge * np.eye(L * K)
        gamma_vec = np.linalg.solve(A, b)
        Gamma = gamma_vec.reshape(L, K)

        for t in range(T_eff):
            beta_t = Z[t] @ Gamma
            BtB = beta_t.T @ beta_t + ridge * np.eye(K)
            F[t] = np.linalg.solve(BtB, beta_t.T @ R[t])

        resid_sq = 0.0
        for t in range(T_eff):
            e = R[t] - Z[t] @ Gamma @ F[t]
            resid_sq += float(np.sum(e * e))

        if it > 0:
            rel = abs(prev_obj - resid_sq) / abs(prev_obj)
            if rel < tol:
                converged = True
                break
        prev_obj = resid_sq

    als_iterations = it + 1

    FtF = F.T @ F / T_eff
    eigvals, eigvecs = np.linalg.eigh(FtF)
    order = np.argsort(-eigvals)
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    F = F @ eigvecs
    Gamma = Gamma @ eigvecs

    for k in range(K):
        if F[:, k].mean() < 0:
            F[:, k] *= -1
            Gamma[:, k] *= -1

    total_ss = float(np.sum(R * R))
    resid_ss = 0.0
    cs_r2 = []
    for t in range(T_eff):
        pred = Z[t] @ Gamma @ F[t]
        e = R[t] - pred
        resid_ss += float(np.sum(e * e))
        denom = float(np.sum((R[t] - R[t].mean()) ** 2))
        if denom > 0:
            cs_r2.append(1.0 - float(np.sum(e * e)) / denom)

    total_r2 = 1.0 - resid_ss / total_ss
    mean_cs_r2 = float(np.mean(cs_r2))

    sharpes = []
    for k in range(K):
        mu = float(np.mean(F[:, k]))
        sd = float(np.std(F[:, k], ddof=1))
        sharpes.append(mu / sd * math.sqrt(annualization) if sd > 0 else 0.0)

    alphas = np.zeros(N)
    eps = np.zeros((T_eff, N))
    X = np.column_stack([np.ones(T_eff), F])
    for i in range(N):
        coef = np.linalg.lstsq(X, R[:, i], rcond=None)[0]
        alphas[i] = coef[0]
        eps[:, i] = R[:, i] - X @ coef

    Sigma = (eps.T @ eps) / (T_eff - K - 1)
    mu_F = F.mean(axis=0)
    Omega = np.cov(F.T, ddof=1)

    try:
        inv_s = np.linalg.inv(Sigma + 1e-10 * np.eye(N))
        inv_o = np.linalg.inv(Omega + 1e-10 * np.eye(K))
        coef_factor = (T_eff - N - K) / N
        sharpe_term = 1.0 + mu_F @ inv_o @ mu_F
        grs = float(coef_factor / sharpe_term * (alphas @ inv_s @ alphas))
        grs_p = float(1.0 - f_dist.cdf(grs, N, T_eff - N - K))
    except np.linalg.LinAlgError:
        grs = float("nan")
        grs_p = float("nan")

    svs = np.linalg.svd(Gamma, compute_uv=False)
    eigsum = float(eigvals.sum())
    values = {
        "nan_count": nan_count,
        "als_converged": bool(converged),
        "als_iterations": int(als_iterations),
        "gamma_frobenius_norm": float(np.linalg.norm(Gamma, "fro")),
        "gamma_largest_sv": float(svs[0]),
        "factor_var_ratio_1": float(eigvals[0] / eigsum) if eigsum > 0 else 0.0,
        "factor_var_ratio_2": float(eigvals[1] / eigsum) if eigsum > 0 and K >= 2 else 0.0,
        "factor_corr_12": float(np.corrcoef(F[:, 0], F[:, 1])[0, 1]) if K >= 2 else 0.0,
        "mean_alpha": float(np.mean(alphas)),
        "mean_abs_alpha": float(np.mean(np.abs(alphas))),
        "total_r_squared": float(total_r2),
        "factor_var_ratio_3": float(eigvals[2] / eigsum) if eigsum > 0 and K >= 3 else 0.0,
        "factor_corr_13": float(np.corrcoef(F[:, 0], F[:, 2])[0, 1]) if K >= 3 else 0.0,
        "factor_corr_23": float(np.corrcoef(F[:, 1], F[:, 2])[0, 1]) if K >= 3 else 0.0,
        "max_factor_return_abs": float(np.max(np.abs(F))),
        "std_factor1": float(np.std(F[:, 0], ddof=1)),
        "std_factor2": float(np.std(F[:, 1], ddof=1)) if K >= 2 else 0.0,
        "std_factor3": float(np.std(F[:, 2], ddof=1)) if K >= 3 else 0.0,
        "gamma_abs_mean": float(np.mean(np.abs(Gamma))),
        "mean_return_matrix": float(np.mean(R)),
        "max_abs_alpha": float(np.max(np.abs(alphas))),
    }

    results = {
        "total_r_squared": float(total_r2),
        "mean_cs_r_squared": mean_cs_r2,
        "grs_statistic": grs,
        "grs_p_value": grs_p,
        "factor1_sharpe_annual": float(sharpes[0]),
        "factor2_sharpe_annual": float(sharpes[1]),
        "factor3_sharpe_annual": float(sharpes[2]),
    }
    solution = {"intermediates": {k: {"value": v} for k, v in values.items()}}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (out_dir / "solution.json").write_text(json.dumps(solution, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class IpcaLatentFactorsSkill:
    name: str = "ipca-latent-factors-domain"

    def matches(self, *, instruction: str, task_dir: Path) -> bool:
        lowered = instruction.lower()
        return (
            "ipca" in lowered
            or "instrumented principal component" in lowered
        ) and (
            "latent factor" in lowered
            or "characteristic" in lowered
            or "alternating least squares" in lowered
        )

    def solve(self, *, instruction: str, task_dir: Path, out_dir: Path, seed: int) -> None:
        del instruction, seed
        _solve(task_dir, out_dir)
