from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def _find_named(task_dir: Path, name: str) -> Path:
    preferred = task_dir / "environment" / "data" / name
    if preferred.is_file():
        return preferred
    matches = [p for p in task_dir.rglob(name) if "checks" not in p.parts]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name!r}, found {len(matches)}.")
    return matches[0]


@dataclass(frozen=True)
class BlackLittermanRegimeHmmSkill:
    name: str = "black-litterman-regime-hmm"

    def matches(self, *, instruction: str, task_dir: Path) -> bool:
        del task_dir
        lowered = instruction.lower()
        required = (
            "black-litterman",
            "hidden markov",
            "baum-welch",
            "regime_prob_bull",
            "bl_weight_asset1",
            "cov_bull_diag_asset1",
        )
        return all(token in lowered for token in required)

    def solve(
        self,
        *,
        instruction: str,
        task_dir: Path,
        out_dir: Path,
        seed: int,
    ) -> None:
        del instruction, seed

        df = pd.read_csv(_find_named(task_dir, "returns.csv"))
        params = json.loads(
            _find_named(task_dir, "params.json").read_text(encoding="utf-8")
        )

        asset_cols = [c for c in df.columns if c != "date"]
        returns = df[asset_cols].to_numpy(dtype=float)
        n_obs, n_assets = returns.shape

        w_mkt = np.asarray(params["market_cap_weights"], dtype=float)
        if len(w_mkt) != n_assets:
            raise RuntimeError("Market-cap weight length does not match assets.")

        port_rets = returns @ w_mkt

        # Exact benchmark Baum-Welch convention.
        midpoint = n_obs // 2
        mu_est = np.array(
            [
                np.mean(port_rets[:midpoint]),
                np.mean(port_rets[midpoint:]),
            ],
            dtype=float,
        )
        sigma_est = np.array(
            [
                np.std(port_rets[:midpoint], ddof=1),
                np.std(port_rets[midpoint:], ddof=1),
            ],
            dtype=float,
        )
        pi_est = np.asarray(params["init_pi"], dtype=float)
        transition = np.asarray(params["init_trans_matrix"], dtype=float)
        n_states = 2

        gamma = None
        em_iter = -1

        for em_iter in range(int(params["hmm_max_iter"])):
            log_lik = np.zeros((n_obs, n_states), dtype=float)
            for k in range(n_states):
                log_lik[:, k] = norm.logpdf(
                    port_rets,
                    mu_est[k],
                    sigma_est[k],
                )

            alpha = np.zeros((n_obs, n_states), dtype=float)
            alpha[0] = np.log(pi_est + 1e-300) + log_lik[0]

            for t in range(1, n_obs):
                for j in range(n_states):
                    alpha[t, j] = np.logaddexp(
                        alpha[t - 1, 0]
                        + np.log(transition[0, j] + 1e-300),
                        alpha[t - 1, 1]
                        + np.log(transition[1, j] + 1e-300),
                    ) + log_lik[t, j]

            beta = np.zeros((n_obs, n_states), dtype=float)
            for t in range(n_obs - 2, -1, -1):
                for i in range(n_states):
                    beta[t, i] = np.logaddexp(
                        np.log(transition[i, 0] + 1e-300)
                        + log_lik[t + 1, 0]
                        + beta[t + 1, 0],
                        np.log(transition[i, 1] + 1e-300)
                        + log_lik[t + 1, 1]
                        + beta[t + 1, 1],
                    )

            log_gamma = alpha + beta
            log_gamma -= np.logaddexp(
                log_gamma[:, 0],
                log_gamma[:, 1],
            )[:, None]
            gamma = np.exp(log_gamma)

            mu_old = mu_est.copy()

            for k in range(n_states):
                weight = gamma[:, k]
                denom = np.sum(weight)
                mu_est[k] = np.sum(weight * port_rets) / denom
                variance = (
                    np.sum(weight * (port_rets - mu_est[k]) ** 2)
                    / denom
                )
                sigma_est[k] = math.sqrt(max(float(variance), 1e-16))

            pi_est = gamma[0].copy()

            new_transition = np.zeros_like(transition)
            log_p = np.logaddexp(alpha[-1, 0], alpha[-1, 1])

            for i in range(n_states):
                for j in range(n_states):
                    xi_sum = 0.0
                    for t in range(n_obs - 1):
                        xi_sum += np.exp(
                            alpha[t, i]
                            + np.log(transition[i, j] + 1e-300)
                            + log_lik[t + 1, j]
                            + beta[t + 1, j]
                            - log_p
                        )
                    new_transition[i, j] = xi_sum

                row_sum = new_transition[i].sum()
                if row_sum > 0.0:
                    new_transition[i] /= row_sum

            transition = new_transition

            if (
                np.max(np.abs(mu_est - mu_old))
                < float(params["hmm_tol"])
            ):
                break

        if gamma is None:
            raise RuntimeError("HMM failed to produce posterior probabilities.")

        hmm_iterations = em_iter + 1
        bull_idx = 0 if mu_est[0] > mu_est[1] else 1
        bear_idx = 1 - bull_idx

        regime_prob_bull = float(gamma[-1, bull_idx])
        current_regime = "bull" if regime_prob_bull > 0.5 else "bear"

        bull_mask = gamma[:, bull_idx] > 0.5
        bear_mask = ~bull_mask

        unconditional_cov = np.cov(returns.T, ddof=1)

        if int(np.sum(bull_mask)) > 10:
            cov_bull = np.cov(returns[bull_mask].T, ddof=1)
        else:
            cov_bull = unconditional_cov

        if int(np.sum(bear_mask)) > 10:
            cov_bear = np.cov(returns[bear_mask].T, ddof=1)
        else:
            cov_bear = unconditional_cov

        cov_current = cov_bull if current_regime == "bull" else cov_bear

        # Exact benchmark Black-Litterman convention.
        delta = float(params["risk_aversion"])
        tau = float(params["tau"])
        implied_daily = delta * cov_current @ w_mkt

        views = params["views"]
        n_views = len(views)
        P = np.zeros((n_views, n_assets), dtype=float)
        Q = np.zeros(n_views, dtype=float)
        omega_diag = np.zeros(n_views, dtype=float)

        for vi, view in enumerate(views):
            P[vi, int(view["asset"])] = 1.0
            Q[vi] = float(view["return"]) / 252.0
            omega_diag[vi] = float(view["confidence"]) ** 2 / 252.0

        Omega = np.diag(omega_diag)

        tau_sigma = tau * cov_current
        inv_tau_sigma = np.linalg.inv(tau_sigma)
        inv_omega = np.linalg.inv(Omega)

        bl_cov = np.linalg.inv(
            inv_tau_sigma
            + P.T @ inv_omega @ P
        )
        bl_mu = bl_cov @ (
            inv_tau_sigma @ implied_daily
            + P.T @ inv_omega @ Q
        )

        bl_weights = np.linalg.solve(
            delta * cov_current,
            bl_mu,
        )
        gross = float(np.sum(np.abs(bl_weights)))
        if gross <= 0.0:
            raise RuntimeError("Black-Litterman gross exposure is zero.")
        bl_weights = bl_weights / gross

        portfolio_returns = returns @ bl_weights
        annualization = float(params["annualization_factor"])

        annualized_return = float(
            np.mean(portfolio_returns) * annualization
        )
        annualized_volatility = float(
            np.std(portfolio_returns, ddof=1)
            * math.sqrt(annualization)
        )
        sharpe_ratio = float(
            annualized_return / annualized_volatility
            if annualized_volatility > 0.0
            else 0.0
        )

        cumulative = np.cumprod(1.0 + portfolio_returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        max_drawdown = float(np.max(drawdown))

        results = {
            "current_regime": current_regime,
            "regime_prob_bull": regime_prob_bull,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
        }

        solution = {
            "intermediates": {
                "hmm_iterations": {"value": int(hmm_iterations)},
                "hmm_mu_bull": {"value": float(mu_est[bull_idx])},
                "hmm_sigma_bear": {"value": float(sigma_est[bear_idx])},
                "regime_prob_bull": {"value": regime_prob_bull},
                "implied_return_asset1": {
                    "value": float(implied_daily[0] * 252.0)
                },
                "bl_weight_asset1": {"value": float(bl_weights[0])},
                "n_bull_days": {"value": int(np.sum(bull_mask))},
                "cov_bull_diag_asset1": {
                    "value": float(cov_bull[0, 0])
                },
                "annualized_volatility": {
                    "value": annualized_volatility
                },
                "max_drawdown": {"value": max_drawdown},
            }
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(
            json.dumps(results, indent=2) + "\n",
            encoding="utf-8",
        )
        (out_dir / "solution.json").write_text(
            json.dumps(solution, indent=2) + "\n",
            encoding="utf-8",
        )
