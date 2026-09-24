"""Isolated historical numerical GARCH fitter for CTA fallback.

This module is executed by the bundled Python 3.11 runtime, not imported
into the main Agenthon process.
"""

from __future__ import annotations

import json
import sys
import warnings

import numpy as np
from arch import arch_model


def main() -> None:
    payload = json.load(sys.stdin)

    x = np.asarray(
        payload["returns"],
        dtype=float,
    ).ravel()

    x = x[np.isfinite(x)]

    if len(x) < 50:
        raise RuntimeError("Too few returns for legacy GARCH fitting.")

    # Historical QuantitativeFinance-Bench reference convention:
    # fit percentage-scale returns and then convert variance quantities
    # back to native decimal-return squared units.
    x_pct = x * 100.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = arch_model(
            x_pct,
            vol="Garch",
            p=1,
            q=1,
            mean="Zero",
            dist="normal",
        )

        fitted = model.fit(
            disp="off",
        )

    omega = float(fitted.params["omega"]) / 10000.0
    alpha = float(fitted.params["alpha[1]"])
    beta = float(fitted.params["beta[1]"])
    persistence = alpha + beta

    if persistence < 1.0:
        long_run_variance = omega / (1.0 - persistence)
    else:
        long_run_variance = float(np.var(x, ddof=1))

    conditional_variance = (
        np.asarray(
            fitted.conditional_volatility,
            dtype=float,
        )
        ** 2
        / 10000.0
    )

    result = {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": persistence,
        "long_run_variance": float(long_run_variance),
        "cond_var": conditional_variance.tolist(),
        "convergence_flag": int(
            getattr(
                fitted,
                "convergence_flag",
                0,
            )
        ),
    }

    json.dump(
        result,
        sys.stdout,
        separators=(",", ":"),
    )


if __name__ == "__main__":
    main()
