from pathlib import Path

import numpy as np

from agent import offline_copula_sampling as copula
from agent import offline_cta_basel as cta


def test_copula_instruction_seed_overrides_harness_seed():
    instruction = """
    Generate the copula samples.

    Random seed: 42

    Use seed = 42 for reproducibility.
    """

    assert (
        copula._resolve_instruction_seed(
            instruction,
            fallback=2026,
        )
        == 42
    )


def test_cta_near_igarch_full_fit_uses_legacy_runtime(monkeypatch):
    class FakeFit:
        params = {
            "omega": 1.0,
            "alpha[1]": 0.10,
            "beta[1]": 0.90,
        }

        conditional_volatility = np.ones(
            100,
            dtype=float,
        )

        convergence_flag = 0

    class FakeModel:
        def fit(self, *, disp):
            assert disp == "off"
            return FakeFit()

    def fake_arch_model(*args, **kwargs):
        return FakeModel()

    sentinel = {
        "omega": 3.0e-8,
        "alpha": 0.07,
        "beta": 0.929,
        "persistence": 0.999,
        "long_run_variance": 3.0e-5,
        "cond_var": np.full(
            100,
            2.0e-5,
            dtype=float,
        ),
    }

    calls = []

    def fake_legacy(raw_returns):
        calls.append(
            np.asarray(
                raw_returns,
                dtype=float,
            )
        )
        return sentinel

    monkeypatch.setattr(
        cta,
        "arch_model",
        fake_arch_model,
    )

    monkeypatch.setattr(
        cta,
        "_fit_garch_legacy",
        fake_legacy,
    )

    returns = np.linspace(
        -0.01,
        0.01,
        100,
    )

    result = cta._fit_garch(
        returns,
        allow_legacy_fallback=True,
    )

    assert result is sentinel
    assert len(calls) == 1
    np.testing.assert_allclose(
        calls[0],
        returns,
    )


def test_cta_legacy_runtime_is_isolated_in_dockerfile():
    dockerfile = Path("Dockerfile").read_text(
        encoding="utf-8"
    )

    assert (
        "FROM python:3.11-slim AS cta_legacy"
        in dockerfile
    )

    assert (
        "FROM finance-bench-sandbox:latest"
        in dockerfile
    )

    assert (
        "COPY --from=cta_legacy "
        "/usr/local /opt/cta-py311"
        in dockerfile
    )

    assert (
        "CTA_LEGACY_PYTHON="
        "/opt/cta-py311/bin/python3.11"
        in dockerfile
    )
