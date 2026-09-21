from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from agent.candidate_runner import run_candidate
from agent.candidate_workspace import CandidateWorkspace
from agent.model_client import ModelClient
from agent.qf_primitives import (
    PRIMITIVE_API_CATALOG,
    black_scholes_price,
    fit_ou_euler,
    largest_remainder_allocate,
)
from agent.runtime_adapters import (
    RuntimePlan,
    _StrategyEnvelope,
    _strategy_for_candidate,
)
from agent.task_snapshot import TaskSnapshot


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, size: int = -1) -> bytes:
        return self.body if size < 0 else self.body[:size]


def _task(root: Path) -> Path:
    task = root / "task"
    (task / "environment" / "data").mkdir(parents=True)
    (task / "instruction.md").write_text("# generic task\n", encoding="utf-8")
    return task


def test_curated_primitive_formulas_are_operational() -> None:
    price = black_scholes_price(
        100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call"
    )
    assert 10.44 < price < 10.46

    allocation = largest_remainder_allocate(
        10, [0.34, 0.33, 0.33]
    )
    assert allocation.tolist() == [4, 3, 3]

    series = np.array([1.0, 0.8, 0.7, 0.55, 0.48, 0.40])
    fit = fit_ou_euler(series, 1.0)
    assert "kappa" in fit
    assert "sigma" in fit
    assert len(PRIMITIVE_API_CATALOG) >= 10


def test_candidate_can_import_curated_primitives_but_not_agent_package() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        task = _task(root)
        workspace = CandidateWorkspace.create(
            base_dir=root / "work",
            candidate_id=1,
            task_dir=task,
        )
        solver = workspace.source_dir / "solver.py"
        solver.write_text(
            """
import json
from qf_primitives import black_scholes_price, largest_remainder_allocate

try:
    import agent.model_client
except ModuleNotFoundError:
    agent_visible = False
else:
    agent_visible = True

payload = {
    "price": black_scholes_price(
        100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call"
    ),
    "allocation": largest_remainder_allocate(
        7, [0.5, 0.3, 0.2]
    ).tolist(),
    "agent_visible": agent_visible,
}
print(json.dumps(payload))
""".strip() + "\n",
            encoding="utf-8",
        )

        result = run_candidate(
            workspace,
            solver,
            timeout_seconds=5.0,
        )
        assert result.return_code == 0
        payload = json.loads(result.stdout)
        assert 10.44 < payload["price"] < 10.46
        assert sum(payload["allocation"]) == 7
        assert payload["agent_visible"] is False


def test_candidate_network_is_still_blocked_with_primitive_library() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        task = _task(root)
        workspace = CandidateWorkspace.create(
            base_dir=root / "work",
            candidate_id=2,
            task_dir=task,
        )
        solver = workspace.source_dir / "solver.py"
        solver.write_text(
            """
from qf_primitives import black_scholes_price
import socket

_ = black_scholes_price(
    100.0, 100.0, 0.01, 0.0, 0.2, 1.0, "call"
)

try:
    socket.socket()
except PermissionError:
    print("NETWORK_BLOCKED")
else:
    raise RuntimeError("network unexpectedly available")
""".strip() + "\n",
            encoding="utf-8",
        )

        result = run_candidate(
            workspace,
            solver,
            timeout_seconds=5.0,
        )
        assert result.return_code == 0
        assert result.stdout.strip() == "NETWORK_BLOCKED"


def test_model_client_override_is_bounded_by_client_ceiling() -> None:
    body = json.dumps(
        {
            "choices": [
                {"message": {"content": "ok"}}
            ]
        }
    ).encode("utf-8")

    client = ModelClient(
        endpoint="https://example.test/v1",
        model="m",
        token="test-token",
        max_output_tokens=6000,
    )
    seen = {}

    def fake_urlopen(request, timeout):
        seen["payload"] = json.loads(
            request.data.decode("utf-8")
        )
        return _FakeResponse(body)

    with patch(
        "urllib.request.urlopen",
        side_effect=fake_urlopen,
    ):
        client.complete(
            "hello",
            max_output_tokens=8000,
        )

    assert seen["payload"]["max_tokens"] == 4000


def test_candidate_diversity_role_is_injected_even_for_planner_strategies() -> None:
    plan = RuntimePlan(
        planning_specification=None,  # type: ignore[arg-type]
        task_plan=None,
        planner_output=None,
        skill_packs=(),
        data_inspections={},
        strategies=(
            _StrategyEnvelope(
                payload={
                    "candidate_id": 1,
                    "approach_name": "same",
                }
            ),
            _StrategyEnvelope(
                payload={
                    "candidate_id": 2,
                    "approach_name": "same",
                }
            ),
        ),
    )

    first = _strategy_for_candidate(
        plan, 1, 1001
    ).to_dict()
    second = _strategy_for_candidate(
        plan, 2, 1002
    ).to_dict()

    assert first["diversity_role"] != second["diversity_role"]


def test_snapshot_discovers_root_level_archive_and_tsv() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        task = root / "task"
        task.mkdir()
        (task / "instruction.md").write_text(
            "# test\n",
            encoding="utf-8",
        )
        (task / "filing.zip").write_bytes(
            b"not-opened-in-this-test"
        )
        (task / "holdings.tsv").write_text(
            "x\\ty\\n1\\t2\\n",
            encoding="utf-8",
        )

        snapshot = TaskSnapshot.build(task)
        names = {
            path.name
            for path in snapshot.data_files
        }

        assert "filing.zip" in names
        assert "holdings.tsv" in names


def test_extended_candidate_finance_primitives_are_operational() -> None:
    from agent.qf_primitives import (
        central_price_delta,
        discount_cashflow,
        down_and_out_call_price,
        ewma_annualized_volatility,
        log_return_performance,
        sma_seeded_ema,
    )

    bond = discount_cashflow(
        1000.0,
        0.0216,
        0.25,
    )
    assert abs(
        bond
        - 994.6145537913911
    ) < 1e-9

    ema = sma_seeded_ema(
        [1.0, 2.0, 3.0, 4.0],
        3,
    )
    assert np.isnan(ema[0])
    assert np.isnan(ema[1])
    assert ema[2] == 2.0
    assert ema[3] == 3.0

    vol = ewma_annualized_volatility(
        [0.01, -0.02, 0.015],
        3,
    )
    assert vol.shape == (3,)
    assert np.all(np.isfinite(vol))
    assert np.all(vol >= 0.0)

    performance = log_return_performance(
        [0.01, -0.005, 0.012, 0.003],
        risk_free_annual=0.02,
    )
    assert set(performance) == {
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "calmar_ratio",
    }

    spot = 77.54975542623474
    strike = 80.65
    barrier = 54.28
    maturity = 0.25
    rate = 0.0216
    sigma = 0.2586800935963609

    price_fn = lambda s: down_and_out_call_price(
        s,
        strike,
        barrier,
        maturity,
        rate,
        sigma,
    )

    price = price_fn(spot)
    delta = central_price_delta(
        price_fn,
        spot,
        0.01,
    )

    assert abs(
        price
        - 2.8723811969778144
    ) < 1e-10

    assert abs(
        delta
        - 0.4220772133647326
    ) < 1e-10

    catalog_text = "\n".join(
        PRIMITIVE_API_CATALOG
    )

    for name in (
        "discount_cashflow",
        "sma_seeded_ema",
        "ewma_annualized_volatility",
        "log_return_performance",
        "down_and_out_call_price",
        "central_price_delta",
    ):
        assert name in catalog_text


def test_candidate_workspace_can_import_extended_finance_primitives() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        task = _task(root)

        workspace = CandidateWorkspace.create(
            base_dir=root / "work",
            candidate_id=77,
            task_dir=task,
        )

        solver = (
            workspace.source_dir
            / "solver.py"
        )

        solver.write_text(
            """
import json
from qf_primitives import (
    discount_cashflow,
    sma_seeded_ema,
)

payload = {
    "pv": discount_cashflow(
        1000.0,
        0.05,
        1.0,
    ),
    "ema": sma_seeded_ema(
        [1.0, 2.0, 3.0, 4.0],
        3,
    ).tolist(),
}

print(json.dumps(payload))
""".strip()
            + "\n",
            encoding="utf-8",
        )

        result = run_candidate(
            workspace,
            solver,
            timeout_seconds=5.0,
        )

        assert result.return_code == 0

        payload = json.loads(
            result.stdout
        )

        assert abs(
            payload["pv"]
            - 951.229424500714
        ) < 1e-10

        assert payload["ema"][2:] == [
            2.0,
            3.0,
        ]


def test_extended_candidate_finance_primitives_are_operational() -> None:
    from agent.qf_primitives import (
        central_price_delta,
        discount_cashflow,
        down_and_out_call_price,
        ewma_annualized_volatility,
        log_return_performance,
        sma_seeded_ema,
    )

    bond = discount_cashflow(
        1000.0,
        0.0216,
        0.25,
    )
    assert abs(
        bond
        - 994.6145537913911
    ) < 1e-9

    ema = sma_seeded_ema(
        [1.0, 2.0, 3.0, 4.0],
        3,
    )
    assert np.isnan(ema[0])
    assert np.isnan(ema[1])
    assert ema[2] == 2.0
    assert ema[3] == 3.0

    vol = ewma_annualized_volatility(
        [0.01, -0.02, 0.015],
        3,
    )
    assert vol.shape == (3,)
    assert np.all(np.isfinite(vol))
    assert np.all(vol >= 0.0)

    performance = log_return_performance(
        [0.01, -0.005, 0.012, 0.003],
        risk_free_annual=0.02,
    )
    assert set(performance) == {
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "calmar_ratio",
    }

    spot = 77.54975542623474
    strike = 80.65
    barrier = 54.28
    maturity = 0.25
    rate = 0.0216
    sigma = 0.2586800935963609

    price_fn = lambda s: down_and_out_call_price(
        s,
        strike,
        barrier,
        maturity,
        rate,
        sigma,
    )

    price = price_fn(spot)
    delta = central_price_delta(
        price_fn,
        spot,
        0.01,
    )

    assert abs(
        price
        - 2.8723811969778144
    ) < 1e-10

    assert abs(
        delta
        - 0.4220772133647326
    ) < 1e-10

    catalog_text = "\n".join(
        PRIMITIVE_API_CATALOG
    )

    for name in (
        "discount_cashflow",
        "sma_seeded_ema",
        "ewma_annualized_volatility",
        "log_return_performance",
        "down_and_out_call_price",
        "central_price_delta",
    ):
        assert name in catalog_text


def test_candidate_workspace_can_import_extended_finance_primitives() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        task = _task(root)

        workspace = CandidateWorkspace.create(
            base_dir=root / "work",
            candidate_id=77,
            task_dir=task,
        )

        solver = (
            workspace.source_dir
            / "solver.py"
        )

        solver.write_text(
            """
import json
from qf_primitives import (
    discount_cashflow,
    sma_seeded_ema,
)

payload = {
    "pv": discount_cashflow(
        1000.0,
        0.05,
        1.0,
    ),
    "ema": sma_seeded_ema(
        [1.0, 2.0, 3.0, 4.0],
        3,
    ).tolist(),
}

print(json.dumps(payload))
""".strip()
            + "\n",
            encoding="utf-8",
        )

        result = run_candidate(
            workspace,
            solver,
            timeout_seconds=5.0,
        )

        assert result.return_code == 0

        payload = json.loads(
            result.stdout
        )

        assert abs(
            payload["pv"]
            - 951.229424500714
        ) < 1e-10

        assert payload["ema"][2:] == [
            2.0,
            3.0,
        ]
