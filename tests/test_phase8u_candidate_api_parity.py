from __future__ import annotations

from dataclasses import asdict, is_dataclass
import importlib
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from agent.candidate_workspace import CandidateWorkspace
from agent.capability_bridge import CANDIDATE_LIBRARY_EXPORTS


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def capability_modules(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, tuple[ModuleType, ModuleType]]:
    root = tmp_path_factory.mktemp("phase8u-api-parity")
    task = root / "task"
    (task / "environment" / "data").mkdir(parents=True)
    (task / "instruction.md").write_text(
        "Exercise general finance computations.",
        encoding="utf-8",
    )
    workspace = CandidateWorkspace.create(
        base_dir=root / "work",
        candidate_id=1,
        task_dir=task,
    )
    curated_root = workspace.root_dir / "lib" / "offline_common"

    result: dict[str, tuple[ModuleType, ModuleType]] = {}
    for filename, exports in CANDIDATE_LIBRARY_EXPORTS.items():
        stem = filename.removesuffix(".py")
        original = importlib.import_module(f"agent.offline_common.{stem}")
        curated = _load_module(
            f"_phase8u_curated_{stem}",
            curated_root / filename,
        )
        assert tuple(curated.__all__) == exports
        result[stem] = (original, curated)
    return result


def _assert_equivalent(first: Any, second: Any) -> None:
    if is_dataclass(first) and not isinstance(first, type):
        assert is_dataclass(second) and not isinstance(second, type)
        _assert_equivalent(asdict(first), asdict(second))
    elif isinstance(first, pd.DataFrame):
        pd.testing.assert_frame_equal(first, second)
    elif isinstance(first, pd.Series):
        pd.testing.assert_series_equal(first, second)
    elif isinstance(first, np.ndarray):
        np.testing.assert_allclose(first, second, equal_nan=True)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_equivalent(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert type(first) is type(second)
        assert len(first) == len(second)
        for left, right in zip(first, second):
            _assert_equivalent(left, right)
    elif isinstance(first, (float, np.floating)):
        assert math.isclose(
            float(first),
            float(second),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    else:
        assert first == second


def _assert_all_exports_exercised(
    module: ModuleType,
    outputs: dict[str, Any],
) -> None:
    assert set(outputs) == set(module.__all__)


def _asset_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=90)
    index = np.arange(len(dates), dtype=float)
    first = 100.0 * np.exp(
        0.0005 * index + 0.015 * np.sin(index / 5.0)
    )
    second = 95.0 * np.exp(
        0.0003 * index + 0.012 * np.sin(index / 6.0 + 0.4)
    )
    return (
        pd.DataFrame({"date": dates, "close": first}),
        pd.DataFrame({"date": dates, "close": second}),
    )


def _derivative_outputs(module: ModuleType) -> dict[str, Any]:
    first, second = _asset_frames()
    calibration = module.calibrate_two_asset_gbm(first, second)
    exchange = module.exchange_volatility(0.25, 0.20, 0.4)
    margrabe = module.margrabe_price(
        S1=120.0,
        S2=100.0,
        sigma1=0.25,
        sigma2=0.20,
        rho=0.4,
        T=0.5,
        q1=0.01,
        q2=0.01,
    )
    kirk = module.kirk_spread_price(
        S1=120.0,
        S2=100.0,
        K=0.0,
        sigma1=0.25,
        sigma2=0.20,
        rho=0.4,
        T=0.5,
        r=0.05,
        q1=0.01,
        q2=0.01,
    )
    monte_carlo = module.monte_carlo_spread_prices(
        S1=120.0,
        S2=100.0,
        strikes=[0.0, 10.0],
        sigma1=0.25,
        sigma2=0.20,
        rho=0.4,
        T=0.5,
        r=0.05,
        q1=0.01,
        q2=0.01,
        n_paths=2_000,
        rng=np.random.default_rng(1234),
    )
    return {
        "TwoAssetCalibration": calibration,
        "calibrate_two_asset_gbm": calibration,
        "exchange_volatility": exchange,
        "margrabe_price": margrabe,
        "kirk_spread_price": kirk,
        "monte_carlo_spread_prices": monte_carlo,
    }


def test_derivative_api_parity_and_price_sanity(
    capability_modules: dict[str, tuple[ModuleType, ModuleType]],
) -> None:
    original, curated = capability_modules["derivatives"]
    original_outputs = _derivative_outputs(original)
    curated_outputs = _derivative_outputs(curated)

    _assert_all_exports_exercised(curated, curated_outputs)
    _assert_equivalent(original_outputs, curated_outputs)
    margrabe_price = curated_outputs["margrabe_price"][0]
    assert margrabe_price >= 0.0
    assert math.isclose(
        margrabe_price,
        curated_outputs["kirk_spread_price"],
        rel_tol=1e-10,
        abs_tol=1e-10,
    )
    assert all(
        price >= 0.0 and error >= 0.0
        for price, error in curated_outputs["monte_carlo_spread_prices"]
    )


def _event_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2020-01-02", periods=260)
    index = np.arange(len(dates), dtype=float)
    market_returns = 0.0002 + 0.008 * np.sin(index / 7.0)
    market_prices = 100.0 * np.exp(np.cumsum(market_returns))
    stock_prices = pd.DataFrame(
        {
            "date": dates,
            "AAA": 80.0
            * np.exp(
                np.cumsum(
                    0.0003
                    + 1.2 * market_returns
                    + 0.003 * np.sin(index / 3.0)
                )
            ),
            "BBB": 120.0
            * np.exp(
                np.cumsum(
                    -0.0001
                    + 0.8 * market_returns
                    + 0.004 * np.cos(index / 5.0)
                )
            ),
        }
    )
    market = pd.DataFrame({"date": dates, "close": market_prices})
    events = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "AAA"],
            "event_date": [dates[160], dates[180], dates[220]],
        }
    )
    return stock_prices, market, events


def _event_outputs(module: ModuleType) -> dict[str, Any]:
    stocks, market, events = _event_frames()
    spec = module.EventStudySpec()
    merged, tickers = module.prepare_event_study_returns(stocks, market)
    records = module.build_event_records(merged, events, spec=spec)
    return {
        "EventStudySpec": spec,
        "EventRecord": records,
        "prepare_event_study_returns": (merged, tickers),
        "build_event_records": records,
        "corrado_rank_statistics": module.corrado_rank_statistics(records),
        "average_pairwise_residual_correlation": (
            module.average_pairwise_residual_correlation(records)
        ),
        "kolari_pynnonen_statistics": (
            module.kolari_pynnonen_statistics(records)
        ),
    }


def test_event_study_api_parity_and_consistency(
    capability_modules: dict[str, tuple[ModuleType, ModuleType]],
) -> None:
    original, curated = capability_modules["event_study"]
    original_outputs = _event_outputs(original)
    curated_outputs = _event_outputs(curated)

    _assert_all_exports_exercised(curated, curated_outputs)
    _assert_equivalent(original_outputs, curated_outputs)
    records = curated_outputs["build_event_records"]
    assert len(records) == 3
    assert all(len(record.event_abnormal_returns) == 11 for record in records)
    statistics = (
        *curated_outputs["corrado_rank_statistics"],
        curated_outputs["average_pairwise_residual_correlation"],
        *curated_outputs["kolari_pynnonen_statistics"],
    )
    assert np.isfinite(statistics).all()


def _fixed_income_outputs(module: ModuleType) -> dict[str, Any]:
    maturities = [1.0, 2.0, 3.0, 5.0]
    par_rates = [0.04, 0.043, 0.046, 0.05]
    curve = module.bootstrap_par_curve(
        maturities,
        par_rates,
        coupon_frequency=2,
    )
    repriced = [
        module.reprice_par_bond(
            maturity=maturity,
            par_rate=par_rate,
            coupon_frequency=2,
            curve=curve,
        )
        for maturity, par_rate in zip(maturities, par_rates)
    ]
    return {
        "BootstrappedCurve": curve,
        "discount_factor_from_zero": (
            module.discount_factor_from_zero(0.04, 2.0)
        ),
        "interpolate_zero_rate": module.interpolate_zero_rate(
            2.5,
            known_maturities=[1.0, 2.0, 3.0],
            known_zero_rates=[0.03, 0.04, 0.05],
        ),
        "bootstrap_par_curve": curve,
        "reprice_par_bond": repriced,
    }


def test_fixed_income_api_parity_and_curve_repricing(
    capability_modules: dict[str, tuple[ModuleType, ModuleType]],
) -> None:
    original, curated = capability_modules["fixed_income"]
    original_outputs = _fixed_income_outputs(original)
    curated_outputs = _fixed_income_outputs(curated)

    _assert_all_exports_exercised(curated, curated_outputs)
    _assert_equivalent(original_outputs, curated_outputs)
    np.testing.assert_allclose(
        curated_outputs["reprice_par_bond"],
        np.ones(4),
        atol=1e-10,
    )


def _portfolio_inputs() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=18, freq="ME")
    index = np.arange(len(dates), dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "A": 100.0 * np.exp(0.02 * index),
            "B": 100.0 * np.exp(-0.01 * index),
            "C": 100.0 * np.exp(0.005 * index + 0.01 * np.sin(index)),
        }
    )


def _portfolio_outputs(module: ModuleType) -> dict[str, Any]:
    spec = module.MomentumSpec(4, 2, 4, 1, 1, 12)
    cleaned, assets = module.clean_price_panel(_portfolio_inputs())
    returns = module.simple_returns_from_prices(
        cleaned,
        asset_columns=assets,
    )
    path = module.cross_sectional_momentum_path(
        returns,
        asset_columns=assets,
        spec=spec,
    )
    summary = module.summarize_return_path(path, periods_per_year=12)
    return {
        "MomentumSpec": spec,
        "clean_price_panel": (cleaned, assets),
        "simple_returns_from_prices": returns,
        "cross_sectional_momentum_path": path,
        "summarize_return_path": summary,
    }


def test_portfolio_api_parity_and_no_lookahead(
    capability_modules: dict[str, tuple[ModuleType, ModuleType]],
) -> None:
    original, curated = capability_modules["portfolio"]
    original_outputs = _portfolio_outputs(original)
    curated_outputs = _portfolio_outputs(curated)
    _assert_all_exports_exercised(curated, curated_outputs)
    _assert_equivalent(original_outputs, curated_outputs)

    returns = curated_outputs["simple_returns_from_prices"].copy()
    assets = ["A", "B", "C"]
    spec = curated.MomentumSpec(4, 2, 4, 1, 1, 12)
    baseline = curated.cross_sectional_momentum_path(
        returns,
        asset_columns=assets,
        spec=spec,
    )
    changed = returns.copy()
    changed.loc[10, assets] = [10.0, -10.0, 5.0]
    revised = curated.cross_sectional_momentum_path(
        changed,
        asset_columns=assets,
        spec=spec,
    )
    pd.testing.assert_frame_equal(
        baseline[baseline["date"] < str(returns.loc[10, "date"].date())],
        revised[revised["date"] < str(returns.loc[10, "date"].date())],
    )


def _risk_market_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=80)
    rows = []
    for index, date in enumerate(dates):
        for symbol, scale in (("AAA", 1.0), ("BBB", 0.8)):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "close": 100.0
                    * np.exp(
                        0.0004 * index
                        + scale * 0.01 * np.sin(index / 5.0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _risk_outputs(module: ModuleType) -> dict[str, Any]:
    sample = 0.015 * np.sin(np.arange(240, dtype=float) / 7.0)
    asset_returns = np.column_stack((sample, 0.7 * sample + 0.002))
    weights = module.exponential_weights(len(sample), decay=0.98)
    covariance = module.ewma_covariance(asset_returns, decay=0.94)
    return {
        "aligned_log_returns": module.aligned_log_returns(
            _risk_market_frame(),
            date_column="date",
            symbol_column="symbol",
            close_column="close",
        ),
        "historical_var_es": module.historical_var_es(
            sample,
            alpha=0.95,
            notional=1_000_000.0,
        ),
        "normal_var_es": module.normal_var_es(
            sample,
            alpha=0.95,
            notional=1_000_000.0,
        ),
        "student_t_var_es": module.student_t_var_es(
            sample,
            alpha=0.95,
            notional=1_000_000.0,
        ),
        "exponential_weights": weights,
        "weighted_quantile": module.weighted_quantile(
            sample,
            weights,
            probability=0.05,
        ),
        "age_weighted_var_es": module.age_weighted_var_es(
            sample,
            alpha=0.95,
            notional=1_000_000.0,
            decay=0.98,
        ),
        "ewma_covariance": covariance,
        "ewma_portfolio_volatility": (
            module.ewma_portfolio_volatility(
                asset_returns,
                decay=0.94,
                weights=np.asarray([0.6, 0.4]),
            )
        ),
        "overlapping_horizon_returns": module.overlapping_horizon_returns(
            sample,
            horizon=10,
        ),
        "expanding_historical_backtest": (
            module.expanding_historical_backtest(
                sample,
                alpha=0.95,
                minimum_observations=60,
            )
        ),
    }


def test_risk_api_parity_and_invariants(
    capability_modules: dict[str, tuple[ModuleType, ModuleType]],
) -> None:
    original, curated = capability_modules["risk"]
    original_outputs = _risk_outputs(original)
    curated_outputs = _risk_outputs(curated)
    _assert_all_exports_exercised(curated, curated_outputs)
    _assert_equivalent(original_outputs, curated_outputs)

    for name in (
        "historical_var_es",
        "normal_var_es",
        "student_t_var_es",
        "age_weighted_var_es",
    ):
        value_at_risk, expected_shortfall = curated_outputs[name]
        assert np.isfinite([value_at_risk, expected_shortfall]).all()
        assert value_at_risk >= 0.0
        assert expected_shortfall >= value_at_risk
    covariance = curated_outputs["ewma_covariance"]
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-14)


def _ohlc_frame(periods: int = 90) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=periods)
    index = np.arange(periods, dtype=float)
    close = 100.0 * np.exp(
        0.0004 * index + 0.01 * np.sin(index / 5.0)
    )
    open_ = close * (1.0 + 0.002 * np.sin(index / 3.0))
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": np.maximum(open_, close) * 1.012,
            "low": np.minimum(open_, close) * 0.988,
            "close": close,
        }
    )


def _volatility_outputs(module: ModuleType) -> dict[str, Any]:
    frame = _ohlc_frame()
    estimate = module.ohlc_variance_estimators(frame)
    rolling = module.rolling_ohlc_estimators(frame, window=21)
    arrays = module.estimator_arrays(rolling)
    return {
        "OhlcVarianceEstimate": estimate,
        "validate_ohlc": module.validate_ohlc(frame),
        "ohlc_variance_estimators": estimate,
        "annualized_volatility": (
            module.annualized_volatility(estimate.close_to_close)
        ),
        "rolling_ohlc_estimators": rolling,
        "estimator_arrays": arrays,
        "efficiency_ratios": module.efficiency_ratios(arrays),
    }


def test_volatility_api_parity_and_nonnegative_variance(
    capability_modules: dict[str, tuple[ModuleType, ModuleType]],
) -> None:
    original, curated = capability_modules["volatility"]
    original_outputs = _volatility_outputs(original)
    curated_outputs = _volatility_outputs(curated)
    _assert_all_exports_exercised(curated, curated_outputs)
    _assert_equivalent(original_outputs, curated_outputs)

    estimate = curated_outputs["ohlc_variance_estimators"]
    assert all(
        value >= 0.0
        for value in asdict(estimate).values()
    )
    assert curated_outputs["annualized_volatility"] >= 0.0
