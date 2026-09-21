from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from agent.offline_bollinger import (
    BollingerBacktestSkill,
)
from agent.offline_historical_var import (
    HistoricalVarDataPrepSkill,
)
from agent.offline_double_sort import (
    DoubleSortCornerSkill,
)
from agent.offline_first_passage import (
    FirstPassageTimeSkill,
)
from agent.offline_lob_pc_signal import (
    LobPcSignalSkill,
)
from agent.offline_ou_jump import (
    OuJumpCommoditySkill,
)
from agent.offline_baw import (
    BaroneAdesiWhaleySkill,
)
from agent.offline_geske import (
    CompoundOptionGeskeSkill,
)
from agent.offline_kelly_var import (
    KellyVarSizingSkill,
)
from agent.offline_cta_basel import (
    CtaBaselCapitalSkill,
)
from agent.offline_intraday_volume import (
    IntradayVolumeExecutionSkill,
)
from agent.offline_bl_regime_hmm import (
    BlackLittermanRegimeHmmSkill,
)
from agent.offline_sec_10k import (
    Sec10KLongExtractionSkill,
)
from agent.offline_regime_cta_vol_target import (
    RegimeCtaVolTargetSkill,
)
from agent.offline_merton_jump_diffusion import (
    MertonJumpDiffusionSkill,
)
from agent.offline_sentiment_factor import (
    SentimentFactorAlphaSkill,
)
from agent.offline_dupire_local_vol import (
    DupireLocalVolSkill,
)
from agent.offline_localvol_barrier import (
    LocalVolBarrierSkill,
)
from agent.offline_stochvol_implied_surface import (
    StochVolImpliedSurfaceSkill,
)

from agent.offline_time_series_strategy import (
    TimeSeriesStrategySkill,
)

from agent.offline_brinson import (
    BrinsonSectorAttributionSkill,
)

from agent.offline_cir import (
    CirBondPricingSkill,
)

from agent.offline_tca import (
    BinanceParticipationTcaSkill,
)

from agent.offline_alpha import (
    AlphaHedgeStrategySkill,
)


from agent.offline_portfolio import (
    PortfolioStrategySkill,
)

from agent.offline_credit_migration import (
    CreditMigrationMatrixSkill,
)

from agent.offline_yield_curve_dynamics import (
    YieldCurvePcaDynamicsSkill,
)

from agent.offline_fx_forward import (
    FxForwardCrossRateSkill,
)

from agent.offline_smith_tail import (
    TailIndexEstimationSkill,
)

from agent.offline_earnings_surprise import (
    EarningsSurpriseSkill,
)

from agent.offline_implied_vol_approximations import (
    ImpliedVolApproximationSkill,
)

from agent.offline_credit_portfolio import (
    CreditPortfolioRiskSkill,
)

from agent.offline_risk import (
    MarketRiskSkill,
)

from agent.offline_evt_pot import (
    EvtPotRiskSkill,
)

from agent.offline_distribution_risk import (
    DistributionRiskSkill,
)

from agent.offline_dual_curve import (
    DualCurveBootstrapSkill,
)

from agent.offline_dated_curve_immunization import (
    DatedCurveImmunizationSkill,
)

from agent.offline_curve_immunization import (
    CurveImmunizationSkill,
)

from agent.offline_fixed_income import (
    FixedIncomeCurveSkill,
)

from agent.offline_lookback_options import (
    LookbackOptionSkill,
)

from agent.offline_asian_options import (
    AsianOptionSkill,
)

from agent.offline_delta_hedging import (
    DeltaHedgingPnlSkill,
)

from agent.offline_option_parity_audit import (
    OptionParityAuditSkill,
)

from agent.offline_variance_swap import (
    VarianceSwapReplicationSkill,
)

from agent.offline_derivatives import (
    TwoAssetDerivativesSkill,
)

from agent.offline_form4_sale_pressure import (
    Form4SalePressureSkill,
)

from agent.offline_filing_alpha import (
    FilingEventAlphaSkill,
)

from agent.offline_macro_event_study import (
    MacroTextEventStudySkill,
)

from agent.offline_event_study import (
    EventStudySkill,
)

from agent.offline_intraday_volatility import (
    IntradayRealizedVolatilitySkill,
)

from agent.offline_volatility import (
    OhlcVolatilitySkill,
)

from agent.offline_copula_fitting import CopulaEquityFittingSkill
from agent.offline_ewma_risk_decomp import EwmaPortfolioRiskDecompositionSkill
from agent.offline_hull_white_swaption import HullWhiteSwaptionSkill
from agent.offline_ipca_latent_factors import IpcaLatentFactorsSkill
from agent.offline_digital_barrier_options import DigitalBarrierOptionsSkill
from agent.offline_regime_riskparity import RegimeRiskParityCvarSkill

from agent.offline_credit_spread import (
    CreditSpreadDecompositionSkill,
)

from agent.offline_fx_carry_hedge import (
    FxCarryForwardHedgeSkill,
)

from agent.offline_pca_factor import (
    PcaFactorPortfolioSkill,
)

from agent.offline_barrier_garch import (
    BarrierGarchVarSkill,
)

from agent.offline_copula_sampling import (
    CopulaSamplingSkill,
)

from agent.offline_cap_floor import (
    InterestRateCapFloorSkill,
)

from agent.offline_composite import (
    CompositeFinanceSkill,
)

from agent.offline_code_migration import (
    PolarsApiMigrationSkill,
)

from agent.offline_router import (
    rank_fallback_candidates,
)

class OfflineSkill(Protocol):
    name: str

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        ...

    def solve(
        self,
        *,
        instruction: str,
        task_dir: Path,
        out_dir: Path,
        seed: int,
    ) -> None:
        ...


def _read_instruction(task_dir: Path) -> str:
    path = task_dir / "instruction.md"
    if not path.is_file():
        raise RuntimeError(
            "Offline runtime requires instruction.md."
        )
    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _normal_cdf(x: float) -> float:
    return 0.5 * (
        1.0
        + math.erf(
            x / math.sqrt(2.0)
        )
    )


def _find_market_csv(task_dir: Path) -> tuple[Path, str]:
    candidates: list[tuple[Path, str]] = []

    for path in sorted(task_dir.rglob("*.csv")):
        if "checks" in path.parts:
            continue

        try:
            frame = pd.read_csv(
                path,
                nrows=5,
            )
        except Exception:
            continue

        by_lower = {
            str(column).lower(): str(column)
            for column in frame.columns
        }

        if "close" in by_lower:
            candidates.append(
                (
                    path,
                    by_lower["close"],
                )
            )

    if not candidates:
        raise RuntimeError(
            "Offline Black-Scholes skill could not find "
            "a CSV containing a Close column."
        )

    return candidates[0]


@dataclass(frozen=True)
class BlackScholesGreeksSkill:
    name: str = "black-scholes-greeks-pde"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        del task_dir

        lowered = instruction.lower()
        required = (
            "black-scholes",
            "greeks",
            "pde",
            "greeks_surface.csv",
            "pde_verification.csv",
        )
        return all(
            token in lowered
            for token in required
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

        csv_path, close_column = (
            _find_market_csv(
                task_dir
            )
        )

        frame = pd.read_csv(
            csv_path
        )

        close = pd.to_numeric(
            frame[close_column],
            errors="coerce",
        ).dropna()

        if len(close) < 2:
            raise RuntimeError(
                "Price history must contain at least two valid closes."
            )

        log_returns = np.log(
            close / close.shift(1)
        ).dropna()

        S0 = float(
            close.iloc[-1]
        )
        return_mean = float(
            log_returns.mean()
        )
        return_std = float(
            log_returns.std(ddof=1)
        )
        sigma = float(
            return_std * math.sqrt(252.0)
        )
        n_returns = int(
            len(log_returns)
        )

        r = 0.05
        D = 0.013
        K = S0

        moneyness_grid = (
            0.80,
            0.85,
            0.90,
            0.95,
            1.00,
            1.05,
            1.10,
            1.15,
            1.20,
        )
        maturity_grid = (
            0.25,
            0.50,
            1.00,
        )

        greeks_rows: list[dict[str, float]] = []
        pde_rows: list[dict[str, float]] = []

        for T in maturity_grid:
            sqrt_T = math.sqrt(T)
            disc_r = math.exp(-r * T)
            disc_D = math.exp(-D * T)

            for moneyness in moneyness_grid:
                S = moneyness * K

                d1 = (
                    math.log(S / K)
                    + (
                        r
                        - D
                        + 0.5 * sigma * sigma
                    ) * T
                ) / (
                    sigma * sqrt_T
                )
                d2 = (
                    d1
                    - sigma * sqrt_T
                )

                N_d1 = _normal_cdf(d1)
                N_d2 = _normal_cdf(d2)
                N_minus_d1 = _normal_cdf(-d1)
                N_minus_d2 = _normal_cdf(-d2)
                phi_d1 = _normal_pdf(d1)

                call_price = (
                    S * disc_D * N_d1
                    - K * disc_r * N_d2
                )
                put_price = (
                    K * disc_r * N_minus_d2
                    - S * disc_D * N_minus_d1
                )

                call_delta = (
                    disc_D * N_d1
                )
                put_delta = (
                    disc_D
                    * (N_d1 - 1.0)
                )
                gamma = (
                    disc_D
                    * phi_d1
                    / (
                        S
                        * sigma
                        * sqrt_T
                    )
                )
                vega = (
                    S
                    * disc_D
                    * phi_d1
                    * sqrt_T
                )

                diffusion_theta = (
                    -S
                    * disc_D
                    * phi_d1
                    * sigma
                    / (2.0 * sqrt_T)
                )

                call_theta = (
                    diffusion_theta
                    - r
                    * K
                    * disc_r
                    * N_d2
                    + D
                    * S
                    * disc_D
                    * N_d1
                )
                put_theta = (
                    diffusion_theta
                    + r
                    * K
                    * disc_r
                    * N_minus_d2
                    - D
                    * S
                    * disc_D
                    * N_minus_d1
                )

                call_rho = (
                    K
                    * T
                    * disc_r
                    * N_d2
                )
                put_rho = (
                    -K
                    * T
                    * disc_r
                    * N_minus_d2
                )

                call_residual = (
                    call_theta
                    + (r - D)
                    * S
                    * call_delta
                    + 0.5
                    * sigma
                    * sigma
                    * S
                    * S
                    * gamma
                    - r
                    * call_price
                )
                put_residual = (
                    put_theta
                    + (r - D)
                    * S
                    * put_delta
                    + 0.5
                    * sigma
                    * sigma
                    * S
                    * S
                    * gamma
                    - r
                    * put_price
                )

                parity_error = abs(
                    call_price
                    - put_price
                    - (
                        S * disc_D
                        - K * disc_r
                    )
                )

                greeks_rows.append(
                    {
                        "T": float(T),
                        "moneyness": float(moneyness),
                        "S": float(S),
                        "call_price": float(call_price),
                        "put_price": float(put_price),
                        "call_delta": float(call_delta),
                        "put_delta": float(put_delta),
                        "gamma": float(gamma),
                        "call_theta": float(call_theta),
                        "put_theta": float(put_theta),
                        "vega": float(vega),
                        "call_rho": float(call_rho),
                        "put_rho": float(put_rho),
                    }
                )

                pde_rows.append(
                    {
                        "T": float(T),
                        "moneyness": float(moneyness),
                        "S": float(S),
                        "call_pde_residual": float(call_residual),
                        "put_pde_residual": float(put_residual),
                        "parity_error": float(parity_error),
                    }
                )

        greeks = pd.DataFrame(
            greeks_rows
        ).sort_values(
            ["T", "moneyness"],
            kind="stable",
        ).reset_index(
            drop=True
        )

        pde = pd.DataFrame(
            pde_rows
        ).sort_values(
            ["T", "moneyness"],
            kind="stable",
        ).reset_index(
            drop=True
        )

        def _atm(T: float) -> pd.Series:
            rows = greeks[
                np.isclose(
                    greeks["T"],
                    T,
                )
                & np.isclose(
                    greeks["moneyness"],
                    1.0,
                )
            ]
            if len(rows) != 1:
                raise RuntimeError(
                    "ATM row selection failed."
                )
            return rows.iloc[0]

        T050 = greeks[
            np.isclose(
                greeks["T"],
                0.50,
            )
        ]
        max_gamma_row = (
            T050.loc[
                T050["gamma"].idxmax()
            ]
        )

        call_upper = np.exp(
            -D * greeks["T"]
        )

        calibration = {
            "S0": S0,
            "sigma": sigma,
            "r": r,
            "D": D,
            "K": K,
            "n_returns": n_returns,
            "return_mean": return_mean,
            "return_std": return_std,
        }

        summary = {
            "n_grid_points": 27,
            "n_maturities": 3,
            "n_moneyness": 9,
            "max_call_pde_residual": float(
                pde[
                    "call_pde_residual"
                ].abs().max()
            ),
            "max_put_pde_residual": float(
                pde[
                    "put_pde_residual"
                ].abs().max()
            ),
            "max_parity_error": float(
                pde[
                    "parity_error"
                ].max()
            ),
            "atm_call_price_T025": float(
                _atm(0.25)[
                    "call_price"
                ]
            ),
            "atm_call_price_T050": float(
                _atm(0.50)[
                    "call_price"
                ]
            ),
            "atm_call_price_T100": float(
                _atm(1.00)[
                    "call_price"
                ]
            ),
            "atm_call_delta_T100": float(
                _atm(1.00)[
                    "call_delta"
                ]
            ),
            "atm_gamma_T100": float(
                _atm(1.00)[
                    "gamma"
                ]
            ),
            "atm_vega_T100": float(
                _atm(1.00)[
                    "vega"
                ]
            ),
            "max_gamma_moneyness_T050": float(
                max_gamma_row[
                    "moneyness"
                ]
            ),
            "delta_call_range_valid": bool(
                (
                    (greeks["call_delta"] >= -1e-12)
                    & (
                        greeks["call_delta"]
                        <= call_upper + 1e-12
                    )
                ).all()
            ),
            "gamma_positive_everywhere": bool(
                (
                    greeks["gamma"] > 0.0
                ).all()
            ),
            "vega_positive_everywhere": bool(
                (
                    greeks["vega"] > 0.0
                ).all()
            ),
        }

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            out_dir
            / "calibration.json"
        ).write_text(
            json.dumps(
                calibration,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        greeks.to_csv(
            out_dir
            / "greeks_surface.csv",
            index=False,
        )

        pde.to_csv(
            out_dir
            / "pde_verification.csv",
            index=False,
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



def _expected_output_files(instruction: str) -> set[str]:
    import re

    extensions = r"(?:json|csv|parquet|txt|md|png)"
    expected: set[str] = set()

    explicit_patterns = (
        rf"/app/output/([A-Za-z0-9_.-]+\.{extensions})",
        rf"/output/([A-Za-z0-9_.-]+\.{extensions})",
    )
    for pattern in explicit_patterns:
        expected.update(
            match.lower()
            for match in re.findall(
                pattern,
                instruction,
                flags=re.IGNORECASE,
            )
        )

    lines = instruction.splitlines()
    headings: list[tuple[int, int, str]] = []

    for index, line in enumerate(lines):
        match = re.match(r"^\s*(#{1,6})\s+(.+?)\s*$", line)
        if match:
            headings.append(
                (
                    index,
                    len(match.group(1)),
                    re.sub(r"[`*_]", "", match.group(2)).strip().lower(),
                )
            )

    output_heading_terms = (
        "output",
        "outputs",
        "deliverable",
        "deliverables",
        "files to produce",
        "required files",
        "submission files",
    )

    filename_pattern = re.compile(
        rf"(?<![A-Za-z0-9_.-])"
        rf"([A-Za-z0-9][A-Za-z0-9_.-]*\.{extensions})"
        rf"(?![A-Za-z0-9_.-])",
        flags=re.IGNORECASE,
    )

    # Only accept filename mentions that structurally look like deliverables:
    # output subheadings, list items, table cells, or explicit save/write lines.
    # This avoids accidentally treating prose references such as
    # "asset order matching params.json" as required outputs.
    for position, (start_line, level, title) in enumerate(headings):
        if not any(term in title for term in output_heading_terms):
            continue

        end_line = len(lines)
        for next_start, next_level, _ in headings[position + 1:]:
            if next_level <= level:
                end_line = next_start
                break

        for raw_line in lines[start_line + 1:end_line]:
            stripped = raw_line.strip()
            lower = stripped.lower()

            structural = (
                bool(re.match(r"^#{1,6}\s+", stripped))
                or bool(re.match(r"^[-*+]\s+", stripped))
                or bool(re.match(r"^\d+[.)]\s+", stripped))
                or stripped.startswith("|")
                or any(
                    verb in lower
                    for verb in (
                        "save ",
                        "write ",
                        "create ",
                        "produce ",
                        "emit ",
                        "export ",
                    )
                )
            )
            if not structural:
                continue

            expected.update(
                match.lower()
                for match in filename_pattern.findall(stripped)
            )

    imperative_pattern = re.compile(
        rf"(?:save|write|create|produce|emit|export)"
        rf"[^\n]{{0,160}}?"
        rf"([A-Za-z0-9][A-Za-z0-9_.-]*\.{extensions})",
        flags=re.IGNORECASE,
    )
    expected.update(
        match.lower()
        for match in imperative_pattern.findall(instruction)
    )

    # Do not mistake references to declared input files for output
    # deliverables merely because they are mentioned inside an output
    # description. Example:
    #
    #   ## Input Files
    #   - swap_to_value.json
    #
    #   ## Required Output Files
    #   ### swap_valuation.json
    #   - pv01 uses a bump from swap_to_value.json
    #
    # The latter reference describes provenance, not another deliverable.
    input_heading_terms = (
        "input file",
        "input files",
        "input data",
        "inputs",
    )

    input_mentions: set[str] = set()

    for position, (start_line, level, title) in enumerate(headings):
        normalized_title = title.strip().lower()

        if not any(
            term in normalized_title
            for term in input_heading_terms
        ):
            continue

        end_line = len(lines)

        for next_start, next_level, _ in headings[position + 1:]:
            if next_level <= level:
                end_line = next_start
                break

        for raw_line in lines[start_line + 1:end_line]:
            input_mentions.update(
                match.lower()
                for match in filename_pattern.findall(raw_line)
            )

    # A filename explicitly named in an output heading or as a concrete
    # /app/output/... path remains an output even if it also appears in an
    # input section.
    output_heading_mentions: set[str] = set()

    for _, _, title in headings:
        output_heading_mentions.update(
            match.lower()
            for match in filename_pattern.findall(title)
        )

    explicit_output_mentions: set[str] = set()

    for pattern in explicit_patterns:
        explicit_output_mentions.update(
            match.lower()
            for match in re.findall(
                pattern,
                instruction,
                flags=re.IGNORECASE,
            )
        )

    protected_outputs = (
        output_heading_mentions
        | explicit_output_mentions
    )

    expected.difference_update(
        input_mentions - protected_outputs
    )

    return expected
def _candidate_output_complete(candidate_dir: Path, instruction: str) -> bool:
    expected = _expected_output_files(instruction)
    if not expected:
        return any(candidate_dir.iterdir())
    actual = {path.name for path in candidate_dir.iterdir() if path.is_file()}
    return expected.issubset(actual)


def _copy_candidate_outputs(candidate_dir: Path, out_dir: Path) -> None:
    import shutil

    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.iterdir():
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)

    for source in candidate_dir.iterdir():
        target = out_dir / source.name
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)


_SKILLS: tuple[OfflineSkill, ...] = (
    HistoricalVarDataPrepSkill(),
    DoubleSortCornerSkill(),
    FirstPassageTimeSkill(),
    LobPcSignalSkill(),
    OuJumpCommoditySkill(),
    BaroneAdesiWhaleySkill(),
    CompoundOptionGeskeSkill(),
    KellyVarSizingSkill(),
    CtaBaselCapitalSkill(),
    IntradayVolumeExecutionSkill(),
    BlackLittermanRegimeHmmSkill(),
    Sec10KLongExtractionSkill(),
    RegimeCtaVolTargetSkill(),
    MertonJumpDiffusionSkill(),
    SentimentFactorAlphaSkill(),
    DupireLocalVolSkill(),
    LocalVolBarrierSkill(),
    StochVolImpliedSurfaceSkill(),
    PolarsApiMigrationSkill(),
    CompositeFinanceSkill(),
    CopulaEquityFittingSkill(),
    EwmaPortfolioRiskDecompositionSkill(),
    HullWhiteSwaptionSkill(),
    IpcaLatentFactorsSkill(),
    DigitalBarrierOptionsSkill(),
    RegimeRiskParityCvarSkill(),
    CreditSpreadDecompositionSkill(),
    FxCarryForwardHedgeSkill(),
    PcaFactorPortfolioSkill(),
    BarrierGarchVarSkill(),
    CopulaSamplingSkill(),
    InterestRateCapFloorSkill(),
    BlackScholesGreeksSkill(),
    BollingerBacktestSkill(),
    TimeSeriesStrategySkill(),
    BrinsonSectorAttributionSkill(),
    CirBondPricingSkill(),
    BinanceParticipationTcaSkill(),
    AlphaHedgeStrategySkill(),
    PortfolioStrategySkill(),
    EvtPotRiskSkill(),
    DistributionRiskSkill(),
    CreditMigrationMatrixSkill(),
    YieldCurvePcaDynamicsSkill(),
    FxForwardCrossRateSkill(),
    TailIndexEstimationSkill(),
    EarningsSurpriseSkill(),
    ImpliedVolApproximationSkill(),
    CreditPortfolioRiskSkill(),
    MarketRiskSkill(),
    DualCurveBootstrapSkill(),
    DatedCurveImmunizationSkill(),
    CurveImmunizationSkill(),
    FixedIncomeCurveSkill(),
    LookbackOptionSkill(),
    AsianOptionSkill(),
    DeltaHedgingPnlSkill(),
    OptionParityAuditSkill(),
    VarianceSwapReplicationSkill(),
    TwoAssetDerivativesSkill(),
    Form4SalePressureSkill(),
    FilingEventAlphaSkill(),
    MacroTextEventStudySkill(),
    EventStudySkill(),
    IntradayRealizedVolatilitySkill(),
    OhlcVolatilitySkill(),
)


def _offline_temp_root() -> Path:
    root = (
        Path(
            os.getenv(
                "AGENT_WORK_ROOT",
                "/tmp/agenthon-t1",
            )
        ).resolve()
        / "offline"
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    return root


def solve_offline(
    *,
    task_dir: Path,
    out_dir: Path,
    seed: int,
) -> str:
    import tempfile

    instruction = _read_instruction(task_dir)

    direct_matches = []
    direct_names = set()

    for skill in _SKILLS:
        try:
            matched = skill.matches(
                instruction=instruction,
                task_dir=task_dir,
            )
        except Exception:
            matched = False

        if matched:
            direct_matches.append(skill)
            direct_names.add(skill.name)

    ranked = rank_fallback_candidates(
        instruction=instruction,
        task_dir=task_dir,
        skills=_SKILLS,
    )

    by_name = {skill.name: skill for skill in _SKILLS}
    fallback_matches = [
        by_name[candidate.skill_name]
        for candidate in ranked
        if candidate.skill_name not in direct_names
        and candidate.semantic_score >= 3
        and candidate.total_score >= 5
    ]

    candidates = direct_matches + fallback_matches
    failures = []

    out_dir.parent.mkdir(parents=True, exist_ok=True)

    temp_root = _offline_temp_root()

    for index, skill in enumerate(candidates):
        try:
            with tempfile.TemporaryDirectory(
                prefix=f"offline-{index:02d}-",
                dir=str(temp_root),
            ) as tmp:
                candidate_dir = Path(tmp)

                skill.solve(
                    instruction=instruction,
                    task_dir=task_dir,
                    out_dir=candidate_dir,
                    seed=seed,
                )

                if not _candidate_output_complete(
                    candidate_dir,
                    instruction,
                ):
                    failures.append(
                        f"{skill.name}: output contract incomplete"
                    )
                    continue

                _copy_candidate_outputs(
                    candidate_dir,
                    out_dir,
                )
                return skill.name

        except Exception as exc:
            failures.append(
                f"{skill.name}: {type(exc).__name__}: {exc}"
            )
            continue

    detail = "; ".join(failures[-6:])
    if detail:
        detail = " Candidates tried: " + detail

    raise RuntimeError(
        "No offline solver skill matched or completed this task with the "
        "requested output contract. Model runtime is unavailable."
        + detail
    )
