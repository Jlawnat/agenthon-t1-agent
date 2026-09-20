from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg as la
from scipy.stats import chi2


RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "Default"]
N_RATINGS = 8
N_NON_DEFAULT = 7
RATING_IDX = {rating: i for i, rating in enumerate(RATINGS)}
SP_MAP = {
    "AAA": "AAA",
    "AA": "AA",
    "A": "A",
    "BBB": "BBB",
    "BB": "BB",
    "B": "B",
    "CCC": "CCC",
    "CC": "CCC",
    "C": "CCC",
    "R": "Default",
    "SD": "Default",
    "D": "Default",
}
EXCLUDE_CATEGORIES = {"NR", "Withdrawals"}


def _find_data_dir(task_dir: Path) -> Path:
    for directory in sorted(
        {p.parent for p in task_dir.rglob("*") if p.is_file() and "checks" not in p.parts},
        key=str,
    ):
        if (
            (directory / "config.json").is_file()
            and (list(directory.glob("Transition-Matrix-*.csv")))
            and (list(directory.glob("Default-Rates-*.csv")))
            and (
                (directory / "historical_transition_matrix.csv").is_file()
                or (directory / "reference_transition_matrix.csv").is_file()
            )
        ):
            return directory
    raise RuntimeError("Could not discover credit-migration cohort inputs.")


def _parse_cerep_file(filepath: Path) -> np.ndarray:
    df = pd.read_csv(filepath, sep=";", index_col=0)
    counts = np.zeros((N_RATINGS, N_RATINGS), dtype=float)

    for bop_rating in df.index:
        bop = str(bop_rating).strip()
        if bop in EXCLUDE_CATEGORIES:
            continue
        bop_mapped = SP_MAP.get(bop)
        if bop_mapped is None:
            continue
        i = RATING_IDX[bop_mapped]

        for eop_rating in df.columns:
            eop = str(eop_rating).strip()
            if eop in EXCLUDE_CATEGORIES:
                continue
            eop_mapped = SP_MAP.get(eop)
            if eop_mapped is None:
                continue
            j = RATING_IDX[eop_mapped]
            counts[i, j] += float(df.loc[bop_rating, eop_rating])

    return counts.astype(int)


def _load_true_defaults(path: Path) -> dict[int, int]:
    df = pd.read_csv(path)
    totals: dict[int, int] = {}
    for _, row in df.iterrows():
        source = str(row["rating"]).strip()
        mapped = SP_MAP.get(source)
        if mapped is None or mapped == "Default":
            continue
        i = RATING_IDX[mapped]
        totals[i] = totals.get(i, 0) + int(row["n_defaults"])
    return totals


def _correct_defaults(counts: np.ndarray, true_defaults: dict[int, int]) -> np.ndarray:
    fixed = counts.copy().astype(float)
    for i in range(N_NON_DEFAULT):
        visible = float(counts[i, -1])
        true_count = float(true_defaults.get(i, 0))
        hidden = true_count - visible
        if hidden > 0:
            fixed[i, -1] += hidden
    return fixed.astype(int)


def _year(path: Path) -> int:
    match = re.search(r"(20\d{2})", path.name)
    if not match:
        raise RuntimeError(f"Could not parse year from {path.name}")
    return int(match.group(1))


def _load_cohorts(data_dir: Path) -> tuple[list[int], dict[int, np.ndarray]]:
    transition_files = {_year(p): p for p in data_dir.glob("Transition-Matrix-*.csv")}
    default_files = {_year(p): p for p in data_dir.glob("Default-Rates-*.csv")}
    years = sorted(set(transition_files) & set(default_files))
    cohorts: dict[int, np.ndarray] = {}

    for year in years:
        counts = _parse_cerep_file(transition_files[year])
        counts = _correct_defaults(counts, _load_true_defaults(default_files[year]))
        cohorts[year] = counts

    if not years:
        raise RuntimeError("No matched transition/default-rate cohort files found.")
    return years, cohorts


def _average_matrix(cohorts: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    total_c = np.zeros((N_RATINGS, N_RATINGS), dtype=float)
    for counts in cohorts.values():
        total_c += counts

    p = np.zeros_like(total_c)
    for i in range(N_RATINGS):
        row_sum = float(total_c[i].sum())
        if row_sum > 0:
            p[i] = total_c[i] / row_sum
        else:
            p[i, i] = 1.0

    p[-1] = 0.0
    p[-1, -1] = 1.0
    return p, total_c


def _cumulative_default(p: np.ndarray, horizons: list[int]) -> dict[int, np.ndarray]:
    return {
        int(h): np.linalg.matrix_power(p, int(h))[:N_NON_DEFAULT, -1]
        for h in horizons
    }


def _markov_test(
    years: list[int],
    cohorts: dict[int, np.ndarray],
    alpha: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for i in range(N_NON_DEFAULT):
        table = np.stack([cohorts[y][i] for y in years], axis=0).astype(float)
        valid_cols = table.sum(axis=0) >= 5.0
        filtered = table[:, valid_cols]
        valid_rows = filtered.sum(axis=1) > 0.0
        filtered = filtered[valid_rows]

        n_rows, n_cols = filtered.shape
        if n_rows < 2 or n_cols < 2:
            stat, df, p_value = 0.0, 0, 1.0
        else:
            row_totals = filtered.sum(axis=1)
            col_totals = filtered.sum(axis=0)
            grand = float(filtered.sum())
            expected = np.outer(row_totals, col_totals) / grand
            mask = expected > 0.0
            stat = float(np.sum((filtered[mask] - expected[mask]) ** 2 / expected[mask]))
            df = int((n_rows - 1) * (n_cols - 1))
            p_value = float(1.0 - chi2.cdf(stat, df)) if df > 0 else 1.0

        rows.append(
            {
                "rating": RATINGS[i],
                "chi2_stat": stat,
                "df": df,
                "p_value": p_value,
                "reject_H0": bool(p_value < alpha),
            }
        )
    return rows


def _generator(p: np.ndarray) -> np.ndarray:
    q = la.logm(p).real
    for i in range(N_RATINGS):
        for j in range(N_RATINGS):
            if i != j and q[i, j] < 0:
                q[i, j] = 0.0
    for i in range(N_RATINGS):
        q[i, i] = 0.0
        q[i, i] = -q[i].sum()
    q[-1] = 0.0
    return q


def _reference_matrix(data_dir: Path) -> np.ndarray:
    path = data_dir / "historical_transition_matrix.csv"
    if not path.is_file():
        path = data_dir / "reference_transition_matrix.csv"

    df = pd.read_csv(path, index_col=0)
    if "NR" in df.columns:
        df = df.drop(columns=["NR"])

    p = df[RATINGS].values.astype(float) / 100.0
    for i in range(p.shape[0]):
        row_sum = float(p[i].sum())
        if row_sum > 0:
            p[i] /= row_sum
    p[-1] = 0.0
    p[-1, -1] = 1.0
    return p


def _cohort_default_rates(years: list[int], cohorts: dict[int, np.ndarray]) -> list[dict[str, object]]:
    rows = []
    for year in years:
        counts = cohorts[year]
        total_issuers = int(sum(counts[i].sum() for i in range(N_NON_DEFAULT)))
        total_defaults = int(sum(counts[i, -1] for i in range(N_NON_DEFAULT)))
        row: dict[str, object] = {
            "year": year,
            "total_issuers": total_issuers,
            "total_defaults": total_defaults,
            "default_rate": float(total_defaults / total_issuers if total_issuers else 0.0),
        }
        for i, rating in enumerate(RATINGS[:N_NON_DEFAULT]):
            row[f"{rating}_defaults"] = int(counts[i, -1])
        rows.append(row)
    return rows


def _solve(task_dir: Path, out_dir: Path) -> None:
    data_dir = _find_data_dir(task_dir)
    config = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
    horizons = [int(x) for x in config["projection_horizons"]]
    alpha = float(config["significance_level"])

    years, cohorts = _load_cohorts(data_dir)
    p_avg, total_c = _average_matrix(cohorts)
    cumulative = _cumulative_default(p_avg, horizons)
    markov = _markov_test(years, cohorts, alpha)
    q = _generator(p_avg)
    deviation = p_avg - _reference_matrix(data_dir)
    cohort_rates = _cohort_default_rates(years, cohorts)

    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "average_transition_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rating"] + RATINGS)
        for i in range(N_RATINGS):
            w.writerow([RATINGS[i]] + [f"{p_avg[i, j]:.6f}" for j in range(N_RATINGS)])

    with (out_dir / "cumulative_default_probs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rating"] + [f"horizon_{h}" for h in horizons])
        for i in range(N_NON_DEFAULT):
            w.writerow([RATINGS[i]] + [f"{cumulative[h][i]:.6f}" for h in horizons])

    with (out_dir / "markov_test.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rating", "chi2_stat", "df", "p_value", "reject_H0"])
        for row in markov:
            w.writerow(
                [
                    row["rating"],
                    f"{float(row['chi2_stat']):.6f}",
                    row["df"],
                    f"{float(row['p_value']):.6f}",
                    row["reject_H0"],
                ]
            )

    with (out_dir / "generator_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rating"] + RATINGS)
        for i in range(N_RATINGS):
            w.writerow([RATINGS[i]] + [f"{q[i, j]:.6f}" for j in range(N_RATINGS)])

    with (out_dir / "cohort_sizes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["year"] + RATINGS[:N_NON_DEFAULT])
        for year in years:
            counts = cohorts[year]
            w.writerow([year] + [int(counts[i].sum()) for i in range(N_NON_DEFAULT)])

    with (out_dir / "transition_matrix_deviation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rating"] + RATINGS)
        for i in range(N_NON_DEFAULT):
            w.writerow([RATINGS[i]] + [f"{deviation[i, j]:.6f}" for j in range(N_RATINGS)])

    rating_default_cols = [f"{r}_defaults" for r in RATINGS[:N_NON_DEFAULT]]
    with (out_dir / "cohort_default_rates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["year", "total_issuers", "total_defaults", "default_rate"] + rating_default_cols)
        for row in cohort_rates:
            w.writerow(
                [
                    row["year"],
                    row["total_issuers"],
                    row["total_defaults"],
                    f"{float(row['default_rate']):.6f}",
                ]
                + [row[col] for col in rating_default_cols]
            )

    pd_1y = cumulative[1]
    pd_5y = cumulative[5]
    offdiag_mask = ~np.eye(N_RATINGS, dtype=bool)
    max_idx = np.unravel_index(
        np.abs(deviation[:N_NON_DEFAULT]).argmax(),
        deviation[:N_NON_DEFAULT].shape,
    )
    peak = max(cohort_rates, key=lambda row: float(row["default_rate"]))
    trough = min(cohort_rates, key=lambda row: float(row["default_rate"]))

    summary = {
        "total_transitions": int(total_c[:N_NON_DEFAULT].sum()),
        "n_cohorts": len(years),
        "avg_default_rate_1y": float(pd_1y.mean()),
        "avg_default_rate_5y": float(pd_5y.mean()),
        "investment_grade_1y_pd": float(pd_1y[:4].mean()),
        "speculative_grade_1y_pd": float(pd_1y[4:].mean()),
        "n_markov_rejections": int(sum(bool(row["reject_H0"]) for row in markov)),
        "generator_max_offdiag": float(q[offdiag_mask].max()),
        "generator_min_diag": float(np.diag(q).min()),
        "max_absolute_deviation": float(np.abs(deviation[:N_NON_DEFAULT]).max()),
        "max_deviation_cell": f"{RATINGS[max_idx[0]]}->{RATINGS[max_idx[1]]}",
        "mean_absolute_deviation": float(np.abs(deviation[:N_NON_DEFAULT]).mean()),
        "peak_default_year": int(peak["year"]),
        "peak_default_rate": float(peak["default_rate"]),
        "lowest_default_year": int(trough["year"]),
        "lowest_default_rate": float(trough["default_rate"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class CreditMigrationMatrixSkill:
    name: str = "credit-migration-matrix-domain"

    def matches(self, *, instruction: str, task_dir: Path) -> bool:
        lowered = instruction.lower()
        file_hint = any(task_dir.rglob("Transition-Matrix-*.csv"))
        semantic = (
            ("transition matrix" in lowered or "rating migration" in lowered)
            and "default" in lowered
            and ("markov" in lowered or "generator" in lowered or "cumulative" in lowered)
        )
        return bool(file_hint or semantic)

    def solve(self, *, instruction: str, task_dir: Path, out_dir: Path, seed: int) -> None:
        del instruction, seed
        _solve(task_dir, out_dir)
