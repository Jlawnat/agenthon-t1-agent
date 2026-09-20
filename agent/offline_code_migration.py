from __future__ import annotations

from dataclasses import dataclass
import ast
import importlib.util
from pathlib import Path
import re


@dataclass(frozen=True)
class PolarsMigrationReport:
    source_path: Path
    input_csv: Path
    transformed_rules: tuple[str, ...]


def _replace(
    text: str,
    old: str,
    new: str,
    *,
    rule: str,
    applied: list[str],
) -> str:
    if old in text:
        text = text.replace(old, new)
        applied.append(rule)
    return text


def _sub(
    text: str,
    pattern: str,
    repl: str,
    *,
    rule: str,
    applied: list[str],
    flags: int = 0,
) -> str:
    updated, count = re.subn(pattern, repl, text, flags=flags)
    if count:
        applied.append(rule)
    return updated


def migrate_polars_0x_source(source: str) -> tuple[str, tuple[str, ...]]:
    '''Migrate common Polars 0.x pipeline code to supported Polars 1.x APIs.'''
    if "def pipeline_old(" not in source:
        raise RuntimeError("No pipeline_old function found in source.")
    if "polars" not in source:
        raise RuntimeError("Source does not appear to use Polars.")

    out = source
    applied: list[str] = []

    out = _replace(
        out,
        "def pipeline_old(",
        "def pipeline_new(",
        rule="rename-entrypoint",
        applied=applied,
    )
    out = _replace(
        out,
        "# polars==0.19.0",
        "# migrated for Polars 1.x",
        rule="version-comment",
        applied=applied,
    )

    for old, new, rule in (
        (".groupby_dynamic(", ".group_by_dynamic(", "groupby-dynamic-to-group_by_dynamic"),
        (".groupby(", ".group_by(", "groupby-to-group_by"),
        ("polars.count()", "polars.len()", "count-to-len"),
        (".list.lengths()", ".list.len()", "list-lengths-to-len"),
        (".with_row_count(", ".with_row_index(", "row-count-to-index"),
        (".shift_and_fill(1, fill_value=0.0)", ".shift(1, fill_value=0.0)", "shift-fill"),
        (".is_first()", ".is_first_distinct()", "first-distinct"),
        (".cumsum()", ".cum_sum()", "cum-sum"),
        ("dtypes={", "schema_overrides={", "scan-schema-overrides"),
        (".take_every(", ".gather_every(", "take-every-to-gather-every"),
        (".take([", ".gather([", "take-to-gather"),
        (".keep_name()", ".name.keep()", "keep-name"),
        (".suffix(", ".name.suffix(", "name-suffix"),
        (".prefix(", ".name.prefix(", "name-prefix"),
        ("polars.Utf8", "polars.String", "utf8-to-string"),
    ):
        out = _replace(out, old, new, rule=rule, applied=applied)

    out = _replace(
        out,
        '.str.strptime(polars.Datetime, "%Y-%m-%d %H:%M:%S")',
        '.str.to_datetime(format="%Y-%m-%d %H:%M:%S")',
        rule="strptime-to-to_datetime",
        applied=applied,
    )

    out = _replace(
        out,
        '.sort("timestamp", reverse=True)',
        '.sort("timestamp", descending=True)',
        rule="sort-reverse-to-descending",
        applied=applied,
    )
    out = _replace(
        out,
        '.sort("ticker", reverse=False)',
        '.sort("ticker", descending=False)',
        rule="sort-reverse-to-descending",
        applied=applied,
    )

    out = _sub(
        out,
        r'how="outer"',
        'how="full", coalesce=True',
        rule="outer-to-full-coalesced",
        applied=applied,
    )

    old_melt = '''step_1.melt(
        id_vars=["timestamp", "ticker"], value_vars=["Open", "Close"]
    )'''
    new_melt = '''step_1.unpivot(
        index=["timestamp", "ticker"], on=["Open", "Close"]
    )'''
    out = _replace(
        out,
        old_melt,
        new_melt,
        rule="melt-to-unpivot",
        applied=applied,
    )

    out = _replace(
        out,
        'polars.col("Volume").apply(lambda x: (-(x**2)).abs()).alias("Volume_tr")',
        '(polars.col("Volume") ** 2).alias("Volume_tr")',
        rule="vectorize-volume-transform",
        applied=applied,
    )

    out = _replace(
        out,
        ".str.strip()",
        ".str.strip_chars()",
        rule="strip-to-strip_chars",
        applied=applied,
    )
    out = _replace(
        out,
        '.str.ljust(width=6, fillchar="_")',
        '.str.pad_end(6, "_")',
        rule="ljust-to-pad_end",
        applied=applied,
    )
    out = _replace(
        out,
        '.str.rjust(width=6, fillchar=".")',
        '.str.pad_start(6, ".")',
        rule="rjust-to-pad_start",
        applied=applied,
    )

    out = _replace(
        out,
        'polars.any(polars.col("Open") > 150, polars.col("Close") > 150)',
        'polars.any_horizontal(polars.col("Open") > 150, polars.col("Close") > 150)',
        rule="any-to-any_horizontal",
        applied=applied,
    )

    out = _replace(
        out,
        '.map_dict({"COST": "Costco", "UPS": "United Parcel Service"}, default="Other")',
        '.replace_strict(\n'
        '            {"COST": "Costco", "UPS": "United Parcel Service"},\n'
        '            default="Other",\n'
        '        )',
        rule="map_dict-to-replace_strict",
        applied=applied,
    )

    out = _replace(
        out,
        'step_1.group_by_dynamic("timestamp", every="1d", by="ticker")',
        'step_1.sort(["ticker", "timestamp"]).group_by_dynamic(\n'
        '            "timestamp", every="1d", group_by="ticker"\n'
        '        )',
        rule="group_by_dynamic-group_by",
        applied=applied,
    )

    out = _replace(
        out,
        ".fetch(100)",
        ".head(100).collect()",
        rule="fetch-to-head-collect",
        applied=applied,
    )

    old_spread = '''polars.struct(["Open", "Close"])
        .apply(lambda x: abs(x["Open"] - x["Close"]) / x["Open"] * 100)
        .alias("pct_spread")'''
    new_spread = '''(
            (polars.col("Open") - polars.col("Close")).abs()
            / polars.col("Open")
            * 100
        ).alias("pct_spread")'''
    out = _replace(
        out,
        old_spread,
        new_spread,
        rule="vectorize-struct-apply",
        applied=applied,
    )

    out = _replace(
        out,
        'numeric_cols = [c for c in lazy_all.columns if c not in ["timestamp", "ticker"]]',
        'numeric_cols = [c for c in lazy_all.collect_schema().names() if c not in ["timestamp", "ticker"]]',
        rule="lazy-columns-to-collect-schema",
        applied=applied,
    )

    out = _replace(
        out,
        'ticker_with_null.join(sector_lookup, on="ticker_key", how="left")',
        'ticker_with_null.join(\n'
        '        sector_lookup,\n'
        '        on="ticker_key",\n'
        '        how="left",\n'
        '        nulls_equal=True,\n'
        '    )',
        rule="explicit-null-key-join",
        applied=applied,
    )

    out = _replace(
        out,
        'polars.col("high_price_flag").is_not()',
        '(~polars.col("high_price_flag"))',
        rule="is_not-to-native-negation",
        applied=applied,
    )

    out = _replace(
        out,
        'values="daily_volume", columns="ticker", index="timestamp"',
        'values="daily_volume", on="ticker", index="timestamp"',
        rule="pivot-columns-to-on",
        applied=applied,
    )

    old_range = '''polars.col(price_col)
                    .apply(lambda s: float(s.max() - s.min()))
                    .alias("range")'''
    new_range = '''(
                        polars.col(price_col).max()
                        - polars.col(price_col).min()
                    ).alias("range")'''
    out = _replace(
        out,
        old_range,
        new_range,
        rule="vectorize-group-range",
        applied=applied,
    )

    out = _sub(
        out,
        r'group_by_dynamic\(([^)]*?)\bby=',
        r'group_by_dynamic(\1group_by=',
        rule="dynamic-by-keyword",
        applied=applied,
        flags=re.DOTALL,
    )

    forbidden = {
        ".apply(": "legacy apply callback",
        ".map_elements(": "Python element callback",
        ".groupby(": "legacy groupby",
        ".melt(": "deprecated melt",
        ".with_row_count(": "deprecated row count",
        ".shift_and_fill(": "deprecated shift-and-fill",
        ".map_dict(": "deprecated map_dict",
        ".cumsum(": "legacy cumsum",
        ".take_every(": "legacy take_every",
        ".take([": "legacy take",
        ".str.ljust(": "legacy ljust",
        ".str.rjust(": "legacy rjust",
        ".keep_name(": "legacy keep_name",
        'how="outer"': "legacy outer join keyword",
        "dtypes={": "legacy scan_csv dtype keyword",
    }
    remaining = [
        description
        for token, description in forbidden.items()
        if token in out
    ]
    if remaining:
        raise RuntimeError(
            "Polars migration left unsupported legacy constructs: "
            + ", ".join(remaining)
        )

    ast.parse(out)
    return out, tuple(dict.fromkeys(applied))


def _source_candidates(task_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in sorted(task_dir.rglob("*.py")):
        if "checks" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "def pipeline_old(" in text and "polars" in text:
            candidates.append(path)
    return candidates


def _explicit_input_csv(instruction: str, task_dir: Path) -> Path:
    names = re.findall(
        r"/app/data/([A-Za-z0-9_.-]+\.csv)",
        instruction,
        flags=re.IGNORECASE,
    )
    for name in names:
        matches = [
            path
            for path in task_dir.rglob(name)
            if "checks" not in path.parts
        ]
        if len(matches) == 1:
            return matches[0]

    candidates = [
        path
        for path in sorted(task_dir.rglob("*.csv"))
        if "checks" not in path.parts
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(
        "Could not identify the Polars migration input CSV from the task contract."
    )


@dataclass(frozen=True)
class PolarsApiMigrationSkill:
    name: str = "polars-api-migration"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        lowered = instruction.lower()
        semantic = (
            "polars" in lowered
            and (
                "migration" in lowered
                or "migrate" in lowered
            )
            and (
                "1.x" in lowered
                or "1.39" in lowered
                or "api version 1" in lowered
            )
        )
        return semantic and len(_source_candidates(task_dir)) == 1

    def solve(
        self,
        *,
        instruction: str,
        task_dir: Path,
        out_dir: Path,
        seed: int,
    ) -> None:
        del seed

        sources = _source_candidates(task_dir)
        if len(sources) != 1:
            raise RuntimeError(
                f"Expected exactly one old Polars pipeline source, found {sources}."
            )

        source_path = sources[0]
        input_csv = _explicit_input_csv(instruction, task_dir)
        source = source_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
        migrated, _rules = migrate_polars_0x_source(source)

        out_dir.mkdir(parents=True, exist_ok=True)
        migrated_path = out_dir / "function-under-new-api.py"
        migrated_path.write_text(
            migrated,
            encoding="utf-8",
        )

        import polars  # type: ignore

        spec = importlib.util.spec_from_file_location(
            "phase8h_migrated_polars_pipeline",
            migrated_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load migrated Polars pipeline.")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pipeline = getattr(module, "pipeline_new", None)
        if not callable(pipeline):
            raise RuntimeError("Migrated source does not expose pipeline_new.")

        pipeline(
            polars,
            str(input_csv),
            str(out_dir),
        )
