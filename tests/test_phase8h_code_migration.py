from __future__ import annotations

from pathlib import Path

from agent.offline_code_migration import (
    PolarsApiMigrationSkill,
    migrate_polars_0x_source,
)


def test_migration_renames_common_polars_0x_apis() -> None:
    source = '''
def pipeline_old(polars, INPUT_FILE, OUTPUT_DIR):
    df = polars.read_csv(INPUT_FILE)
    out = df.groupby("ticker").agg(polars.count().alias("n"))
    out = out.with_row_count("row_nr")
    out = out.with_columns(polars.col("x").cumsum().alias("cs"))
    out.write_csv(f"{OUTPUT_DIR}/x.csv")
'''
    migrated, rules = migrate_polars_0x_source(source)
    assert "def pipeline_new(" in migrated
    assert ".group_by(" in migrated
    assert "polars.len()" in migrated
    assert ".with_row_index(" in migrated
    assert ".cum_sum()" in migrated
    assert ".groupby(" not in migrated
    assert ".with_row_count(" not in migrated
    assert rules


def test_migration_vectorizes_known_apply_patterns() -> None:
    source = '''
def pipeline_old(polars, INPUT_FILE, OUTPUT_DIR):
    df = polars.read_csv(INPUT_FILE)
    step_6 = df.with_columns(
        polars.col("Volume").apply(lambda x: (-(x**2)).abs()).alias("Volume_tr")
    )
    step_15 = df.with_columns(
        polars.struct(["Open", "Close"])
        .apply(lambda x: abs(x["Open"] - x["Close"]) / x["Open"] * 100)
        .alias("pct_spread")
    )
'''
    migrated, _ = migrate_polars_0x_source(source)
    assert ".apply(" not in migrated
    assert '(polars.col("Volume") ** 2)' in migrated
    assert '(polars.col("Open") - polars.col("Close")).abs()' in migrated


def test_migration_handles_join_and_string_namespace_changes() -> None:
    source = '''
def pipeline_old(polars, INPUT_FILE, OUTPUT_DIR):
    df = polars.read_csv(INPUT_FILE)
    x = df.join(df, on="ticker", how="outer")
    x = x.with_columns(polars.col("ticker").str.strip())
    x = x.with_columns(polars.col("ticker").str.ljust(width=6, fillchar="_"))
    x = x.with_columns(polars.col("ticker").str.rjust(width=6, fillchar="."))
'''
    migrated, _ = migrate_polars_0x_source(source)
    assert 'how="full", coalesce=True' in migrated
    assert ".str.strip_chars()" in migrated
    assert '.str.pad_end(6, "_")' in migrated
    assert '.str.pad_start(6, ".")' in migrated


def test_skill_matches_semantics_plus_old_pipeline_source(tmp_path: Path) -> None:
    task = tmp_path / "task"
    data = task / "environment" / "data"
    data.mkdir(parents=True)
    (data / "legacy.py").write_text(
        "def pipeline_old(polars, INPUT_FILE, OUTPUT_DIR):\n"
        "    return polars.read_csv(INPUT_FILE)\n",
        encoding="utf-8",
    )
    (data / "pilot.csv").write_text(
        "timestamp,ticker,x\n2026-01-01,A,1\n",
        encoding="utf-8",
    )
    skill = PolarsApiMigrationSkill()
    assert skill.matches(
        instruction=(
            "Migrate this Polars API version 0.x pipeline to Polars 1.x "
            "and write /app/output/function-under-new-api.py."
        ),
        task_dir=task,
    )


def test_migration_handles_groupby_dynamic_and_by_keyword() -> None:
    source = '''
def pipeline_old(polars, INPUT_FILE, OUTPUT_DIR):
    df = polars.read_csv(INPUT_FILE)
    out = (
        df.groupby_dynamic(
            "timestamp",
            every="1d",
            by="ticker",
        )
        .agg(polars.col("Volume").sum())
    )
'''
    migrated, _ = migrate_polars_0x_source(source)

    assert ".group_by_dynamic(" in migrated
    assert ".groupby_dynamic(" not in migrated
    assert 'group_by="ticker"' in migrated
