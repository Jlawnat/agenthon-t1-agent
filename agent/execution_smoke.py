from pathlib import Path


def main() -> None:
    output_dir = Path("output")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / "execution_smoke.txt"
    )

    output_file.write_text(
        "execution layer works\n",
        encoding="utf-8",
    )

    print("candidate executed successfully")


if __name__ == "__main__":
    main()