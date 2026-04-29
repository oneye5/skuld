from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from skuld_research.raw_data_analysis.dataset import load_analysis_dataset
from skuld_research.raw_data_analysis.pipeline import run_raw_data_analysis
from skuld_research.raw_data_analysis.report import write_raw_data_report


def _parse_run_date_folder(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("run-date must be an exact YYYY-MM-DD date") from exc

    if parsed.isoformat() != value:
        raise argparse.ArgumentTypeError("run-date must be an exact YYYY-MM-DD date")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Skuld raw data analysis workflow")
    parser.add_argument("--data", type=Path, required=True, help="Path to data_long.csv")
    parser.add_argument("--out", type=Path, required=True, help="Base output directory")
    parser.add_argument(
        "--run-date",
        type=_parse_run_date_folder,
        default=date.today().isoformat(),
        help="Run date folder name in YYYY-MM-DD format",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return 1 if exc.code is None else int(exc.code)

    if not args.data.exists():
        print(f"ERROR: data file not found: {args.data}", file=sys.stderr)
        return 1
    if not args.data.is_file():
        print(f"ERROR: data path is not a file: {args.data}", file=sys.stderr)
        return 1

    legend_path = args.data.with_name("source_legend.csv")
    if not legend_path.exists():
        print(f"ERROR: source legend file not found: {legend_path}", file=sys.stderr)
        return 1
    if not legend_path.is_file():
        print(f"ERROR: source legend path is not a file: {legend_path}", file=sys.stderr)
        return 1

    try:
        dataset = load_analysis_dataset(args.data)
        result = run_raw_data_analysis(dataset)
        out_dir = args.out / args.run_date
        report_path, summary_path = write_raw_data_report(result, out_dir)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Report written to {report_path}")
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
