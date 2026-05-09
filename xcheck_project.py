#!/usr/bin/env python3
"""Run intraday-prices smoke checks.

The default checks are portable and use only files committed under samples/.
The local-data checks exercise the fuller Yahoo/Stooq workflow, but they depend
on user-specific data directories and are intended for the maintainer's machine
or similarly prepared environments.

This script does not fetch market data.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_FILES = [
    "market_constants.py",
    "io_intraday.py",
    "file_utils.py",
    "intraday_bars.py",
    "intraday_returns.py",
    "data_quality.py",
    "vendor_compare.py",
    "xcache_intraday.py",
    "xcheck_intraday.py",
    "xcheck_project.py",
    "xcompare_vendors.py",
    "xconsolidate.py",
    "xsplice_prices.py",
    "xread_prices.py",
    "xfetch_yahoo_intraday.py",
    "xfetch_databento_intraday.py",
]

SAMPLE_FILES = [
    ("samples/yahoo_intraday_sample.csv", "yahoo"),
    ("samples/stooq_intraday_sample.txt", "stooq"),
    ("samples/kibot_intraday_sample.txt", "kibot"),
    ("samples/polygon_intraday_sample.csv", "polygon"),
    ("samples/databento_intraday_sample.csv", "databento"),
    ("samples/quantquote_intraday_sample.csv", "quantquote"),
    ("samples/portara_intraday_sample.txt", "portara"),
    ("samples/first_rate_intraday_sample.csv", "first_rate"),
]

# These defaults match the maintainer's local Windows data layout. Override
# them when running in a different environment.
DEFAULT_TEST_ROOT = Path(r"C:\data\intraday_prices_test")
DEFAULT_CACHE_ROOT = Path(r"C:\data\intraday_cache\intraday_prices_test")
DEFAULT_YAHOO_ROOT = Path(r"C:\python\code\stocks\5_minute_prices")
DEFAULT_YAHOO_SAMPLE = DEFAULT_YAHOO_ROOT / "20260503"
DEFAULT_STOOQ_ROOT = Path(r"C:\data\stooq")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run intraday-prices smoke checks without fetching market data."
    )
    parser.add_argument(
        "--local-data",
        action="store_true",
        help=(
            "Run checks that depend on local Yahoo/Stooq data directories. "
            "By default only portable sample checks are run."
        ),
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=DEFAULT_TEST_ROOT,
        help=f"Directory for temporary test outputs. Defaults to {DEFAULT_TEST_ROOT}.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help=f"Directory for temporary cache outputs. Defaults to {DEFAULT_CACHE_ROOT}.",
    )
    parser.add_argument(
        "--yahoo-root",
        type=Path,
        default=DEFAULT_YAHOO_ROOT,
        help=f"Yahoo dated snapshot root for --local-data. Defaults to {DEFAULT_YAHOO_ROOT}.",
    )
    parser.add_argument(
        "--yahoo-sample",
        type=Path,
        default=DEFAULT_YAHOO_SAMPLE,
        help=f"Yahoo sample snapshot directory for --local-data. Defaults to {DEFAULT_YAHOO_SAMPLE}.",
    )
    parser.add_argument(
        "--stooq-root",
        type=Path,
        default=DEFAULT_STOOQ_ROOT,
        help=f"Stooq dated snapshot root for --local-data. Defaults to {DEFAULT_STOOQ_ROOT}.",
    )
    return parser.parse_args()


def run_step(title: str, cmd: list[str], cwd: Path) -> None:
    print()
    print(f"== {title} ==")
    print(" ".join(cmd))
    sys.stdout.flush()
    subprocess.run(cmd, cwd=cwd, check=True)


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise SystemExit(
            f"missing {description}: {path}\n"
            "Use portable sample checks without --local-data, or pass an override path."
        )


def remove_if_exists(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def run_compile(project_dir: Path) -> None:
    run_step(
        "Compile project Python files",
        [sys.executable, "-m", "py_compile", *PROJECT_FILES],
        project_dir,
    )


def run_sample_checks(project_dir: Path, test_root: Path) -> None:
    sample_summary = test_root / "sample_daily_summary.csv"
    remove_if_exists(sample_summary)

    for sample_file, vendor in SAMPLE_FILES:
        run_step(
            f"Read sample {vendor}",
            [
                sys.executable,
                "xcheck_intraday.py",
                sample_file,
                "--format",
                vendor,
                "--no-realized-vol",
                "--no-bars-per-day",
            ],
            project_dir,
        )

    run_step(
        "Sample daily summary export",
        [
            sys.executable,
            "xcheck_intraday.py",
            "samples/yahoo_intraday_sample.csv",
            "samples/databento_intraday_sample.csv",
            "--no-realized-vol",
            "--no-bars-per-day",
            "--show-bad-days",
            "5",
            "--output-daily-summary",
            str(sample_summary),
        ],
        project_dir,
    )
    require_path(sample_summary, "sample daily summary output")

    run_step(
        "Compare identical Yahoo sample",
        [
            sys.executable,
            "xcompare_vendors.py",
            "samples/yahoo_intraday_sample.csv",
            "samples/yahoo_intraday_sample.csv",
            "--left-format",
            "yahoo",
            "--right-format",
            "yahoo",
            "--return-horizons",
            "5",
            "--max-rows",
            "2",
        ],
        project_dir,
    )


def run_local_data_checks(args: argparse.Namespace, project_dir: Path) -> None:
    require_path(args.yahoo_sample, "Yahoo sample snapshot directory")
    require_path(args.yahoo_root, "Yahoo root directory")
    require_path(args.stooq_root, "Stooq root directory")

    test_root = args.test_root
    cache_root = args.cache_root
    test_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    run_step(
        "Yahoo read speed smoke test",
        [sys.executable, "xread_prices.py", "--input-dir", str(args.yahoo_sample), "--limit", "25"],
        project_dir,
    )

    run_step(
        "Yahoo data check smoke test",
        [
            sys.executable,
            "xcheck_intraday.py",
            "--input-dir",
            str(args.yahoo_sample),
            "--format",
            "yahoo",
            "--symbols",
            "SPY",
            "TLT",
            "HYG",
            "VXX",
            "--no-quality",
            "--no-bars-per-day",
        ],
        project_dir,
    )

    daily_summary = test_root / "daily_summary_yahoo.csv"
    run_step(
        "Yahoo daily summary and anomaly smoke test",
        [
            sys.executable,
            "xcheck_intraday.py",
            "--input-dir",
            str(args.yahoo_sample),
            "--format",
            "yahoo",
            "--symbols",
            "SPY",
            "TLT",
            "--show-bad-days",
            "5",
            "--output-daily-summary",
            str(daily_summary),
            "--no-realized-vol",
            "--no-bars-per-day",
        ],
        project_dir,
    )
    require_path(daily_summary, "Yahoo daily summary output")

    run_step(
        "Cache Yahoo sample files",
        [
            sys.executable,
            "xcache_intraday.py",
            "--input-dir",
            str(args.yahoo_sample),
            "--format",
            "yahoo",
            "--symbols",
            "SPY",
            "TLT",
            "--output-dir",
            str(cache_root),
            "--overwrite",
        ],
        project_dir,
    )

    run_step(
        "Check cached files",
        [sys.executable, "xcheck_intraday.py", str(cache_root / "*.parquet"), "--format", "auto"],
        project_dir,
    )

    consolidated_yahoo = test_root / "consolidated_yahoo_5min"
    run_step(
        "Consolidate Yahoo snapshots",
        [
            sys.executable,
            "xconsolidate.py",
            "--vendor",
            "yahoo",
            "--interval",
            "5min",
            "SPY",
            "TLT",
            "--root",
            str(args.yahoo_root),
            "--file-date-min",
            "20260426",
            "--file-date-max",
            "20260503",
            "--output-dir",
            str(consolidated_yahoo),
        ],
        project_dir,
    )

    consolidated_stooq = test_root / "consolidated_stooq_hourly"
    run_step(
        "Consolidate Stooq hourly snapshots",
        [
            sys.executable,
            "xconsolidate.py",
            "--vendor",
            "stooq",
            "--interval",
            "hourly",
            "SPY",
            "TLT",
            "--root",
            str(args.stooq_root),
            "--file-date-min",
            "20250303",
            "--file-date-max",
            "20250317",
            "--output-dir",
            str(consolidated_stooq),
        ],
        project_dir,
    )

    spliced_yahoo = test_root / "spliced_yahoo_5min"
    run_step(
        "Splice Yahoo snapshots with overlap adjustment",
        [
            sys.executable,
            "xsplice_prices.py",
            "--vendor",
            "yahoo",
            "--interval",
            "5min",
            "SPY",
            "TLT",
            "--root",
            str(args.yahoo_root),
            "--file-date-min",
            "20260426",
            "--file-date-max",
            "20260503",
            "--output-dir",
            str(spliced_yahoo),
        ],
        project_dir,
    )

    run_step(
        "Splice Yahoo snapshots without overlap adjustment",
        [
            sys.executable,
            "xsplice_prices.py",
            "--vendor",
            "yahoo",
            "--interval",
            "5min",
            "SPY",
            "TLT",
            "--root",
            str(args.yahoo_root),
            "--file-date-min",
            "20260426",
            "--file-date-max",
            "20260503",
            "--output-dir",
            str(test_root / "spliced_yahoo_5min_raw"),
            "--no-adjust-overlap",
        ],
        project_dir,
    )

    run_step(
        "Check consolidated Yahoo output",
        [
            sys.executable,
            "xcheck_intraday.py",
            str(consolidated_yahoo / "SPY.csv"),
            str(consolidated_yahoo / "TLT.csv"),
            "--format",
            "yahoo",
        ],
        project_dir,
    )

    run_step(
        "Check spliced Yahoo output",
        [
            sys.executable,
            "xcheck_intraday.py",
            str(spliced_yahoo / "SPY.csv"),
            str(spliced_yahoo / "TLT.csv"),
            "--format",
            "yahoo",
        ],
        project_dir,
    )

    run_step(
        "Check consolidated Stooq output",
        [
            sys.executable,
            "xcheck_intraday.py",
            str(consolidated_stooq / "spy.us.txt"),
            str(consolidated_stooq / "tlt.us.txt"),
            "--format",
            "stooq",
        ],
        project_dir,
    )

    compare_out = test_root / "vendor_compare_self"
    run_step(
        "Compare identical consolidated Yahoo files",
        [
            sys.executable,
            "xcompare_vendors.py",
            str(consolidated_yahoo / "SPY.csv"),
            str(consolidated_yahoo / "SPY.csv"),
            "--left-format",
            "yahoo",
            "--right-format",
            "yahoo",
            "--return-horizons",
            "5",
            "--max-rows",
            "2",
            "--output-dir",
            str(compare_out),
        ],
        project_dir,
    )
    require_path(compare_out / "SPY_bars.csv", "vendor comparison bars output")


def main() -> int:
    start = time.perf_counter()
    args = parse_args()
    project_dir = Path(__file__).resolve().parent

    run_compile(project_dir)
    run_sample_checks(project_dir, args.test_root)
    if args.local_data:
        run_local_data_checks(args, project_dir)

    print()
    print("All requested intraday-prices smoke checks completed successfully.")
    print(f"time elapsed (s): {time.perf_counter() - start:.2f}")
    if args.local_data:
        print("Test outputs were left in:")
        print(f"  {args.test_root}")
        print(f"  {args.cache_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
