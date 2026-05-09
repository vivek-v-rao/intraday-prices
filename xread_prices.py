"""Benchmark reading intraday price CSV files."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from file_utils import expand_file_patterns


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Read price files and report basic counts.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(r"C:\python\code\stocks\5_minute_prices\20260503"),
        help="Directory containing price CSV files.",
    )
    parser.add_argument("--input-glob", default="*.csv", help="Input glob. Defaults to *.csv.")
    parser.add_argument("--limit", type=int, help="Limit number of files read.")
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Concatenate files into one DataFrame instead of only counting rows.",
    )
    return parser.parse_args()


def main() -> None:
    """Read the files and print benchmark statistics."""
    args = parse_args()
    files = expand_file_patterns([str(args.input_dir / args.input_glob)])
    if args.limit is not None:
        files = files[: args.limit]

    t0 = time.perf_counter()
    if args.combine:
        frames = []
        for path in files:
            df = pd.read_csv(path)
            df["symbol"] = path.stem
            frames.append(df)
        prices = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        rows = len(prices)
    else:
        rows = 0
        for path in files:
            rows += len(pd.read_csv(path))
    elapsed = time.perf_counter() - t0

    print(f"files: {len(files)}")
    print(f"rows: {rows}")
    print(f"avg rows/file: {rows / len(files) if files else 0}")
    print(f"elapsed seconds: {elapsed:.2f}")


if __name__ == "__main__":
    main()
