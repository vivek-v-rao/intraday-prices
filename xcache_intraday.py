"""Build a normalized cache from intraday vendor text/CSV files."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from file_utils import expand_file_patterns
from intraday_returns import asset_label, infer_bar_interval_minutes
from io_intraday import INTRADAY_VENDORS, read_intraday_prices


CANONICAL_COLUMNS = [
    "Datetime",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Dividends",
    "Stock Splits",
    "Capital Gains",
    "date",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Normalize intraday files and write one cache file per symbol."
    )
    parser.add_argument("prices_files", nargs="*", help="Input files or glob patterns.")
    parser.add_argument("--input-dir", type=Path, help="Directory containing input files.")
    parser.add_argument("--symbols", nargs="+", help="Symbols to read from --input-dir.")
    parser.add_argument(
        "--symbol-file-template",
        default="{symbol}.csv",
        help="Filename template used with --input-dir/--symbols. Defaults to {symbol}.csv.",
    )
    parser.add_argument(
        "--input-glob",
        default="*.csv",
        help="Glob used inside --input-dir when --symbols is omitted. Defaults to *.csv.",
    )
    parser.add_argument("--limit", type=int, help="Limit the number of resolved input files.")
    parser.add_argument(
        "--format",
        choices=INTRADAY_VENDORS,
        default="auto",
        help="Input file format/vendor. Defaults to auto.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Cache output directory.")
    parser.add_argument(
        "--compression",
        default="snappy",
        help="Parquet compression codec. Defaults to snappy. Ignored with --pickle.",
    )
    parser.add_argument(
        "--pickle",
        action="store_true",
        help="Write pandas pickle files (.pkl) instead of Parquet files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cache files.",
    )
    return parser.parse_args()


def symbol_price_files(input_dir: Path, symbols: list[str], template: str) -> list[Path]:
    """Return price file paths built from an input directory and symbols."""
    files = []
    for symbol in symbols:
        filename = template.format(symbol=symbol, SYMBOL=symbol.upper(), lower=symbol.lower())
        files.append(input_dir / filename)
    return files


def apply_file_limit(files: list[Path], limit: int | None) -> list[Path]:
    """Return at most limit files after validating the limit."""
    if limit is None:
        return files
    if limit < 1:
        raise ValueError("--limit must be at least 1")
    return files[:limit]


def resolve_price_files(args: argparse.Namespace) -> list[Path]:
    """Resolve positional file patterns and --input-dir/--symbols into paths."""
    if args.input_dir is not None or args.symbols is not None:
        if args.input_dir is None:
            raise ValueError("--symbols requires --input-dir")
        if args.prices_files:
            raise ValueError("use either positional prices_files or --input-dir/--symbols, not both")
        if args.symbols:
            files = symbol_price_files(args.input_dir, args.symbols, args.symbol_file_template)
            missing = [path for path in files if not path.is_file()]
            if missing:
                missing_text = "\n".join(str(path) for path in missing)
                raise FileNotFoundError(f"file not found:\n{missing_text}")
            return apply_file_limit([path.resolve() for path in files], args.limit)
        return apply_file_limit(
            expand_file_patterns([str(args.input_dir / args.input_glob)]),
            args.limit,
        )
    if not args.prices_files:
        raise ValueError("provide prices_files or use --input-dir")
    return apply_file_limit(expand_file_patterns(args.prices_files), args.limit)


def output_path(output_dir: Path, symbol: str, use_pickle: bool) -> Path:
    """Return the cache path for one symbol."""
    suffix = ".pkl" if use_pickle else ".parquet"
    return output_dir / f"{symbol}{suffix}"


def cache_one_file(
    source_path: Path,
    output_dir: Path,
    vendor: str,
    compression: str,
    use_pickle: bool,
    overwrite: bool,
) -> dict[str, object]:
    """Normalize one source file and write its cache file."""
    symbol = asset_label(source_path)
    out_path = output_path(output_dir, symbol, use_pickle)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists; pass --overwrite to replace it")

    prices = read_intraday_prices(source_path, vendor=vendor)
    prices = prices[CANONICAL_COLUMNS].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    if use_pickle:
        prices.to_pickle(out_path)
    else:
        prices.to_parquet(out_path, index=False, compression=compression)

    try:
        bar_minutes = infer_bar_interval_minutes(prices)
    except ValueError:
        bar_minutes = pd.NA

    return {
        "symbol": symbol,
        "source_file": str(source_path),
        "cache_file": str(out_path),
        "vendor": vendor,
        "cache_format": "pickle" if use_pickle else "parquet",
        "rows": len(prices),
        "days": prices["date"].nunique(),
        "bar_minutes": bar_minutes,
        "first_dt": prices["Datetime"].min(),
        "last_dt": prices["Datetime"].max(),
    }


def main() -> None:
    """Build the cache and manifest."""
    start = time.perf_counter()
    args = parse_args()
    files = resolve_price_files(args)

    rows = []
    for path in files:
        rows.append(
            cache_one_file(
                path,
                args.output_dir,
                args.format,
                args.compression,
                args.pickle,
                args.overwrite,
            )
        )

    manifest = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    elapsed = time.perf_counter() - start

    print(manifest.to_string(index=False))
    print()
    print(f"symbols cached: {len(manifest)}")
    print(f"avg obs per symbol: {manifest['rows'].mean():.1f}" if not manifest.empty else "avg obs per symbol: 0.0")
    print(f"manifest: {manifest_path}")
    print(f"time elapsed (s): {elapsed:.2f}")


if __name__ == "__main__":
    main()
