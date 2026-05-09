#!/usr/bin/env python3
"""
Splice intraday price snapshots from multiple vendors.

Supported layouts:

* Stooq: dated subdirectories containing zip snapshots such as 5_us_txt.zip
  or h_us_txt.zip. Symbol files are found inside the zip by basename.
* Yahoo Finance: dated subdirectories containing one CSV per symbol, such as
  20260503/SPY.csv.

Stooq snapshots are consolidated as-is. Yahoo Finance snapshots can be spliced
with overlap-based price rebasing so older rolling histories are put on the
same adjustment basis as newer snapshots before older non-overlapping bars are
added.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path


DATE_RE = re.compile(r"^\d{8}$")
STOOQ_INTERVAL_DEFAULTS = {
    "5min": ("5_us_txt.zip", "consolidated_5min"),
    "hourly": ("h_us_txt.zip", "consolidated_hourly"),
}
YAHOO_INTERVAL_DEFAULTS = {
    "1min": "consolidated_yahoo_1min",
    "5min": "consolidated_yahoo_5min",
}


@dataclass(frozen=True)
class Snapshot:
    file_date: str
    path: Path


@dataclass
class SymbolStats:
    symbol_file: str
    snapshots_seen: int = 0
    snapshots_missing: int = 0
    input_rows: int = 0
    duplicate_bars: int = 0
    changed_duplicate_bars: int = 0
    output_rows: int = 0
    first_bar: str = ""
    last_bar: str = ""
    first_snapshot: str = ""
    last_snapshot: str = ""
    first_input_file: str = ""
    last_input_file: str = ""
    adjusted_snapshots: int = 0
    min_overlap_bars: int = 0
    min_adjustment_ratio: str = ""
    max_adjustment_ratio: str = ""
    max_ratio_mad: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Splice selected symbols from dated intraday price snapshots."
    )
    parser.add_argument(
        "symbols",
        nargs="*",
        help="Symbols or file names to consolidate, for example SPY TLT or SPY.csv.",
    )
    parser.add_argument(
        "--vendor",
        choices=["stooq", "yahoo"],
        required=True,
        help="Input vendor/source layout.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory containing dated subdirectories.",
    )
    parser.add_argument(
        "--interval",
        choices=["1min", "5min", "hourly"],
        default="5min",
        help="Data interval. For Stooq, supported values are 5min and hourly.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for consolidated files. Defaults depend on vendor and interval.",
    )
    parser.add_argument(
        "--file-date-min",
        help="Minimum dated subdirectory to process, inclusive, in YYYYMMDD format.",
    )
    parser.add_argument(
        "--file-date-max",
        help="Maximum dated subdirectory to process, inclusive, in YYYYMMDD format.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Consolidate all symbol files found in the selected snapshots.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of selected symbols. Useful with --all for dry runs.",
    )
    parser.add_argument(
        "--archive-glob",
        help="Stooq zip glob inside each dated subdirectory. Defaults from --interval.",
    )
    parser.add_argument(
        "--input-glob",
        default="*.csv",
        help="Yahoo CSV glob inside each dated subdirectory. Defaults to *.csv.",
    )
    parser.add_argument(
        "--symbol-file-template",
        default="{symbol}.csv",
        help="Yahoo filename template used for explicit symbols. Defaults to {symbol}.csv.",
    )
    parser.add_argument(
        "--no-adjust-overlap",
        action="store_true",
        help="For Yahoo, splice raw rows without rebasing older snapshots to newer overlaps.",
    )
    parser.add_argument(
        "--min-overlap-bars",
        type=int,
        default=100,
        help="Minimum Yahoo overlap bars required to estimate an adjustment ratio.",
    )
    parser.add_argument(
        "--max-ratio-mad",
        type=float,
        default=0.001,
        help=(
            "Maximum median absolute relative deviation of overlap price ratios. "
            "Defaults to 0.001."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be processed without writing consolidated files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    validate_date_arg("--file-date-min", args.file_date_min)
    validate_date_arg("--file-date-max", args.file_date_max)
    if (
        args.file_date_min is not None
        and args.file_date_max is not None
        and args.file_date_min > args.file_date_max
    ):
        raise SystemExit("--file-date-min cannot be greater than --file-date-max")
    if args.vendor == "stooq" and args.interval not in STOOQ_INTERVAL_DEFAULTS:
        raise SystemExit("Stooq supports --interval 5min or --interval hourly")
    if args.vendor == "yahoo" and args.interval not in YAHOO_INTERVAL_DEFAULTS:
        raise SystemExit("Yahoo supports --interval 1min or --interval 5min")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.min_overlap_bars < 1:
        raise SystemExit("--min-overlap-bars must be positive")
    if args.max_ratio_mad < 0:
        raise SystemExit("--max-ratio-mad cannot be negative")
    if not args.symbols and not args.all:
        raise SystemExit("provide symbols or use --all")
    if args.symbols and args.all:
        raise SystemExit("use explicit symbols or --all, not both")


def validate_date_arg(name: str, value: str | None) -> None:
    if value is not None and not DATE_RE.match(value):
        raise SystemExit(f"{name} must be in YYYYMMDD format")


def selected_dated_dirs(
    root: Path,
    file_date_min: str | None,
    file_date_max: str | None,
) -> list[Path]:
    dated_dirs: list[Path] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        file_date = path.name
        if not DATE_RE.match(file_date):
            continue
        if file_date_min is not None and file_date < file_date_min:
            continue
        if file_date_max is not None and file_date > file_date_max:
            continue
        dated_dirs.append(path)
    return sorted(dated_dirs, key=lambda path: path.name)


def apply_limit(symbol_files: list[str], limit: int | None) -> list[str]:
    if limit is None:
        return symbol_files
    return symbol_files[:limit]


def normalize_stooq_symbol(symbol: str) -> str:
    value = Path(symbol.strip().lower()).name
    if value.endswith(".txt"):
        return value
    if "." not in value:
        return f"{value}.us.txt"
    return f"{value}.txt"


def normalize_yahoo_symbol(symbol: str, template: str) -> str:
    value = Path(symbol.strip()).name
    if value.lower().endswith(".csv"):
        return value
    return template.format(symbol=value.upper())


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    if args.vendor == "stooq":
        return Path(STOOQ_INTERVAL_DEFAULTS[args.interval][1])
    return Path(YAHOO_INTERVAL_DEFAULTS[args.interval])


def stooq_archive_glob(args: argparse.Namespace) -> str:
    return args.archive_glob or STOOQ_INTERVAL_DEFAULTS[args.interval][0]


def find_stooq_snapshots(args: argparse.Namespace) -> list[Snapshot]:
    archive_glob = stooq_archive_glob(args)
    snapshots: list[Snapshot] = []
    for dated_dir in selected_dated_dirs(args.root, args.file_date_min, args.file_date_max):
        for path in sorted(dated_dir.glob(archive_glob)):
            if path.is_file():
                snapshots.append(Snapshot(file_date=dated_dir.name, path=path))
    if not snapshots:
        raise SystemExit(f"No matching Stooq zip files found for archive glob {archive_glob!r}.")
    return snapshots


def find_yahoo_snapshots(args: argparse.Namespace) -> list[Snapshot]:
    snapshots = [
        Snapshot(file_date=dated_dir.name, path=dated_dir)
        for dated_dir in selected_dated_dirs(args.root, args.file_date_min, args.file_date_max)
    ]
    snapshots = [snapshot for snapshot in snapshots if any(snapshot.path.glob(args.input_glob))]
    if not snapshots:
        raise SystemExit(f"No Yahoo dated directories found with files matching {args.input_glob!r}.")
    return snapshots


def find_zip_member(zip_file: zipfile.ZipFile, symbol_file: str) -> str | None:
    matches = [
        name
        for name in zip_file.namelist()
        if not name.endswith("/") and Path(name).name.lower() == symbol_file
    ]
    if len(matches) > 1:
        joined = ", ".join(matches)
        raise ValueError(f"{symbol_file} is ambiguous in {zip_file.filename}: {joined}")
    return matches[0] if matches else None


def read_stooq_rows(zip_path: Path, member_name: str) -> tuple[list[str], list[dict[str, str]]]:
    with zipfile.ZipFile(zip_path) as zip_file:
        text = zip_file.read(member_name).decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError(f"{zip_path}: {member_name} has no header")
    for column in ["<DATE>", "<TIME>"]:
        if column not in reader.fieldnames:
            raise ValueError(f"{zip_path}: {member_name} has no {column} column")
    return list(reader.fieldnames), list(reader)


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header")
        return list(reader.fieldnames), list(reader)


def stooq_bar_key(row: dict[str, str]) -> tuple[str, str]:
    return row["<DATE>"], row["<TIME>"]


def yahoo_bar_key(row: dict[str, str]) -> str:
    if "Datetime" not in row:
        raise ValueError('Yahoo CSV has no "Datetime" column')
    return row["Datetime"]


def list_stooq_symbols(snapshots: list[Snapshot]) -> list[str]:
    symbols: set[str] = set()
    for snapshot in snapshots:
        with zipfile.ZipFile(snapshot.path) as zip_file:
            symbols.update(
                Path(name).name.lower()
                for name in zip_file.namelist()
                if not name.endswith("/") and Path(name).suffix.lower() == ".txt"
            )
    return sorted(symbols)


def list_yahoo_symbols(snapshots: list[Snapshot], input_glob: str) -> list[str]:
    symbols: dict[str, str] = {}
    for snapshot in snapshots:
        for path in sorted(snapshot.path.glob(input_glob), key=lambda item: item.name.lower()):
            key = path.name.lower()
            symbols.setdefault(key, path.name)
    return [symbols[key] for key in sorted(symbols)]


def write_rows(
    output_path: Path,
    fieldnames: list[str],
    rows_by_key: dict[object, dict[str, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(output_dir: Path, stats: list[SymbolStats]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "symbol_file",
        "snapshots_seen",
        "snapshots_missing",
        "input_rows",
        "duplicate_bars",
        "changed_duplicate_bars",
        "output_rows",
        "first_bar",
        "last_bar",
        "first_snapshot",
        "last_snapshot",
        "first_input_file",
        "last_input_file",
        "adjusted_snapshots",
        "min_overlap_bars",
        "min_adjustment_ratio",
        "max_adjustment_ratio",
        "max_ratio_mad",
    ]
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for item in stats:
            writer.writerow(item.__dict__)


def update_rows(
    stats: SymbolStats,
    rows_by_key: dict[object, dict[str, str]],
    rows: list[dict[str, str]],
    key_func,
) -> None:
    stats.input_rows += len(rows)
    for row in rows:
        key = key_func(row)
        old_row = rows_by_key.get(key)
        if old_row is not None:
            stats.duplicate_bars += 1
            if old_row != row:
                stats.changed_duplicate_bars += 1
        rows_by_key[key] = row


def add_new_rows_only(
    stats: SymbolStats,
    rows_by_key: dict[object, dict[str, str]],
    rows: list[dict[str, str]],
    key_func,
) -> None:
    stats.input_rows += len(rows)
    for row in rows:
        key = key_func(row)
        old_row = rows_by_key.get(key)
        if old_row is not None:
            stats.duplicate_bars += 1
            if old_row != row:
                stats.changed_duplicate_bars += 1
            continue
        rows_by_key[key] = row


def finalize_stats(stats: SymbolStats, rows_by_key: dict[object, dict[str, str]]) -> None:
    keys = sorted(rows_by_key)
    stats.output_rows = len(keys)
    if keys:
        first = keys[0]
        last = keys[-1]
        stats.first_bar = " ".join(first) if isinstance(first, tuple) else str(first)
        stats.last_bar = " ".join(last) if isinstance(last, tuple) else str(last)


def consolidate_stooq_symbol(
    symbol_file: str, snapshots: list[Snapshot]
) -> tuple[SymbolStats, list[str], dict[object, dict[str, str]]]:
    stats = SymbolStats(symbol_file=symbol_file)
    rows_by_key: dict[object, dict[str, str]] = {}
    fieldnames: list[str] | None = None

    for snapshot in snapshots:
        with zipfile.ZipFile(snapshot.path) as zip_file:
            member_name = find_zip_member(zip_file, symbol_file)
        if member_name is None:
            stats.snapshots_missing += 1
            continue

        current_fieldnames, rows = read_stooq_rows(snapshot.path, member_name)
        if fieldnames is None:
            fieldnames = current_fieldnames
        elif current_fieldnames != fieldnames:
            raise ValueError(f"{snapshot.path}: {member_name} header differs for {symbol_file}")

        stats.snapshots_seen += 1
        stats.first_snapshot = stats.first_snapshot or snapshot.file_date
        stats.last_snapshot = snapshot.file_date
        stats.first_input_file = stats.first_input_file or f"{snapshot.path}!{member_name}"
        stats.last_input_file = f"{snapshot.path}!{member_name}"
        update_rows(stats, rows_by_key, rows, stooq_bar_key)

    if fieldnames is None:
        raise ValueError(f"{symbol_file} was not found in any selected Stooq archive")
    finalize_stats(stats, rows_by_key)
    return stats, fieldnames, rows_by_key


def consolidate_yahoo_symbol(
    symbol_file: str, snapshots: list[Snapshot]
) -> tuple[SymbolStats, list[str], dict[object, dict[str, str]]]:
    stats = SymbolStats(symbol_file=symbol_file)
    rows_by_key: dict[object, dict[str, str]] = {}
    fieldnames: list[str] | None = None

    for snapshot in snapshots:
        path = snapshot.path / symbol_file
        if not path.is_file():
            stats.snapshots_missing += 1
            continue

        current_fieldnames, rows = read_csv_rows(path)
        if "Datetime" not in current_fieldnames:
            raise ValueError(f'{path} has no "Datetime" column')
        if fieldnames is None:
            fieldnames = current_fieldnames
        elif current_fieldnames != fieldnames:
            raise ValueError(f"{path} header differs for {symbol_file}")

        stats.snapshots_seen += 1
        stats.first_snapshot = stats.first_snapshot or snapshot.file_date
        stats.last_snapshot = snapshot.file_date
        stats.first_input_file = stats.first_input_file or str(path)
        stats.last_input_file = str(path)
        update_rows(stats, rows_by_key, rows, yahoo_bar_key)

    if fieldnames is None:
        raise ValueError(f"{symbol_file} was not found in any selected Yahoo directory")
    finalize_stats(stats, rows_by_key)
    return stats, fieldnames, rows_by_key


PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close"]


def parse_positive_float(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number


def median_value(values) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("cannot compute median of an empty sequence")
    middle = n // 2
    if n % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def format_adjusted_value(value: str, ratio: float) -> str:
    number = parse_positive_float(value)
    if number is None:
        return value
    return f"{number * ratio:.10g}"


def estimate_overlap_ratio(
    symbol_file: str,
    snapshot: Snapshot,
    rows: list[dict[str, str]],
    rows_by_key: dict[object, dict[str, str]],
    min_overlap_bars: int,
    max_ratio_mad: float,
) -> tuple[float, int, float]:
    ratios: list[float] = []
    for row in rows:
        key = yahoo_bar_key(row)
        newer_row = rows_by_key.get(key)
        if newer_row is None:
            continue
        old_close = parse_positive_float(row.get("Close", ""))
        new_close = parse_positive_float(newer_row.get("Close", ""))
        if old_close is not None and new_close is not None:
            ratios.append(new_close / old_close)

    overlap_bars = len(ratios)
    if overlap_bars < min_overlap_bars:
        raise ValueError(
            f"{symbol_file}: {snapshot.file_date} has only {overlap_bars} usable overlap bars; "
            f"need at least {min_overlap_bars}"
        )

    ratio = median_value(ratios)
    ratio_mad = median_value(abs(item / ratio - 1.0) for item in ratios)
    if ratio_mad > max_ratio_mad:
        raise ValueError(
            f"{symbol_file}: {snapshot.file_date} overlap ratios are unstable "
            f"(MAD={ratio_mad:.6g}, limit={max_ratio_mad:.6g})"
        )
    return ratio, overlap_bars, ratio_mad


def adjust_price_rows(rows: list[dict[str, str]], ratio: float) -> list[dict[str, str]]:
    adjusted_rows: list[dict[str, str]] = []
    for row in rows:
        adjusted = row.copy()
        for column in PRICE_COLUMNS:
            if column in adjusted:
                adjusted[column] = format_adjusted_value(adjusted[column], ratio)
        adjusted_rows.append(adjusted)
    return adjusted_rows


def consolidate_yahoo_symbol_spliced(
    symbol_file: str,
    snapshots: list[Snapshot],
    min_overlap_bars: int,
    max_ratio_mad: float,
) -> tuple[SymbolStats, list[str], dict[object, dict[str, str]]]:
    stats = SymbolStats(symbol_file=symbol_file)
    rows_by_key: dict[object, dict[str, str]] = {}
    fieldnames: list[str] | None = None
    overlap_counts: list[int] = []
    adjustment_ratios: list[float] = []
    ratio_mads: list[float] = []

    for snapshot in reversed(snapshots):
        path = snapshot.path / symbol_file
        if not path.is_file():
            stats.snapshots_missing += 1
            continue

        current_fieldnames, rows = read_csv_rows(path)
        if "Datetime" not in current_fieldnames:
            raise ValueError(f'{path} has no "Datetime" column')
        if fieldnames is None:
            fieldnames = current_fieldnames
        elif current_fieldnames != fieldnames:
            raise ValueError(f"{path} header differs for {symbol_file}")

        stats.snapshots_seen += 1
        stats.first_input_file = stats.first_input_file or str(path)
        stats.last_input_file = str(path)

        if rows_by_key:
            ratio, overlap_bars, ratio_mad = estimate_overlap_ratio(
                symbol_file,
                snapshot,
                rows,
                rows_by_key,
                min_overlap_bars,
                max_ratio_mad,
            )
            rows = adjust_price_rows(rows, ratio)
            stats.adjusted_snapshots += 1
            overlap_counts.append(overlap_bars)
            adjustment_ratios.append(ratio)
            ratio_mads.append(ratio_mad)

        add_new_rows_only(stats, rows_by_key, rows, yahoo_bar_key)

    if fieldnames is None:
        raise ValueError(f"{symbol_file} was not found in any selected Yahoo directory")

    seen_dates = sorted(
        snapshot.file_date
        for snapshot in snapshots
        if (snapshot.path / symbol_file).is_file()
    )
    if seen_dates:
        stats.first_snapshot = seen_dates[0]
        stats.last_snapshot = seen_dates[-1]
    if overlap_counts:
        stats.min_overlap_bars = min(overlap_counts)
    if adjustment_ratios:
        stats.min_adjustment_ratio = f"{min(adjustment_ratios):.10g}"
        stats.max_adjustment_ratio = f"{max(adjustment_ratios):.10g}"
    if ratio_mads:
        stats.max_ratio_mad = f"{max(ratio_mads):.6g}"

    finalize_stats(stats, rows_by_key)
    return stats, fieldnames, rows_by_key


def run_stooq(args: argparse.Namespace, output_dir: Path) -> None:
    snapshots = find_stooq_snapshots(args)
    symbol_files = (
        list_stooq_symbols(snapshots)
        if args.all
        else [normalize_stooq_symbol(symbol) for symbol in args.symbols]
    )
    symbol_files = apply_limit(symbol_files, args.limit)

    print(
        f"Snapshots selected: {len(snapshots)} "
        f"({snapshots[0].file_date} through {snapshots[-1].file_date})"
    )
    print(f"Vendor: stooq")
    print(f"Archive glob: {stooq_archive_glob(args)}")
    write_outputs(args, output_dir, symbol_files, snapshots, consolidate_stooq_symbol)


def run_yahoo(args: argparse.Namespace, output_dir: Path) -> None:
    snapshots = find_yahoo_snapshots(args)
    symbol_files = (
        list_yahoo_symbols(snapshots, args.input_glob)
        if args.all
        else [normalize_yahoo_symbol(symbol, args.symbol_file_template) for symbol in args.symbols]
    )
    symbol_files = apply_limit(symbol_files, args.limit)

    print(
        f"Snapshots selected: {len(snapshots)} "
        f"({snapshots[0].file_date} through {snapshots[-1].file_date})"
    )
    print(f"Vendor: yahoo")
    print(f"Input glob: {args.input_glob}")
    if args.no_adjust_overlap:
        print("Overlap adjustment: off")
        consolidate_func = consolidate_yahoo_symbol
    else:
        print(
            "Overlap adjustment: on "
            f"(min_overlap_bars={args.min_overlap_bars}, max_ratio_mad={args.max_ratio_mad})"
        )

        def consolidate_func(symbol_file: str, selected_snapshots: list[Snapshot]):
            return consolidate_yahoo_symbol_spliced(
                symbol_file,
                selected_snapshots,
                args.min_overlap_bars,
                args.max_ratio_mad,
            )

    write_outputs(args, output_dir, symbol_files, snapshots, consolidate_func)


def write_outputs(
    args: argparse.Namespace,
    output_dir: Path,
    symbol_files: list[str],
    snapshots: list[Snapshot],
    consolidate_func,
) -> None:
    print(f"Symbols selected: {len(symbol_files)}")
    if symbol_files:
        print(f"First symbols: {', '.join(symbol_files[:10])}")
    if args.dry_run:
        print(f"Dry run: would write consolidated files to {output_dir}")
    else:
        print(f"Writing consolidated files to {output_dir}")

    all_stats: list[SymbolStats] = []
    for symbol_file in symbol_files:
        stats, fieldnames, rows_by_key = consolidate_func(symbol_file, snapshots)
        all_stats.append(stats)
        if not args.dry_run:
            write_rows(output_dir / symbol_file, fieldnames, rows_by_key)
        print(
            f"{symbol_file}: output_rows={stats.output_rows} "
            f"first={stats.first_bar} last={stats.last_bar} "
            f"changed_duplicates={stats.changed_duplicate_bars}"
        )

    if not args.dry_run:
        write_manifest(output_dir, all_stats)
        print(f"Manifest written to {output_dir / 'manifest.csv'}")


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.root = args.root.resolve()
    output_dir = resolve_output_dir(args).resolve()

    if args.vendor == "stooq":
        run_stooq(args, output_dir)
    else:
        run_yahoo(args, output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
