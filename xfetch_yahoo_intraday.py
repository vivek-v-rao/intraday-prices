#!/usr/bin/env python3
"""
Fetch Yahoo Finance intraday prices into dated snapshot directories.

The output layout matches the consolidation and splicing tools:

    output-root/5_minute_prices/YYYYMMDD/SPY.csv
    output-root/1_minute_prices/YYYYMMDD/SPY.csv

Yahoo intraday history may be adjusted to the download date's adjustment basis.
For rolling snapshots, use xsplice_prices.py to splice overlapping histories.
"""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


INTERVAL_DIRS = {
    "1m": "1_minute_prices",
    "5m": "5_minute_prices",
}
DEFAULT_PERIODS = {
    "1m": "5d",
    "5m": "max",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Yahoo Finance intraday bars into dated snapshot directories."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to fetch, for example SPY TLT HYG.",
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        help="Text file containing one symbol per line. Blank lines and # comments are ignored.",
    )
    parser.add_argument(
        "--interval",
        choices=sorted(INTERVAL_DIRS),
        nargs="+",
        default=["5m"],
        help="Interval(s) to fetch. Defaults to 5m.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("stocks"),
        help="Root for interval/date output directories. Defaults to stocks.",
    )
    parser.add_argument(
        "--date",
        default=date.today().strftime("%Y%m%d"),
        help="Snapshot date directory in YYYYMMDD format. Defaults to today.",
    )
    parser.add_argument(
        "--period",
        help="Yahoo period passed to yfinance. Defaults to 5d for 1m and max for 5m.",
    )
    parser.add_argument(
        "--start",
        help="Optional Yahoo start date, YYYY-MM-DD. If set, period is not passed.",
    )
    parser.add_argument(
        "--end",
        help="Optional Yahoo end date, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--days-prior",
        type=int,
        help="Set start to today minus this many calendar days. Ignored if --start is set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of symbols fetched.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing symbol CSV files.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="Seconds to sleep between symbols. Defaults to 0.",
    )
    return parser.parse_args()


def read_symbols_file(path: Path) -> list[str]:
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        symbols.append(value)
    return symbols


def resolve_symbols(args: argparse.Namespace) -> list[str]:
    symbols: list[str] = []
    if args.symbols:
        symbols.extend(args.symbols)
    if args.symbols_file:
        symbols.extend(read_symbols_file(args.symbols_file))
    if not symbols:
        raise SystemExit("provide --symbols or --symbols-file")

    deduped: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        value = symbol.strip().upper()
        if value and value not in seen:
            deduped.append(value)
            seen.add(value)

    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        deduped = deduped[: args.limit]
    return deduped


def output_dir(output_root: Path, interval: str, snapshot_date: str) -> Path:
    return output_root / INTERVAL_DIRS[interval] / snapshot_date


def fetch_one_symbol(
    symbol: str,
    interval: str,
    period: str | None,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    kwargs = {"interval": interval, "end": end}
    if start is not None:
        kwargs["start"] = start
    else:
        kwargs["period"] = period or DEFAULT_PERIODS[interval]
    return ticker.history(**kwargs)


def summarize_frame(symbol: str, frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "symbol": symbol,
            "rows": 0,
            "days": 0,
            "first_dt": "",
            "last_dt": "",
            "status": "empty",
        }
    return {
        "symbol": symbol,
        "rows": len(frame),
        "days": len(pd.Index(frame.index.date).unique()),
        "first_dt": frame.index[0],
        "last_dt": frame.index[-1],
        "status": "ok",
    }


def fetch_interval(args: argparse.Namespace, symbols: list[str], interval: str) -> list[dict[str, object]]:
    out_dir = output_dir(args.output_root, interval, args.date)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = args.start
    if start is None and args.days_prior is not None:
        start = (date.today() - timedelta(days=args.days_prior)).strftime("%Y-%m-%d")
    period = args.period or DEFAULT_PERIODS[interval]

    print()
    print(f"interval: {interval}")
    print(f"output dir: {out_dir}")
    print(f"period: {period if start is None else ''}")
    print(f"start: {start or ''}")
    print(f"end: {args.end or ''}")

    summaries: list[dict[str, object]] = []
    for symbol in symbols:
        out_file = out_dir / f"{symbol}.csv"
        if out_file.exists() and not args.overwrite:
            print(f"{symbol}: exists, skipped")
            summaries.append(
                {
                    "symbol": symbol,
                    "rows": "",
                    "days": "",
                    "first_dt": "",
                    "last_dt": "",
                    "status": "exists",
                }
            )
            continue

        try:
            frame = fetch_one_symbol(symbol, interval, period, start, args.end)
        except Exception as exc:
            print(f"{symbol}: error: {exc}")
            summaries.append(
                {
                    "symbol": symbol,
                    "rows": "",
                    "days": "",
                    "first_dt": "",
                    "last_dt": "",
                    "status": f"error: {exc}",
                }
            )
            continue

        summary = summarize_frame(symbol, frame)
        summaries.append(summary)
        if not frame.empty:
            frame.to_csv(out_file)
        print(f"{symbol}: rows={summary['rows']} days={summary['days']}")
        if args.pause:
            time.sleep(args.pause)

    return summaries


def print_summary(summaries: list[dict[str, object]]) -> None:
    if not summaries:
        return
    summary_df = pd.DataFrame(summaries)
    print()
    print(summary_df.to_string(index=False))


def main() -> int:
    t0 = time.perf_counter()
    args = parse_args()
    symbols = resolve_symbols(args)

    all_summaries: list[dict[str, object]] = []
    for interval in args.interval:
        all_summaries.extend(fetch_interval(args, symbols, interval))

    print_summary(all_summaries)
    ok_count = sum(item.get("status") == "ok" for item in all_summaries)
    print()
    print(f"symbols requested: {len(symbols)}")
    print(f"files written: {ok_count}")
    print(f"time elapsed (s): {time.perf_counter() - t0:.2f}")

    return 0 if ok_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
