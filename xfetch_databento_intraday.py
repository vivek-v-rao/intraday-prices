#!/usr/bin/env python3
"""
Fetch Databento intraday OHLCV prices into dated snapshot directories.

The output layout matches the consolidation and splicing tools:

    output-root/1_minute_prices/YYYYMMDD/SPY.csv

The default API key is the placeholder "your-api-key" so this script can be
checked into a public repository. Pass a real key with --api-key or set
DATABENTO_API_KEY in your environment before running.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import date, timedelta
from pathlib import Path

import databento as db
import pandas as pd


INTERVAL_CONFIG = {
    "1m": {
        "directory": "1_minute_prices",
        "schema": "ohlcv-1m",
    },
}
DEFAULT_API_KEY = "your-api-key"
DEFAULT_DATASET = "EQUS.MINI"
DEFAULT_START = "2023-03-28"
CSV_COLUMNS = ["ts_event", "symbol", "open", "high", "low", "close", "volume"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Databento intraday OHLCV bars into dated snapshot directories."
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
        choices=sorted(INTERVAL_CONFIG),
        nargs="+",
        default=["1m"],
        help="Interval(s) to fetch. Defaults to 1m.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Databento dataset. Defaults to {DEFAULT_DATASET}.",
    )
    parser.add_argument(
        "--stype-in",
        default="raw_symbol",
        help="Databento input symbol type. Defaults to raw_symbol.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("DATABENTO_API_KEY", DEFAULT_API_KEY),
        help=(
            "Databento API key. Defaults to DATABENTO_API_KEY, or the placeholder "
            f"{DEFAULT_API_KEY!r}."
        ),
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
        "--start",
        default=DEFAULT_START,
        help=f"Databento start date/time. Defaults to {DEFAULT_START}.",
    )
    parser.add_argument(
        "--end",
        help="Optional Databento end date/time. Defaults to today plus one day.",
    )
    parser.add_argument(
        "--days-prior",
        type=int,
        help="Set start to today minus this many calendar days. Overrides --start.",
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
    return output_root / INTERVAL_CONFIG[interval]["directory"] / snapshot_date


def default_end() -> str:
    return (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")


def effective_start(args: argparse.Namespace) -> str:
    if args.days_prior is not None:
        return (date.today() - timedelta(days=args.days_prior)).strftime("%Y-%m-%d")
    return args.start


def normalize_databento_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    out = frame.reset_index()
    keep = [col for col in CSV_COLUMNS if col in out.columns]
    return out[keep].copy()


def fetch_one_symbol(
    client: db.Historical,
    symbol: str,
    dataset: str,
    schema: str,
    stype_in: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    data = client.timeseries.get_range(
        dataset=dataset,
        schema=schema,
        symbols=[symbol],
        stype_in=stype_in,
        start=start,
        end=end,
    )
    return normalize_databento_frame(data.to_df())


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

    ts = pd.to_datetime(frame["ts_event"], errors="coerce") if "ts_event" in frame else pd.Series()
    return {
        "symbol": symbol,
        "rows": len(frame),
        "days": len(pd.Index(ts.dt.date).dropna().unique()) if not ts.empty else "",
        "first_dt": ts.min() if not ts.empty else "",
        "last_dt": ts.max() if not ts.empty else "",
        "status": "ok",
    }


def fetch_interval(
    args: argparse.Namespace,
    client: db.Historical,
    symbols: list[str],
    interval: str,
) -> list[dict[str, object]]:
    out_dir = output_dir(args.output_root, interval, args.date)
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = INTERVAL_CONFIG[interval]["schema"]
    start = effective_start(args)
    end = args.end or default_end()

    print()
    print(f"interval: {interval}")
    print(f"dataset: {args.dataset}")
    print(f"schema: {schema}")
    print(f"output dir: {out_dir}")
    print(f"start: {start}")
    print(f"end: {end}")

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
            frame = fetch_one_symbol(
                client,
                symbol,
                args.dataset,
                schema,
                args.stype_in,
                start,
                end,
            )
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
            frame.to_csv(out_file, index=False)
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
    client = db.Historical(args.api_key)

    all_summaries: list[dict[str, object]] = []
    for interval in args.interval:
        all_summaries.extend(fetch_interval(args, client, symbols, interval))

    print_summary(all_summaries)
    ok_count = sum(item.get("status") == "ok" for item in all_summaries)
    print()
    print(f"symbols requested: {len(symbols)}")
    print(f"files written: {ok_count}")
    print(f"time elapsed (s): {time.perf_counter() - t0:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
