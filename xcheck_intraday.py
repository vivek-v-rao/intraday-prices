"""Quickly validate and summarize intraday price files."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_quality import DEFAULT_MAX_RANGE
from data_quality import find_bad_ohlcv
from file_utils import expand_file_patterns
from intraday_returns import asset_label
from intraday_returns import infer_bar_interval_minutes
from io_intraday import INTRADAY_VENDORS, read_intraday_prices
from market_constants import TRADING_DAYS


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Read intraday price files and print lightweight validity stats."
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
    parser.add_argument(
        "--frequency",
        choices=["intraday", "daily"],
        default="intraday",
        help="Threshold profile for data-quality checks. Defaults to intraday.",
    )
    parser.add_argument(
        "--show-bad",
        type=int,
        default=0,
        help="Print up to N bad rows per file. Defaults to 0.",
    )
    parser.add_argument(
        "--show-bad-days",
        type=int,
        default=0,
        help=(
            "Print up to N symbol/date combinations with anomalous daily max_range. "
            "Defaults to 0."
        ),
    )
    parser.add_argument(
        "--daily-max-range-threshold",
        type=float,
        help=(
            "Flag a symbol/date when its largest intraday High/Low - 1 exceeds "
            "this threshold. Defaults to the --frequency max_range threshold."
        ),
    )
    parser.add_argument(
        "--output-daily-summary",
        type=Path,
        help=(
            "Optional CSV path for one row per symbol/date with daily OHLCV, "
            "range, open/close returns, and intraday realized volatility."
        ),
    )
    parser.add_argument(
        "--no-realized-vol",
        action="store_true",
        help="Skip realized-volatility calculation.",
    )
    parser.add_argument(
        "--realized-vol-method",
        choices=["rms-bar", "daily-rv"],
        default="rms-bar",
        help=(
            "Method for realized_vol. rms-bar uses RMS consecutive bar returns; "
            "daily-rv uses sqrt of mean daily realized variance. Defaults to rms-bar."
        ),
    )
    parser.add_argument(
        "--no-quality",
        action="store_true",
        help="Skip OHLCV data-quality checks.",
    )
    parser.add_argument(
        "--no-bars-per-day",
        action="store_true",
        help="Skip bar interval and bars-per-day statistics.",
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


def daily_rv_realized_vol(prices: pd.DataFrame) -> float:
    """Return annualized volatility from mean daily realized variance."""
    work = prices[["Datetime", "date", "Close"]].copy()
    work = work.dropna(subset=["Datetime", "date", "Close"])
    work = work.drop_duplicates(subset=["Datetime"], keep="last")
    work = work.sort_values(["date", "Datetime"])
    log_close = np.log(work["Close"])
    work["ret"] = log_close.diff()
    daily_var = work.dropna(subset=["ret"]).groupby("date")["ret"].apply(lambda x: np.sum(x * x))
    if daily_var.empty:
        return np.nan
    return float(np.sqrt(TRADING_DAYS * daily_var.mean()))


def rms_bar_realized_vol(prices: pd.DataFrame) -> float:
    """Return annualized volatility from RMS consecutive bar returns."""
    work = prices[["Datetime", "date", "Close"]].copy()
    work = work.dropna(subset=["Datetime", "date", "Close"])
    work = work.drop_duplicates(subset=["Datetime"], keep="last")
    work = work.sort_values(["date", "Datetime"])
    returns = np.log(work["Close"]).diff().dropna()
    if returns.empty:
        return np.nan
    bars_per_day = work.groupby("date").size().median()
    periods_per_year = TRADING_DAYS * bars_per_day
    return float(np.sqrt(periods_per_year * np.mean(np.square(returns))))


def realized_vol(prices: pd.DataFrame, method: str) -> float:
    """Return annualized realized volatility using the requested method."""
    if method == "rms-bar":
        return rms_bar_realized_vol(prices)
    if method == "daily-rv":
        return daily_rv_realized_vol(prices)
    raise ValueError(f"unknown realized vol method {method!r}")


def close_to_close_vol(prices: pd.DataFrame) -> float:
    """Return annualized RMS close-to-close volatility from daily closing prices."""
    daily_close = (
        prices.dropna(subset=["date", "Close"])
        .sort_values(["date", "Datetime"])
        .groupby("date")["Close"]
        .last()
    )
    returns = np.log(daily_close).diff().dropna()
    if returns.empty:
        return np.nan
    return float(100.0 * np.sqrt(TRADING_DAYS * np.mean(np.square(returns))))


def max_range_threshold(frequency: str, override: float | None) -> float:
    """Return the daily max-range anomaly threshold."""
    if override is not None:
        return override
    if frequency not in DEFAULT_MAX_RANGE:
        raise ValueError(f"unknown frequency {frequency!r}; expected one of {sorted(DEFAULT_MAX_RANGE)}")
    return DEFAULT_MAX_RANGE[frequency]


def daily_max_range_anomalies(
    path: Path,
    prices: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Return symbol/date rows whose largest intraday high-low range is anomalous."""
    required = {"Datetime", "date", "Open", "High", "Low", "Close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    work = prices.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work["bar_range"] = work["High"] / work["Low"] - 1.0
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", "bar_range"])
    if work.empty:
        return pd.DataFrame()

    grouped = work.groupby("date")
    idx = grouped["bar_range"].idxmax()
    high_idx = grouped["High"].idxmax()
    low_idx = grouped["Low"].idxmin()
    counts = work.loc[work["bar_range"] > threshold].groupby("date").size()

    worst = work.loc[idx].copy()
    worst = worst.loc[worst["bar_range"] > threshold]
    if worst.empty:
        return pd.DataFrame()

    high_times = work.loc[high_idx, ["date", "Datetime"]].set_index("date")["Datetime"]
    low_times = work.loc[low_idx, ["date", "Datetime"]].set_index("date")["Datetime"]
    worst["time_High"] = worst["date"].map(high_times)
    worst["time_Low"] = worst["date"].map(low_times)

    out_cols = [
        "symbol",
        "date",
        "max_range",
        "range_bars",
        "Datetime",
        "time_High",
        "time_Low",
        "Open",
        "High",
        "Low",
        "Close",
    ]
    if "Volume" in worst.columns:
        out_cols.append("Volume")
    worst["symbol"] = asset_label(path)
    worst["max_range"] = worst["bar_range"]
    worst["range_bars"] = worst["date"].map(counts).fillna(0).astype(int)
    return worst[out_cols].sort_values(["max_range", "symbol", "date"], ascending=[False, True, True])


def daily_summary_table(path: Path, prices: pd.DataFrame) -> pd.DataFrame:
    """Return one daily summary row per symbol/date."""
    required = {"Datetime", "date", "Open", "High", "Low", "Close", "Volume"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    work = prices[["Datetime", "date", "Open", "High", "Low", "Close", "Volume"]].copy()
    work = work.dropna(subset=["Datetime", "date", "Open", "High", "Low", "Close"])
    work = work.sort_values(["date", "Datetime"])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Open", "High", "Low", "Close"]
    )
    if work.empty:
        return pd.DataFrame()

    work["bar_range"] = work["High"] / work["Low"] - 1.0
    work["bar_ret"] = np.log(work["Close"]).groupby(work["date"]).diff()
    grouped = work.groupby("date", sort=True)
    high_idx = grouped["High"].idxmax()
    low_idx = grouped["Low"].idxmin()
    max_bar_range_idx = grouped["bar_range"].idxmax()

    out = grouped.agg(
        n_bars=("Datetime", "size"),
        first_dt=("Datetime", "first"),
        last_dt=("Datetime", "last"),
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
        realized_var=("bar_ret", lambda x: float(np.sum(np.square(x.dropna())))),
        n_returns=("bar_ret", lambda x: int(x.notna().sum())),
        max_bar_range=("bar_range", "max"),
    ).reset_index()

    high_times = work.loc[high_idx, ["date", "Datetime"]].set_index("date")["Datetime"]
    low_times = work.loc[low_idx, ["date", "Datetime"]].set_index("date")["Datetime"]
    max_range_times = work.loc[max_bar_range_idx, ["date", "Datetime"]].set_index("date")[
        "Datetime"
    ]
    out["time_High"] = out["date"].map(high_times)
    out["time_Low"] = out["date"].map(low_times)
    out["time_max_bar_range"] = out["date"].map(max_range_times)

    prev_close = out["Close"].shift(1)
    out["range"] = out["High"] / out["Low"] - 1.0
    out["log_range"] = np.log(out["High"] / out["Low"])
    out["co_return"] = np.log(out["Open"] / prev_close)
    out["oc_return"] = np.log(out["Close"] / out["Open"])
    out["cc_return"] = np.log(out["Close"] / prev_close)
    out["realized_vol_ann"] = np.sqrt(TRADING_DAYS * out["realized_var"])
    out["realized_vol_ann_pct"] = 100.0 * out["realized_vol_ann"]
    out.insert(0, "symbol", asset_label(path))

    ordered_cols = [
        "symbol",
        "date",
        "n_bars",
        "n_returns",
        "first_dt",
        "last_dt",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "range",
        "log_range",
        "co_return",
        "oc_return",
        "cc_return",
        "realized_var",
        "realized_vol_ann",
        "realized_vol_ann_pct",
        "max_bar_range",
        "time_High",
        "time_Low",
        "time_max_bar_range",
    ]
    return out[ordered_cols]


def summarize_prices(
    path: Path,
    prices: pd.DataFrame,
    frequency: str,
    daily_max_range_threshold: float,
    compute_realized_vol: bool,
    realized_vol_method: str,
    compute_quality: bool,
    compute_bars_per_day: bool,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Return one lightweight summary row and bad-row details for one file."""
    bad = find_bad_ohlcv(prices, frequency=frequency) if compute_quality else pd.DataFrame()
    bad_days = (
        daily_max_range_anomalies(path, prices, daily_max_range_threshold)
        if compute_quality
        else pd.DataFrame()
    )
    summary = {
        "symbol": asset_label(path),
        "rows": len(prices),
        "days": int(prices["date"].nunique()),
        "first_dt": prices["Datetime"].min(),
        "last_dt": prices["Datetime"].max(),
    }
    if compute_bars_per_day:
        days = prices.groupby("date").size()
        try:
            bar_minutes = infer_bar_interval_minutes(prices)
        except ValueError:
            bar_minutes = np.nan
        summary.update(
            {
                "bar_min": bar_minutes,
                "median_bars_day": float(days.median()) if not days.empty else np.nan,
                "min_bars_day": int(days.min()) if not days.empty else 0,
                "max_bars_day": int(days.max()) if not days.empty else 0,
            }
        )
    if compute_quality:
        high_low_range = prices["High"] / prices["Low"] - 1.0
        oc_return = np.log(prices["Close"] / prices["Open"]).abs()
        summary.update(
            {
                "bad_rows": len(bad),
                "bad_days": len(bad_days),
                "max_range": float(high_low_range.max()),
                "max_abs_oc": float(oc_return.max()),
            }
        )
    if compute_realized_vol:
        summary["realized_vol"] = 100.0 * realized_vol(prices, realized_vol_method)
        summary["cc_vol"] = close_to_close_vol(prices)
    return summary, bad, bad_days


def main() -> None:
    """Run the intraday validity scan."""
    start = time.perf_counter()
    args = parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    files = resolve_price_files(args)
    daily_range_threshold = max_range_threshold(args.frequency, args.daily_max_range_threshold)
    summaries = []
    bad_examples: list[tuple[Path, pd.DataFrame]] = []
    bad_day_examples: list[pd.DataFrame] = []
    daily_summaries: list[pd.DataFrame] = []
    for path in files:
        prices = read_intraday_prices(path, vendor=args.format)
        if args.output_daily_summary:
            daily_summaries.append(daily_summary_table(path, prices))
        summary, bad, bad_days = summarize_prices(
            path,
            prices,
            args.frequency,
            daily_range_threshold,
            compute_realized_vol=not args.no_realized_vol,
            realized_vol_method=args.realized_vol_method,
            compute_quality=not args.no_quality,
            compute_bars_per_day=not args.no_bars_per_day,
        )
        summaries.append(summary)
        if args.show_bad and not args.no_quality and not bad.empty:
            bad_examples.append((path, bad.head(args.show_bad)))
        if args.show_bad_days and not args.no_quality and not bad_days.empty:
            bad_day_examples.append(bad_days)

    summary_df = pd.DataFrame(summaries)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    for path, bad in bad_examples:
        print(f"\nBad row examples: {path}")
        print(bad.to_string(index=False))

    if bad_day_examples:
        bad_days_df = pd.concat(bad_day_examples, ignore_index=True)
        bad_days_df = bad_days_df.sort_values(
            ["max_range", "symbol", "date"],
            ascending=[False, True, True],
        ).head(args.show_bad_days)
        print(f"\nAnomalous symbol/date max_range rows (threshold={daily_range_threshold:.6g})")
        print(bad_days_df.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

    if args.output_daily_summary:
        daily_summary = (
            pd.concat(daily_summaries, ignore_index=True)
            if daily_summaries
            else pd.DataFrame()
        )
        args.output_daily_summary.parent.mkdir(parents=True, exist_ok=True)
        daily_summary.to_csv(args.output_daily_summary, index=False)
        print(f"\nwrote daily summary: {args.output_daily_summary}")

    elapsed = time.perf_counter() - start
    n_symbols = len(summary_df)
    avg_obs = summary_df["rows"].mean() if n_symbols else 0.0
    print()
    print(f"symbols processed: {n_symbols}")
    print(f"avg obs per symbol: {avg_obs:.1f}")
    if n_symbols and "realized_vol" in summary_df.columns:
        rv = summary_df["realized_vol"].dropna()
        if not rv.empty:
            min_symbol = summary_df.loc[rv.idxmin(), "symbol"]
            max_symbol = summary_df.loc[rv.idxmax(), "symbol"]
            print(
                "realized vol stats (%): "
                f"median={rv.median():.3f}, "
                f"mean={rv.mean():.3f}, "
                f"sd={rv.std():.3f}, "
                f"min={rv.min():.3f} ({min_symbol}), "
                f"max={rv.max():.3f} ({max_symbol})"
            )
    if n_symbols:
        earliest = summary_df["first_dt"].min()
        latest = summary_df["last_dt"].max()
        calendar_days = (latest.date() - earliest.date()).days
        print(f"datetime range: {earliest} to {latest} ({calendar_days} calendar days)")
    print(f"time elapsed (s): {elapsed:.2f}")


if __name__ == "__main__":
    main()
