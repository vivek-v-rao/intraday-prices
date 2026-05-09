"""Compare same-symbol intraday price files from two data sources."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from io_intraday import INTRADAY_VENDORS, read_intraday_prices
from vendor_compare import compare_bars
from vendor_compare import compare_intraday_returns
from vendor_compare import compare_realized_vol
from vendor_compare import resample_pair


DAILY_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
DAILY_METRIC_COLUMNS = [
    "high_low_range",
    "close_open_ret",
    "open_close_ret",
    "close_close_ret",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare same-symbol intraday price files from two vendors/sources. "
            "Use positional files for one pair, or --left-dir/--right-dir/--symbols "
            "for a batch."
        )
    )
    parser.add_argument("left_file", nargs="?", type=Path, help="Left/source-A price file.")
    parser.add_argument("right_file", nargs="?", type=Path, help="Right/source-B price file.")
    parser.add_argument("--left-dir", type=Path, help="Directory containing left/source-A files.")
    parser.add_argument("--right-dir", type=Path, help="Directory containing right/source-B files.")
    parser.add_argument("--symbols", nargs="+", help="Symbols to compare in directory mode.")
    parser.add_argument(
        "--left-template",
        default="{symbol}.csv",
        help="Filename template for --left-dir. Defaults to {symbol}.csv.",
    )
    parser.add_argument(
        "--right-template",
        default="{symbol}.csv",
        help="Filename template for --right-dir. Defaults to {symbol}.csv.",
    )
    parser.add_argument(
        "--left-format",
        choices=INTRADAY_VENDORS,
        default="auto",
        help="Left input format/vendor. Defaults to auto.",
    )
    parser.add_argument(
        "--right-format",
        choices=INTRADAY_VENDORS,
        default="auto",
        help="Right input format/vendor. Defaults to auto.",
    )
    parser.add_argument("--left-label", default="left", help="Display label for left source.")
    parser.add_argument("--right-label", default="right", help="Display label for right source.")
    parser.add_argument(
        "--resample",
        help="Optional pandas frequency such as 5min or 15min; applied before comparison.",
    )
    parser.add_argument(
        "--return-horizons",
        nargs="+",
        type=int,
        default=[5, 15, 30],
        help="Intraday return horizons in minutes. Defaults to 5 15 30.",
    )
    parser.add_argument(
        "--price-tolerance-bp",
        type=float,
        default=1.0,
        help="Flag bars whose max OHLC difference exceeds this many basis points.",
    )
    parser.add_argument(
        "--volume-tolerance",
        type=float,
        default=0.05,
        help="Flag bars whose relative volume difference exceeds this threshold.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for detailed CSV comparison tables.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=10,
        help="Rows of largest differences to print for each section. Defaults to 10.",
    )
    parser.add_argument(
        "--no-annualize",
        action="store_true",
        help="Do not annualize daily realized volatilities.",
    )
    parser.add_argument(
        "--float-format",
        default="{:.6g}",
        help="Python format string for printed floats. Defaults to {:.6g}.",
    )
    return parser.parse_args()


def filename_from_template(template: str, symbol: str) -> str:
    """Return a filename from a symbol template."""
    return template.format(symbol=symbol, SYMBOL=symbol.upper(), lower=symbol.lower())


def resolve_pairs(args: argparse.Namespace) -> list[tuple[str, Path, Path]]:
    """Resolve command-line inputs to labeled left/right file pairs."""
    directory_mode = args.left_dir is not None or args.right_dir is not None or args.symbols
    if directory_mode:
        if args.left_file or args.right_file:
            raise ValueError("use either positional files or directory mode, not both")
        if args.left_dir is None or args.right_dir is None or not args.symbols:
            raise ValueError("directory mode requires --left-dir, --right-dir, and --symbols")
        pairs = []
        for symbol in args.symbols:
            left = args.left_dir / filename_from_template(args.left_template, symbol)
            right = args.right_dir / filename_from_template(args.right_template, symbol)
            pairs.append((symbol.upper(), left, right))
    else:
        if args.left_file is None or args.right_file is None:
            raise ValueError("provide left_file and right_file, or use directory mode")
        symbol = args.left_file.stem
        pairs = [(symbol, args.left_file, args.right_file)]

    missing = [path for _, left, right in pairs for path in (left, right) if not path.is_file()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"file not found:\n{missing_text}")
    return [(symbol, left.resolve(), right.resolve()) for symbol, left, right in pairs]


def daily_ohlcv(prices: pd.DataFrame) -> pd.DataFrame:
    """Aggregate intraday bars to daily OHLCV."""
    required = {"Datetime", "date", *DAILY_PRICE_COLUMNS}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    work = prices.dropna(subset=["Datetime", "date", "Open", "High", "Low", "Close"]).copy()
    work = work.sort_values(["date", "Datetime"])
    out = (
        work.groupby("date", sort=True)
        .agg(
            Open=("Open", "first"),
            High=("High", "max"),
            Low=("Low", "min"),
            Close=("Close", "last"),
            Volume=("Volume", "sum"),
        )
        .reset_index()
    )
    prev_close = out["Close"].shift(1)
    out["high_low_range"] = np.log(out["High"] / out["Low"])
    out["close_open_ret"] = np.log(out["Open"] / prev_close)
    out["open_close_ret"] = np.log(out["Close"] / out["Open"])
    out["close_close_ret"] = np.log(out["Close"] / prev_close)
    return out


def compare_daily_metrics(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Compare daily OHLCV levels and daily return/range metrics."""
    left_daily = daily_ohlcv(left).rename(
        columns={col: f"{col}_left" for col in [*DAILY_PRICE_COLUMNS, *DAILY_METRIC_COLUMNS]}
    )
    right_daily = daily_ohlcv(right).rename(
        columns={col: f"{col}_right" for col in [*DAILY_PRICE_COLUMNS, *DAILY_METRIC_COLUMNS]}
    )
    out = left_daily.merge(right_daily, on="date", how="inner")

    max_daily_price_diff_bp = pd.Series(0.0, index=out.index)
    for col in ["Open", "High", "Low", "Close"]:
        avg = (out[f"{col}_left"].abs() + out[f"{col}_right"].abs()) / 2.0
        out[f"{col}_diff_bp"] = 10000.0 * (out[f"{col}_left"] - out[f"{col}_right"]) / avg
        max_daily_price_diff_bp = np.maximum(max_daily_price_diff_bp, out[f"{col}_diff_bp"].abs())
    out["max_daily_price_diff_bp"] = max_daily_price_diff_bp

    vol_avg = (out["Volume_left"].abs() + out["Volume_right"].abs()) / 2.0
    out["Volume_rel_diff"] = (out["Volume_left"] - out["Volume_right"]) / vol_avg.replace(
        0.0,
        np.nan,
    )
    for col in DAILY_METRIC_COLUMNS:
        out[f"{col}_diff"] = out[f"{col}_left"] - out[f"{col}_right"]
    return out


def summarize_series(values: pd.Series) -> dict[str, float]:
    """Return compact absolute-difference summary stats."""
    clean = values.replace([np.inf, -np.inf], np.nan).dropna().abs()
    if clean.empty:
        return {"mean_abs": np.nan, "median_abs": np.nan, "p95_abs": np.nan, "max_abs": np.nan}
    return {
        "mean_abs": float(clean.mean()),
        "median_abs": float(clean.median()),
        "p95_abs": float(clean.quantile(0.95)),
        "max_abs": float(clean.max()),
    }


def diff_summary(df: pd.DataFrame, diff_columns: list[str]) -> pd.DataFrame:
    """Summarize absolute differences for selected columns."""
    rows = []
    for col in diff_columns:
        if col in df.columns:
            row = {"metric": col, **summarize_series(df[col])}
            rows.append(row)
    return pd.DataFrame(rows)


def print_table(title: str, df: pd.DataFrame, float_format: str) -> None:
    """Print a DataFrame with consistent formatting."""
    print(f"\n{title}")
    if df.empty:
        print("(empty)")
        return
    fmt = lambda x: float_format.format(x) if np.isfinite(x) else "nan"
    print(df.to_string(index=False, float_format=fmt))


def write_csv(path: Path, df: pd.DataFrame) -> None:
    """Write a CSV, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def output_path(output_dir: Path, symbol: str, name: str) -> Path:
    """Return an output CSV path for a symbol and table name."""
    safe_symbol = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in symbol)
    return output_dir / f"{safe_symbol}_{name}.csv"


def largest_abs_rows(df: pd.DataFrame, column: str, max_rows: int) -> pd.DataFrame:
    """Return rows with largest absolute values in a column."""
    if max_rows <= 0 or column not in df.columns or df.empty:
        return pd.DataFrame()
    work = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[column]).copy()
    if work.empty:
        return pd.DataFrame()
    return work.loc[work[column].abs().sort_values(ascending=False).head(max_rows).index]


def compare_pair(
    symbol: str,
    left_path: Path,
    right_path: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Run all comparisons for one left/right pair and print results."""
    print(f"\n{'=' * 100}")
    print(f"symbol: {symbol}")
    print(f"{args.left_label}: {left_path}")
    print(f"{args.right_label}: {right_path}")

    left = read_intraday_prices(left_path, vendor=args.left_format)
    right = read_intraday_prices(right_path, vendor=args.right_format)
    if args.resample:
        left, right = resample_pair(left, right, args.resample)

    print(
        "rows: "
        f"{args.left_label}={len(left)}, {args.right_label}={len(right)}; "
        "days: "
        f"{args.left_label}={left['date'].nunique()}, {args.right_label}={right['date'].nunique()}"
    )

    bars = compare_bars(
        left,
        right,
        price_tolerance_bp=args.price_tolerance_bp,
        volume_tolerance=args.volume_tolerance,
    )
    bar_summary = pd.DataFrame(
        [
            {
                "aligned_bars": len(bars),
                "flag_price": int(bars["flag_price"].sum()),
                "flag_volume": int(bars["flag_volume"].sum()),
                "median_abs_price_diff_bp": bars["max_price_diff_bp"].abs().median(),
                "p95_abs_price_diff_bp": bars["max_price_diff_bp"].abs().quantile(0.95),
                "max_abs_price_diff_bp": bars["max_price_diff_bp"].abs().max(),
                "median_abs_volume_rel_diff": bars["Volume_rel_diff"].abs().median(),
                "max_abs_volume_rel_diff": bars["Volume_rel_diff"].abs().max(),
            }
        ]
    )
    print_table("Bar comparison summary", bar_summary, args.float_format)
    top_bars = largest_abs_rows(bars, "max_price_diff_bp", args.max_rows)
    print_table("Largest bar price differences", top_bars, args.float_format)

    daily = compare_daily_metrics(left, right)
    daily_summary = diff_summary(
        daily,
        [
            "max_daily_price_diff_bp",
            "Volume_rel_diff",
            *[f"{col}_diff" for col in DAILY_METRIC_COLUMNS],
        ],
    )
    print_table("Daily metric difference summary", daily_summary, args.float_format)
    top_daily = largest_abs_rows(daily, "max_daily_price_diff_bp", args.max_rows)
    print_table("Largest daily OHLC price differences", top_daily, args.float_format)

    annualize = not args.no_annualize
    rv_tables = {"bar": compare_realized_vol(left, right, horizon_minutes=None, annualize=annualize)}
    for horizon in args.return_horizons:
        returns = compare_intraday_returns(left, right, horizon)
        ret_summary = diff_summary(returns, ["return_diff"])
        print_table(f"{horizon}-minute return difference summary", ret_summary, args.float_format)
        if args.output_dir:
            write_csv(output_path(args.output_dir, symbol, f"returns_{horizon}min"), returns)
        rv_tables[f"{horizon}min"] = compare_realized_vol(
            left,
            right,
            horizon_minutes=horizon,
            annualize=annualize,
        )

    rv_summaries = []
    for name, rv in rv_tables.items():
        row = {"horizon": name, "aligned_days": len(rv)}
        row.update({f"rv_vol_{key}": value for key, value in summarize_series(rv["rv_vol_diff"]).items()})
        ratio = rv["rv_vol_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        row["rv_vol_ratio_median"] = float(ratio.median()) if not ratio.empty else np.nan
        row["rv_vol_ratio_min"] = float(ratio.min()) if not ratio.empty else np.nan
        row["rv_vol_ratio_max"] = float(ratio.max()) if not ratio.empty else np.nan
        rv_summaries.append(row)
        if args.output_dir:
            write_csv(output_path(args.output_dir, symbol, f"rv_{name}"), rv)
    rv_summary = pd.DataFrame(rv_summaries)
    print_table("Daily realized volatility difference summary", rv_summary, args.float_format)

    if args.output_dir:
        write_csv(output_path(args.output_dir, symbol, "bars"), bars)
        write_csv(output_path(args.output_dir, symbol, "daily_metrics"), daily)

    return {
        "symbol": symbol,
        "left_rows": len(left),
        "right_rows": len(right),
        "aligned_bars": len(bars),
        "aligned_days": len(daily),
        "flag_price": int(bars["flag_price"].sum()),
        "flag_volume": int(bars["flag_volume"].sum()),
        "max_abs_price_diff_bp": bars["max_price_diff_bp"].abs().max(),
        "max_abs_daily_price_diff_bp": daily["max_daily_price_diff_bp"].abs().max(),
    }


def main() -> None:
    """Run same-symbol vendor/source comparisons."""
    args = parse_args()
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 120)

    pairs = resolve_pairs(args)
    summaries = [compare_pair(symbol, left, right, args) for symbol, left, right in pairs]
    if len(summaries) > 1:
        print_table("Batch summary", pd.DataFrame(summaries), args.float_format)


if __name__ == "__main__":
    main()
