from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from io_intraday import read_intraday_prices


def asset_label(path: Path) -> str:
    """Return a compact asset label inferred from a price file name."""
    label = path.stem
    if label.lower().endswith(".us"):
        label = label[:-3]
    return label.upper()


def infer_bar_interval_minutes(prices: pd.DataFrame) -> int:
    """Infer the regular intraday bar interval in minutes from timestamps."""
    work = prices[["Datetime", "date"]].dropna().drop_duplicates()
    diffs = (
        work.sort_values(["date", "Datetime"])
        .groupby("date")["Datetime"]
        .diff()
        .dropna()
        .dt.total_seconds()
        .div(60.0)
    )
    diffs = diffs[diffs > 0]
    if diffs.empty:
        raise ValueError("cannot infer bar interval from fewer than two intraday bars")
    interval = float(diffs.median())
    rounded = int(round(interval))
    if not np.isclose(interval, rounded, atol=1.0e-9):
        raise ValueError(f"cannot infer integer-minute bar interval from median gap {interval}")
    return rounded


def steps_for_horizon(bar_interval_minutes: int, horizon_minutes: int) -> int:
    """Return the number of input bars needed for a return horizon."""
    if horizon_minutes % bar_interval_minutes != 0:
        raise ValueError(
            f"{horizon_minutes}-minute returns cannot be computed exactly from "
            f"{bar_interval_minutes}-minute bars"
        )
    return horizon_minutes // bar_interval_minutes


def intraday_bar_returns(prices: pd.DataFrame, step: int) -> pd.Series:
    """Return same-day close-to-close log returns over step input bars."""
    work = prices[["Datetime", "date", "Close"]].copy()
    work = work.dropna(subset=["Datetime", "date", "Close"])
    work = work.drop_duplicates(subset=["Datetime"], keep="last")
    work = work.sort_values(["date", "Datetime"])
    log_close = np.log(work["Close"])
    ret = log_close.groupby(work["date"]).diff(step)
    out = pd.Series(ret.to_numpy(), index=work["Datetime"], name="ret")
    return out.dropna()


def intraday_horizon_returns(
    prices: pd.DataFrame,
    horizon_minutes: int,
) -> pd.Series:
    """Return same-day log returns over a horizon expressed in minutes."""
    interval = infer_bar_interval_minutes(prices)
    step = steps_for_horizon(interval, horizon_minutes)
    return intraday_bar_returns(prices, step)


def daily_returns(prices: pd.DataFrame) -> pd.Series:
    """Return daily close-to-close log returns from the last bar of each date."""
    daily_close = (
        prices.dropna(subset=["date", "Close"])
        .sort_values(["date", "Datetime"])
        .groupby("date")["Close"]
        .last()
    )
    return np.log(daily_close).diff().dropna()


def cross_asset_returns(
    prices_files: list[Path],
    frequency: str,
    vendor: str,
    horizon_minutes: int | None = None,
) -> pd.DataFrame:
    """Return aligned cross-asset log returns for one sampling frequency."""
    cols: dict[str, pd.Series] = {}
    for path in prices_files:
        prices = read_intraday_prices(path, vendor=vendor)
        if horizon_minutes is None:
            ret = daily_returns(prices)
        else:
            ret = intraday_horizon_returns(prices, horizon_minutes)
        cols[asset_label(path)] = ret
    out = pd.concat(cols, axis=1, join="inner").dropna(how="any")
    out.columns.name = frequency
    return out
