from __future__ import annotations

import numpy as np
import pandas as pd

from intraday_bars import resample_intraday_bars
from intraday_returns import intraday_horizon_returns
from market_constants import TRADING_DAYS


PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
COMPARE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def comparable_datetimes(values: pd.Series) -> pd.Series:
    """Return datetimes normalized to timezone-naive values for comparison."""
    dt = pd.to_datetime(values, errors="coerce")
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    return dt


def align_bars(
    left: pd.DataFrame,
    right: pd.DataFrame,
    suffixes: tuple[str, str] = ("_left", "_right"),
) -> pd.DataFrame:
    """Return bars from two vendors aligned by Datetime."""
    left_cols = ["Datetime", "date", *COMPARE_COLUMNS]
    right_cols = ["Datetime", "date", *COMPARE_COLUMNS]
    missing_left = set(left_cols) - set(left.columns)
    missing_right = set(right_cols) - set(right.columns)
    if missing_left:
        raise ValueError(f"left data missing columns: {sorted(missing_left)}")
    if missing_right:
        raise ValueError(f"right data missing columns: {sorted(missing_right)}")

    left_work = left[left_cols].copy()
    right_work = right[right_cols].copy()
    left_work["Datetime"] = comparable_datetimes(left_work["Datetime"])
    right_work["Datetime"] = comparable_datetimes(right_work["Datetime"])
    left_work = left_work.drop_duplicates(subset=["Datetime"], keep="last")
    right_work = right_work.drop_duplicates(subset=["Datetime"], keep="last")
    return left_work.merge(right_work, on="Datetime", how="inner", suffixes=suffixes)


def compare_bars(
    left: pd.DataFrame,
    right: pd.DataFrame,
    price_tolerance_bp: float = 1.0,
    volume_tolerance: float = 0.05,
) -> pd.DataFrame:
    """Compare aligned OHLCV bars and flag material price/volume differences."""
    aligned = align_bars(left, right)
    out = aligned[["Datetime", "date_left", "date_right"]].copy()

    max_price_diff_bp = pd.Series(0.0, index=aligned.index)
    for col in PRICE_COLUMNS:
        lcol = f"{col}_left"
        rcol = f"{col}_right"
        avg = (aligned[lcol].abs() + aligned[rcol].abs()) / 2.0
        diff_bp = 10000.0 * (aligned[lcol] - aligned[rcol]) / avg
        out[f"{col}_left"] = aligned[lcol]
        out[f"{col}_right"] = aligned[rcol]
        out[f"{col}_diff_bp"] = diff_bp
        max_price_diff_bp = np.maximum(max_price_diff_bp, diff_bp.abs())

    lvol = aligned["Volume_left"]
    rvol = aligned["Volume_right"]
    vol_avg = (lvol.abs() + rvol.abs()) / 2.0
    vol_rel_diff = (lvol - rvol) / vol_avg.replace(0.0, np.nan)
    out["Volume_left"] = lvol
    out["Volume_right"] = rvol
    out["Volume_rel_diff"] = vol_rel_diff
    out["max_price_diff_bp"] = max_price_diff_bp
    out["flag_price"] = max_price_diff_bp > price_tolerance_bp
    out["flag_volume"] = vol_rel_diff.abs() > volume_tolerance
    return out


def compare_intraday_returns(
    left: pd.DataFrame,
    right: pd.DataFrame,
    horizon_minutes: int,
) -> pd.DataFrame:
    """Compare aligned intraday log returns for one horizon."""
    left_ret = intraday_horizon_returns(left, horizon_minutes).rename("return_left")
    right_ret = intraday_horizon_returns(right, horizon_minutes).rename("return_right")
    left_ret.index = comparable_datetimes(pd.Series(left_ret.index))
    right_ret.index = comparable_datetimes(pd.Series(right_ret.index))
    out = pd.concat([left_ret, right_ret], axis=1, join="inner").dropna()
    out["return_diff"] = out["return_left"] - out["return_right"]
    out["abs_return_diff"] = out["return_diff"].abs()
    out.index.name = "Datetime"
    return out.reset_index()


def realized_vol_by_day(
    prices: pd.DataFrame,
    horizon_minutes: int | None = None,
    annualize: bool = True,
) -> pd.DataFrame:
    """Compute simple daily realized volatility from intraday log returns."""
    if horizon_minutes is None:
        work = prices[["Datetime", "date", "Close"]].dropna().copy()
        work = work.drop_duplicates(subset=["Datetime"], keep="last")
        work = work.sort_values(["date", "Datetime"])
        log_close = np.log(work["Close"])
        work["ret"] = log_close.groupby(work["date"]).diff()
    else:
        ret = intraday_horizon_returns(prices, horizon_minutes)
        work = pd.DataFrame({"Datetime": ret.index, "ret": ret.to_numpy()})
        dates = prices[["Datetime", "date"]].drop_duplicates(subset=["Datetime"], keep="last")
        work = work.merge(dates, on="Datetime", how="left")

    grouped = work.dropna(subset=["ret"]).groupby("date")["ret"]
    out = grouped.agg(n_returns="count", rv_var=lambda x: float(np.sum(np.square(x))))
    scale = TRADING_DAYS if annualize else 1.0
    out["rv_vol"] = np.sqrt(scale * out["rv_var"])
    return out.reset_index()


def compare_realized_vol(
    left: pd.DataFrame,
    right: pd.DataFrame,
    horizon_minutes: int | None = None,
    annualize: bool = True,
) -> pd.DataFrame:
    """Compare daily realized volatility from two vendors."""
    left_rv = realized_vol_by_day(left, horizon_minutes, annualize).rename(
        columns={
            "n_returns": "n_returns_left",
            "rv_var": "rv_var_left",
            "rv_vol": "rv_vol_left",
        }
    )
    right_rv = realized_vol_by_day(right, horizon_minutes, annualize).rename(
        columns={
            "n_returns": "n_returns_right",
            "rv_var": "rv_var_right",
            "rv_vol": "rv_vol_right",
        }
    )
    out = left_rv.merge(right_rv, on="date", how="inner")
    out["rv_vol_diff"] = out["rv_vol_left"] - out["rv_vol_right"]
    out["rv_vol_ratio"] = out["rv_vol_left"] / out["rv_vol_right"].replace(0.0, np.nan)
    out["rv_var_diff"] = out["rv_var_left"] - out["rv_var_right"]
    return out


def compare_realized_vol_horizons(
    left: pd.DataFrame,
    right: pd.DataFrame,
    horizons_minutes: tuple[int, ...] = (5, 15, 30),
    annualize: bool = True,
) -> dict[str, pd.DataFrame]:
    """Compare daily realized volatility for several return horizons."""
    out = {
        f"{horizon}-minute": compare_realized_vol(
            left,
            right,
            horizon_minutes=horizon,
            annualize=annualize,
        )
        for horizon in horizons_minutes
    }
    out["bar"] = compare_realized_vol(left, right, horizon_minutes=None, annualize=annualize)
    return out


def resample_pair(
    left: pd.DataFrame,
    right: pd.DataFrame,
    freq: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample two canonical intraday datasets to the same OHLCV frequency."""
    return resample_intraday_bars(left, freq), resample_intraday_bars(right, freq)
