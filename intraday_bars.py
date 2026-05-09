from __future__ import annotations

import pandas as pd


OHLCV_AGG = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum",
}
CORPORATE_ACTION_AGG = {
    "Dividends": "sum",
    "Stock Splits": "sum",
    "Capital Gains": "sum",
}


def resample_intraday_bars(prices: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample canonical intraday OHLCV bars within each trading date.

    Parameters
    ----------
    prices
        DataFrame with canonical columns from ``io_intraday.read_intraday_prices``:
        Datetime, date, Open, High, Low, Close, and Volume. Corporate action
        columns are preserved when present.
    freq
        Pandas offset alias such as "5min", "15min", or "30min".

    Returns
    -------
    pd.DataFrame
        Resampled canonical intraday bars. Each output timestamp is the left
        edge of the resampling bin.
    """
    required = {"Datetime", "date", *OHLCV_AGG}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    work = prices.copy()
    work["Datetime"] = pd.to_datetime(work["Datetime"], errors="coerce")
    if work["Datetime"].isna().any():
        raise ValueError("Datetime column contains unparsable values")

    agg = dict(OHLCV_AGG)
    for col, func in CORPORATE_ACTION_AGG.items():
        if col in work.columns:
            agg[col] = func

    pieces = []
    for date, day in work.sort_values(["date", "Datetime"]).groupby("date", sort=True):
        day = day.set_index("Datetime")
        resampled = day.resample(freq, origin="start_day", label="left", closed="left").agg(agg)
        resampled = resampled.dropna(subset=["Open", "High", "Low", "Close"])
        if resampled.empty:
            continue
        resampled["date"] = date
        pieces.append(resampled.reset_index())

    if not pieces:
        columns = ["Datetime", *agg, "date"]
        return pd.DataFrame(columns=columns)

    out = pd.concat(pieces, ignore_index=True)
    ordered_cols = [
        "Datetime",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        *[col for col in CORPORATE_ACTION_AGG if col in out.columns],
        "date",
    ]
    return out[ordered_cols].sort_values("Datetime").reset_index(drop=True)
