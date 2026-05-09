from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_MAX_RANGE = {
    "intraday": 0.05,
    "daily": 0.20,
}
DEFAULT_MAX_OC_RETURN = {
    "intraday": 0.03,
    "daily": 0.15,
}
DEFAULT_MAX_PREV_CLOSE_RETURN = {
    "intraday": 0.05,
    "daily": 0.20,
}
DEFAULT_MIN_LOW_TO_OC = {
    "intraday": 0.95,
    "daily": 0.80,
}
DEFAULT_MAX_HIGH_TO_OC = {
    "intraday": 1.05,
    "daily": 1.20,
}


def _threshold(defaults: dict[str, float], frequency: str, override: float | None) -> float:
    if override is not None:
        return override
    if frequency not in defaults:
        raise ValueError(f"unknown frequency {frequency!r}; expected one of {sorted(defaults)}")
    return defaults[frequency]


def _id_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in ["<TICKER>", "Ticker", "ticker", "<DATE>", "date", "Datetime", "<TIME>", "time"]
        if col in df.columns
    ]


def _flag_rows(df: pd.DataFrame, mask: pd.Series, reason: str, details: pd.Series) -> pd.DataFrame:
    if not mask.any():
        return pd.DataFrame()
    cols = [
        *_id_columns(df),
        "Open",
        "High",
        "Low",
        "Close",
        *[col for col in ["Volume"] if col in df.columns],
    ]
    out = df.loc[mask, cols].copy()
    out.insert(0, "reason", reason)
    out["detail"] = details.loc[mask].astype(str)
    return out


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with common vendor OHLCV columns mapped to canonical names."""
    rename = {
        "<OPEN>": "Open",
        "<HIGH>": "High",
        "<LOW>": "Low",
        "<CLOSE>": "Close",
        "<VOL>": "Volume",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    out = df.rename(columns={k: v for k, v in rename.items() if k in df.columns}).copy()
    return out


def find_bad_ohlcv(
    df: pd.DataFrame,
    frequency: str = "intraday",
    max_range: float | None = None,
    max_oc_return: float | None = None,
    max_prev_close_return: float | None = None,
    min_low_to_oc: float | None = None,
    max_high_to_oc: float | None = None,
) -> pd.DataFrame:
    """Return rows with suspicious OHLCV values and reason codes.

    The checks are deliberately conservative and intended to flag rows for
    review, not to repair data automatically.
    """
    work = normalize_ohlcv_columns(df)
    required = {"Open", "High", "Low", "Close"}
    missing = required - set(work.columns)
    if missing:
        raise ValueError(f"missing OHLC columns: {sorted(missing)}")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    max_range = _threshold(DEFAULT_MAX_RANGE, frequency, max_range)
    max_oc_return = _threshold(DEFAULT_MAX_OC_RETURN, frequency, max_oc_return)
    max_prev_close_return = _threshold(
        DEFAULT_MAX_PREV_CLOSE_RETURN,
        frequency,
        max_prev_close_return,
    )
    min_low_to_oc = _threshold(DEFAULT_MIN_LOW_TO_OC, frequency, min_low_to_oc)
    max_high_to_oc = _threshold(DEFAULT_MAX_HIGH_TO_OC, frequency, max_high_to_oc)

    pieces = []
    o = work["Open"]
    h = work["High"]
    l = work["Low"]
    c = work["Close"]

    nonpositive = (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)
    nonpositive = nonpositive | o.isna() | h.isna() | l.isna() | c.isna()
    pieces.append(_flag_rows(work, nonpositive, "nonpositive_or_missing_ohlc", pd.Series("", index=work.index)))

    ordering = (h < l) | (h < o) | (h < c) | (l > o) | (l > c)
    pieces.append(_flag_rows(work, ordering, "ohlc_order_violation", pd.Series("", index=work.index)))

    valid = ~(nonpositive | ordering)
    bar_range = h / l - 1.0
    pieces.append(
        _flag_rows(
            work,
            valid & (bar_range > max_range),
            "extreme_high_low_range",
            bar_range,
        )
    )

    oc_return = np.log(c / o).abs()
    pieces.append(
        _flag_rows(
            work,
            valid & (oc_return > max_oc_return),
            "extreme_open_close_return",
            oc_return,
        )
    )

    oc_min = pd.concat([o, c], axis=1).min(axis=1)
    oc_max = pd.concat([o, c], axis=1).max(axis=1)
    low_to_oc = l / oc_min
    high_to_oc = h / oc_max
    pieces.append(
        _flag_rows(
            work,
            valid & (low_to_oc < min_low_to_oc),
            "low_far_from_open_close",
            low_to_oc,
        )
    )
    pieces.append(
        _flag_rows(
            work,
            valid & (high_to_oc > max_high_to_oc),
            "high_far_from_open_close",
            high_to_oc,
        )
    )

    prev_close = c.shift(1)
    prev_ret = np.log(c / prev_close).abs()
    pieces.append(
        _flag_rows(
            work,
            valid & prev_close.notna() & (prev_close > 0) & (prev_ret > max_prev_close_return),
            "extreme_close_to_prev_close_return",
            prev_ret,
        )
    )

    if "Volume" in work.columns:
        bad_volume = work["Volume"].isna() | (work["Volume"] < 0)
        pieces.append(_flag_rows(work, bad_volume, "negative_or_missing_volume", pd.Series("", index=work.index)))

    nonempty = [piece for piece in pieces if not piece.empty]
    if not nonempty:
        return pd.DataFrame()
    return pd.concat(nonempty, ignore_index=True)
