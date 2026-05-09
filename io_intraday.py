from __future__ import annotations

from pathlib import Path

import pandas as pd


VENDOR_AUTO = "auto"
VENDOR_YAHOO = "yahoo"
VENDOR_STOOQ = "stooq"
VENDOR_KIBOT = "kibot"
VENDOR_POLYGON = "polygon"
VENDOR_QUANTQUOTE = "quantquote"
VENDOR_PORTARA = "portara"
VENDOR_DATABENTO = "databento"
VENDOR_GENERIC = "generic"
VENDOR_FIRST_RATE = "first_rate"
VENDOR_PARQUET = "parquet"
VENDOR_PICKLE = "pickle"
INTRADAY_VENDORS = (
    VENDOR_AUTO,
    VENDOR_YAHOO,
    VENDOR_STOOQ,
    VENDOR_KIBOT,
    VENDOR_POLYGON,
    VENDOR_QUANTQUOTE,
    VENDOR_PORTARA,
    VENDOR_DATABENTO,
    VENDOR_GENERIC,
    VENDOR_FIRST_RATE,
    VENDOR_PARQUET,
    VENDOR_PICKLE,
)

CANONICAL_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
CORPORATE_ACTION_COLUMNS = ["Dividends", "Stock Splits", "Capital Gains"]
YAHOO_REQUIRED_COLUMNS = {"Datetime", "Open", "High", "Low", "Close"}
STOOQ_COLUMN_MAP = {
    "<OPEN>": "Open",
    "<HIGH>": "High",
    "<LOW>": "Low",
    "<CLOSE>": "Close",
    "<VOL>": "Volume",
}
STOOQ_REQUIRED_COLUMNS = {"<DATE>", "<TIME>", *STOOQ_COLUMN_MAP}
KIBOT_COLUMNS = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]
QUANTQUOTE_COLUMNS = [
    "Date",
    "Time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]
PORTARA_COLUMNS = [
    "Symbol",
    "Date",
    "Time",
    "Open",
    "High",
    "Low",
    "Close",
    "TickCount",
    "Volume",
]
POLYGON_COLUMN_MAP = {
    "datetime": "Datetime",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}
POLYGON_REQUIRED_COLUMNS = set(POLYGON_COLUMN_MAP)
DATABENTO_COLUMN_MAP = {
    "ts_event": "Datetime",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}
DATABENTO_REQUIRED_COLUMNS = set(DATABENTO_COLUMN_MAP)
GENERIC_COLUMN_MAP = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}
GENERIC_TIME_COLUMNS = ("timestamp", "datetime")
GENERIC_REQUIRED_COLUMNS = set(GENERIC_COLUMN_MAP)


def infer_intraday_vendor(columns: list[str] | pd.Index) -> str:
    """Infer an intraday file vendor from its column names."""
    column_set = set(columns)
    if STOOQ_REQUIRED_COLUMNS <= column_set:
        return VENDOR_STOOQ
    if YAHOO_REQUIRED_COLUMNS <= column_set:
        return VENDOR_YAHOO
    if POLYGON_REQUIRED_COLUMNS <= column_set:
        return VENDOR_POLYGON
    if DATABENTO_REQUIRED_COLUMNS <= column_set:
        return VENDOR_DATABENTO
    if GENERIC_REQUIRED_COLUMNS <= column_set and any(
        time_col in column_set for time_col in GENERIC_TIME_COLUMNS
    ):
        return VENDOR_GENERIC
    raise ValueError(
        "could not auto-detect intraday file format; pass vendor='stooq' "
        "or vendor='yahoo' or vendor='kibot' or vendor='polygon' "
        "or vendor='quantquote' or vendor='portara' or vendor='databento' "
        "or vendor='generic'"
    )


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [
        col
        for col in [*CANONICAL_PRICE_COLUMNS, *CORPORATE_ACTION_COLUMNS]
        if col in out.columns
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _ensure_corporate_action_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in CORPORATE_ACTION_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
    return out


def _finalize_intraday(df: pd.DataFrame) -> pd.DataFrame:
    out = _coerce_numeric(_ensure_corporate_action_columns(df))
    missing_prices = set(CANONICAL_PRICE_COLUMNS) - set(out.columns)
    if missing_prices:
        raise ValueError(f"missing normalized price columns: {sorted(missing_prices)}")
    if out["Datetime"].isna().any():
        raise ValueError("Datetime column contains unparsable values")
    out = out.sort_values("Datetime").reset_index(drop=True)
    out["date"] = out["Datetime"].dt.date
    return out


def normalize_yahoo_intraday(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a Yahoo-style intraday CSV."""
    missing = YAHOO_REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"missing Yahoo columns: {sorted(missing)}")

    out = raw.copy()
    dt = pd.to_datetime(out["Datetime"], errors="coerce", utc=True)
    if dt.isna().any():
        raise ValueError("Datetime column contains unparsable values")
    out["Datetime"] = dt.dt.tz_convert("America/New_York")
    return _finalize_intraday(out)


def normalize_stooq_intraday(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a Stooq intraday text file."""
    missing = STOOQ_REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"missing Stooq columns: {sorted(missing)}")

    out = raw.rename(columns=STOOQ_COLUMN_MAP).copy()
    time_text = out["<TIME>"].astype(str).str.zfill(6)
    out["Datetime"] = pd.to_datetime(
        out["<DATE>"].astype(str) + time_text,
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )
    return _finalize_intraday(out)


def looks_like_kibot_intraday(raw: pd.DataFrame) -> bool:
    """Return True when a headerless CSV looks like Kibot intraday data."""
    if raw.shape[1] != len(KIBOT_COLUMNS) or raw.empty:
        return False
    sample = raw.head(10)
    dates = pd.to_datetime(sample.iloc[:, 0], format="%m/%d/%Y", errors="coerce")
    times = pd.to_datetime(sample.iloc[:, 1].astype(str), format="%H:%M", errors="coerce")
    return bool(dates.notna().all() and times.notna().all())


def normalize_kibot_intraday(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a Kibot headerless intraday text file."""
    if raw.shape[1] != len(KIBOT_COLUMNS):
        raise ValueError(
            f"Kibot intraday files should have {len(KIBOT_COLUMNS)} columns: "
            "Date, Time, Open, High, Low, Close, Volume"
        )

    out = raw.copy()
    out.columns = KIBOT_COLUMNS
    out["Datetime"] = pd.to_datetime(
        out["Date"].astype(str) + " " + out["Time"].astype(str),
        format="%m/%d/%Y %H:%M",
        errors="coerce",
    )
    return _finalize_intraday(out)


def looks_like_quantquote_intraday(raw: pd.DataFrame) -> bool:
    """Return True when a headerless CSV looks like QuantQuote intraday data."""
    if raw.shape[1] < len(QUANTQUOTE_COLUMNS) or raw.empty:
        return False
    sample = raw.head(10)
    date_text = sample.iloc[:, 0].astype(str).str.replace(r"\.0$", "", regex=True)
    time_text = sample.iloc[:, 1].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    dates = pd.to_datetime(date_text, format="%Y%m%d", errors="coerce")
    times = pd.to_datetime(time_text, format="%H%M", errors="coerce")
    return bool(dates.notna().all() and times.notna().all())


def normalize_quantquote_intraday(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a QuantQuote headerless intraday CSV."""
    if raw.shape[1] < len(QUANTQUOTE_COLUMNS):
        raise ValueError(
            f"QuantQuote intraday files should have at least {len(QUANTQUOTE_COLUMNS)} "
            "columns: Date, Time, Open, High, Low, Close, Volume"
        )

    out = raw.iloc[:, : len(QUANTQUOTE_COLUMNS)].copy()
    out.columns = QUANTQUOTE_COLUMNS
    date_text = out["Date"].astype(str).str.replace(r"\.0$", "", regex=True)
    time_text = out["Time"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    out["Datetime"] = pd.to_datetime(
        date_text + time_text,
        format="%Y%m%d%H%M",
        errors="coerce",
    )
    return _finalize_intraday(out)


def looks_like_portara_intraday(raw: pd.DataFrame) -> bool:
    """Return True when a headerless CSV looks like Portara intraday data."""
    if raw.shape[1] != len(PORTARA_COLUMNS) or raw.empty:
        return False
    sample = raw.head(10)
    date_text = sample.iloc[:, 1].astype(str).str.replace(r"\.0$", "", regex=True)
    time_text = sample.iloc[:, 2].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    dates = pd.to_datetime(date_text, format="%Y%m%d", errors="coerce")
    times = pd.to_datetime(time_text, format="%H%M", errors="coerce")
    return bool(dates.notna().all() and times.notna().all())


def normalize_portara_intraday(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a Portara headerless intraday text file."""
    if raw.shape[1] != len(PORTARA_COLUMNS):
        raise ValueError(
            f"Portara intraday files should have {len(PORTARA_COLUMNS)} columns: "
            "Symbol, Date, Time, Open, High, Low, Close, TickCount, Volume"
        )

    out = raw.copy()
    out.columns = PORTARA_COLUMNS
    date_text = out["Date"].astype(str).str.replace(r"\.0$", "", regex=True)
    time_text = out["Time"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    out["Datetime"] = pd.to_datetime(
        date_text + time_text,
        format="%Y%m%d%H%M",
        errors="coerce",
    )
    return _finalize_intraday(out)


def normalize_polygon_intraday(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a Polygon-style intraday CSV."""
    missing = POLYGON_REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"missing Polygon columns: {sorted(missing)}")

    out = raw.rename(columns=POLYGON_COLUMN_MAP).copy()
    dt = pd.to_datetime(out["Datetime"], errors="coerce")
    if dt.isna().any():
        raise ValueError("Datetime column contains unparsable values")
    out["Datetime"] = dt
    return _finalize_intraday(out)


def normalize_databento_intraday(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a Databento OHLCV intraday CSV."""
    missing = DATABENTO_REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"missing Databento columns: {sorted(missing)}")

    out = raw.rename(columns=DATABENTO_COLUMN_MAP).copy()
    dt = pd.to_datetime(out["Datetime"], errors="coerce", utc=True)
    if dt.isna().any():
        raise ValueError("Datetime column contains unparsable values")
    out["Datetime"] = dt.dt.tz_convert("America/New_York")
    return _finalize_intraday(out)


def normalize_generic_intraday(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a generic lowercase OHLCV intraday CSV."""
    missing = GENERIC_REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"missing generic OHLCV columns: {sorted(missing)}")

    time_col = next((col for col in GENERIC_TIME_COLUMNS if col in raw.columns), None)
    if time_col is None:
        raise ValueError("generic OHLCV files require a timestamp or datetime column")

    out = raw.rename(columns={**GENERIC_COLUMN_MAP, time_col: "Datetime"}).copy()
    dt = pd.to_datetime(out["Datetime"], errors="coerce")
    if dt.isna().any():
        raise ValueError("Datetime column contains unparsable values")
    out["Datetime"] = dt
    return _finalize_intraday(out)


def normalize_parquet_intraday(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a cached canonical intraday Parquet file."""
    if "Datetime" not in raw.columns:
        raise ValueError("Parquet intraday files require a Datetime column")
    return _finalize_intraday(raw)


def normalize_pickle_intraday(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a cached canonical intraday pickle file."""
    if "Datetime" not in raw.columns:
        raise ValueError("Pickle intraday files require a Datetime column")
    return _finalize_intraday(raw)


def read_intraday_prices(path: str | Path, vendor: str = VENDOR_AUTO) -> pd.DataFrame:
    """Read intraday prices from a supported vendor format.

    The returned DataFrame uses the project's canonical columns:
    Datetime, Open, High, Low, Close, Volume, Dividends, Stock Splits,
    Capital Gains, and date.
    """
    path = Path(path)
    vendor = vendor.lower()
    if vendor == VENDOR_FIRST_RATE:
        vendor = VENDOR_GENERIC
    if vendor not in INTRADAY_VENDORS:
        raise ValueError(f"unknown vendor {vendor!r}; expected one of {INTRADAY_VENDORS}")

    if vendor == VENDOR_KIBOT:
        return normalize_kibot_intraday(pd.read_csv(path, header=None))
    if vendor == VENDOR_QUANTQUOTE:
        return normalize_quantquote_intraday(pd.read_csv(path, header=None))
    if vendor == VENDOR_PORTARA:
        return normalize_portara_intraday(pd.read_csv(path, header=None))
    if vendor == VENDOR_PARQUET:
        return normalize_parquet_intraday(pd.read_parquet(path))
    if vendor == VENDOR_PICKLE:
        return normalize_pickle_intraday(pd.read_pickle(path))
    if vendor == VENDOR_AUTO and path.suffix.lower() == ".parquet":
        return normalize_parquet_intraday(pd.read_parquet(path))
    if vendor == VENDOR_AUTO and path.suffix.lower() in {".pkl", ".pickle"}:
        return normalize_pickle_intraday(pd.read_pickle(path))

    raw = pd.read_csv(path)
    if vendor == VENDOR_AUTO:
        try:
            vendor = infer_intraday_vendor(raw.columns)
        except ValueError:
            raw_no_header = pd.read_csv(path, header=None)
            if looks_like_kibot_intraday(raw_no_header):
                return normalize_kibot_intraday(raw_no_header)
            if looks_like_quantquote_intraday(raw_no_header):
                return normalize_quantquote_intraday(raw_no_header)
            if looks_like_portara_intraday(raw_no_header):
                return normalize_portara_intraday(raw_no_header)
            raise
    if vendor == VENDOR_YAHOO:
        return normalize_yahoo_intraday(raw)
    if vendor == VENDOR_STOOQ:
        return normalize_stooq_intraday(raw)
    if vendor == VENDOR_POLYGON:
        return normalize_polygon_intraday(raw)
    if vendor == VENDOR_DATABENTO:
        return normalize_databento_intraday(raw)
    if vendor == VENDOR_GENERIC:
        return normalize_generic_intraday(raw)
    raise AssertionError(f"unhandled vendor {vendor!r}")


def read_intraday_csv(path: str | Path) -> pd.DataFrame:
    """Compatibility wrapper for older scripts."""
    return read_intraday_prices(path, vendor=VENDOR_AUTO)
