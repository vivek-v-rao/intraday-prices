# intraday-prices

Utilities for fetching, reading, validating, caching, consolidating, and splicing intraday OHLCV price files from multiple vendors.

The project is focused on price data hygiene rather than volatility forecasting. It is intended to prepare clean intraday price files that can be used by downstream analytics projects.

## Features

- Read intraday files from Yahoo Finance, Stooq, Kibot, Polygon, QuantQuote, Databento, Portara, FirstRate Data, Parquet, and pickle.
- Fetch Yahoo Finance intraday snapshots using `yfinance`.
- Fetch Databento 1-minute OHLCV snapshots.
- Cache normalized files to Parquet or pickle.
- Check intraday price files for common data-quality problems.
- Export one-row-per-symbol-date daily summaries for cross-source comparison.
- Consolidate dated snapshot folders into one file per symbol.
- Splice rolling Yahoo Finance snapshots with overlap-based price rebasing.
- Compare prices, returns, and simple realized volatility across vendors.
- Resample intraday bars to lower frequencies.

## Data Layouts

Yahoo Finance snapshots are expected in dated folders:

```text
C:\python\code\stocks\5_minute_prices\20260503\SPY.csv
C:\python\code\stocks\5_minute_prices\20260503\TLT.csv
```

Stooq snapshots are expected in dated folders with zip files:

```text
C:\data\stooq\20250303\5_us_txt.zip
C:\data\stooq\20250303\h_us_txt.zip
```

Tiny sample files for each supported text/CSV vendor format are in:

```text
samples\
```

Each sample includes the header row when the original format has one, plus five price rows. These files are included only to document and test parsers; they are not usable market datasets.

## Install

Core dependencies:

```powershell
pip install pandas numpy pyarrow
```

For Yahoo fetching:

```powershell
pip install yfinance
```

For Databento fetching:

```powershell
pip install databento
```

## Quick Check

The portable Python smoke-test runner uses only committed files under `samples/` by default:

```powershell
cd C:\python\intraday_prices
python xcheck_project.py
```

To run the fuller maintainer workflow against local Yahoo/Stooq data directories:

```powershell
python xcheck_project.py --local-data
```

The test writes temporary outputs to:

```text
C:\data\intraday_prices_test
C:\data\intraday_cache\intraday_prices_test
```

## Fetch Yahoo Intraday Data

Fetch 5-minute data for a few symbols:

```powershell
python xfetch_yahoo_intraday.py --symbols SPY TLT HYG --interval 5m --output-root C:\python\code\stocks --overwrite
```

Fetch both 1-minute and 5-minute data:

```powershell
python xfetch_yahoo_intraday.py --symbols SPY TLT --interval 1m 5m --output-root C:\python\code\stocks --overwrite
```

Fetch from a symbol file:

```powershell
python xfetch_yahoo_intraday.py --symbols-file etf_symbols.txt --interval 5m --output-root C:\python\code\stocks --limit 25
```

## Fetch Databento Intraday Data

`xfetch_databento_intraday.py` downloads Databento OHLCV bars into the same dated snapshot layout used by the consolidation and splicing tools.

The checked-in script uses the placeholder API key `your-api-key`. Pass a real key on the command line or set `DATABENTO_API_KEY` in your environment.

Fetch 1-minute data for selected symbols:

```powershell
python xfetch_databento_intraday.py --symbols SPY TLT HYG --interval 1m --output-root C:\python\code\stocks --api-key YOUR_REAL_KEY --overwrite
```

Using an environment variable:

```powershell
set DATABENTO_API_KEY=YOUR_REAL_KEY
python xfetch_databento_intraday.py --symbols-file etf_symbols.txt --interval 1m --output-root C:\python\code\stocks --limit 25
```

Useful options:

```powershell
python xfetch_databento_intraday.py --symbols SPY --dataset EQUS.MINI --start 2023-03-28 --end 2026-05-07 --output-root C:\python\code\stocks
```

The output files contain:

```text
ts_event,symbol,open,high,low,close,volume
```

## Check Intraday Files

Check selected Yahoo files:

```powershell
python xcheck_intraday.py --input-dir C:\python\code\stocks\5_minute_prices\20260503 --format yahoo --symbols SPY TLT HYG VXX
```

Read all files in a directory, with a limit for a quick run:

```powershell
python xcheck_intraday.py --input-dir C:\python\code\stocks\5_minute_prices\20260503 --format yahoo --limit 100
```

Skip slower checks while experimenting:

```powershell
python xcheck_intraday.py --input-dir C:\python\code\stocks\5_minute_prices\20260503 --format yahoo --no-quality --no-bars-per-day
```

Show symbol/date combinations with anomalous daily maximum intraday ranges:

```powershell
python xcheck_intraday.py C:\data\stooq\consolidated_5min\*.txt --format stooq --show-bad-days 20
```

The anomalous-day table includes `time_High`, `time_Low`, and `time_max_bar_range` to help locate suspicious high or low prices.

Write a daily summary CSV with one row per `symbol,date`:

```powershell
python xcheck_intraday.py C:\data\stooq\consolidated_5min\*.txt --format stooq --output-daily-summary stooq_daily_summary.csv
```

The daily summary includes:

```text
symbol,date,n_bars,n_returns,first_dt,last_dt,
Open,High,Low,Close,Volume,
range,log_range,
co_return,oc_return,cc_return,
realized_var,realized_vol_ann,realized_vol_ann_pct,
max_bar_range,time_High,time_Low,time_max_bar_range
```

This is useful for comparing two data sources after running `xcheck_intraday.py` separately on each source.

## Cache Normalized Files

Write Parquet cache files:

```powershell
python xcache_intraday.py --input-dir C:\python\code\stocks\5_minute_prices\20260503 --format yahoo --output-dir C:\data\intraday_cache\yahoo_5min_20260503 --overwrite
```

Write pickle cache files:

```powershell
python xcache_intraday.py --input-dir C:\python\code\stocks\5_minute_prices\20260503 --format yahoo --output-dir C:\data\intraday_cache\yahoo_5min_20260503_pickle --pickle --overwrite
```

Check cached files:

```powershell
python xcheck_intraday.py C:\data\intraday_cache\yahoo_5min_20260503\*.parquet --format auto
```

## Consolidate Snapshots

`xconsolidate.py` combines dated snapshots as-is. When duplicate bars exist, newer snapshots replace older snapshots. It does not adjust prices.

Yahoo:

```powershell
python xconsolidate.py --vendor yahoo --interval 5min SPY TLT --root C:\python\code\stocks\5_minute_prices --output-dir C:\data\intraday_prices\consolidated_yahoo_5min
```

Stooq 5-minute:

```powershell
python xconsolidate.py --vendor stooq --interval 5min SPY TLT --root C:\data\stooq --output-dir C:\data\intraday_prices\consolidated_stooq_5min
```

Stooq hourly:

```powershell
python xconsolidate.py --vendor stooq --interval hourly SPY TLT --root C:\data\stooq --output-dir C:\data\intraday_prices\consolidated_stooq_hourly
```

Restrict dated folders:

```powershell
python xconsolidate.py --vendor yahoo --interval 5min SPY TLT --root C:\python\code\stocks\5_minute_prices --file-date-min 20260426 --file-date-max 20260503 --output-dir C:\data\intraday_prices_test\consolidated_yahoo_5min
```

## Splice Yahoo Snapshots

Yahoo Finance intraday snapshots may be adjusted to the download date's adjustment basis. That means old snapshots and new snapshots may not be directly spliceable without rebasing.

`xsplice_prices.py` is the safer tool for rolling Yahoo snapshots. By default it:

- processes snapshots newest to oldest,
- estimates `newer Close / older Close` over overlapping bars,
- applies that multiplicative ratio to older `Open`, `High`, `Low`, and `Close`,
- adds only older non-overlapping bars.

```powershell
python xsplice_prices.py --vendor yahoo --interval 5min SPY TLT --root C:\python\code\stocks\5_minute_prices --output-dir C:\data\intraday_prices\spliced_yahoo_5min
```

Controls:

```powershell
python xsplice_prices.py --vendor yahoo --interval 5min SPY TLT --root C:\python\code\stocks\5_minute_prices --min-overlap-bars 100 --max-ratio-mad 0.001 --output-dir C:\data\intraday_prices\spliced_yahoo_5min
```

Raw splice without overlap adjustment:

```powershell
python xsplice_prices.py --vendor yahoo --interval 5min SPY TLT --root C:\python\code\stocks\5_minute_prices --no-adjust-overlap --output-dir C:\data\intraday_prices\spliced_yahoo_5min_raw
```

For Stooq, `xsplice_prices.py` behaves like as-is consolidation.

## Compare Vendors

`xcompare_vendors.py` compares two same-symbol price files from different sources. It reports aligned OHLCV bar differences, intraday return differences, daily range and open/close return differences, and daily realized-volatility differences.

Compare one Stooq file to one Yahoo file:

```powershell
python xcompare_vendors.py C:\data\stooq\consolidated_5min\spy.us.txt C:\data\intraday_prices\spliced_yahoo_5min\SPY.csv --left-format stooq --right-format yahoo
```

Write detailed CSV comparison tables:

```powershell
python xcompare_vendors.py C:\data\stooq\consolidated_5min\spy.us.txt C:\data\intraday_prices\spliced_yahoo_5min\SPY.csv --left-format stooq --right-format yahoo --output-dir C:\data\intraday_compare\spy
```

Batch mode by symbol:

```powershell
python xcompare_vendors.py --left-dir C:\data\stooq\consolidated_5min --right-dir C:\data\intraday_prices\spliced_yahoo_5min --symbols SPY TLT --left-template {lower}.us.txt --right-template {symbol}.csv --left-format stooq --right-format yahoo
```

The lower-level `vendor_compare.py` module still provides the reusable comparison functions.

## Main Files

| File | Purpose |
|---|---|
| `xfetch_yahoo_intraday.py` | Fetch Yahoo intraday snapshots with `yfinance` |
| `xfetch_databento_intraday.py` | Fetch Databento 1-minute OHLCV snapshots |
| `xcheck_intraday.py` | Validate files, show anomalous days, and export daily summaries |
| `xcheck_project.py` | Run portable sample checks and optional local-data smoke tests |
| `xcompare_vendors.py` | Compare same-symbol files from two sources |
| `xcache_intraday.py` | Normalize and cache files to Parquet or pickle |
| `xconsolidate.py` | Consolidate dated snapshots as-is |
| `xsplice_prices.py` | Splice rolling snapshots, with Yahoo overlap rebasing |
| `xread_prices.py` | Minimal read-speed benchmark |
| `io_intraday.py` | Vendor readers and normalization |
| `data_quality.py` | Bad OHLCV checks |
| `intraday_bars.py` | Intraday bar resampling |
| `intraday_returns.py` | Return and bar-interval helpers |
| `vendor_compare.py` | Cross-vendor comparisons |
| `file_utils.py` | Windows-friendly file glob expansion |

## Notes

- `xconsolidate.py` preserves vendor data as-is and does not adjust for splits or dividends.
- `xsplice_prices.py` adjusts Yahoo price columns using overlap ratios, but it does not adjust volume.
- Always inspect `manifest.csv` after consolidation or splicing. For Yahoo splicing, it reports overlap counts, adjustment ratio ranges, and ratio stability.
- This project prepares price data. Realized volatility modelling and forecasting should live in a separate project that consumes these cleaned files.
