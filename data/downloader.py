"""
Central Data Download Module
Downloads and caches data for all stock tickers once.
Other files use this cached data.
"""
import sys, os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.join(_HERE, '..')
sys.path.insert(0, _os.path.join(_ROOT, 'core'))
_os.chdir(_ROOT)

import pandas as pd
import yfinance as yf
import os
from datetime import datetime

# Single source of truth: tickers.csv (loaded via core/tickers.py)
from tickers import TICKERS

CACHE_DIR       = "data_cache"
PRICE_DIR       = "data_cache/prices"
TRENDS_DIR      = "data_cache/trends"

def _adjust_prices(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    if 'Adj Close' in df.columns:
        adj_factor = df['Adj Close'] / df['Close']
        df['Open']  = df['Open']  * adj_factor
        df['High']  = df['High']  * adj_factor
        df['Low']   = df['Low']   * adj_factor
        df['Close'] = df['Adj Close']
        df.drop(columns=['Adj Close'], inplace=True, errors='ignore')
    return df

def download_and_cache_data(ticker, force_refresh=False):
    os.makedirs(PRICE_DIR, exist_ok=True)
    cache_file = os.path.join(PRICE_DIR, f"{ticker}.parquet")

    if force_refresh and os.path.exists(cache_file):
        os.remove(cache_file)

    # --- INCREMENTAL UPDATE ---
    if os.path.exists(cache_file):
        try:
            df_cached = pd.read_parquet(cache_file)
            last_date = df_cached.index.max()
            today = pd.Timestamp.today().normalize()

            if last_date >= today - pd.Timedelta(days=1):
                print(f"[CACHE] {ticker} already up to date ({last_date.date()})")
                return df_cached

            fetch_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"[UPDATE] {ticker}: fetching {fetch_start} → today...")
            df_new = yf.download(ticker, start=fetch_start, interval="1d", progress=False, auto_adjust=False)

            if df_new.empty:
                print(f"[CACHE] {ticker} no new data, using cache ({len(df_cached)} rows)")
                return df_cached

            if isinstance(df_new.columns, pd.MultiIndex):
                df_new.columns = df_new.columns.get_level_values(0)
            df_new = _adjust_prices(df_new)

            df_combined = pd.concat([df_cached, df_new])
            df_combined = df_combined[~df_combined.index.duplicated(keep="last")].sort_index()
            df_combined.to_parquet(cache_file)
            print(f"[UPDATED] {ticker}: +{len(df_new)} rows → {len(df_combined)} total")
            return df_combined

        except Exception as e:
            print(f"[ERROR] {ticker} cache update failed: {e}. Re-downloading...")

    # --- FULL DOWNLOAD ---
    print(f"[DOWNLOADING] {ticker} full history...")
    try:
        df = yf.download(ticker, period="max", interval="1d", progress=False, auto_adjust=False)

        if df.empty:
            print(f"[WARNING] No data found for {ticker}!")
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = _adjust_prices(df)

        df.to_parquet(cache_file)
        print(f"[SAVED] {ticker} saved to cache ({len(df)} rows)")
        return df

    except Exception as e:
        print(f"[ERROR] {ticker} download error: {e}")
        return pd.DataFrame()

def download_trends_for_ticker(ticker, force_refresh=False):
    import social_data as sd
    cache_file = os.path.join(TRENDS_DIR, f"{ticker}_trends.parquet")
    if not force_refresh and os.path.exists(cache_file):
        print(f"[CACHE] {ticker} trends already cached, skipping.")
        return
    sd.get_google_trends(ticker, force_refresh=force_refresh)

def download_all_tickers(force_refresh=False, include_trends=True):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os
    print("="*60)
    print("CENTRAL DATA DOWNLOAD STARTING")
    print("="*60)
    print(f"Number of Tickers: {len(TICKERS)}")
    print(f"Cache Directory: {CACHE_DIR}")
    if force_refresh:
        print("[WARNING] All caches will be recreated!")
    print("="*60)

    success_count = 0
    failed_tickers = []

    import time as _time
    for t in TICKERS:
        try:
            df = download_and_cache_data(t, force_refresh)
            if not df.empty:
                success_count += 1
            else:
                failed_tickers.append(t)
        except Exception as e:
            print(f"[ERROR] {t}: {e}")
            failed_tickers.append(t)
        _time.sleep(0.3)

    print("\n" + "="*60)
    print("PRICE DATA DOWNLOAD COMPLETED")
    print("="*60)
    print(f"Successful: {success_count}/{len(TICKERS)}")
    if failed_tickers:
        print(f"Failed Tickers: {', '.join(failed_tickers)}")

    if include_trends:
        print("\n" + "="*60)
        print("GOOGLE TRENDS DOWNLOAD STARTING")
        print("="*60)
        trends_success = 0
        trends_failed = []
        for i, ticker in enumerate(TICKERS, 1):
            print(f"\n[{i}/{len(TICKERS)}] Trends for {ticker}...")
            try:
                download_trends_for_ticker(ticker, force_refresh=force_refresh)
                trends_success += 1
            except Exception as e:
                print(f"[ERROR] {ticker} trends failed: {e}")
                trends_failed.append(ticker)
        print("\n" + "="*60)
        print("TRENDS DOWNLOAD COMPLETED")
        print("="*60)
        print(f"Successful: {trends_success}/{len(TICKERS)}")
        if trends_failed:
            print(f"Failed: {', '.join(trends_failed)}")

    print(f"\nCache Directory: {CACHE_DIR}/")
    print("="*60)

def get_cache_info():
    """Shows cache status."""
    if not os.path.exists(CACHE_DIR):
        print(f"Cache directory not found: {CACHE_DIR}")
        return

    subdirs = [
        ("prices",       PRICE_DIR),
        ("trends",       TRENDS_DIR),
        ("sectors",      "data_cache/sectors"),
        ("earnings",     "data_cache/earnings"),
        ("fundamentals", "data_cache/fundamentals"),
    ]

    for label, subdir in subdirs:
        if not os.path.exists(subdir):
            continue
        files = [f for f in os.listdir(subdir) if f.endswith('.parquet')]
        if not files:
            continue
        print(f"\nCache Status — {label}/ ({len(files)} files):")
        print("-" * 60)
        for file in sorted(files):
            ticker = file.replace('.parquet', '')
            filepath = os.path.join(subdir, file)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            try:
                df = pd.read_parquet(filepath)
                print(f"{ticker:<30} | {len(df):>6} rows | {size_mb:>6.2f} MB | {mod_time.strftime('%Y-%m-%d %H:%M')}")
            except:
                print(f"{ticker:<30} | [ERROR] File could not be read")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--refresh":
        download_all_tickers(force_refresh=True, include_trends=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "--no-trends":
        download_all_tickers(force_refresh=False, include_trends=False)
    elif len(sys.argv) > 1 and sys.argv[1] == "--info":
        get_cache_info()
    else:
        download_all_tickers(force_refresh=False, include_trends=True)
        print("\nNote: To refresh all caches: python data_downloader.py --refresh")
        print("      Price data only:        python data_downloader.py --no-trends")
        print("      Cache info:             python data_downloader.py --info")
