import os
import time
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

CACHE_DIR = "data_cache/trends"

# Tickers that had a different symbol in the past.
# List is [current_search_term, old1, old2, ...] — tried in order per time chunk.
# Add new entries whenever a major SP500 ticker changes name/symbol.
TICKER_ALIASES = {
    "META":  ["META stock",  "FB stock"],        # Facebook → Meta (Oct 2021)
    "GOOGL": ["GOOGL stock", "GOOG stock"],      # GOOG Class A; GOOGL added Apr 2014
    "PARA":  ["PARA stock",  "VIAC stock"],      # ViacomCBS → Paramount (Feb 2022)
    "WBD":   ["WBD stock",   "DISCA stock"],     # Discovery → Warner Bros. Discovery (Apr 2022)
    "BRK-B": ["BRK-B stock", "BRK.B stock"],
    "BF-B":  ["BF-B stock",  "BF.B stock"],
}


def get_google_trends(ticker: str, start_year: int = 2012,
                      force_refresh: bool = False,
                      cache_only: bool = False) -> pd.DataFrame:
    cache_file = os.path.join(CACHE_DIR, f"{ticker}_trends.parquet")

    if os.path.exists(cache_file):
        try:
            df_cached = pd.read_parquet(cache_file)
            if cache_only:
                return df_cached

            last_date = df_cached.index.max()

            if not force_refresh:
                print(f"[SOCIAL] {ticker} trends updating from {last_date.date()}...")
                start_year = last_date.year
        except Exception:
            if cache_only:
                return pd.DataFrame()

    if cache_only:
        return pd.DataFrame()

    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("[SOCIAL] pytrends not installed. Run: pip install pytrends")
        return pd.DataFrame()

    print(f"[SOCIAL] Downloading Google Trends for {ticker} (this may take a moment)...")

    try:
        import urllib3.util.retry as _retry_mod
        _orig_retry_init = _retry_mod.Retry.__init__

        def _patched_retry_init(self, *args, **kwargs):
            if "method_whitelist" in kwargs:
                kwargs.setdefault("allowed_methods", kwargs.pop("method_whitelist"))
            _orig_retry_init(self, *args, **kwargs)

        _retry_mod.Retry.__init__ = _patched_retry_init
    except Exception:
        pass

    try:
        pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25), retries=2, backoff_factor=0.5)
    except TypeError:
        try:
            pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25))
        except Exception as e:
            print(f"[SOCIAL] Could not create TrendReq: {e}")
            return pd.DataFrame()
    except Exception as e:
        print(f"[SOCIAL] Could not create TrendReq: {e}")
        return pd.DataFrame()

    all_chunks: list[pd.DataFrame] = []
    search_terms = TICKER_ALIASES.get(ticker, [f"{ticker} stock"])

    chunk_start = pd.Timestamp(f"{start_year}-01-01")
    today       = pd.Timestamp.today().normalize()

    while chunk_start < today:
        chunk_end = min(chunk_start + pd.DateOffset(years=4), today)
        timeframe = f"{chunk_start.strftime('%Y-%m-%d')} {chunk_end.strftime('%Y-%m-%d')}"

        MAX_RETRIES = 4
        chunk_added = False

        for search_term in search_terms:
            if chunk_added:
                break
            for attempt in range(MAX_RETRIES):
                try:
                    pytrends.build_payload([search_term], cat=0, timeframe=timeframe, geo="", gprop="")
                    df_chunk = pytrends.interest_over_time()

                    if df_chunk is not None and not df_chunk.empty and search_term in df_chunk.columns:
                        series = df_chunk[[search_term]].rename(columns={search_term: "search_interest"})
                        if series["search_interest"].sum() > 0:
                            all_chunks.append(series)
                            chunk_added = True
                    break

                except Exception as e:
                    is_rate_limit = "429" in str(e) or "too many" in str(e).lower()

                    if is_rate_limit and attempt < MAX_RETRIES - 1:
                        wait = 30 * (2 ** attempt)
                        print(f"[SOCIAL] Rate limited on {timeframe}. Waiting {wait}s (retry {attempt + 1}/{MAX_RETRIES - 1})...")
                        time.sleep(wait)
                    else:
                        if not is_rate_limit:
                            print(f"[SOCIAL] Trends chunk {timeframe} failed: {e}")
                        else:
                            print(f"[SOCIAL] Gave up on {timeframe} after {MAX_RETRIES} attempts.")
                        break

            if not chunk_added and len(search_terms) > 1:
                time.sleep(3)

        chunk_start = chunk_end
        time.sleep(6)

    if not all_chunks:
        print(f"[SOCIAL] No Trends data collected for {ticker}")
        return pd.DataFrame()

    df_trends = pd.concat(all_chunks)
    df_trends = df_trends[~df_trends.index.duplicated(keep="last")].sort_index()

    max_val = df_trends["search_interest"].max()
    if max_val > 0:
        df_trends["search_interest"] = df_trends["search_interest"] / max_val

    daily_idx = pd.date_range(start=df_trends.index[0], end=df_trends.index[-1], freq="D")
    df_daily  = df_trends.reindex(daily_idx).interpolate(method="linear")
    df_daily.index.name = "Date"

    os.makedirs(CACHE_DIR, exist_ok=True)
    df_daily.to_parquet(cache_file)
    print(f"[SOCIAL] {ticker}: Google Trends saved ({len(df_daily)} daily rows).")

    return df_daily


def merge_trends(df_price: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df_trends = get_google_trends(ticker, cache_only=True)

    if df_trends.empty:
        return df_price

    aligned = df_trends.reindex(df_price.index).ffill().fillna(0.0)

    if aligned["search_interest"].notna().any():
        result = df_price.copy()
        for window in [14, 30]:
            rolling_mean = aligned["search_interest"].rolling(window, min_periods=1).mean()
            chg = (aligned["search_interest"] - rolling_mean) / rolling_mean.replace(0, float("nan"))
            result[f"search_interest_chg_{window}d"] = chg.fillna(0.0).clip(-2, 2)
        return result

    return df_price


if __name__ == "__main__":
    import sys

    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "NVDA"]

    for t in tickers:
        print(f"\n=== {t} ===")
        df = get_google_trends(t, force_refresh=True)
        if not df.empty:
            print(df.tail(5))
