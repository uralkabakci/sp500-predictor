import os
import sys
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..')
sys.path.insert(0, os.path.join(_ROOT, 'core'))

_ET = ZoneInfo("America/New_York")

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

import database as db
import predictor
import pandas_market_calendars as mcal

_NYSE_CAL = mcal.get_calendar("NYSE")

# ── Trends refresh ────────────────────────────────────────────────────────────
# Google Trends data is published weekly (Thursdays US ET).
# We pull on Friday 09:00 UTC to ensure the latest week is available.

# ── Price + signal refresh ────────────────────────────────────────────────────
SIGNAL_INTERVAL_MINUTES = 60


def _market_is_open() -> bool:
    now_et = datetime.now(_ET)
    if now_et.weekday() >= 5:
        return False
    if not (dtime(9, 30) <= now_et.time() <= dtime(16, 0)):
        return False
    today_str = now_et.date().isoformat()
    schedule  = _NYSE_CAL.schedule(start_date=today_str, end_date=today_str)
    return not schedule.empty


def refresh_signals():
    if not _market_is_open():
        return
    db.log_event("INFO", "scheduler", "Signal refresh started")
    try:
        def _save_signal(sig):
            db.upsert_signal(
                ticker=sig["ticker"],
                days=sig["days"],
                prob=sig["prob"],
                entry_price=sig["entry_price"],
                target_price=sig["target_price"],
                stop_loss_price=sig["stop_loss_price"],
                signal_date=sig["signal_date"],
                entry_time=sig.get("entry_time"),
            )

        blocked = db.get_stoploss_cooldown_blocked(predictor.STOPLOSS_COOLDOWN)
        signals = predictor.get_latest_signals(log_fn=db.log_event, signal_fn=_save_signal, blocked=blocked)

        # Reuse OHLC already fetched inside get_latest_signals — no second API call
        live_ohlc = predictor._live_ohlc_ref
        for t, bar in live_ohlc.items():
            db.update_price_cache(t, bar["close"])

        db.close_expired_signals(live_ohlc)

        db.log_event(
            "INFO", "scheduler",
            f"Signal refresh done — {len(signals)} signals generated, "
            f"{len(live_ohlc)} prices updated"
        )
    except Exception as e:
        db.log_event("ERROR", "scheduler", f"Signal refresh failed: {e}")


def refresh_price_cache():
    """Daily: fetch incremental price data for all tickers and save to cache."""
    import time
    import yfinance as yf
    import pandas as pd

    db.log_event("INFO", "scheduler", "Daily price cache update started")
    try:
        CACHE_DIR = os.path.join(_ROOT, "data_cache", "prices")
        tickers = predictor.TICKERS
        total   = len(tickers)
        updated = 0
        errors  = 0

        for i, ticker in enumerate(tickers, 1):
            try:
                cache_file = os.path.join(CACHE_DIR, f"{ticker}.parquet")
                if not os.path.exists(cache_file):
                    continue
                df_cached   = pd.read_parquet(cache_file)
                last_date   = df_cached.index.max()
                fetch_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

                new_data = yf.download(ticker, start=fetch_start, interval="1d",
                                       progress=False, auto_adjust=False)
                if new_data.empty:
                    continue
                if isinstance(new_data.columns, pd.MultiIndex):
                    new_data.columns = new_data.columns.get_level_values(0)
                if 'Adj Close' in new_data.columns:
                    adj = new_data['Adj Close'] / new_data['Close']
                    new_data['Open']  = new_data['Open']  * adj
                    new_data['High']  = new_data['High']  * adj
                    new_data['Low']   = new_data['Low']   * adj
                    new_data['Close'] = new_data['Adj Close']
                    new_data.drop(columns=['Adj Close'], inplace=True, errors='ignore')

                df = pd.concat([df_cached, new_data[~new_data.index.isin(df_cached.index)]])
                df.sort_index(inplace=True)
                df.to_parquet(cache_file)
                updated += 1

            except Exception as e:
                errors += 1
                db.log_event("WARN", "scheduler", f"Price cache skip {ticker}: {e}")

            if i % 25 == 0:
                db.log_event("INFO", "scheduler",
                             f"Price cache progress: {i}/{total}, updated={updated}, errors={errors}")

            time.sleep(0.2)  # avoid yfinance rate limit

        db.log_event("INFO", "scheduler",
                     f"Daily price cache done — {updated}/{total} tickers saved, {errors} errors")
    except Exception as e:
        db.log_event("ERROR", "scheduler", f"Daily price cache failed: {e}")


def refresh_reference_etfs_daily():
    """
    Daily: incrementally update SPY, GLD, and all sector ETF parquet caches.
    These are NOT in TICKERS so refresh_price_cache misses them. Without this,
    the cached reference series can lag by days, breaking sector_rel/spy_rel
    features in backtests (live has in-memory override but backtest does not).
    """
    import time
    import yfinance as yf
    import pandas as pd

    db.log_event("INFO", "scheduler", "Daily reference ETF cache update started")
    try:
        sys.path.insert(0, os.path.join(_ROOT, 'core'))
        import sector_data as sd

        SECTOR_DIR = os.path.join(_ROOT, "data_cache", "sectors")
        os.makedirs(SECTOR_DIR, exist_ok=True)

        etfs   = sorted(set(["SPY", "GLD"]) | set(sd.TICKER_SECTOR.values()))
        ok     = 0
        errors = 0

        for etf in etfs:
            cache_file = os.path.join(SECTOR_DIR, f"{etf}_sector.parquet")
            try:
                if os.path.exists(cache_file):
                    df_cached   = pd.read_parquet(cache_file)
                    last_date   = df_cached.index.max()
                    fetch_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                else:
                    df_cached   = None
                    fetch_start = "2010-01-01"

                new_data = yf.download(etf, start=fetch_start, interval="1d",
                                       progress=False, auto_adjust=True)
                if new_data.empty:
                    if df_cached is None:
                        errors += 1
                    continue
                if isinstance(new_data.columns, pd.MultiIndex):
                    new_data.columns = new_data.columns.get_level_values(0)

                # Don't .squeeze() — single-row Series collapses to scalar.
                close = new_data["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                if not isinstance(close, pd.Series) or close.empty:
                    continue
                close.index = pd.to_datetime(close.index).normalize()
                close.name  = "Close"
                close_df    = close.to_frame()

                if df_cached is not None:
                    merged = pd.concat([df_cached, close_df[~close_df.index.isin(df_cached.index)]])
                    merged.sort_index(inplace=True)
                else:
                    merged = close_df

                merged.to_parquet(cache_file)
                ok += 1
            except Exception as e:
                errors += 1
                db.log_event("WARN", "scheduler", f"Reference ETF skip {etf}: {e}")

            time.sleep(0.4)  # be gentle on yfinance

        db.log_event("INFO", "scheduler",
                     f"Daily reference ETF cache done — {ok}/{len(etfs)} ETFs updated, {errors} errors")
    except Exception as e:
        db.log_event("ERROR", "scheduler", f"Daily reference ETF cache failed: {e}")


def refresh_sector_daily():
    """Daily: recompute sector/SPY/Gold relative metrics (local, no rate limit)."""
    db.log_event("INFO", "scheduler", "Daily sector refresh started")
    try:
        sys.path.insert(0, os.path.join(_ROOT, 'core'))
        import sector_data as sec
        import data_processor as dp
        ok = 0
        for ticker in predictor.TICKERS:
            try:
                raw = dp.get_data(ticker)
                if raw is not None and not raw.empty:
                    sec.merge_sector_relative(raw, ticker, windows=[14, 30])
                    sec.merge_spy_relative(raw, windows=[14, 30])
                    sec.merge_gold_spy(raw, windows=[14, 30])
                    ok += 1
            except Exception:
                pass
        db.log_event("INFO", "scheduler",
                     f"Daily sector refresh done — {ok}/{len(predictor.TICKERS)} tickers")
    except Exception as e:
        db.log_event("ERROR", "scheduler", f"Daily sector refresh failed: {e}")


def refresh_trends_daily():
    """
    Daily: refresh Google Trends for all tickers.
    Rate-limit aware: retries failed tickers with exponential backoff until all succeed.
    """
    import time, random
    db.log_event("INFO", "scheduler", "Daily trends refresh started")
    try:
        sys.path.insert(0, os.path.join(_ROOT, 'core'))
        import social_data as sd

        tickers     = list(predictor.TICKERS)
        pending     = list(tickers)
        succeeded   = set()
        max_passes  = 6
        base_sleep  = 2.0   # seconds between OK requests
        backoff     = 30.0  # initial backoff on rate-limit (seconds)

        for pass_num in range(1, max_passes + 1):
            if not pending:
                break
            db.log_event("INFO", "scheduler",
                         f"Trends pass {pass_num}/{max_passes}: {len(pending)} tickers")
            failed = []
            for i, ticker in enumerate(pending, 1):
                try:
                    sd.get_google_trends(ticker, force_refresh=True)
                    succeeded.add(ticker)
                    time.sleep(base_sleep + random.random())  # jitter
                except Exception as e:
                    msg = str(e).lower()
                    if "429" in msg or "too many" in msg or "rate" in msg:
                        db.log_event("WARN", "scheduler",
                                     f"Trends rate-limited on {ticker}; sleeping {backoff:.0f}s")
                        time.sleep(backoff)
                        backoff = min(backoff * 1.5, 600.0)  # cap 10 min
                    failed.append(ticker)

                if i % 50 == 0:
                    db.log_event("INFO", "scheduler",
                                 f"Trends pass {pass_num}: {i}/{len(pending)} processed, {len(failed)} failed so far")

            pending = failed
            if pending:
                wait = backoff * 2
                db.log_event("INFO", "scheduler",
                             f"Trends pass {pass_num} done. Failed: {len(pending)}. Waiting {wait:.0f}s before retry")
                time.sleep(wait)

        if pending:
            db.log_event("ERROR", "scheduler",
                         f"Trends refresh INCOMPLETE: {len(pending)} tickers still failing. Sample: {pending[:10]}")
        else:
            db.log_event("INFO", "scheduler",
                         f"Daily trends refresh complete — {len(succeeded)}/{len(tickers)} tickers")
    except Exception as e:
        db.log_event("ERROR", "scheduler", f"Daily trends refresh failed: {e}")


def refresh_earnings_daily():
    """Daily: update earnings/fundamentals data for all tickers after market close."""
    db.log_event("INFO", "scheduler", "Daily earnings refresh started")
    try:
        import earnings_data as ed
        errors = 0
        for ticker in predictor.TICKERS:
            try:
                ed.get_earnings_features(ticker, force_refresh=True)
            except Exception:
                errors += 1
        db.log_event("INFO", "scheduler",
                     f"Daily earnings refresh done — {len(predictor.TICKERS) - errors}/{len(predictor.TICKERS)} tickers updated")
    except Exception as e:
        db.log_event("ERROR", "scheduler", f"Daily earnings refresh failed: {e}")


_scheduler = None


def start():
    global _scheduler
    db.init_db()

    _scheduler = BackgroundScheduler(timezone="UTC", daemon=True)

    _scheduler.add_job(
        refresh_signals,
        trigger=IntervalTrigger(minutes=SIGNAL_INTERVAL_MINUTES),
        id="signal_refresh",
        replace_existing=True,
    )

    # Daily at 21:00 UTC (NYSE close + 1h): save updated prices to disk
    _scheduler.add_job(
        refresh_price_cache,
        trigger=CronTrigger(hour=21, minute=0),
        id="price_cache_daily",
        replace_existing=True,
    )

    # Daily at 21:30 UTC: SPY/GLD/sector ETF cache update (used by sector_rel features)
    _scheduler.add_job(
        refresh_reference_etfs_daily,
        trigger=CronTrigger(hour=21, minute=30),
        id="reference_etfs_daily",
        replace_existing=True,
    )

    # Daily at 22:00 UTC (01:00 TR): earnings/fundamentals refresh
    _scheduler.add_job(
        refresh_earnings_daily,
        trigger=CronTrigger(hour=22, minute=0),
        id="earnings_daily",
        replace_existing=True,
    )

    # Daily at 23:00 UTC (02:00 TR): sector/SPY/Gold relative refresh (fast, local)
    _scheduler.add_job(
        refresh_sector_daily,
        trigger=CronTrigger(hour=23, minute=0),
        id="sector_daily",
        replace_existing=True,
    )

    # Daily at 00:00 UTC (03:00 TR): Google Trends refresh (slow, rate-limit aware)
    _scheduler.add_job(
        refresh_trends_daily,
        trigger=CronTrigger(hour=0, minute=0),
        id="trends_daily",
        replace_existing=True,
    )

    _scheduler.start()
    db.log_event("INFO", "scheduler", "Scheduler started")
    return _scheduler


def stop():
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        db.log_event("INFO", "scheduler", "Scheduler stopped")


def run_signal_refresh_now():
    """Manual trigger — called from API endpoint."""
    refresh_signals()
