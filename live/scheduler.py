import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..')
sys.path.insert(0, os.path.join(_ROOT, 'core'))

from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

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
TRENDS_DAY_OF_WEEK  = "fri"
TRENDS_HOUR_UTC     = 9
TRENDS_MINUTE_UTC   = 0

# ── Price + signal refresh ────────────────────────────────────────────────────
SIGNAL_INTERVAL_MINUTES = 30


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

        signals = predictor.get_latest_signals(log_fn=db.log_event, signal_fn=_save_signal)

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


def refresh_market_data():
    """Weekly: refresh price cache, Google Trends, sector/SPY/Gold, and fundamentals."""
    db.log_event("INFO", "scheduler", "Weekly market data refresh started")
    try:
        sys.path.insert(0, os.path.join(_ROOT, 'data'))
        import downloader as dd
        import predictor as pred

        # Price + Trends for all tickers
        dd.download_all_tickers(force_refresh=False, include_trends=True)
        db.log_event("INFO", "scheduler", "Price + Trends download done")

        # Sector relative (SPY, Gold) — fiyat bazlı, haftalık yeterli
        try:
            import sector_data as sec
            import data_processor as dp
            for ticker in pred.TICKERS:
                try:
                    raw = dp.get_data(ticker)
                    if raw is not None and not raw.empty:
                        sec.merge_sector_relative(raw, ticker, windows=[14, 30])
                        sec.merge_spy_relative(raw, windows=[14, 30])
                        sec.merge_gold_spy(raw, windows=[14, 30])
                except Exception:
                    pass
            db.log_event("INFO", "scheduler", "Sector/SPY/Gold refresh done")
        except Exception as e:
            db.log_event("WARN", "scheduler", f"Sector refresh partial error: {e}")

        # Fundamentals (EPS, revenue) — çeyrek bazlı, haftalık yeterli
        try:
            import earnings_data as ed
            for ticker in pred.TICKERS:
                try:
                    ed.get_earnings_features(ticker)
                except Exception:
                    pass
            db.log_event("INFO", "scheduler", "Fundamentals refresh done")
        except Exception as e:
            db.log_event("WARN", "scheduler", f"Fundamentals refresh partial error: {e}")

        db.log_event("INFO", "scheduler", "Weekly market data refresh complete")
    except Exception as e:
        db.log_event("ERROR", "scheduler", f"Weekly market data refresh failed: {e}")


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

    # Weekly Friday 09:00 UTC: Trends + Sector + Fundamentals
    _scheduler.add_job(
        refresh_market_data,
        trigger=CronTrigger(
            day_of_week=TRENDS_DAY_OF_WEEK,
            hour=TRENDS_HOUR_UTC,
            minute=TRENDS_MINUTE_UTC,
        ),
        id="market_data_refresh",
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
