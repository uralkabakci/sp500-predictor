import sqlite3
import os
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))

from datetime import datetime
import pandas_market_calendars as mcal

_NYSE = mcal.get_calendar("NYSE")


def _trading_days_elapsed(signal_date, today) -> int:
    """Count NYSE trading days from signal_date (exclusive) to today (inclusive)."""
    if today <= signal_date:
        return 0
    schedule = _NYSE.schedule(
        start_date=signal_date.isoformat(),
        end_date=today.isoformat(),
    )
    return max(0, len(schedule) - 1)


DB_PATH = os.path.join(os.path.dirname(__file__), "data", "metrade.db")


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker           TEXT    NOT NULL,
            days             INTEGER NOT NULL,
            prob             REAL    NOT NULL,
            entry_price      REAL,
            target_price     REAL,
            stop_loss_price  REAL,
            signal_date      TEXT    NOT NULL,
            entry_time       TEXT,
            status           TEXT    NOT NULL DEFAULT 'active',
            exit_price       REAL,
            exit_date        TEXT,
            exit_reason      TEXT,
            created_at       TEXT    NOT NULL,
            updated_at       TEXT    NOT NULL
        )
    """)
    # Migrate existing DB: add stop_loss_price if missing
    cols = [row[1] for row in c.execute("PRAGMA table_info(signals)").fetchall()]
    if "stop_loss_price" not in cols:
        c.execute("ALTER TABLE signals ADD COLUMN stop_loss_price REAL")

    c.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            level      TEXT NOT NULL,
            source     TEXT NOT NULL,
            message    TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS price_cache (
            ticker     TEXT NOT NULL,
            price      REAL NOT NULL,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (ticker)
        )
    """)

    conn.commit()
    conn.close()


def log_event(level: str, source: str, message: str):
    conn = get_conn()
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO logs (level, source, message, created_at) VALUES (?, ?, ?, ?)",
        (level, source, message, now)
    )
    conn.commit()
    conn.close()


def upsert_signal(ticker, days, prob, entry_price, target_price, stop_loss_price,
                  signal_date, entry_time=None):
    """Insert a new signal only if no active signal exists for same (ticker, days)."""
    conn = get_conn()
    now = datetime.utcnow().isoformat()
    existing = conn.execute(
        "SELECT id FROM signals WHERE ticker=? AND days=? AND status='active'",
        (ticker, days)
    ).fetchone()
    if not existing:
        conn.execute(
            """INSERT INTO signals
               (ticker, days, prob, entry_price, target_price, stop_loss_price,
                signal_date, entry_time, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)""",
            (ticker, days, prob, entry_price, target_price, stop_loss_price,
             signal_date, entry_time, now, now)
        )
        conn.commit()
    conn.close()


def close_expired_signals(ohlc: dict):
    """
    Called every 30 min during market hours.
    ohlc: {ticker: {"open", "high", "low", "close"}}

    Rules (in priority order):
    1. open >= target  → close at open  (gap up)
    2. open <= stop    → close at open  (gap down)
    3. high >= target  → close at target (intraday target hit)
    4. low  <= stop    → close at stop   (intraday stop hit)
    5. elapsed >= days → close at close  (expired)

    Signals created within the last 30 minutes are skipped — their creation
    bar's High/Low may predate the actual entry.
    """
    from datetime import timedelta
    conn       = get_conn()
    now_dt     = datetime.utcnow()
    now        = now_dt.isoformat()
    today      = now_dt.date()
    cutoff     = (now_dt - timedelta(minutes=30)).isoformat()
    active = conn.execute("SELECT * FROM signals WHERE status='active'").fetchall()

    for sig in active:
        ticker          = sig["ticker"]
        days            = sig["days"]
        entry_price     = sig["entry_price"]
        target_price    = sig["target_price"]
        stop_loss_price = sig["stop_loss_price"]
        signal_date     = sig["signal_date"]
        entry_time      = sig["entry_time"]

        if not entry_price or not target_price:
            continue

        # Skip signals created in the current 30-min bar
        if entry_time and entry_time > cutoff:
            continue

        bar       = ohlc.get(ticker)
        signal_dt = datetime.fromisoformat(signal_date).date()
        elapsed   = _trading_days_elapsed(signal_dt, today)

        exit_price  = None
        exit_reason = None

        if bar:
            o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
            if o >= target_price:
                exit_price, exit_reason = o, "target_hit"
            elif stop_loss_price and o <= stop_loss_price:
                exit_price, exit_reason = o, "stop_loss_hit"
            elif h >= target_price:
                exit_price, exit_reason = target_price, "target_hit"
            elif stop_loss_price and l <= stop_loss_price:
                exit_price, exit_reason = stop_loss_price, "stop_loss_hit"

        if not exit_reason and elapsed >= days:
            exit_price  = bar["close"] if bar else entry_price
            exit_reason = "expired"

        if exit_reason:
            conn.execute(
                """UPDATE signals SET status='closed', exit_price=?, exit_date=?,
                   exit_reason=?, updated_at=? WHERE id=?""",
                (exit_price, now, exit_reason, now, sig["id"])
            )

    conn.commit()
    conn.close()


def get_active_signals():
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM signals WHERE status='active' ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_history(ticker=None, tickers=None, since_days=None):
    conn = get_conn()
    query = "SELECT * FROM signals WHERE status='closed'"
    params = []
    if tickers:
        placeholders = ",".join("?" * len(tickers))
        query += f" AND ticker IN ({placeholders})"
        params.extend(tickers)
    elif ticker:
        query += " AND ticker=?"
        params.append(ticker)
    if since_days:
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=since_days)
        query += " AND exit_date >= ?"
        params.append(cutoff.isoformat())
    query += " ORDER BY exit_date DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_logs(limit=500):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM logs ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_price_cache(ticker, price):
    conn = get_conn()
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO price_cache (ticker, price, fetched_at) VALUES (?, ?, ?)",
        (ticker, price, now)
    )
    conn.commit()
    conn.close()


def get_price_cache():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM price_cache").fetchall()
    conn.close()
    return {r["ticker"]: r["price"] for r in rows}


def get_stoploss_cooldown_blocked(cooldown_days: dict) -> set:
    """
    Returns a set of (ticker, days) pairs that are in stop-loss cooldown.
    cooldown_days: {days: n_calendar_days_cooldown}
    """
    from datetime import date, timedelta
    conn    = get_conn()
    blocked = set()
    today   = date.today()
    for days, n in cooldown_days.items():
        cutoff = (today - timedelta(days=n)).isoformat()
        rows = conn.execute(
            """SELECT DISTINCT ticker FROM signals
               WHERE days=? AND exit_reason='stop_loss_hit' AND exit_date >= ?""",
            (days, cutoff)
        ).fetchall()
        for r in rows:
            blocked.add((r["ticker"], days))
    conn.close()
    return blocked
