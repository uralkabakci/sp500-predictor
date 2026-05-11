import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys as _sys
_HERE_FILE = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.abspath(os.path.join(_HERE_FILE, '..'))
_sys.path.insert(0, os.path.join(_ROOT_DIR, 'core'))
os.chdir(_ROOT_DIR)   # so data_processor & friends find data_cache/, saved_models/

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import uvicorn

import database as db
import scheduler as sched
import predictor

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))

@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    sched.start()
    # Run an initial signal refresh on startup (non-blocking via scheduler)
    import threading
    threading.Thread(target=sched.run_signal_refresh_now, daemon=True).start()
    yield
    sched.stop()


app = FastAPI(title="MeTrade Live", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


# ── Pages ─────────────────────────────────────────────────────────────────────

_INDEX_HTML = open(os.path.join(BASE_DIR, "templates", "index.html"), encoding="utf-8").read()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return HTMLResponse(content=_INDEX_HTML)


# ── API: Signals ──────────────────────────────────────────────────────────────

@app.get("/api/signals")
async def api_signals():
    signals = db.get_active_signals()
    prices  = db.get_price_cache()
    for s in signals:
        t       = s["ticker"]
        current = prices.get(t)
        if current and s["entry_price"]:
            s["current_price"] = current
            s["pnl_pct"] = round((current - s["entry_price"]) / s["entry_price"] * 100, 2)
        else:
            s["current_price"] = None
            s["pnl_pct"]       = None
    return JSONResponse(signals)


# ── API: History ──────────────────────────────────────────────────────────────

PERIOD_MAP = {
    "24h":  1,
    "1w":   7,
    "1m":   30,
    "3m":   90,
    "6m":   180,
    "1y":   365,
    "all":  None,
}

@app.get("/api/history")
async def api_history(
    ticker: str = Query(default=None),
    tickers: str = Query(default=None),
    period: str = Query(default="1m"),
):
    since_days   = PERIOD_MAP.get(period, 30)
    ticker_list  = None
    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    elif ticker:
        ticker_list = [ticker.upper()]
    rows = db.get_history(tickers=ticker_list, since_days=since_days)
    return JSONResponse(rows)


# ── API: Backtest ─────────────────────────────────────────────────────────────

def _load_backtest_trades(csv_path):
    """Load trade records from signal_analysis_trades.csv."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    trades = []
    for _, row in df.iterrows():
        entry = float(row["entry_price"])
        exit_ = float(row["exit_price"])
        trades.append({
            "ticker":      str(row["ticker"]),
            "days":        int(row["days"]),
            "entry_date":  str(row["signal_date"]),
            "exit_date":   str(row["exit_date"]),
            "entry_price": round(entry, 4),
            "exit_price":  round(exit_, 4),
            "pnl":         round(exit_ - entry, 4),
            "pnl_pct":     round(float(row["pnl_pct"]), 4),
            "exit_reason": str(row["exit_reason"]),
        })
    return trades


@app.get("/api/backtest")
async def api_backtest(
    ticker: str = Query(default=None),
    tickers: str = Query(default=None),
    period: str = Query(default="all"),
):
    """Return trade records from signal_analysis_trades.csv."""
    csv_path = os.path.join(ROOT_DIR, "signal_analysis_trades.csv")
    if not os.path.exists(csv_path):
        return JSONResponse(
            {"error": "Backtest data not found. Run signal_analysis.py first."},
            status_code=404
        )
    try:
        trades      = _load_backtest_trades(csv_path)
        ticker_list = None
        if tickers:
            ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        elif ticker:
            ticker_list = [ticker.upper()]
        if ticker_list:
            trades = [t for t in trades if t["ticker"].upper() in ticker_list]
        since_days = PERIOD_MAP.get(period, None)
        if since_days:
            from datetime import datetime, timedelta
            cutoff = (datetime.utcnow() - timedelta(days=since_days)).strftime("%Y-%m-%d")
            trades = [t for t in trades if t["entry_date"] >= cutoff]
        trades.sort(key=lambda t: t["entry_date"], reverse=True)
        return JSONResponse(trades)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/backtest/stats")
async def api_backtest_stats():
    """Return aggregate stats from signal_analysis_trades.csv."""
    csv_path = os.path.join(ROOT_DIR, "signal_analysis_trades.csv")
    if not os.path.exists(csv_path):
        return JSONResponse({"error": "Backtest data not found."}, status_code=404)
    try:
        trades    = _load_backtest_trades(csv_path)
        total     = len(trades)
        wins      = sum(1 for t in trades if t["pnl"] > 0)
        total_pnl = sum(t["pnl"] for t in trades)
        win_rate  = round(wins / total * 100, 1) if total > 0 else 0
        return JSONResponse({
            "total_trades": total,
            "wins":         wins,
            "win_rate_pct": win_rate,
            "total_pnl":    round(total_pnl, 2),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── API: Tickers ──────────────────────────────────────────────────────────────

@app.get("/api/tickers")
async def api_tickers():
    return JSONResponse(sorted(predictor.TICKERS))


# ── API: Win Rates ────────────────────────────────────────────────────────────

@app.get("/api/winrates")
async def api_winrates():
    """Win rates from live signal history grouped by ticker."""
    rows = db.get_history(since_days=None)
    return JSONResponse(_calc_winrates(rows, pnl_key=None, use_exit_reason=True))


@app.get("/api/backtest/winrates")
async def api_backtest_winrates():
    """Win rates from signal_analysis_trades.csv grouped by ticker."""
    csv_path = os.path.join(ROOT_DIR, "signal_analysis_trades.csv")
    if not os.path.exists(csv_path):
        return JSONResponse({"error": "Backtest data not found."}, status_code=404)
    try:
        trades = _load_backtest_trades(csv_path)
        return JSONResponse(_calc_winrates(trades, pnl_key=None, use_exit_reason=True))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


def _calc_winrates(rows, pnl_key, use_exit_reason):
    from collections import defaultdict
    stats = defaultdict(lambda: {"wins": 0, "total": 0})

    for r in rows:
        ticker = r.get("ticker", "?")
        if use_exit_reason:
            win = r.get("exit_reason") == "target_hit"
        else:
            pnl = r.get(pnl_key)
            win = pnl is not None and float(pnl) > 0
        stats[ticker]["total"] += 1
        if win:
            stats[ticker]["wins"] += 1

    result = []
    g_wins = g_total = 0
    for ticker, s in sorted(stats.items()):
        t = s["total"]
        w = s["wins"]
        g_wins  += w
        g_total += t
        result.append({
            "ticker":   ticker,
            "wins":     w,
            "total":    t,
            "win_rate": round(w / t * 100, 1) if t else 0,
        })

    result.sort(key=lambda x: x["total"], reverse=True)
    result.append({
        "ticker":   "TOTAL",
        "wins":     g_wins,
        "total":    g_total,
        "win_rate": round(g_wins / g_total * 100, 1) if g_total else 0,
    })
    return result


# ── API: Manual refresh ───────────────────────────────────────────────────────

@app.post("/api/refresh")
async def api_refresh():
    import threading
    threading.Thread(target=sched.run_signal_refresh_now, daemon=True).start()
    db.log_event("INFO", "api", "Manual refresh triggered")
    return JSONResponse({"status": "refresh started"})


# ── API: Logs ─────────────────────────────────────────────────────────────────

@app.get("/api/logs")
async def api_logs(limit: int = Query(default=500, le=2000)):
    rows = db.get_logs(limit=limit)
    return JSONResponse(rows)


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
