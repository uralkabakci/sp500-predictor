import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys as _sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..')
_sys.path.insert(0, os.path.join(_ROOT, 'core'))

MODELS_DIR = os.path.join(_ROOT, 'saved_models')
PARAMS_DIR  = os.path.join(_ROOT, 'params_log')

import glob
import joblib
import warnings
import pandas as pd
import numpy as np

import data_processor as dp
import model_utils  # noqa: F401
from tickers import TICKERS as _ALL_TICKERS

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG — must match training config
# ============================================================================

ENSEMBLE_K   = 3
MIN_PRECISION = 0.0

TP_PCT = {10: 0.04, 15: 0.05, 20: 0.07}
SL_PCT = {10: 0.02, 15: 0.02, 20: 0.02}

TICKERS = _ALL_TICKERS

TARGET_DAYS = [10, 15, 20]

# ============================================================================


def _load_selected_index():
    path = os.path.join(PARAMS_DIR, "report_selected.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    idx = {}
    for _, row in df.iterrows():
        key = (row["ticker"], int(row["days"]), int(row["fold_year"]))
        idx.setdefault(key, []).append((int(row["ensemble_rank"]), int(row["pool_rank"])))
    for key in idx:
        idx[key].sort()
    return idx


_SELECTED_INDEX      = None
_SELECTED_INDEX_DATE = None


def _get_selected_index():
    global _SELECTED_INDEX, _SELECTED_INDEX_DATE
    today = pd.Timestamp.today().date()
    if _SELECTED_INDEX is None or _SELECTED_INDEX_DATE != today:
        _SELECTED_INDEX      = _load_selected_index()
        _SELECTED_INDEX_DATE = today
    return _SELECTED_INDEX


def _load_models_for_ticker(ticker):
    selected = _get_selected_index()
    models = {}
    for days in TARGET_DAYS:
        pct_tag = int(TP_PCT[days] * 100)
        fold_map = {}

        if selected:
            fold_years = {fy for (t, d, fy) in selected if t == ticker and d == days}
            for fold_year in fold_years:
                key = (ticker, days, fold_year)
                pkgs = []
                for ensemble_rank, pool_rank in selected.get(key, []):
                    fname = os.path.join(
                        MODELS_DIR,
                        f"model_{ticker}_{days}d_{pct_tag}pct_fold{fold_year}_cand{pool_rank}.pkl"
                    )
                    if os.path.exists(fname):
                        try:
                            pkg = joblib.load(fname)
                            pkg['ensemble_rank'] = ensemble_rank
                            if pkg['metrics']['precision'] >= MIN_PRECISION:
                                pkgs.append(pkg)
                        except Exception:
                            pass
                if pkgs:
                    fold_map[fold_year] = sorted(pkgs, key=lambda p: p.get('ensemble_rank', 99))

        if not fold_map:
            rank_by_fold = {}
            for fname in sorted(glob.glob(
                    os.path.join(MODELS_DIR, f"model_{ticker}_{days}d_{pct_tag}pct_fold*_cand*.pkl"))):
                try:
                    pkg = joblib.load(fname)
                    fy = pkg.get('fold_year')
                    if fy and pkg['metrics']['precision'] >= MIN_PRECISION:
                        rank_by_fold.setdefault(fy, []).append(pkg)
                except Exception:
                    pass
            for fy, pkgs in rank_by_fold.items():
                for i, p in enumerate(sorted(pkgs, key=lambda p: p.get('rank', 1)), 1):
                    p['ensemble_rank'] = i
                fold_map[fy] = sorted(pkgs, key=lambda p: p.get('ensemble_rank', 99))

        if fold_map:
            models[days] = fold_map
    return models


def _get_fold_packages(models, days, date):
    fold_map = models.get(days)
    if not fold_map:
        return []
    year = date.year
    pkgs = fold_map.get(year)
    if pkgs is None:
        earlier = [y for y in fold_map if y <= year]
        pkgs = fold_map[max(earlier)] if earlier else []
    return [p for p in pkgs if p.get('ensemble_rank', 1) <= ENSEMBLE_K]


def _predict_prob(pkgs, row):
    if not pkgs:
        return 0.0
    probs = []
    for pkg in pkgs:
        try:
            X = row[pkg['features']].values
            X_s = pkg['scaler'].transform(X)
            prob = pkg['model'].predict_proba(X_s)[0][1]
            if prob < pkg['threshold']:
                return 0.0
            probs.append(prob)
        except KeyError:
            continue
    if not probs:
        return 0.0
    return float(np.mean(probs))


_live_ohlc_ref = {}  # shared ref so scheduler can reuse without second fetch


def get_current_ohlc(tickers=None):
    """
    Fetch latest 30-min bar OHLC in batches of 50.
    Returns {ticker: {"open", "high", "low", "close"}}.
    """
    import yfinance as yf
    import time as _time
    tickers    = list(tickers or TICKERS)
    result     = {}
    batch_size = 50

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(batch, period="1d", interval="30m",
                               auto_adjust=True, progress=False, threads=False)
            if data.empty:
                continue
            for t in batch:
                try:
                    result[t] = {
                        "open":  round(float(data["Open"][t].dropna().iloc[-1]),  4),
                        "high":  round(float(data["High"][t].dropna().iloc[-1]),  4),
                        "low":   round(float(data["Low"][t].dropna().iloc[-1]),   4),
                        "close": round(float(data["Close"][t].dropna().iloc[-1]), 4),
                    }
                except Exception:
                    pass
        except Exception:
            pass
        if i + batch_size < len(tickers):
            _time.sleep(1.0)

    return result


def get_latest_signals(log_fn=None, signal_fn=None):
    """
    Returns list of dicts: {ticker, days, prob, entry_price, target_price, signal_date}
    Uses today's latest data row. Runs for all TICKERS.
    """
    import time
    signals = []
    total   = len(TICKERS)

    # Fetch 30-min OHLC for all tickers upfront
    if log_fn:
        log_fn("INFO", "predictor", "Fetching live OHLC (30m) for all tickers...")
    live_ohlc = get_current_ohlc(TICKERS)
    _live_ohlc_ref.clear()
    _live_ohlc_ref.update(live_ohlc)

    for i, ticker in enumerate(TICKERS, 1):
        try:
            models = _load_models_for_ticker(ticker)
            if not models:
                if log_fn and i % 20 == 0:
                    log_fn("INFO", "predictor", f"Progress: {i}/{total} tickers processed, {len(signals)} signals so far")
                continue

            raw_df = dp.get_data(ticker)
            if raw_df is None or raw_df.empty:
                continue
            df = dp.create_full_feature_universe(raw_df, ticker=ticker)
            if df is None or df.empty:
                continue

            last_date  = df.index[-1]
            row        = df.loc[[last_date]]

            # Use live close as entry; fall back to last daily close if unavailable
            bar = live_ohlc.get(ticker)
            entry_price = (bar["close"] if bar else None) or float(df['Close'].iloc[-1])

            for days in TARGET_DAYS:
                pkgs = _get_fold_packages(models, days, last_date)
                if not pkgs:
                    continue
                prob = _predict_prob(pkgs, row)
                if prob <= 0.0:
                    continue

                tp_pct = TP_PCT[days]
                sl_pct = SL_PCT[days]
                sig = {
                    "ticker":           ticker,
                    "days":             days,
                    "prob":             round(prob, 4),
                    "entry_price":      round(entry_price, 4),
                    "target_price":     round(entry_price * (1 + tp_pct), 4),
                    "stop_loss_price":  round(entry_price * (1 - sl_pct), 4),
                    "signal_date":      pd.Timestamp.today().strftime("%Y-%m-%d"),
                    "entry_time":       pd.Timestamp.utcnow().isoformat(),
                    "tp_pct":           tp_pct,
                    "sl_pct":           sl_pct,
                }
                signals.append(sig)
                if signal_fn:
                    signal_fn(sig)

        except Exception as e:
            if log_fn:
                log_fn("ERROR", "predictor", f"{ticker}: {e}")

        # Progress log every 25 tickers
        if log_fn and i % 25 == 0:
            log_fn("INFO", "predictor",
                   f"Progress: {i}/{total} tickers processed, {len(signals)} signals so far")

        # Small delay to avoid yfinance rate limiting
        time.sleep(0.15)

    return signals


