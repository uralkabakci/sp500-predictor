import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys as _sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..')
_sys.path.insert(0, os.path.join(_ROOT, 'core'))
os.chdir(_ROOT)

import pandas as pd
import numpy as np
import glob
import joblib
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import data_processor as dp
import model_utils  # noqa: F401
from tickers import TICKERS

warnings.filterwarnings("ignore")

# ============================================================================
# PARAMETERS — must match predictor.py / live system
# ============================================================================

TEST_START_DATE = "2025-01-01"
TEST_END_DATE   = None

ENSEMBLE_K    = 3
MIN_PRECISION = 0.0

TP_PCT = {10: 0.04, 15: 0.05, 20: 0.07}
SL_PCT = {10: 0.02, 15: 0.02, 20: 0.02}

STOPLOSS_COOLDOWN = {10: 1, 15: 2, 20: 3}  # calendar days after SL hit

TARGET_DAYS = [10, 15, 20]

TRADES_CSV = "signal_analysis_trades.csv"

# ============================================================================

def _load_selected_index():
    path = os.path.join("params_log", "report_selected.csv")
    if not os.path.exists(path):
        return {}
    df  = pd.read_csv(path)
    idx = {}
    for _, row in df.iterrows():
        key = (row["ticker"], int(row["days"]), int(row["fold_year"]))
        idx.setdefault(key, []).append((int(row["ensemble_rank"]), int(row["pool_rank"])))
    for key in idx:
        idx[key].sort()
    return idx


def _load_models(ticker, selected_index):
    models = {}
    for days in TARGET_DAYS:
        pct_tag  = int(TP_PCT[days] * 100)
        fold_map = {}
        if selected_index:
            fold_years = {fy for (t, d, fy) in selected_index if t == ticker and d == days}
            for fold_year in fold_years:
                key      = (ticker, days, fold_year)
                selected = selected_index.get(key, [])
                pkgs     = []
                for ensemble_rank, pool_rank in selected:
                    fname = os.path.join(
                        "saved_models",
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
                    f"saved_models/model_{ticker}_{days}d_{pct_tag}pct_fold*_cand*.pkl")):
                try:
                    pkg = joblib.load(fname)
                    fy  = pkg.get('fold_year')
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
    if days not in models:
        return []
    fold_map = models[days]
    year     = date.year
    pkgs     = fold_map.get(year)
    if pkgs is None:
        earlier = [y for y in fold_map if y <= year]
        pkgs    = fold_map[max(earlier)] if earlier else []
    return [p for p in pkgs if p.get('ensemble_rank', 1) <= ENSEMBLE_K]


def _predict_prob(pkgs, row):
    if not pkgs:
        return 0.0
    probs = []
    for pkg in pkgs:
        try:
            X   = row[pkg['features']].values
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


def _check_outcome(df, signal_idx, days, tp_pct, sl_pct):
    """
    Walk forward day-by-day using OHLC.
    Contested (same-bar TP+SL hit) → stop_loss_hit (pessimistic).
    Returns (exit_date, exit_price, exit_reason) or (None, None, None) if pending.
    """
    entry  = float(df['Close'].iloc[signal_idx])
    target = entry * (1 + tp_pct)
    stop   = entry * (1 - sl_pct)
    future = df.iloc[signal_idx + 1 : signal_idx + 1 + days]

    for date, bar in future.iterrows():
        hit_tp = float(bar['High']) >= target
        hit_sl = float(bar['Low'])  <= stop

        if hit_tp and hit_sl:
            return date, round(stop, 4), "stop_loss_hit"
        if hit_tp:
            return date, round(target, 4), "target_hit"
        if hit_sl:
            return date, round(stop, 4), "stop_loss_hit"

    # No early hit. Need a FULL horizon of forward data to call it expired.
    # Otherwise the signal is still pending (right-censoring guard).
    if len(future) < days:
        return None, None, None
    last = future.iloc[-1]
    return future.index[-1], round(float(last['Close']), 4), "expired"


# ── Worker (runs in a subprocess) ────────────────────────────────────────────

def _worker(args):
    """Process a single ticker. Returns (ticker, trades, agg_stats, baseline_stats)."""
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    import warnings; warnings.filterwarnings("ignore")

    ticker, start_str, end_str, selected_index, work_dir = args
    os.chdir(work_dir)

    import data_processor as dp
    import model_utils  # noqa
    import pandas as pd

    start = pd.Timestamp(start_str)
    end   = pd.Timestamp(end_str) if end_str else pd.Timestamp.today()

    agg      = {d: {'signals': 0, 'hits': 0, 'pending': 0} for d in TARGET_DAYS}
    baseline = {d: {'total': 0, 'hits': 0} for d in TARGET_DAYS}
    trades   = []

    try:
        models = _load_models(ticker, selected_index)
        if not models:
            return ticker, trades, agg, baseline

        raw_df = dp.get_data(ticker)
        if raw_df is None or raw_df.empty:
            return ticker, trades, agg, baseline
        df = dp.create_full_feature_universe(raw_df, ticker=ticker)
        if df is None or df.empty:
            return ticker, trades, agg, baseline
        df = df[df.index >= start - pd.Timedelta(days=120)]

        sim_dates = df.index[(df.index >= start) & (df.index <= end)]

        for days in TARGET_DAYS:
            tp_pct = TP_PCT[days]
            sl_pct = SL_PCT[days]

            # ── Baseline: every trading day, no model ────────────────────
            for date in sim_dates:
                idx = df.index.get_loc(date)
                _, _, reason = _check_outcome(df, idx, days, tp_pct, sl_pct)
                if reason is not None:
                    baseline[days]['total'] += 1
                    if reason == 'target_hit':
                        baseline[days]['hits'] += 1

            # ── Model signals ────────────────────────────────────────────
            if days not in models:
                continue

            active_signal_end = None  # track open signal's exit date
            cooldown_until    = None  # block new signals after stop loss
            cooldown_n        = pd.Timedelta(days=STOPLOSS_COOLDOWN[days])

            for date in sim_dates:
                # Skip if a signal is still active
                if active_signal_end is not None and date <= active_signal_end:
                    continue
                # Skip if we're in stop-loss cooldown
                if cooldown_until is not None and date <= cooldown_until:
                    continue

                pkgs = _get_fold_packages(models, days, date)
                if not pkgs:
                    continue
                prob = _predict_prob(pkgs, df.loc[[date]])
                if prob <= 0.0:
                    continue

                signal_idx  = df.index.get_loc(date)
                entry_price = round(float(df['Close'].iloc[signal_idx]), 4)
                exit_date, exit_price, exit_reason = _check_outcome(
                    df, signal_idx, days, tp_pct, sl_pct
                )

                # Mark signal as active until it closes
                active_signal_end = exit_date if exit_date is not None else sim_dates[-1]
                # If stop loss hit, enter cooldown
                if exit_reason == 'stop_loss_hit' and exit_date is not None:
                    cooldown_until = exit_date + cooldown_n
                else:
                    cooldown_until = None

                agg[days]['signals'] += 1
                if exit_reason is None:
                    agg[days]['pending'] += 1
                elif exit_reason == 'target_hit':
                    agg[days]['hits'] += 1

                if exit_reason is not None:
                    pnl_pct = round((exit_price - entry_price) / entry_price * 100, 4)
                    trades.append({
                        'signal_date':  date.strftime('%Y-%m-%d'),
                        'ticker':       ticker,
                        'days':         days,
                        'prob':         round(float(prob), 4),
                        'entry_price':  entry_price,
                        'target_price': round(entry_price * (1 + tp_pct), 4),
                        'stop_price':   round(entry_price * (1 - sl_pct), 4),
                        'exit_date':    exit_date.strftime('%Y-%m-%d'),
                        'exit_price':   exit_price,
                        'exit_reason':  exit_reason,
                        'pnl_pct':      pnl_pct,
                    })

    except Exception as e:
        print(f"\n  [ERROR] {ticker}: {e}", flush=True)

    return ticker, trades, agg, baseline


# ── Main ──────────────────────────────────────────────────────────────────────

def run_analysis(save_csv=True, workers=None):
    import multiprocessing
    if workers is None:
        workers = min(multiprocessing.cpu_count(), 16)

    selected_index = _load_selected_index()
    work_dir       = os.path.abspath(_ROOT)
    tickers        = TICKERS

    if os.environ.get("ABLATION_TICKERS"):
        tickers = os.environ["ABLATION_TICKERS"].split(",")

    end_str = TEST_END_DATE

    print(f"Analysing {len(tickers)} tickers | {TEST_START_DATE} → {end_str or 'today'} | {workers} workers")
    print(f"Ensemble K={ENSEMBLE_K}  TP={TP_PCT}  SL={SL_PCT}")
    print("-" * 60)

    args_list = [
        (ticker, TEST_START_DATE, end_str, selected_index, work_dir)
        for ticker in tickers
    ]

    all_trades = []
    results    = {t: {d: {'signals': 0, 'hits': 0, 'pending': 0} for d in TARGET_DAYS} for t in tickers}
    baselines  = {t: {d: {'total': 0, 'hits': 0} for d in TARGET_DAYS} for t in tickers}
    done       = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, args): args[0] for args in args_list}
        for future in as_completed(futures):
            ticker, trades, agg, baseline = future.result()
            all_trades.extend(trades)
            results[ticker]   = agg
            baselines[ticker] = baseline
            done += 1
            print(f"  [{done:>3}/{len(tickers)}] {ticker:<6}  +{len(trades)} trades", end="\r", flush=True)

    print(f"\nDone — {len(all_trades)} completed trades across {len(tickers)} tickers.\n")
    _print_report(results, baselines, tickers)

    if save_csv and all_trades:
        df_out = pd.DataFrame(all_trades).sort_values(['signal_date', 'ticker'])
        df_out.to_csv(TRADES_CSV, index=False)
        print(f"\nSaved → {TRADES_CSV}  ({len(all_trades)} rows)")

    return all_trades


def _print_report(results, baselines, tickers):
    W = 7  # column width per horizon block
    # Header: Ticker | [10d: Sig  Hit%  Base%] | [15d: ...] | [20d: ...] | Total Hit%  Base%
    print("=" * 120)
    print(f"SIGNAL ACCURACY vs RANDOM BASELINE  |  K={ENSEMBLE_K}  |  TP={TP_PCT}  |  SL={SL_PCT}")
    print(f"Period: {TEST_START_DATE} → {TEST_END_DATE or 'today'}")
    print("=" * 120)

    header = f"{'Ticker':<8}"
    for d in TARGET_DAYS:
        header += f"  {'─── ' + str(d) + 'd ───':>22}"
    header += f"  {'── Total ──':>22}"
    print(header)

    subhdr = f"{'':8}"
    for d in TARGET_DAYS:
        subhdr += f"  {'Signals':>7}  {'Hit%':>6}  {'Base%':>6}"
    subhdr += f"  {'Signals':>7}  {'Hit%':>6}  {'Base%':>6}"
    print(subhdr)
    print("-" * 120)

    ticker_rows = []
    for ticker in tickers:
        total_sig  = sum(results[ticker][d]['signals'] for d in TARGET_DAYS)
        total_hits = sum(results[ticker][d]['hits']    for d in TARGET_DAYS)
        total_pend = sum(results[ticker][d]['pending'] for d in TARGET_DAYS)
        evaluated  = total_sig - total_pend
        total_pct  = (total_hits / evaluated * 100) if evaluated > 0 else 0.0
        bl_tot     = sum(baselines[ticker][d]['total'] for d in TARGET_DAYS)
        bl_hits    = sum(baselines[ticker][d]['hits']  for d in TARGET_DAYS)
        bl_pct     = (bl_hits / bl_tot * 100) if bl_tot > 0 else 0.0
        ticker_rows.append((ticker, total_sig, total_hits, total_pend, evaluated, total_pct, bl_pct))
    ticker_rows.sort(key=lambda x: x[1], reverse=True)

    grand_sig  = {d: 0 for d in TARGET_DAYS}
    grand_hits = {d: 0 for d in TARGET_DAYS}
    grand_pend = {d: 0 for d in TARGET_DAYS}
    grand_bl_t = {d: 0 for d in TARGET_DAYS}
    grand_bl_h = {d: 0 for d in TARGET_DAYS}

    for ticker, total_sig, total_hits, total_pend, evaluated, total_pct, bl_pct in ticker_rows:
        if total_sig == 0:
            continue
        row = f"{ticker:<8}"
        for d in TARGET_DAYS:
            sig  = results[ticker][d]['signals']
            hits = results[ticker][d]['hits']
            pend = results[ticker][d]['pending']
            evl  = sig - pend
            pct  = (hits / evl * 100) if evl > 0 else 0.0
            blt  = baselines[ticker][d]['total']
            blh  = baselines[ticker][d]['hits']
            bpct = (blh / blt * 100) if blt > 0 else 0.0
            edge = pct - bpct
            edge_str = f"({edge:+.1f})" if evl > 0 else ""
            row += f"  {sig:>7}  {pct:>5.1f}%  {bpct:>5.1f}%"
            grand_sig[d]  += sig
            grand_hits[d] += hits
            grand_pend[d] += pend
            grand_bl_t[d] += blt
            grand_bl_h[d] += blh
        flag = " ⚠" if evaluated > 0 and total_pct < 40 else ""
        row += f"  {total_sig:>7}  {total_pct:>5.1f}%  {bl_pct:>5.1f}%{flag}"
        print(row)

    print("-" * 120)
    total_row = f"{'TOTAL':<8}"
    g_sig = g_hits = g_pend = 0
    g_blt = g_blh  = 0
    for d in TARGET_DAYS:
        evl  = grand_sig[d] - grand_pend[d]
        pct  = (grand_hits[d] / evl * 100) if evl > 0 else 0.0
        bpct = (grand_bl_h[d] / grand_bl_t[d] * 100) if grand_bl_t[d] > 0 else 0.0
        total_row += f"  {grand_sig[d]:>7}  {pct:>5.1f}%  {bpct:>5.1f}%"
        g_sig  += grand_sig[d]; g_hits += grand_hits[d]; g_pend += grand_pend[d]
        g_blt  += grand_bl_t[d]; g_blh += grand_bl_h[d]
    g_evl  = g_sig - g_pend
    g_pct  = (g_hits / g_evl * 100) if g_evl > 0 else 0.0
    g_bpct = (g_blh  / g_blt  * 100) if g_blt  > 0 else 0.0
    total_row += f"  {g_sig:>7}  {g_pct:>5.1f}%  {g_bpct:>5.1f}%"
    print(total_row)
    print("=" * 120)
    print(f"  Hit%  = target_hit rate on model signal days")
    print(f"  Base% = target_hit rate if buying every trading day (random baseline)")
    print(f"  ⚠     = hit rate < 40%")


if __name__ == "__main__":
    run_analysis(save_csv=True)
