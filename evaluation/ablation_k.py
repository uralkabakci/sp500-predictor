"""
Ablation: test ENSEMBLE_K = 1..5 in a single pass.

Each ticker's models are loaded once. For every signal date we record
each model's probability. Then we simulate K=1..5 without re-running.

A signal fires for a given K if the top-K models ALL exceed their thresholds.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys as _sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..')
_sys.path.insert(0, os.path.join(_ROOT, 'core'))
os.chdir(_ROOT)

import warnings
warnings.filterwarnings("ignore")

import glob
import joblib
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

import data_processor as dp
import model_utils  # noqa: F401
from tickers import TICKERS

TEST_START_DATE = "2025-01-01"
TEST_END_DATE   = None
MAX_K           = 5
MIN_PRECISION   = 0.0

TP_PCT      = {10: 0.04, 15: 0.05, 20: 0.07}
SL_PCT      = {10: 0.02, 15: 0.02, 20: 0.02}
TARGET_DAYS = [10, 15, 20]


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
                key  = (ticker, days, fold_year)
                pkgs = []
                for ensemble_rank, pool_rank in selected_index.get(key, []):
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


def _get_fold_packages(models, days, date, k):
    fold_map = models.get(days)
    if not fold_map:
        return []
    year = date.year
    pkgs = fold_map.get(year)
    if pkgs is None:
        earlier = [y for y in fold_map if y <= year]
        pkgs    = fold_map[max(earlier)] if earlier else []
    return [p for p in pkgs if p.get('ensemble_rank', 1) <= k]


def _check_outcome(close_series, signal_idx, days, tp_pct, sl_pct):
    entry  = float(close_series.iloc[signal_idx])
    target = entry * (1 + tp_pct)
    stop   = entry * (1 - sl_pct)
    future = close_series.iloc[signal_idx + 1: signal_idx + 1 + days]
    if len(future) == 0:
        return None, None, None
    for date, price in future.items():
        price = float(price)
        if price >= target:
            return date, round(target, 4), "target_hit"
        if price <= stop:
            return date, round(stop, 4), "stop_loss_hit"
    return future.index[-1], round(float(future.iloc[-1]), 4), "expired"


def _worker(args):
    import os; os.environ["OMP_NUM_THREADS"] = "1"
    import warnings; warnings.filterwarnings("ignore")

    ticker, start_str, end_str, selected_index, work_dir = args
    os.chdir(work_dir)

    import data_processor as dp
    import model_utils  # noqa
    import pandas as pd
    import numpy as np

    start = pd.Timestamp(start_str)
    end   = pd.Timestamp(end_str) if end_str else pd.Timestamp.today()

    # stats[k][days] = {signals, hits, pending}
    stats    = {k: {d: {'signals': 0, 'hits': 0, 'pending': 0} for d in TARGET_DAYS} for k in range(1, MAX_K + 1)}
    baseline = {d: {'total': 0, 'hits': 0} for d in TARGET_DAYS}

    try:
        models = _load_models(ticker, selected_index)
        if not models:
            return ticker, stats, baseline

        raw_df = dp.get_data(ticker)
        if raw_df is None or raw_df.empty:
            return ticker, stats, baseline
        df = dp.create_full_feature_universe(raw_df, ticker=ticker)
        if df is None or df.empty:
            return ticker, stats, baseline
        df = df[df.index >= start - pd.Timedelta(days=60)]

        close_arr = df['Close']
        sim_dates = df.index[(df.index >= start) & (df.index <= end)]

        for days in TARGET_DAYS:
            tp_pct = TP_PCT[days]
            sl_pct = SL_PCT[days]

            # Baseline
            for date in sim_dates:
                idx = close_arr.index.get_loc(date)
                _, _, reason = _check_outcome(close_arr, idx, days, tp_pct, sl_pct)
                if reason is not None:
                    baseline[days]['total'] += 1
                    if reason == 'target_hit':
                        baseline[days]['hits'] += 1

            if days not in models:
                continue

            for date in sim_dates:
                # Get MAX_K packages, compute all probs once
                pkgs_all = _get_fold_packages(models, days, date, MAX_K)
                if not pkgs_all:
                    continue

                row = df.loc[[date]]

                # Compute per-model (prob, threshold, rank) sorted by rank
                model_results = []
                for pkg in pkgs_all:
                    try:
                        X    = row[pkg['features']].values
                        X_s  = pkg['scaler'].transform(X)
                        prob = float(pkg['model'].predict_proba(X_s)[0][1])
                        model_results.append((pkg.get('ensemble_rank', 99), prob, pkg['threshold']))
                    except KeyError:
                        continue

                model_results.sort(key=lambda x: x[0])

                # Simulate K=1..MAX_K
                signal_idx = close_arr.index.get_loc(date)
                exit_date, exit_price, exit_reason = _check_outcome(
                    close_arr, signal_idx, days, tp_pct, sl_pct
                )

                for k in range(1, MAX_K + 1):
                    top_k = model_results[:k]
                    if len(top_k) < k:
                        continue
                    # All top-k must exceed their threshold
                    if all(prob >= thr for _, prob, thr in top_k):
                        stats[k][days]['signals'] += 1
                        if exit_reason is None:
                            stats[k][days]['pending'] += 1
                        elif exit_reason == 'target_hit':
                            stats[k][days]['hits'] += 1

    except Exception as e:
        print(f"\n  [ERROR] {ticker}: {e}", flush=True)

    return ticker, stats, baseline


def run_ablation():
    import multiprocessing
    workers = min(multiprocessing.cpu_count(), 16)

    selected_index = _load_selected_index()
    work_dir       = os.path.abspath(_ROOT)
    end_str        = TEST_END_DATE

    print(f"Ablation K=1..{MAX_K} | {len(TICKERS)} tickers | {TEST_START_DATE} → {end_str or 'today'} | {workers} workers")
    print("-" * 60)

    args_list = [(t, TEST_START_DATE, end_str, selected_index, work_dir) for t in TICKERS]

    # Accumulators
    grand = {k: {d: {'signals': 0, 'hits': 0, 'pending': 0} for d in TARGET_DAYS} for k in range(1, MAX_K + 1)}
    bl    = {d: {'total': 0, 'hits': 0} for d in TARGET_DAYS}
    done  = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, a): a[0] for a in args_list}
        for future in as_completed(futures):
            ticker, stats, baseline = future.result()
            for k in range(1, MAX_K + 1):
                for d in TARGET_DAYS:
                    for key in ('signals', 'hits', 'pending'):
                        grand[k][d][key] += stats[k][d][key]
            for d in TARGET_DAYS:
                bl[d]['total'] += baseline[d]['total']
                bl[d]['hits']  += baseline[d]['hits']
            done += 1
            print(f"  [{done:>3}/{len(TICKERS)}] {ticker}", end="\r", flush=True)

    print(f"\nDone.\n")
    _print_report(grand, bl)


def _print_report(grand, bl):
    DAYS  = TARGET_DAYS
    w     = 9

    # Baseline
    bl_overall_hits  = sum(bl[d]['hits']  for d in DAYS)
    bl_overall_total = sum(bl[d]['total'] for d in DAYS)
    bl_pct = bl_overall_hits / bl_overall_total * 100 if bl_overall_total else 0

    print("=" * 80)
    print(f"  ENSEMBLE K ABLATION  |  TP={TP_PCT}  SL={SL_PCT}")
    print(f"  Baseline (random every day): {bl_pct:.1f}%  ({bl_overall_hits}/{bl_overall_total})")
    print("=" * 80)
    header = f"{'K':<4}" + "".join(f"  {'─── ' + str(d) + 'd ───':>{w+6}}" for d in DAYS) + f"  {'── Total ──':>{w+6}}"
    subhdr = f"{'':4}" + "".join(f"  {'Signals':>{w}}  {'Hit%':>6}  {'Base%':>6}" for _ in DAYS) + f"  {'Signals':>{w}}  {'Hit%':>6}  {'Base%':>6}"
    print(header)
    print(subhdr)
    print("-" * 80)

    for k in range(1, MAX_K + 1):
        row   = f"K={k:<2}"
        g_sig = g_hit = g_pend = 0
        for d in DAYS:
            sig  = grand[k][d]['signals']
            hits = grand[k][d]['hits']
            pend = grand[k][d]['pending']
            evl  = sig - pend
            pct  = hits / evl * 100 if evl > 0 else 0.0
            blt  = bl[d]['total']
            bpct = bl[d]['hits'] / blt * 100 if blt > 0 else 0.0
            row  += f"  {sig:>{w}}  {pct:>5.1f}%  {bpct:>5.1f}%"
            g_sig += sig; g_hit += hits; g_pend += pend
        g_evl  = g_sig - g_pend
        g_pct  = g_hit / g_evl * 100 if g_evl > 0 else 0.0
        edge   = g_pct - bl_pct
        row   += f"  {g_sig:>{w}}  {g_pct:>5.1f}%  {bl_pct:>5.1f}%  (edge {edge:+.1f}%)"
        print(row)

    print("=" * 80)


if __name__ == "__main__":
    run_ablation()
