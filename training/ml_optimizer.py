import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys as _sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..')
_sys.path.insert(0, os.path.join(_ROOT, 'core'))
os.chdir(_ROOT)

import pandas as pd
import numpy as np
import random as _rnd
import warnings
import joblib
import time
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score

import data_processor as dp

warnings.filterwarnings("ignore")

# ============================================================================
# PARAMETERS
# ============================================================================

TICKERS = [
    "AAPL", "GOOGL", "MSFT", "NVDA", "META",
    "AMZN", "TSLA", "AMD", "INTC", "NFLX",
    #"QCOM", "AVGO", "TXN", "MU", "AMAT",
    #"CRM", "ORCL", "CSCO", "IBM", "SNOW",
    #"JPM", "BAC", "GS", "V", "MA",
    #"BLK", "MS", "C",
    #"JNJ", "PFE", "ABBV", "MRK", "UNH", "LLY",
    #"COST", "WMT", "HD", "NKE", "MCD", "SBUX",
    #"XOM", "CVX",
    #"BA", "CAT", "DIS", "PYPL", "UBER", "ADBE", "NOW", "PANW", "TMO",
]
if os.environ.get("ABLATION_TICKERS"):
    TICKERS = os.environ["ABLATION_TICKERS"].split(",")

TIME_BUDGET_HOURS = 5.5

TARGETS = [
    (14, 0.05),
    (20, 0.07),
    (30, 0.10),
    (45, 0.13),
    (60, 0.15),
]

TRAIN_START          = pd.Timestamp("2012-01-01")
WF_FIRST_TEST_YEAR   = 2016
TEST_START_GAP_WEEKS = 2
STEP_SIZE_MAPPING    = {14: 4, 20: 8, 30: 10, 45: 15, 60: 20}

THRESHOLDS = [float(os.environ["ABLATION_THRESHOLD"])] if os.environ.get("ABLATION_THRESHOLD") else [0.70]

OUTPUT_FILE                = "modular_search_results_final.csv"
SAVE_FOLD_MODELS_FROM_YEAR = 2022

EXCLUDE_COLS = {"Open", "High", "Low", "Close", "Volume", "Adj Close", "Target", "ATR"}

NUM_WORKERS = os.cpu_count()

# ============================================================================
# JOINT SEARCH SPACE  (feature params + model hyperparams, per horizon)
# ============================================================================

HORIZON_SEARCH_SPACE = {
    14: {
        "feature": {
            "rsi_periods":  [[7], [14], [7, 14], [10, 14], [7, 10, 14], [14, 21]],
            "roc_periods":  [[7], [14], [7, 14]],
            "sma_periods":  [[10, 30], [20, 50], [10, 20, 50], [10, 30, 100], [20, 100]],
            "use_macd":     [True, False],
            "macd_config":  [{"fast": 12, "slow": 26, "signal": 9},
                             {"fast":  8, "slow": 21, "signal": 7}],
            "adx_windows":  [[], [7], [14], [7, 14]],
            "cci_windows":  [[], [7], [14], [7, 14]],
            "atr_windows":  [[7], [14], [7, 14]],
            "bb_windows":   [[], [10], [20], [10, 20]],
            "use_obv":      [True, False],
        },
        "model": {
            "n_estimators":     [100, 200, 300, 400],
            "max_depth":        [3, 4, 5, 6],
            "learning_rate":    [0.03, 0.05, 0.1, 0.15],
            "subsample":        [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "min_child_weight": [1, 3, 5],
            "gamma":            [0, 0.1, 0.3],
        },
    },
    20: {
        "feature": {
            "rsi_periods":  [[14], [10, 14], [7, 14, 21], [14, 21], [10, 14, 21]],
            "roc_periods":  [[7], [14], [7, 14]],
            "sma_periods":  [[20, 50], [10, 30, 100], [20, 50, 100], [30, 100], [10, 50, 100]],
            "use_macd":     [True, False],
            "macd_config":  [{"fast": 12, "slow": 26, "signal": 9},
                             {"fast":  8, "slow": 21, "signal": 7}],
            "adx_windows":  [[], [14], [7, 14], [14, 21]],
            "cci_windows":  [[], [14], [7, 14], [14, 20]],
            "atr_windows":  [[14], [7, 14]],
            "bb_windows":   [[], [20], [10, 20], [20, 30]],
            "use_obv":      [True, False],
        },
        "model": {
            "n_estimators":     [200, 300, 400, 500],
            "max_depth":        [4, 5, 6],
            "learning_rate":    [0.03, 0.05, 0.1, 0.15],
            "subsample":        [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "min_child_weight": [1, 3, 5],
            "gamma":            [0, 0.1, 0.3],
        },
    },
    30: {
        "feature": {
            "rsi_periods":  [[14], [14, 21], [10, 14, 21], [7, 14, 21]],
            "roc_periods":  [[14], [7, 14]],
            "sma_periods":  [[20, 50, 100], [30, 100, 200], [20, 100], [50, 200], [20, 50, 200]],
            "use_macd":     [True, False],
            "macd_config":  [{"fast": 12, "slow": 26, "signal": 9},
                             {"fast":  8, "slow": 21, "signal": 7}],
            "adx_windows":  [[], [14], [14, 21], [7, 14, 21]],
            "cci_windows":  [[], [14], [14, 20]],
            "atr_windows":  [[14], [7, 14]],
            "bb_windows":   [[], [20], [20, 30], [10, 20, 30]],
            "use_obv":      [True, False],
        },
        "model": {
            "n_estimators":     [300, 400, 500, 600],
            "max_depth":        [4, 5, 6, 7],
            "learning_rate":    [0.03, 0.05, 0.1],
            "subsample":        [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "min_child_weight": [1, 3, 5],
            "gamma":            [0, 0.1, 0.3],
        },
    },
    45: {
        "feature": {
            "rsi_periods":  [[14, 21], [10, 14, 21], [14], [7, 14, 21]],
            "roc_periods":  [[14], [7, 14]],
            "sma_periods":  [[50, 100, 200], [20, 100, 200], [30, 100, 200],
                             [100, 200], [20, 50, 200]],
            "use_macd":     [True, False],
            "macd_config":  [{"fast": 12, "slow": 26, "signal": 9},
                             {"fast":  8, "slow": 21, "signal": 7}],
            "adx_windows":  [[14, 21], [7, 14, 21], [14]],
            "cci_windows":  [[14, 20], [14], [20]],
            "atr_windows":  [[14], [7, 14]],
            "bb_windows":   [[], [20], [20, 30]],
            "use_obv":      [True, False],
        },
        "model": {
            "n_estimators":     [400, 500, 600, 700],
            "max_depth":        [5, 6, 7],
            "learning_rate":    [0.03, 0.05, 0.1],
            "subsample":        [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "min_child_weight": [1, 3, 5],
            "gamma":            [0, 0.1, 0.3],
        },
    },
    60: {
        "feature": {
            "rsi_periods":  [[14, 21], [7, 14, 21], [21], [14]],
            "roc_periods":  [[14], [7, 14]],
            "sma_periods":  [[50, 100, 200], [30, 100, 200], [100, 200], [20, 50, 200]],
            "use_macd":     [True],
            "macd_config":  [{"fast": 12, "slow": 26, "signal": 9},
                             {"fast":  8, "slow": 21, "signal": 7}],
            "adx_windows":  [[14, 21], [7, 14, 21]],
            "cci_windows":  [[14, 20], [20]],
            "atr_windows":  [[14], [7, 14]],
            "bb_windows":   [[20, 30], [30]],
            "use_obv":      [True, False],
        },
        "model": {
            "n_estimators":     [500, 600, 700, 800],
            "max_depth":        [5, 6, 7, 8],
            "learning_rate":    [0.03, 0.05, 0.1],
            "subsample":        [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "min_child_weight": [1, 3, 5],
            "gamma":            [0, 0.1, 0.3],
        },
    },
}

# ============================================================================

def _select_features_from_universe(df, fp, exclude_cols):
    cols = []

    for p in fp.get("rsi_periods", []):
        c = f"RSI_{p}"
        if c in df.columns: cols.append(c)

    for p in fp.get("roc_periods", []):
        c = f"ROC_{p}"
        if c in df.columns: cols.append(c)

    if fp.get("use_macd"):
        cfg = fp.get("macd_config", {"fast": 12, "slow": 26, "signal": 9})
        f_, s_, g_ = cfg["fast"], cfg["slow"], cfg["signal"]
        for c in [f"MACD_{f_}_{s_}_{g_}",
                  f"MACD_signal_{f_}_{s_}_{g_}",
                  f"MACD_diff_{f_}_{s_}_{g_}"]:
            if c in df.columns: cols.append(c)

    for p in fp.get("adx_windows", []):
        c = f"ADX_{p}"
        if c in df.columns: cols.append(c)

    for p in fp.get("cci_windows", []):
        c = f"CCI_{p}"
        if c in df.columns: cols.append(c)

    for p in fp.get("atr_windows", []):
        c = f"ATR_{p}"
        if c in df.columns: cols.append(c)

    for p in fp.get("bb_windows", []):
        for c in [f"BB_Width_{p}", f"BB_Pct_{p}"]:
            if c in df.columns: cols.append(c)

    if fp.get("use_obv") and "OBV" in df.columns:
        cols.append("OBV")

    for p in fp.get("sma_periods", []):
        c = f"Dist_SMA_{p}"
        if c in df.columns: cols.append(c)

    for c in df.columns:
        if any(c.startswith(pfx) for pfx in
               ["google_trend", "eps_yoy", "revenue_yoy", "days_since"]):
            cols.append(c)

    seen, result = set(), []
    for c in cols:
        if c not in seen and c not in exclude_cols:
            seen.add(c)
            result.append(c)
    return result


STOP_PCT = 0.02  # must match simulation FIXED_STOP_PCT

def _prepare_target(df_s, d, tp_pct):
    """
    Label = 1 if price reaches tp_pct gain before hitting STOP_PCT loss
    within the next d trading days. Vectorized numpy implementation.
    """
    closes = df_s["Close"].values
    highs  = df_s["High"].values
    lows   = df_s["Low"].values
    n      = len(closes)
    valid  = n - d

    # Build (valid, d) matrices of future highs/lows
    idx          = np.arange(valid)[:, None] + np.arange(1, d + 1)[None, :]
    future_highs = highs[idx]                          # (valid, d)
    future_lows  = lows[idx]                           # (valid, d)

    tp_prices = closes[:valid] * (1 + tp_pct)          # (valid,)
    sl_prices = closes[:valid] * (1 - STOP_PCT)        # (valid,)

    tp_hit = future_highs >= tp_prices[:, None]        # (valid, d)
    sl_hit = future_lows  <= sl_prices[:, None]        # (valid, d)

    tp_day = np.where(tp_hit.any(axis=1), np.argmax(tp_hit, axis=1), d)
    sl_day = np.where(sl_hit.any(axis=1), np.argmax(sl_hit, axis=1), d)

    labels = np.zeros(n, dtype=int)
    labels[:valid] = (tp_day < sl_day).astype(int)

    temp = df_s.copy()
    temp["Target"] = labels
    temp = temp.iloc[:valid]
    temp.dropna(inplace=True)
    return temp


# ============================================================================
# Per-ticker worker
# ============================================================================

def _train_ticker(ticker: str, deadline: float) -> tuple:
    warnings.filterwarnings("ignore")
    os.makedirs("saved_models", exist_ok=True)

    csv_lines   = []
    today       = pd.Timestamp.today().normalize()
    fold_years  = list(range(WF_FIRST_TEST_YEAR, today.year + 1))
    ticker_t0   = time.time()

    # Pre-compute: how many (days, fold) pairs exist — for time budget distribution
    total_slots = len(TARGETS) * len(fold_years)

    try:
        raw_df = dp.get_data(ticker)
        if raw_df.empty:
            print(f"[{ticker}] No data, skipping.")
            return ticker, []

        df_universe = dp.create_full_feature_universe(raw_df, ticker=ticker)
        print(f"[{ticker}] Universe ready ({len(df_universe)} rows). Starting search.")

        slots_done = 0

        for days, pct in TARGETS:
            if time.time() > deadline:
                break

            df_ready     = _prepare_target(df_universe, days, pct)
            search_space = HORIZON_SEARCH_SPACE[days]
            step_size    = STEP_SIZE_MAPPING.get(days, 1)
            final_pkg    = None

            for fold_idx, test_year in enumerate(fold_years):
                if time.time() > deadline:
                    break

                is_final = (fold_idx == len(fold_years) - 1)

                train_end  = pd.Timestamp(f"{test_year - 1}-12-31")
                test_start = pd.Timestamp(f"{test_year}-01-01") + timedelta(weeks=TEST_START_GAP_WEEKS)
                test_end   = today if is_final else pd.Timestamp(f"{test_year}-12-31")

                train_mask = (df_ready.index >= TRAIN_START) & (df_ready.index <= train_end)
                test_mask  = (df_ready.index >= test_start)  & (df_ready.index <= test_end)

                df_train = df_ready[train_mask]
                df_test  = df_ready[test_mask]
                if step_size > 1 and not is_final:
                    df_test = df_test.iloc[::step_size]

                min_test = 2 if is_final else 5
                if len(df_train) < 100 or len(df_test) < min_test:
                    slots_done += 1
                    continue
                if np.sum(df_train["Target"]) < 5:
                    slots_done += 1
                    continue

                # Distribute remaining time evenly across remaining slots
                slots_remaining = max(total_slots - slots_done, 1)
                time_remaining  = max(deadline - time.time(), 0)
                fold_budget     = time_remaining / slots_remaining
                fold_deadline   = time.time() + fold_budget

                min_d  = df_train.index.min()
                max_d  = df_train.index.max()
                span   = max((max_d - min_d).days, 1)
                sample_weights = np.array([
                    0.3 + 0.7 * ((d - min_d).days / span)
                    for d in df_train.index
                ])

                y_train = df_train["Target"].values
                y_test  = df_test["Target"].values

                best_clf          = None
                best_score        = -1.0
                best_feature_cols = None
                best_scaler       = None
                best_params_log   = {}

                fold_lbl  = f"{test_year}(FINAL)" if is_final else str(test_year)
                fold_t0   = time.time()
                iter_count = 0
                last_print = fold_t0

                while time.time() < fold_deadline and time.time() < deadline:
                    fp = {k: _rnd.choice(v) for k, v in search_space["feature"].items()}
                    mp = {k: _rnd.choice(v) for k, v in search_space["model"].items()}

                    feature_cols = _select_features_from_universe(df_train, fp, EXCLUDE_COLS)
                    if len(feature_cols) < 3:
                        iter_count += 1
                        continue

                    try:
                        X_tr = df_train[feature_cols].values
                        X_te = df_test[feature_cols].values

                        scaler    = StandardScaler()
                        X_tr_s    = scaler.fit_transform(X_tr)
                        X_te_s    = scaler.transform(X_te)

                        clf = XGBClassifier(**mp, n_jobs=1, random_state=42, verbosity=0)
                        clf.fit(X_tr_s, y_train, sample_weight=sample_weights)

                        pv = clf.predict_proba(X_te_s)[:, 1]
                        p_ = (pv >= THRESHOLDS[0]).astype(int)
                        sc = 0.0
                        if np.sum(p_) > 0:
                            prec_ = precision_score(y_test, p_, zero_division=0)
                            rec_  = recall_score(y_test,    p_, zero_division=0)
                            ev    = prec_ * pct - (1 - prec_) * STOP_PCT
                            if ev > 0:
                                sc = ev * (rec_ ** 0.25)

                        if sc > best_score:
                            best_score        = sc
                            best_clf          = clf
                            best_feature_cols = feature_cols
                            best_scaler       = scaler
                            best_params_log   = {"model": mp, "features": fp}
                    except Exception:
                        pass

                    iter_count += 1

                    # Progress print every 30 seconds
                    now = time.time()
                    if now - last_print >= 30:
                        last_print = now
                        print(f"  [{ticker}][{fold_lbl}][{days}d] "
                              f"iter={iter_count:,}  best={best_score:.4f}  "
                              f"fold={now - fold_t0:.0f}s")

                slots_done += 1

                if best_clf is None or best_scaler is None:
                    continue

                X_te_s = best_scaler.transform(df_test[best_feature_cols].values)
                probs  = best_clf.predict_proba(X_te_s)[:, 1]
                preds  = (probs >= THRESHOLDS[0]).astype(int)

                n_sig = int(np.sum(preds))
                n_ok  = int(np.sum((preds == 1) & (y_test == 1)))
                prec  = precision_score(y_test, preds, zero_division=0) if n_sig > 0 else 0
                rec   = recall_score(y_test,    preds, zero_division=0) if n_sig > 0 else 0
                acc   = accuracy_score(y_test, preds)
                wr    = n_ok / n_sig if n_sig > 0 else 0

                fold_elapsed = time.time() - fold_t0
                print(f"  [{ticker}][{fold_lbl}] {days}d | "
                      f"Prec:{prec:.3f} Rec:{rec:.3f} Sigs:{n_sig}/{len(y_test)} | "
                      f"iters:{iter_count:,} in {fold_elapsed:.0f}s")

                pkg = {
                    "ticker":      ticker,
                    "model":       best_clf,
                    "scaler":      best_scaler,
                    "threshold":   THRESHOLDS[0],
                    "features":    best_feature_cols,
                    "target_days": days,
                    "target_pct":  pct,
                    "fold_year":   test_year,
                    "metrics":     {"precision": prec, "recall": rec,
                                    "accuracy": acc, "pr_score": prec * rec},
                }

                if is_final:
                    final_pkg = pkg

                if test_year >= SAVE_FOLD_MODELS_FROM_YEAR:
                    fold_fname = (f"saved_models/model_{ticker}_{days}d_"
                                  f"{int(pct * 100)}pct_fold{test_year}.pkl")
                    joblib.dump(pkg, fold_fname)

                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                csv_lines.append(
                    f"{ts},{ticker},{days},{pct},{test_year},XGBoost,"
                    f"{THRESHOLDS[0]},{prec:.4f},{rec:.4f},{acc:.4f},{wr:.4f},"
                    f"{n_ok},{n_sig},\"{best_params_log}\"\n"
                )

            if final_pkg is not None:
                fname = f"saved_models/model_{ticker}_{days}d_{int(pct * 100)}pct.pkl"
                joblib.dump(final_pkg, fname)
                m = final_pkg["metrics"]
                print(f"  *** [{ticker}] SAVED {days}d | "
                      f"Prec:{m['precision']:.4f} Rec:{m['recall']:.4f} ***")

    except Exception as e:
        import traceback
        print(f"[{ticker}] ERROR: {e}")
        traceback.print_exc()

    elapsed = time.time() - ticker_t0
    print(f"[{ticker}] FINISHED in {elapsed / 60:.1f}min")
    return ticker, csv_lines


# ============================================================================
# Main
# ============================================================================

def _search_space_size(sp):
    feat, model = 1, 1
    for v in sp["feature"].values(): feat  *= len(v)
    for v in sp["model"].values():   model *= len(v)
    return feat, model


def run_optimization():
    deadline = time.time() + TIME_BUDGET_HOURS * 3600

    print("=" * 80)
    print(f"OVERNIGHT SEARCH  |  {NUM_WORKERS} parallel workers  |  {TIME_BUDGET_HOURS}h budget")
    print(f"Deadline : {datetime.fromtimestamp(deadline).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tickers  : {len(TICKERS)}  |  Targets: {TARGETS}")
    print("=" * 80)
    print(f"{'Horizon':<10} {'Feature space':>16} {'Model space':>13} {'Total':>16}")
    print("-" * 60)
    for days, _ in TARGETS:
        f, m = _search_space_size(HORIZON_SEARCH_SPACE[days])
        print(f"  {days}d{'':<6} {f:>16,} {m:>13,} {f * m:>16,}")
    print("=" * 80)
    print()

    os.makedirs("saved_models", exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        f.write("Timestamp,Ticker,Target_Days,Target_Pct,Fold_Year,Models,"
                "Threshold,Precision,Recall,Accuracy,Win_Rate,"
                "Correct_Count,Total_Signals,Best_Params\n")

    completed = 0
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(_train_ticker, t, deadline): t for t in TICKERS}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                _, csv_lines = future.result()
                if csv_lines:
                    with open(OUTPUT_FILE, "a") as f:
                        f.writelines(csv_lines)
                completed += 1
                print(f"[{completed}/{len(TICKERS)}] {ticker} done.")
            except Exception as e:
                print(f"[{ticker}] FAILED: {e}")

    print(f"\n[DONE] {completed}/{len(TICKERS)} tickers. Results: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_optimization()
