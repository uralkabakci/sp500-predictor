"""
quick_model_generator.py

Reads report_selected.csv and re-trains only the selected models
without running the full hyperparameter search.

Requires:
    params_log/report_selected.csv   (feature list + selection index)
    data_cache/                      (price + indicator data)

Usage:
    python quick_model_generator.py                  # all tickers
    python quick_model_generator.py AAPL MSFT NVDA   # specific tickers

Output: saved_models/model_<TICKER>_<N>d_<P>pct_fold<Y>_cand<R>.pkl
"""

import sys, os

_HERE    = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.join(_HERE, '..')           # github/ root
sys.path.insert(0, os.path.join(_ROOT, 'core'))
os.chdir(_ROOT)                                # tüm relative path'ler github/ root'tan

import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score
from xgboost import XGBClassifier

import data_processor as dp

warnings.filterwarnings("ignore")

# ── Config (must match ml_optimizer.py) ──────────────────────────────────────

TRAIN_START = pd.Timestamp("2012-01-01")
THRESHOLD   = 0.70
STOP_PCT    = 0.02

TP_PCT = {10: 0.04, 15: 0.05, 20: 0.07}

MODEL_PARAMS = {
    "n_estimators":     400,
    "max_depth":        8,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma":            0.1,
}

PARAMS_LOG_DIR  = "params_log"
SAVED_MODELS_DIR = "saved_models"

# ─────────────────────────────────────────────────────────────────────────────


def _prepare_target(df_s, days, tp_pct):
    closes = df_s["Close"].values
    highs  = df_s["High"].values
    lows   = df_s["Low"].values
    n      = len(closes)
    valid  = n - days

    idx          = np.arange(valid)[:, None] + np.arange(1, days + 1)[None, :]
    future_highs = highs[idx]
    future_lows  = lows[idx]

    tp_prices = closes[:valid] * (1 + tp_pct)
    sl_prices = closes[:valid] * (1 - STOP_PCT)

    tp_hit = future_highs >= tp_prices[:, None]
    sl_hit = future_lows  <= sl_prices[:, None]

    tp_day = np.where(tp_hit.any(axis=1), np.argmax(tp_hit, axis=1), days)
    sl_day = np.where(sl_hit.any(axis=1), np.argmax(sl_hit, axis=1), days)

    labels = np.zeros(n, dtype=int)
    labels[:valid] = (tp_day < sl_day).astype(int)

    temp = df_s.copy()
    temp["Target"] = labels
    temp = temp.iloc[:valid]
    temp.dropna(inplace=True)
    return temp


def _load_selected_index():
    path = os.path.join(PARAMS_LOG_DIR, "report_selected.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run ensemble_selector.py first.")
    df = pd.read_csv(path)
    return df


def _train_one(ticker, days, fold_year, pool_rank, features, df_ready):
    tp_pct = TP_PCT[days]

    df_target  = _prepare_target(df_ready, days, tp_pct)
    train_end  = pd.Timestamp(f"{fold_year - 1}-12-31")
    test_start = pd.Timestamp(f"{fold_year}-01-01")
    test_end   = pd.Timestamp(f"{fold_year}-12-31")

    train_mask = (df_target.index >= TRAIN_START) & (df_target.index <= train_end)
    test_mask  = (df_target.index >= test_start)  & (df_target.index <= test_end)

    df_train = df_target[train_mask]
    df_test  = df_target[test_mask]

    if len(df_train) < 100 or len(df_test) < 20:
        return None, "insufficient data"

    if np.sum(df_train["Target"]) < 5:
        return None, "too few positives"

    # Drop missing features gracefully
    features = [f for f in features if f in df_train.columns]
    if len(features) < 3:
        return None, "too few features"

    X_tr = df_train[features].values
    X_te = df_test[features].values
    y_tr = df_train["Target"].values
    y_te = df_test["Target"].values

    # Time-weighted sample weights
    min_d = df_train.index.min()
    max_d = df_train.index.max()
    span  = max((max_d - min_d).days, 1)
    weights = np.array([0.3 + 0.7 * ((d - min_d).days / span) for d in df_train.index])

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    clf = XGBClassifier(**MODEL_PARAMS, n_jobs=1, random_state=42, verbosity=0)
    clf.fit(X_tr_s, y_tr, sample_weight=weights)

    probs = clf.predict_proba(X_te_s)[:, 1]
    preds = (probs >= THRESHOLD).astype(int)
    n_sig = int(np.sum(preds))

    prec = precision_score(y_te, preds, zero_division=0) if n_sig > 0 else 0.0
    rec  = recall_score(y_te,    preds, zero_division=0) if n_sig > 0 else 0.0
    acc  = accuracy_score(y_te, preds)

    pkg = {
        "ticker":      ticker,
        "model":       clf,
        "scaler":      scaler,
        "threshold":   THRESHOLD,
        "features":    features,
        "target_days": days,
        "target_pct":  tp_pct,
        "fold_year":   fold_year,
        "rank":        pool_rank,
        "metrics":     {"precision": prec, "recall": rec,
                        "accuracy": acc, "pr_score": prec * rec},
    }
    return pkg, f"prec={prec:.3f} rec={rec:.3f} sigs={n_sig}"


def run(tickers_filter=None):
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    selected = _load_selected_index()
    if tickers_filter:
        selected = selected[selected["ticker"].isin(tickers_filter)]

    tickers = selected["ticker"].unique()
    print(f"Generating models for {len(tickers)} tickers "
          f"({len(selected)} model entries)...")
    print("-" * 60)

    ok = skip = fail = 0

    for ticker in tickers:
        raw_df = dp.get_data(ticker)
        if raw_df is None or raw_df.empty:
            print(f"  [{ticker}] SKIP — no data")
            skip += 1
            continue

        df_features = dp.create_full_feature_universe(raw_df, ticker=ticker)
        if df_features is None or df_features.empty:
            print(f"  [{ticker}] SKIP — feature build failed")
            skip += 1
            continue

        rows = selected[selected["ticker"] == ticker]

        for _, row in rows.iterrows():
            days      = int(row["days"])
            fold_year = int(row["fold_year"])
            pool_rank = int(row["pool_rank"])
            pct_tag   = int(TP_PCT[days] * 100)

            out_path = os.path.join(
                SAVED_MODELS_DIR,
                f"model_{ticker}_{days}d_{pct_tag}pct_fold{fold_year}_cand{pool_rank}.pkl"
            )

            if os.path.exists(out_path):
                print(f"  [{ticker}] {days}d fold{fold_year} cand{pool_rank} — already exists, skipping")
                ok += 1
                continue

            features_raw = str(row.get("features", ""))
            features = [f.strip() for f in features_raw.split("|") if f.strip()]
            if not features:
                print(f"  [{ticker}] {days}d fold{fold_year} cand{pool_rank} — no features in report_selected.csv")
                fail += 1
                continue

            pkg, msg = _train_one(ticker, days, fold_year, pool_rank,
                                  features, df_features)
            if pkg is None:
                print(f"  [{ticker}] {days}d fold{fold_year} cand{pool_rank} — FAIL ({msg})")
                fail += 1
                continue

            joblib.dump(pkg, out_path)
            print(f"  [{ticker}] {days}d fold{fold_year} cand{pool_rank} — OK  {msg}")
            ok += 1

    print("-" * 60)
    print(f"Done: {ok} OK  |  {skip} skipped  |  {fail} failed")


if __name__ == "__main__":
    tickers_filter = sys.argv[1:] if len(sys.argv) > 1 else None
    run(tickers_filter)
