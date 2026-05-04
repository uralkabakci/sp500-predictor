import sys, os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.join(_HERE, '..')
sys.path.insert(0, _os.path.join(_ROOT, 'core'))
_os.chdir(_ROOT)

import pandas as pd
import numpy as np
import warnings
from datetime import timedelta

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score

import data_processor as dp

warnings.filterwarnings("ignore")

# ============================================================================
# PARAMETERS
# ============================================================================

TICKERS = [
    # Original 10
    "AAPL", "GOOGL", "MSFT", "NVDA", "META",
    "AMZN", "TSLA", "AMD", "INTC", "NFLX",
    # Tech / Semiconductors
    "QCOM", "AVGO", "TXN", "MU", "AMAT",
    "CRM", "ORCL", "CSCO", "IBM", "SNOW",
    # Finance
    "JPM", "BAC", "GS", "V", "MA",
    "BLK", "MS", "C",
    # Healthcare
    "JNJ", "PFE", "ABBV", "MRK", "UNH", "LLY",
    # Consumer / Retail
    "COST", "WMT", "HD", "NKE", "MCD", "SBUX",
    # Energy
    "XOM", "CVX",
    # Industrial / Other
    "BA", "CAT", "DIS", "PYPL", "UBER", "ADBE", "NOW", "PANW", "TMO",
]

TARGETS = [
    (20, 0.05),   # ~4 takvim haftası (20 iş günü), %5 hedef
    (40, 0.08),   # ~8 takvim haftası (40 iş günü), %8 hedef
]

TRAIN_START       = pd.Timestamp("2000-01-01")
WF_FIRST_TEST_YEAR = 2010   # first year used as test fold
WF_LAST_TEST_YEAR  = 2024   # last year used as test fold
GAP_WEEKS          = 2

BASE_CONFIG     = {'rsi': 14, 'sma_short': 20, 'sma_long': 50}
RSI_LIST        = [14]
RSI_CONFIG      = {'rsi': 14, 'sma_short': 10, 'sma_long': 50}
SMA_LIST        = [14, 30]
SMA_CONFIG_BASE = {'rsi': 14, 'sma_long': 200}

MODEL_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 1.0,
    'n_jobs': -1,
    'random_state': 42,
    'verbosity': 0,
}

STEP_SIZE_MAPPING = {20: 8, 40: 16}

# ============================================================================


def build_features(ticker):
    raw_df = dp.get_data(ticker)
    if raw_df.empty:
        return pd.DataFrame()
    return dp.create_enhanced_features(
        raw_df, ticker=ticker,
        base_config=BASE_CONFIG, rsi_list=RSI_LIST,
        rsi_config=RSI_CONFIG, sma_list=SMA_LIST,
        sma_config_base=SMA_CONFIG_BASE,
    )


def add_target(df, days, pct):
    temp = df.copy()
    next_highs = temp['High'].shift(-1)
    future_max = next_highs[::-1].rolling(window=days, min_periods=1).max()[::-1]
    temp['Target'] = ((future_max / temp['Close'] - 1) > pct).astype(int)
    temp.dropna(inplace=True)
    return temp


def compute_weights(index):
    min_d = index.min()
    max_d = index.max()
    span = max(((max_d - min_d).days), 1)
    return np.array([0.3 + 0.7 * ((d - min_d).days / span) for d in index])


def run_walk_forward():
    print("=" * 80)
    print("WALK-FORWARD VALIDATION")
    print(f"Test years: {WF_FIRST_TEST_YEAR} → {WF_LAST_TEST_YEAR}")
    print("=" * 80)

    exclude_cols = {'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Target'}
    all_results = []

    for ticker in TICKERS:
        print(f"\n{'='*40}\n{ticker}\n{'='*40}")

        df_feat = build_features(ticker)
        if df_feat.empty:
            print(f"  [SKIP] No features for {ticker}")
            continue

        for days, pct in TARGETS:
            df_ready = add_target(df_feat, days, pct)
            if df_ready.empty:
                continue

            feature_cols = [c for c in df_ready.columns if c not in exclude_cols]
            fold_rows = []

            for test_year in range(WF_FIRST_TEST_YEAR, WF_LAST_TEST_YEAR + 1):
                train_end   = pd.Timestamp(f"{test_year - 1}-12-31")
                test_start  = pd.Timestamp(f"{test_year}-01-01") + timedelta(weeks=GAP_WEEKS)
                test_end    = pd.Timestamp(f"{test_year}-12-31")

                train_mask = (df_ready.index >= TRAIN_START) & (df_ready.index <= train_end)
                test_mask  = (df_ready.index >= test_start)  & (df_ready.index <= test_end)

                df_train = df_ready[train_mask]
                df_test  = df_ready[test_mask]

                step = STEP_SIZE_MAPPING.get(days, 1)
                if step > 1:
                    df_test = df_test.iloc[::step]

                if len(df_train) < 100 or len(df_test) < 5:
                    continue
                if np.sum(df_train['Target']) < 5:
                    continue

                X_train = df_train[feature_cols].values
                y_train = df_train['Target'].values
                X_test  = df_test[feature_cols].values
                y_test  = df_test['Target'].values

                weights = compute_weights(df_train.index)

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s  = scaler.transform(X_test)

                clf = XGBClassifier(**MODEL_PARAMS)
                clf.fit(X_train_s, y_train, sample_weight=weights)

                probs = clf.predict_proba(X_test_s)[:, 1]
                preds = (probs >= 0.50).astype(int)

                total_signals = np.sum(preds)
                prec = precision_score(y_test, preds, zero_division=0) if total_signals > 0 else 0
                rec  = recall_score(y_test, preds, zero_division=0)    if total_signals > 0 else 0
                acc  = accuracy_score(y_test, preds)

                fold_rows.append({
                    'year': test_year,
                    'precision': prec,
                    'recall': rec,
                    'accuracy': acc,
                    'signals': int(total_signals),
                    'test_size': len(y_test),
                    'pos_rate': float(y_test.mean()),
                })

                print(f"  {test_year} | {days:2}d | Prec:{prec:.3f} Rec:{rec:.3f} Acc:{acc:.3f} | Signals:{total_signals}/{len(y_test)}")

            if fold_rows:
                df_folds = pd.DataFrame(fold_rows)
                avg_prec = df_folds['precision'].mean()
                avg_rec  = df_folds['recall'].mean()
                avg_acc  = df_folds['accuracy'].mean()
                print(f"  --- {ticker} {days}d AVERAGE | Prec:{avg_prec:.3f} Rec:{avg_rec:.3f} Acc:{avg_acc:.3f} ---")

                for row in fold_rows:
                    all_results.append({
                        'ticker': ticker,
                        'days': days,
                        **row,
                    })

    if all_results:
        df_all = pd.DataFrame(all_results)
        df_all.to_csv("wf_validation_results.csv", index=False)

        print("\n" + "=" * 80)
        print("OVERALL SUMMARY (averaged across all tickers and years)")
        print("=" * 80)
        for days, _ in TARGETS:
            subset = df_all[df_all['days'] == days]
            print(f"\n  {days}-day target:")
            print(f"    Avg Precision : {subset['precision'].mean():.3f}")
            print(f"    Avg Recall    : {subset['recall'].mean():.3f}")
            print(f"    Avg Accuracy  : {subset['accuracy'].mean():.3f}")
            print(f"    Avg Signals/yr: {subset['signals'].mean():.1f}")

            by_year = subset.groupby('year')[['precision', 'recall']].mean()
            print(f"\n    Year-by-year precision:")
            for yr, row in by_year.iterrows():
                bar = "█" * int(row['precision'] * 20)
                print(f"      {yr}: {row['precision']:.3f} {bar}")

        print(f"\nFull results saved to: wf_validation_results.csv")


if __name__ == "__main__":
    run_walk_forward()
