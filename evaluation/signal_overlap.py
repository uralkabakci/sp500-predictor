"""
Signal Overlap Analysis
-----------------------
For each ticker × horizon × fold, loads the selected ensemble models and
answers: if you used only m-model combinations, how many signals on average?

For each m in 1..K:
  - Enumerate all C(K, m) sub-combinations of the selected models
  - For each combination, count days where avg_prob >= THRESHOLD
  - Report the average signal count across all combinations

Output: params_log/signal_overlap.csv
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys as _sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..')
_sys.path.insert(0, os.path.join(_ROOT, 'core'))
os.chdir(_ROOT)

import glob
import itertools
import warnings
from collections import defaultdict
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd

import data_processor as dp

warnings.filterwarnings("ignore")

THRESHOLD            = 0.70
WF_FIRST_TEST_YEAR   = 2022
TEST_START_GAP_WEEKS = 2
MODELS_DIR           = "saved_models"
LOG_DIR              = "params_log"

TP_PCT = {10: 0.04, 15: 0.05, 20: 0.07}


def get_test_period(test_year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    today      = pd.Timestamp.today().normalize()
    fold_years = list(range(WF_FIRST_TEST_YEAR, today.year + 1))
    is_final   = (test_year == fold_years[-1])
    start = pd.Timestamp(f"{test_year}-01-01") + timedelta(weeks=TEST_START_GAP_WEEKS)
    end   = today if is_final else pd.Timestamp(f"{test_year}-12-31")
    return start, end


def build_selected_index(selected_csv: str) -> dict:
    """Returns {(ticker, days, fold_year): [pool_rank, ...]} for selected ensemble members."""
    df  = pd.read_csv(selected_csv)
    idx = defaultdict(list)
    for _, row in df.iterrows():
        key = (row["ticker"], int(row["days"]), int(row["fold_year"]))
        idx[key].append(int(row["pool_rank"]))
    return idx


def load_ensemble_pkgs(ticker: str, days: int, fold_year: int,
                       selected_ranks: list[int]) -> list:
    pct_tag = int(TP_PCT[days] * 100)
    pkgs = []
    for rank in selected_ranks:
        fname = os.path.join(
            MODELS_DIR,
            f"model_{ticker}_{days}d_{pct_tag}pct_fold{fold_year}_cand{rank}.pkl"
        )
        if os.path.exists(fname):
            try:
                pkgs.append(joblib.load(fname))
            except Exception:
                pass
    return sorted(pkgs, key=lambda p: p.get("rank", 99))


def get_probs(pkg: dict, df_test: pd.DataFrame) -> np.ndarray:
    """Return per-day probabilities for a single model. NaN where features missing."""
    probs = np.full(len(df_test), np.nan)
    for i, (_, row) in enumerate(df_test.iterrows()):
        try:
            X   = row[pkg["features"]].values.reshape(1, -1)
            X_s = pkg["scaler"].transform(X)
            probs[i] = pkg["model"].predict_proba(X_s)[0][1]
        except Exception:
            pass
    return probs


def analyse_ticker(ticker: str, df_features: pd.DataFrame,
                   selected_idx: dict) -> list[dict]:
    rows = []
    today      = pd.Timestamp.today().normalize()
    fold_years = list(range(WF_FIRST_TEST_YEAR, today.year + 1))

    for days in TP_PCT:
        for test_year in fold_years:
            key   = (ticker, days, test_year)
            ranks = selected_idx.get(key)
            if not ranks:
                continue
            pkgs = load_ensemble_pkgs(ticker, days, test_year, ranks)
            if not pkgs:
                continue

            test_start, test_end = get_test_period(test_year)
            df_test = df_features[
                (df_features.index >= test_start) &
                (df_features.index <= test_end)
            ]
            if len(df_test) < 5:
                continue

            K          = len(pkgs)
            total_days = len(df_test)

            # Pre-compute per-model probability arrays (shape: K × total_days)
            prob_matrix = np.stack([get_probs(pkg, df_test) for pkg in pkgs])

            record = {
                "ticker":     ticker,
                "days":       days,
                "fold_year":  test_year,
                "K":          K,
                "total_days": total_days,
            }

            # For each ensemble size m, enumerate all C(K, m) combinations
            for m in range(1, K + 1):
                combos     = list(itertools.combinations(range(K), m))
                sig_counts = []
                for combo in combos:
                    avg_probs = np.nanmean(prob_matrix[list(combo), :], axis=0)
                    n_signals = int(np.sum(avg_probs >= THRESHOLD))
                    sig_counts.append(n_signals)
                avg_signals = np.mean(sig_counts)
                record[f"m{m}_combos"]      = len(combos)
                record[f"m{m}_avg_signals"] = round(avg_signals, 1)
                record[f"m{m}_avg_pct"]     = round(avg_signals / total_days * 100, 2)

            rows.append(record)

            summary = "  ".join(
                f"m={m}→{record[f'm{m}_avg_signals']:.1f}sig"
                for m in range(1, K + 1)
            )
            print(f"  {ticker} {days}d fold{test_year}: K={K}  days={total_days}  {summary}")

    return rows


def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    selected_csv = os.path.join(LOG_DIR, "report_selected.csv")
    if not os.path.exists(selected_csv):
        print(f"ERROR: {selected_csv} not found. Run ensemble_selector.py first.")
        return

    selected_idx = build_selected_index(selected_csv)
    tickers = sorted({k[0] for k in selected_idx})
    print(f"Found {len(tickers)} tickers in selected: {tickers}\n")

    all_rows = []
    for ticker in tickers:
        print(f"[{ticker}]")
        try:
            raw_df      = dp.get_data(ticker)
            df_features = dp.create_full_feature_universe(raw_df, ticker=ticker)
        except Exception as e:
            print(f"  WARN: could not build features: {e}")
            continue
        rows = analyse_ticker(ticker, df_features, selected_idx)
        all_rows.extend(rows)

    if not all_rows:
        print("No data collected.")
        return

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["ticker", "days", "fold_year"])

    out = os.path.join(LOG_DIR, "signal_overlap.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}  ({len(df)} rows)")

    # ── Aggregate summary across all tickers/folds ────────────────────────
    max_k = df["K"].max()
    print(f"\n{'='*60}")
    print("Average signals per year by ensemble size")
    print("(weighted by total_days across all tickers and folds)")
    print(f"{'='*60}")
    total_w = df["total_days"].sum()
    for m in range(1, max_k + 1):
        col = f"m{m}_avg_signals"
        if col not in df.columns:
            continue
        w_avg_sig = (df[col] * df["total_days"]).sum() / total_w
        w_avg_pct = w_avg_sig / (df["total_days"] * df["total_days"]).sum() * total_w  # approx
        pct_col   = f"m{m}_avg_pct"
        w_avg_pct = (df[pct_col] * df["total_days"]).sum() / total_w
        n_combos  = df[f"m{m}_combos"].iloc[0] if f"m{m}_combos" in df.columns else "?"
        bar       = "█" * int(w_avg_pct)
        print(f"  m={m}  C(K,{m})={n_combos:>3}  "
              f"avg {w_avg_sig:5.1f} signals  ({w_avg_pct:5.1f}% of days)  {bar}")


if __name__ == "__main__":
    main()
