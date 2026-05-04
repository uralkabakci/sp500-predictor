"""
Feature Importance Analyzer
----------------------------
Loads all saved models from saved_models/ and aggregates XGBoost feature importances.
Run after ml_optimizer.py finishes to find most/least useful features.

Usage:
    python feature_importance.py
    python feature_importance.py --horizon 14
    python feature_importance.py --ticker AAPL
"""

import os
import sys as _sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..')
_sys.path.insert(0, os.path.join(_ROOT, 'core'))
os.chdir(_ROOT)

import glob
import argparse
import joblib
import numpy as np
import pandas as pd

MODELS_DIR = "saved_models"


def load_models(horizon: int = None, ticker: str = None) -> list[dict]:
    pattern = os.path.join(MODELS_DIR, "model_*.pkl")
    files = glob.glob(pattern)

    # Exclude fold models — use only the final per-ticker model
    files = [f for f in files if "fold" not in os.path.basename(f)]

    pkgs = []
    for f in files:
        try:
            pkg = joblib.load(f)
        except Exception as e:
            print(f"  [WARN] Could not load {f}: {e}")
            continue

        if horizon and pkg.get("target_days") != horizon:
            continue
        if ticker and pkg.get("ticker") != ticker:
            continue

        pkgs.append(pkg)

    return pkgs


def extract_importances(pkgs: list[dict]) -> pd.DataFrame:
    rows = []
    for pkg in pkgs:
        model    = pkg["model"]
        features = pkg["features"]

        # sklearn API returns array in same order as features list
        importances = model.feature_importances_
        total = importances.sum() or 1.0

        for feat, gain in zip(features, importances):
            rows.append({
                "feature":  feat,
                "ticker":   pkg["ticker"],
                "horizon":  pkg["target_days"],
                "gain":     gain / total,
                "selected": 1,
            })

    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg = (df.groupby("feature")
             .agg(
                 mean_gain   = ("gain",     "mean"),
                 std_gain    = ("gain",     "std"),
                 max_gain    = ("gain",     "max"),
                 model_count = ("selected", "count"),
             )
             .fillna(0)
             .sort_values("mean_gain", ascending=False)
             .reset_index())
    return agg


def _group_by_prefix(agg: pd.DataFrame) -> pd.DataFrame:
    """Summarize by feature family (rsi, sma, atr, etc.)."""
    def prefix(name):
        for p in ["rsi", "roc", "sma", "ema", "macd", "adx", "cci", "atr",
                  "bb", "obv", "search_interest", "trend_price",
                  "eps_yoy", "revenue_yoy", "days_since"]:
            if name.lower().startswith(p):
                return p
        return name.split("_")[0]

    agg = agg.copy()
    agg["family"] = agg["feature"].apply(prefix)
    return (agg.groupby("family")
               .agg(mean_gain=("mean_gain", "mean"), feature_count=("feature", "count"))
               .sort_values("mean_gain", ascending=False)
               .reset_index())


def print_report(agg: pd.DataFrame, top_n: int = 15):
    total = len(agg)
    print("\n" + "=" * 65)
    print(f"  FEATURE IMPORTANCE REPORT  ({total} unique features)")
    print("=" * 65)

    print(f"\n  TOP {top_n} — keep these in search space")
    print(f"  {'Feature':<35} {'Mean Gain':>10} {'Max Gain':>10} {'Models':>7}")
    print("  " + "-" * 62)
    for _, row in agg.head(top_n).iterrows():
        print(f"  {row['feature']:<35} {row['mean_gain']:>10.4f} "
              f"{row['max_gain']:>10.4f} {row['model_count']:>7.0f}")

    bottom = agg.tail(top_n).iloc[::-1]
    print(f"\n  BOTTOM {top_n} — consider removing from search space")
    print(f"  {'Feature':<35} {'Mean Gain':>10} {'Max Gain':>10} {'Models':>7}")
    print("  " + "-" * 62)
    for _, row in bottom.iterrows():
        print(f"  {row['feature']:<35} {row['mean_gain']:>10.4f} "
              f"{row['max_gain']:>10.4f} {row['model_count']:>7.0f}")

    family = _group_by_prefix(agg)
    print(f"\n  BY FEATURE FAMILY")
    print(f"  {'Family':<25} {'Mean Gain':>10} {'# Features':>12}")
    print("  " + "-" * 50)
    for _, row in family.iterrows():
        print(f"  {row['family']:<25} {row['mean_gain']:>10.4f} {row['feature_count']:>12.0f}")

    zero_gain = agg[agg["mean_gain"] == 0]
    if not zero_gain.empty:
        print(f"\n  ZERO GAIN FEATURES ({len(zero_gain)}) — safe to remove:")
        for feat in zero_gain["feature"].tolist():
            print(f"    - {feat}")

    print()


def save_csv(agg: pd.DataFrame, path: str = "feature_importance_results.csv"):
    agg.to_csv(path, index=False)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=None, help="Filter by horizon (14/20/30)")
    parser.add_argument("--ticker",  type=str, default=None, help="Filter by ticker")
    parser.add_argument("--top",     type=int, default=15,   help="How many top/bottom to show")
    parser.add_argument("--csv",     action="store_true",    help="Save results to CSV")
    args = parser.parse_args()

    pkgs = load_models(horizon=args.horizon, ticker=args.ticker)
    if not pkgs:
        print(f"  No models found in {MODELS_DIR}/")
        print("  Run ml_optimizer.py first.")
        return

    print(f"  Loaded {len(pkgs)} models", end="")
    if args.horizon: print(f" (horizon={args.horizon}d)", end="")
    if args.ticker:  print(f" (ticker={args.ticker})", end="")
    print()

    df  = extract_importances(pkgs)
    agg = aggregate(df)

    print_report(agg, top_n=args.top)

    if args.csv:
        save_csv(agg)


if __name__ == "__main__":
    main()
