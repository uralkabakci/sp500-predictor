"""
Ensemble Selector
-----------------
Loads saved candidate pools (JSON + pkl) from ml_optimizer training output and
selects the best K-model ensemble per fold using:

    obj(S) = Σ quality(cᵢ) + λ × avg_pairwise_jaccard(S)

Re-run this script any time to change K, LAMBDA, or quality metric without
retraining models.

Outputs (in params_log/):
  report_all_candidates.csv   — every above-threshold config with metrics
  report_selected.csv         — the selected K models per fold with metrics

Usage:
    python ensemble_selector.py
    python ensemble_selector.py --tickers AAPL NVDA --k 3 --lambda 0.4
"""

import os
import sys as _sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..')
_sys.path.insert(0, os.path.join(_ROOT, 'core'))
os.chdir(_ROOT)

import json
import glob
import argparse
import itertools
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Configurable parameters ──────────────────────────────────────────────────
K           = 3      # ensemble size (how many models to select per fold)
LAMBDA      = 0.30   # weight of diversity bonus in objective
MODELS_DIR  = "saved_models"
LOG_DIR     = "params_log"

# Tickers / horizons to scan — set to None to scan all found files
TICKERS     = None   # e.g. ["AAPL", "NVDA"] or None for all
TARGET_DAYS = None   # e.g. [7, 14] or None for all
# ─────────────────────────────────────────────────────────────────────────────


def _variable_features(candidates: list[dict]) -> set:
    """Features that are NOT present in every single candidate — the only ones
    that contribute to diversity. Constant features inflate the union and
    suppress Jaccard, making lambda ineffective."""
    all_feat_sets = [set(c["features"]) for c in candidates]
    constant = set.intersection(*all_feat_sets) if all_feat_sets else set()
    return set.union(*all_feat_sets) - constant if all_feat_sets else set()


def _jaccard_variable(feats_a: list, feats_b: list, variable: set) -> float:
    a = set(feats_a) & variable
    b = set(feats_b) & variable
    union = a | b
    if not union:
        return 0.0
    return 1.0 - len(a & b) / len(union)


def _avg_pairwise_jaccard(feature_lists: list[list], variable: set) -> float:
    n = len(feature_lists)
    if n < 2:
        return 0.0
    total = sum(
        _jaccard_variable(feature_lists[i], feature_lists[j], variable)
        for i, j in itertools.combinations(range(n), 2)
    )
    return total / (n * (n - 1) / 2)


def _objective(candidates: list[dict], indices: tuple, lam: float,
               variable: set) -> float:
    subset = [candidates[i] for i in indices]
    quality_sum = sum(c["score"] for c in subset)
    diversity   = _avg_pairwise_jaccard([c["features"] for c in subset], variable)
    return quality_sum + lam * len(subset) * diversity


def find_json_files() -> list[str]:
    return sorted(glob.glob(os.path.join(LOG_DIR, "candidates_*.json")))


def parse_json_fname(fname: str) -> dict | None:
    base = os.path.basename(fname)
    # candidates_{ticker}_{days}d_{pct}pct_fold{year}.json
    try:
        parts = base.replace("candidates_", "").replace(".json", "").split("_")
        # parts = [ticker, "{days}d", "{pct}pct", "fold{year}"]
        ticker   = parts[0]
        days     = int(parts[1].rstrip("d"))
        pct_tag  = int(parts[2].rstrip("pct"))
        fold_year = int(parts[3].replace("fold", ""))
        return {"ticker": ticker, "days": days, "pct_tag": pct_tag, "fold_year": fold_year}
    except Exception:
        return None


def load_model_pool(ticker: str, days: int, pct_tag: int, fold_year: int) -> dict[int, dict]:
    """Load all _cand{N}.pkl files for this fold. Returns {rank: pkg}."""
    pattern = os.path.join(
        MODELS_DIR,
        f"model_{ticker}_{days}d_{pct_tag}pct_fold{fold_year}_cand*.pkl"
    )
    pool = {}
    for fname in glob.glob(pattern):
        try:
            pkg  = joblib.load(fname)
            rank = pkg.get("rank")
            if rank is not None:
                pool[rank] = pkg
        except Exception:
            pass
    return pool


def select_ensemble(candidates: list[dict], k: int, lam: float) -> tuple[list[int], float]:
    """Exhaustive C(N, k) search. Returns (selected_indices, best_obj)."""
    n        = len(candidates)
    k        = min(k, n)
    variable = _variable_features(candidates)
    best_obj  = -1.0
    best_idxs = list(range(k))

    for combo in itertools.combinations(range(n), k):
        obj = _objective(candidates, combo, lam, variable)
        if obj > best_obj:
            best_obj  = obj
            best_idxs = list(combo)

    return best_idxs, best_obj


def run(k: int, lam: float, tickers_filter, days_filter):
    os.makedirs(LOG_DIR, exist_ok=True)

    json_files = find_json_files()
    if not json_files:
        print(f"[selector] No candidate JSON files found in {LOG_DIR}/")
        return

    all_rows      = []  # for report_all_candidates.csv
    selected_rows = []  # for report_selected.csv

    for jf in json_files:
        meta = parse_json_fname(jf)
        if meta is None:
            continue

        ticker    = meta["ticker"]
        days      = meta["days"]
        pct_tag   = meta["pct_tag"]
        fold_year = meta["fold_year"]

        if tickers_filter and ticker not in tickers_filter:
            continue
        if days_filter and days not in days_filter:
            continue

        with open(jf) as f:
            candidates = json.load(f)

        if not candidates:
            continue

        # Append to all-candidates report
        for rank_in_log, c in enumerate(candidates, 1):
            all_rows.append({
                "ticker":    ticker,
                "days":      days,
                "pct_tag":   pct_tag,
                "fold_year": fold_year,
                "log_rank":  rank_in_log,
                "score":     round(c["score"],     4),
                "precision": round(c["precision"], 4),
                "recall":    round(c["recall"],    4),
                "accuracy":  round(c["accuracy"],  4),
                "n_sig":     c["n_sig"],
                "features":  "|".join(c["features"]),
            })

        # ── Ensemble selection ────────────────────────────────────────────
        pool = load_model_pool(ticker, days, pct_tag, fold_year)

        # Use candidates directly — model pool may be a subset (CANDIDATE_POOL_SIZE).
        # Both are ordered by score descending, so index i → pool rank i+1.
        if pool:
            # Re-sort candidates by score desc to match pool rank ordering
            cands_sorted     = sorted(candidates, key=lambda x: x["score"], reverse=True)
            n_pool           = len(pool)
            cands_with_model = cands_sorted[:n_pool]

            sel_idxs, sel_obj = select_ensemble(cands_with_model, k, lam)

            for sel_rank, idx in enumerate(sel_idxs, 1):
                c         = cands_with_model[idx]
                pool_rank = idx + 1   # pool pkl files are 1-indexed
                pkg       = pool.get(pool_rank, {})
                metrics   = pkg.get("metrics", {}) if pkg else {}

                selected_rows.append({
                    "ticker":       ticker,
                    "days":         days,
                    "pct_tag":      pct_tag,
                    "fold_year":    fold_year,
                    "ensemble_rank": sel_rank,
                    "pool_rank":    pool_rank,
                    "obj_score":    round(sel_obj, 4),
                    "score":        round(c["score"],     4),
                    "precision":    round(c["precision"], 4),
                    "recall":       round(c["recall"],    4),
                    "accuracy":     round(c["accuracy"],  4),
                    "n_sig":        c["n_sig"],
                    "features":     "|".join(c["features"]),
                    "has_model_pkl": bool(pkg),
                })

            label = f"{ticker} {days}d fold{fold_year}"
            print(f"  {label:<35}  {len(cands_with_model):>3} cands  "
                  f"selected {len(sel_idxs)}  obj={sel_obj:.4f}")
        else:
            # No pkl files — still write to report_selected so quick_model_generator can train them
            cands_sorted = sorted(candidates, key=lambda x: x["score"], reverse=True)
            sel_idxs, sel_obj = select_ensemble(cands_sorted, k, lam)

            for sel_rank, idx in enumerate(sel_idxs, 1):
                c         = cands_sorted[idx]
                pool_rank = idx + 1

                selected_rows.append({
                    "ticker":        ticker,
                    "days":          days,
                    "pct_tag":       pct_tag,
                    "fold_year":     fold_year,
                    "ensemble_rank": sel_rank,
                    "pool_rank":     pool_rank,
                    "obj_score":     round(sel_obj, 4),
                    "score":         round(c["score"],     4),
                    "precision":     round(c["precision"], 4),
                    "recall":        round(c["recall"],    4),
                    "accuracy":      round(c["accuracy"],  4),
                    "n_sig":         c["n_sig"],
                    "features":      "|".join(c["features"]),
                    "has_model_pkl": False,
                })

            label = f"{ticker} {days}d fold{fold_year}"
            print(f"  {label:<35}  {len(cands_sorted):>3} cands  "
                  f"selected {len(sel_idxs)}  obj={sel_obj:.4f}  (no pkls)")

    # ── Write reports ─────────────────────────────────────────────────────────
    if all_rows:
        df_all = pd.DataFrame(all_rows)
        out_all = os.path.join(LOG_DIR, "report_all_candidates.csv")
        df_all.to_csv(out_all, index=False)
        print(f"\n[selector] All candidates  → {out_all}  ({len(df_all)} rows)")
    else:
        print("\n[selector] No candidate rows to write.")

    if selected_rows:
        df_sel = pd.DataFrame(selected_rows)
        out_sel = os.path.join(LOG_DIR, "report_selected.csv")
        df_sel.to_csv(out_sel, index=False)
        print(f"[selector] Selected models → {out_sel}  ({len(df_sel)} rows)")
    else:
        print("[selector] No selected model rows to write (no pkl files found).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",       type=int,   default=K,      help="Ensemble size")
    parser.add_argument("--lambda",  type=float, default=LAMBDA, dest="lam",
                        help="Diversity weight λ")
    parser.add_argument("--tickers", nargs="+",  default=None,   help="Filter tickers")
    parser.add_argument("--days",    nargs="+",  type=int, default=None,
                        help="Filter target days")
    args = parser.parse_args()

    print("=" * 60)
    print("Ensemble Selector")
    print(f"  K      = {args.k}")
    print(f"  λ      = {args.lam}")
    print(f"  tickers= {args.tickers or 'all'}")
    print(f"  days   = {args.days or 'all'}")
    print("=" * 60)

    run(k=args.k, lam=args.lam, tickers_filter=args.tickers, days_filter=args.days)


if __name__ == "__main__":
    main()
