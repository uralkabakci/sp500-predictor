"""
Single source of truth for the ticker universe.

Usage:
    from tickers import TICKERS, TICKER_NAMES        # all tickers
    from tickers import DELISTED_PAIRS               # set of (ticker, horizon) to skip
    from tickers import is_pair_active               # (ticker, horizon) -> bool

A (ticker, horizon) pair is "delisted" when its 2025 backtest win rate was
<= 40% with at least 3 signals. We skip these pairs at signal generation.

To regenerate delisted_pairs.csv, run signal_analysis.py then a small
walk-forward script (see project notes).
"""

import os
import csv

_CSV_PATH      = os.path.join(os.path.dirname(__file__), "tickers.csv")
_DELISTED_PATH = os.path.join(os.path.dirname(__file__), "delisted_pairs.csv")


def _load_tickers():
    symbols, names = [], {}
    with open(_CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sym = row["symbol"].strip()
            symbols.append(sym)
            names[sym] = row["name"].strip()
    return symbols, names


def _load_delisted_pairs():
    pairs = set()
    if not os.path.exists(_DELISTED_PATH):
        return pairs
    with open(_DELISTED_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                pairs.add((row["ticker"].strip(), int(row["horizon"])))
            except Exception:
                continue
    return pairs


TICKERS, TICKER_NAMES = _load_tickers()
DELISTED_PAIRS        = _load_delisted_pairs()


def is_pair_active(ticker: str, horizon: int) -> bool:
    """Return True if (ticker, horizon) should be used for signal generation."""
    return (ticker, horizon) not in DELISTED_PAIRS
