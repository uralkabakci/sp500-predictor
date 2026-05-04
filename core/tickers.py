"""
Single source of truth for the ticker universe.

Usage:
    from tickers import TICKERS, TICKER_NAMES

To add/remove tickers edit tickers.csv (symbol,name columns).
"""

import os
import csv

_CSV_PATH = os.path.join(os.path.dirname(__file__), "tickers.csv")


def _load():
    symbols, names = [], {}
    with open(_CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sym = row["symbol"].strip()
            symbols.append(sym)
            names[sym] = row["name"].strip()
    return symbols, names


TICKERS, TICKER_NAMES = _load()
