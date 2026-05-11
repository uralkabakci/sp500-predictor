# S&P 500 Signal Generation System

A machine learning system that generates buy signals for the S&P 500 universe using XGBoost ensembles trained on technical indicators, earnings, sector relative strength, and Google Trends. Ships with an OHLC walk-forward backtest, a horizon-level performance filter, and a live FastAPI server that refreshes signals hourly during market hours.

## Quick Start

Price, earnings, trends, and sector data are bundled in `data_cache/`. Model selection parameters for 503 tickers live in `params_log/report_selected.csv`. Install dependencies and generate models:

```bash
pip install -r requirements.txt
python training/quick_model_generator.py             # all tickers
# or specific ones:
python training/quick_model_generator.py AAPL MSFT NVDA
```

Run the live signal server:

```bash
cd live && uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` for the dashboard.

Run a backtest:

```bash
python evaluation/signal_analysis.py
```

This writes `signal_analysis_trades.csv` and `per_ticker_horizon.csv` and prints a per-ticker hit-rate table vs random baseline.

## Project Structure

```
core/
  tickers.csv              506 S&P 500 symbols (single source of truth)
  tickers.py               Loads TICKERS, DELISTED_PAIRS, is_pair_active()
  delisted_pairs.csv       (ticker, horizon) pairs disabled from live signals
  data_processor.py        Cache-aware price loader + feature universe builder
  earnings_data.py         SEC EDGAR fundamentals (eps_yoy, days_since_earnings)
  sector_data.py           Sector/SPY/Gold relative features (with today override)
  social_data.py           Google Trends interest + price alignment features
  model_utils.py           Shared model helpers

data/
  downloader.py            Bootstrap script — fresh full data pull for all 506

data_cache/
  prices/<TICKER>.parquet         Daily OHLCV
  earnings/<TICKER>_earnings.parquet
  trends/<TICKER>_trends.parquet
  sectors/<ETF>_sector.parquet    SPY, GLD, XLB..XLY, SOXX, GDX

training/
  ml_optimizer.py          Walk-forward search over feature/hyperparam combos
  ensemble_selector.py     Picks top-K candidates per (ticker, horizon, fold)

evaluation/
  signal_analysis.py       OHLC walk-forward backtest with cooldown + right-censoring
  wf_validator.py          Walk-forward selection validator
  feature_importance.py    SHAP / gain analysis
  signal_overlap.py        Overlap analysis between horizons

live/
  app.py                   FastAPI dashboard + JSON APIs
  scheduler.py             Hourly signal refresh + daily data refresh jobs
  predictor.py             Per-ticker signal generation (today-bar injection,
                           ETF overrides, DELISTED_PAIRS filter, cooldown)
  database.py              SQLite signals + logs + OHLC exit logic
  templates/index.html     Dashboard (Live Signals, Backtest, History, Win Rates, Logs)
  static/style.css

simulation/
  ablation_k.py            Ensemble-K sweep
  threshold_ablation.py    Probability-threshold sweep
```

## How the System Works

### Training (offline)
`training/ml_optimizer.py` runs a walk-forward search per ticker, horizon (10/15/20 days), and TP (4/5/7%). For each fold year, it trains on 2012 → year-end and validates on the next year. The top-5 candidates per fold by expected value × recall^0.25 are saved to `saved_models/`. Selected ensemble members are recorded in `params_log/report_selected.csv`.

### Backtest
`evaluation/signal_analysis.py` replays signals from `TEST_START_DATE` to today using saved models. For each (ticker, horizon, signal_date) it walks forward day by day on OHLC:

- `High[t+k] >= target` **before** `Low[t+k] <= stop` → win
- `Low[t+k] <= stop` **before** target → loss
- Both same bar → contested (treated as loss, pessimist)
- No hit within full horizon → expired (loss)
- Not enough forward data → pending (excluded from win rate)

An `active_signal_end` guard prevents overlapping trades on the same (ticker, horizon). After a stop-loss or contested bar, a per-horizon cooldown (`STOPLOSS_COOLDOWN = {10: 1, 15: 2, 20: 3}` calendar days) blocks new entries. A random baseline (`RANDOM_SIGNAL_PROB = 5%/day` averaged over 10 seeds) is computed on the same data so the model's edge is visible.

### Live Signal Generation
`live/predictor.py` runs hourly during market hours via `live/scheduler.py`. For each ticker:

1. Fetch today's intraday 60-min bars (one batch per 50 tickers, 4-sec pacing).
2. Aggregate to a synthetic daily bar (today's open, max-high, min-low, latest-close, total-volume) and **inject** it into the cached daily DataFrame so RSI, ROC, OBV, ATR, Dist_SMA all recompute against today's state.
3. Override SPY, GLD, and sector ETF cached close series with their intraday values so `sector_rel_30d`, `spy_rel_30d`, and `gold_spy_rel_30d` reflect today's relative move.
4. For each horizon (10/15/20):
   - Skip if `(ticker, horizon) ∈ DELISTED_PAIRS` (per-pair filter, see below).
   - Skip if a cooldown is active for this pair.
   - Run the ensemble: every selected candidate must clear its own probability threshold (unanimous voting); the mean is reported as `prob`.
5. Save accepted signals to SQLite via `db.upsert_signal` (idempotent: blocks duplicates while an active signal exists on the same pair).

### Trade Exit
`live/database.py.close_expired_signals` runs every hour on active signals using the latest 60-min OHLC bar:

- **Opening bar (9:30–10:00 ET)**: gap up at open → exit at open price; gap down at open → stop at open price.
- **Intraday bars**: `High >= target` → exit exactly at the target price (no slippage), `Low <= stop` → exit exactly at the stop price.
- After `days` trading days without a hit: expired at the latest close.

A 60-minute "skip current bar" cutoff prevents the signal's own creation bar from triggering its own exit.

### Horizon-Level Filtering
Not every (ticker, horizon) pair works equally well. The backtest writes a per-pair win-rate report. Pairs whose **2025** backtest win rate was ≤40% (and had at least 3 signals) are written to `core/delisted_pairs.csv` and skipped at live signal generation. The filter is **not** applied to the backtest itself — backtest reports the raw model behaviour, the live system reports filtered behaviour. Walk-forward validation: pairs KEPT by the 2025 filter scored ~55.9% on 2026, pairs REMOVED scored ~45.2% — a +10.7pp edge confirms the filter generalizes.

## Live Dashboard

The dashboard at `/` has five tabs:

| Tab | Source |
|---|---|
| Live Signals | `/api/signals` — currently open positions with live P&L |
| Backtest | `/api/backtest` — trades from `signal_analysis_trades.csv` |
| History | `/api/history` — closed signals from SQLite |
| Win Rates | `/api/winrates` (live) and `/api/backtest/winrates` |
| Logs | `/api/logs` — scheduler / predictor events |

All endpoints return JSON and can be polled directly (e.g., for mobile integrations). 

## Scheduled Jobs

Times are UTC. NYSE market hours = 14:30 → 21:00 UTC.

| Time | Job | What it does |
|---|---|---|
| Every hour during market hours | `refresh_signals` | Run predictor on all tickers, save accepted signals, close expired |
| 21:00 | `refresh_price_cache` | Incrementally update each ticker's daily parquet |
| 21:30 | `refresh_reference_etfs_daily` | Incrementally update SPY, GLD, and 14 sector ETFs |
| 22:00 | `refresh_earnings_daily` | Fetch latest EDGAR filings per ticker (force refresh) |
| 23:00 | `refresh_sector_daily` | Recompute sector_rel / spy_rel / gold_spy features locally |
| 00:00 | `refresh_trends_daily` | Google Trends with rate-limit-aware retry (max 6 passes, exponential backoff) |

## Requirements

Python 3.12 recommended (3.14 has known incompatibilities with jinja2 and other libs). Install:

```bash
pip install -r requirements.txt
```

Key dependencies: pandas, numpy, xgboost, scikit-learn, ta, yfinance, pytrends, fastapi, uvicorn, apscheduler, pandas_market_calendars, jinja2.

## Notes

- `saved_models/` and `params_log/candidates_*.json` are gitignored — generate them locally via `training/ml_optimizer.py` or `quick_model_generator.py`.
- The live system stores signals in `live/data/metrade.db` (gitignored). Wipe via `rm live/data/metrade.db && restart server`.
- All feature computation runs at signal time on the most recent data; there is no overnight pre-computation. The hourly refresh is the canonical update.
