import os
os.environ["OMP_NUM_THREADS"] = "1"  # prevent LightGBM thread-init hang on Apple Silicon

import sys as _sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..')
_sys.path.insert(0, os.path.join(_ROOT, 'core'))
os.chdir(_ROOT)

import pandas as pd
import numpy as np
import os
import glob
import joblib
import warnings
import matplotlib.pyplot as plt

import data_processor as dp
import model_utils  # noqa: F401 — required for EnsembleModel deserialization

warnings.filterwarnings("ignore")

# ============================================================================
# PARAMETERS
# ============================================================================

INITIAL_CAPITAL    = 10000.0
TRADE_SIZE_PCT     = 0.2
MIN_PRECISION      = 0.0
TEST_START_DATE    = "2025-01-01"
TEST_END_DATE      = None

ENSEMBLE_MODE      = True   # False = use rank1 only; True = average top-K probs
ENSEMBLE_K         = 3      # unanimous voting: top-K models must all agree

FIXED_STOP_PCT     = 0.02
ATR_MULTIPLIER     = 1.5
BREAKEVEN_TRIGGER  = 0.50   # move stop to breakeven when price reaches 50% of target

COMMISSION_PCT     = 0.0005  # 0.05% per trade side (IBKR-style)
SLIPPAGE_PCT       = 0.0005  # 0.05% per trade side (large-cap bid-ask spread)

# 2025-01-01 → bugün en çok düşen 20 hisse (stres testi)
_LOSERS_2025 = [
    'TTD', 'FISV', 'IT', 'GDDY', 'LULU',
    'NOW', 'FDS', 'WDAY', 'ARE', 'FICO',
    'CPB', 'DECK', 'CSGP', 'ACN', 'CRM',
    'CAG', 'EPAM', 'ADBE', 'CMG', 'PYPL',
]

from tickers import TICKERS as _ALL_TICKERS
import random as _random

if os.environ.get("ABLATION_TICKERS"):
    TICKERS = os.environ["ABLATION_TICKERS"].split(",")
elif os.environ.get("RANDOM_N"):
    TICKERS = _random.sample(_ALL_TICKERS, int(os.environ["RANDOM_N"]))
else:
    TICKERS = _LOSERS_2025

TARGET_DAYS = [10, 15, 20]

# ── Load ensemble selection index (from ensemble_selector.py output) ──────────
def _load_selected_index() -> dict:
    """{(ticker, days, fold_year): [(ensemble_rank, pool_rank), ...]} sorted by rank."""
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

SELECTED_INDEX = _load_selected_index()
# ─────────────────────────────────────────────────────────────────────────────

BASE_CONFIG     = {'rsi': 14, 'sma_short': 20, 'sma_long': 50}
RSI_LIST        = [14]
RSI_CONFIG      = {'rsi': 14, 'sma_short': 10, 'sma_long': 50}
SMA_LIST        = [14, 30]
SMA_CONFIG_BASE = {'rsi': 14, 'sma_long': 200}

# ============================================================================


# Percentile of the forward-20d drawdown distribution to use as stop distance.
# 25th percentile = conservative (wider stop, protects against typical dips).
DRAWDOWN_PERCENTILE = 25


class MultiStockSimulation:
    STOPLOSS_COOLDOWN = {10: 1, 15: 2, 20: 3}
    TP_PCT            = {10: 0.04, 15: 0.05, 20: 0.07, 30: 0.10, 45: 0.13, 60: 0.15}

    def __init__(self, strategy: str = "fixed", threshold_override: "float | None" = None):
        assert strategy in ("fixed", "atr", "atr_breakeven", "ticker_avg"), \
            f"Unknown strategy: {strategy}"
        self.strategy            = strategy
        self.threshold_override  = threshold_override  # overrides pkg['threshold'] if set

        # models[ticker][days] = {fold_year: package}
        self.models          = {}
        # ticker_stops[ticker] = stop_pct (float, pre-computed historical median drawdown)
        self.ticker_stops    = {}
        self.data_store      = {}
        self.ledger          = []
        self.completed_trades = []
        self.active_positions = []
        self.cash         = INITIAL_CAPITAL
        self.equity_curve = []
        self.dates        = None
        self.idle_days    = 0
        self.cooldown     = {}
        self.max_k        = 5   # unanimous voting: use top max_k ensemble models

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def _load_ticker_models(self, ticker):
        """Load selected ensemble models for one ticker using SELECTED_INDEX.
        Each fold entry is a list of pkgs tagged with ensemble_rank, sorted 1..K.
        """
        ticker_models = {}
        count = 0
        for days in TARGET_DAYS:
            pct_tag  = int(self.TP_PCT[days] * 100)
            fold_map = {}

            if SELECTED_INDEX:
                fold_years = {fy for (t, d, fy) in SELECTED_INDEX if t == ticker and d == days}
                for fold_year in fold_years:
                    key      = (ticker, days, fold_year)
                    selected = SELECTED_INDEX.get(key, [])
                    pkgs     = []
                    for ensemble_rank, pool_rank in selected:
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
                        count += len(pkgs)

            # Fallback when no SELECTED_INDEX: load all cand files
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
                    count += len(pkgs)

            if fold_map:
                ticker_models[days] = fold_map
        return ticker, ticker_models, count

    def load_models(self):
        print(f"[{self.strategy.upper()}] Loading fold models...")
        loaded = 0
        for i, ticker in enumerate(TICKERS, 1):
            print(f"  [{i}/{len(TICKERS)}] {ticker}...", end="\r")
            _, ticker_models, count = self._load_ticker_models(ticker)
            self.models[ticker] = ticker_models
            loaded += count
        print(f"  {loaded} fold-model entries loaded.          ")

    def _get_fold_packages(self, ticker: str, days: int, date: pd.Timestamp) -> list:
        """Return list of pkgs for the appropriate fold, limited to top max_k by ensemble_rank."""
        fold_map = self.models.get(ticker, {}).get(days)
        if not fold_map:
            return []
        year = date.year
        pkgs = fold_map.get(year)
        if pkgs is None:
            earlier = [y for y in fold_map if y <= year]
            pkgs = fold_map[max(earlier)] if earlier else []
        return [p for p in pkgs if p.get('ensemble_rank', 1) <= self.max_k]

    def _predict_prob(self, pkgs: list, row: pd.DataFrame) -> float:
        """Unanimous voting: return avg prob only if ALL pkgs individually pass their own threshold."""
        if not pkgs:
            return 0.0
        probs = []
        for pkg in pkgs:
            try:
                X   = row[pkg['features']].values
                X_s = pkg['scaler'].transform(X)
                prob = pkg['model'].predict_proba(X_s)[0][1]
                if prob < self._effective_threshold(pkg):
                    return 0.0
                probs.append(prob)
            except KeyError:
                continue
        if not probs:
            return 0.0
        return float(np.mean(probs))

    # ------------------------------------------------------------------
    # Data Preparation
    # ------------------------------------------------------------------

    def _prepare_ticker(self, ticker):
        """Build features for one ticker. Returns (ticker, df_features) or (ticker, None)."""
        try:
            raw_df      = dp.get_data(ticker)
            df_features = self._build_features(raw_df, ticker)
            return ticker, df_features
        except Exception as e:
            print(f"  [WARN] {ticker} feature build failed: {e}")
            return ticker, None

    def prepare_data(self):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print("Preparing simulation data...")
        active_tickers = [t for t in TICKERS if self.models.get(t)]

        combined_dates = set()
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(self._prepare_ticker, t): t for t in active_tickers}
            for future in as_completed(futures):
                ticker, df_features = future.result()
                if df_features is None:
                    continue
                if self.strategy == "ticker_avg":
                    self.ticker_stops[ticker] = self._compute_ticker_stop(
                        df_features, days=20, cutoff=pd.Timestamp(TEST_START_DATE)
                    )
                mask = df_features.index >= pd.Timestamp(TEST_START_DATE)
                if TEST_END_DATE is not None:
                    mask &= df_features.index <= pd.Timestamp(TEST_END_DATE)
                df_features = df_features[mask]
                if not df_features.empty:
                    self.data_store[ticker] = df_features
                    combined_dates.update(df_features.index)
        self.dates = sorted(list(combined_dates))

        if self.strategy == "ticker_avg":
            print("  Ticker-specific stop percentages:")
            for t, pct in sorted(self.ticker_stops.items(), key=lambda x: x[1], reverse=True):
                print(f"    {t:<6} : {pct*100:.2f}%")

        print(f"  Simulation spans {len(self.dates)} trading days.")

    @staticmethod
    def _compute_ticker_stop(df: pd.DataFrame, days: int, cutoff: pd.Timestamp) -> float:
        """
        Compute the Nth-percentile forward max-drawdown over the historical period
        (up to cutoff) as the stop-loss distance for this ticker.
        Uses all dates, not just signal dates — simple and unbiased.
        """
        hist = df[df.index < cutoff].copy()
        if len(hist) < days + 10:
            return FIXED_STOP_PCT  # fallback

        next_lows  = hist['Low'].shift(-1)
        future_min = next_lows[::-1].rolling(window=days, min_periods=1).min()[::-1]
        drawdowns  = (future_min / hist['Close'] - 1).dropna()
        drawdowns  = drawdowns[drawdowns < 0]

        if len(drawdowns) == 0:
            return FIXED_STOP_PCT

        # Use the Nth percentile (e.g. 25th = covers 75% of historical dips)
        stop_pct = abs(float(np.percentile(drawdowns, DRAWDOWN_PERCENTILE)))
        stop_pct = max(stop_pct, 0.01)  # floor 1%
        stop_pct = min(stop_pct, 0.10)  # ceiling 10%
        return stop_pct

    def _build_features(self, raw_df, ticker=None):
        return dp.create_full_feature_universe(raw_df, ticker=ticker)

    # ------------------------------------------------------------------
    # Simulation Loop
    # ------------------------------------------------------------------

    def run(self):
        import time
        if not self.dates:
            print("No data.")
            return
        total = len(self.dates)
        print(f"\n--- [{self.strategy.upper()}] SIMULATION STARTING ({total} days) ---")
        t0 = time.time()
        for i, current_date in enumerate(self.dates, 1):
            self._check_exits(current_date)
            self._check_entries(current_date, self._calculate_equity(current_date))
            final_equity = self._calculate_equity(current_date)
            self.equity_curve.append({'Date': current_date, 'Equity': final_equity})
            if self.cash > 100:
                self.idle_days += 1
            if i % 50 == 0 or i == total:
                elapsed  = time.time() - t0
                eta      = (elapsed / i) * (total - i)
                print(f"  {i}/{total} days  |  {current_date.date()}  |  "
                      f"Equity: ${final_equity:,.0f}  |  ETA: {eta:.0f}s", end="\r")
        print()
        if self.dates:
            self._close_all_positions(self.dates[-1])
        self._save_ledger_csv()
        self._generate_plots()

    # ------------------------------------------------------------------
    # Stop-Loss Calculation
    # ------------------------------------------------------------------

    def _calc_stop(self, entry_price: float, atr: float, ticker: str = "") -> float:
        if self.strategy == "fixed":
            return entry_price * (1 - FIXED_STOP_PCT)
        if self.strategy == "ticker_avg":
            stop_pct = self.ticker_stops.get(ticker, FIXED_STOP_PCT)
            return entry_price * (1 - stop_pct)
        # atr / atr_breakeven
        atr_stop = entry_price - ATR_MULTIPLIER * atr
        return min(atr_stop, entry_price * (1 - 0.005))

    # ------------------------------------------------------------------
    # Entry / Exit Logic
    # ------------------------------------------------------------------

    def _effective_threshold(self, pkg) -> float:
        return self.threshold_override if self.threshold_override is not None else pkg['threshold']

    def _get_signal(self, ticker, date):
        """Return (best_days, best_prob, best_pkg) for ticker on date, or None."""
        if ticker not in self.data_store: return None
        if date not in self.data_store[ticker].index: return None
        row = self.data_store[ticker].loc[[date]]
        best_score, best_prob, best_pkgs, best_days = -1, -1, None, None
        for days in TARGET_DAYS:
            pkgs = self._get_fold_packages(ticker, days, date)
            if not pkgs: continue
            prob = self._predict_prob(pkgs, row)
            if prob < self._effective_threshold(pkgs[0]):
                continue
            score = (self.TP_PCT[days] * prob) / days
            if score > best_score:
                best_score = score
                best_prob  = prob
                best_pkgs  = pkgs
                best_days  = days
        if best_pkgs is None:
            return None
        return best_days, best_prob, best_pkgs[0]

    def _check_entries(self, date, current_equity):
        for ticker in TICKERS:
            if ticker not in self.data_store: continue
            if date not in self.data_store[ticker].index: continue

            # Skip if already in a position for this ticker
            if any(p['ticker'] == ticker for p in self.active_positions):
                continue

            row = self.data_store[ticker].loc[[date]]

            # Find the best signal across all horizons by expected return/day
            best_score = -1
            best_prob  = -1
            best_pkg   = None
            best_days  = None

            for days in TARGET_DAYS:
                if date < self.cooldown.get((ticker, days), pd.Timestamp.min):
                    continue
                pkgs = self._get_fold_packages(ticker, days, date)
                if not pkgs: continue
                prob = self._predict_prob(pkgs, row)
                if prob < self._effective_threshold(pkgs[0]):
                    continue
                score = (self.TP_PCT[days] * prob) / days
                if score > best_score:
                    best_score = score
                    best_prob  = prob
                    best_pkg   = pkgs[0]
                    best_days  = days

            if best_pkg is None:
                continue

            price = float(row['Close'].values[0])
            atr   = float(row['ATR'].values[0]) if 'ATR' in row.columns else price * FIXED_STOP_PCT

            target_size = current_equity * TRADE_SIZE_PCT
            if self.cash < target_size:
                if self.cash > 100:
                    target_size = self.cash
                else:
                    continue

            fill_price       = price * (1 + SLIPPAGE_PCT)
            commission_entry = target_size * COMMISSION_PCT
            tp_price         = fill_price * (1 + self.TP_PCT[best_days or 20])
            stop_price       = self._calc_stop(fill_price, atr, ticker)

            self.active_positions.append({
                'ticker':              ticker,
                'entry_date':          date,
                'entry_price':         fill_price,
                'amount':              target_size / fill_price,
                'stop_loss':           stop_price,
                'target_days':         best_days,
                'tp_price':            tp_price,
                'atr_at_entry':        atr,
                'breakeven_triggered': False,
                'fold_year':           best_pkg['fold_year'],
                'confidence':          round(best_prob * 100, 1),
            })
            self.cash -= (target_size + commission_entry)

            self.ledger.append({
                'Date': date, 'Ticker': ticker, 'Type': 'BUY',
                'Reason': f"Signal {best_days}d ({best_prob*100:.0f}%)",
                'Price': price, 'Profit': 0, 'Return_Pct': 0,
                'Target_Days': best_days, 'Fold_Year': best_pkg['fold_year'],
            })

    def _check_exits(self, date):
        for i in range(len(self.active_positions) - 1, -1, -1):
            pos    = self.active_positions[i]
            ticker = pos['ticker']
            if ticker not in self.data_store: continue
            if date not in self.data_store[ticker].index: continue

            if date <= pos['entry_date']:
                continue

            bar   = self.data_store[ticker].loc[date]
            open_ = float(bar['Open'])
            high  = float(bar['High'])
            low   = float(bar['Low'])
            close = float(bar['Close'])
            tp    = pos['tp_price']
            sl    = pos['stop_loss']
            days  = pos['target_days']
            is_expiry = (date - pos['entry_date']).days >= days

            # 1. Gap open: open already past target or stop
            if open_ >= tp:
                self._close_position(i, date, tp, "TAKE PROFIT")
            elif open_ <= sl:
                self._close_position(i, date, sl, "STOP LOSS")
            # 2. Intraday: stop checked before target (conservative)
            elif low <= sl:
                self._close_position(i, date, sl, "STOP LOSS")
            elif high >= tp:
                self._close_position(i, date, tp, "TAKE PROFIT")
            # 3. Time expired: sell at close
            elif is_expiry:
                self._close_position(i, date, close, "TIME EXPIRED")

    def _close_position(self, i, date, exit_price, reason):
        pos        = self.active_positions[i]
        ticker     = pos['ticker']
        days       = pos['target_days']
        # Slippage on exit: receive slightly below exit price
        fill_exit  = exit_price * (1 - SLIPPAGE_PCT)
        revenue    = fill_exit * pos['amount']
        commission = revenue * COMMISSION_PCT
        self.cash += revenue - commission
        cost       = pos['entry_price'] * pos['amount']
        profit     = (revenue - commission) - cost
        profit_pct = (profit / cost) * 100

        self.ledger.append({
            'Date': date, 'Ticker': ticker, 'Type': 'SELL',
            'Reason': reason, 'Price': exit_price,
            'Profit': profit, 'Return_Pct': profit_pct,
            'Target_Days': days, 'Fold_Year': pos.get('fold_year', ''),
        })
        self.completed_trades.append({
            'Ticker':      ticker,
            'Entry_Date':  pos['entry_date'],
            'Exit_Date':   date,
            'Entry_Price': pos['entry_price'],
            'Exit_Price':  exit_price,
            'Profit':      profit,
            'Return_Pct':  profit_pct,
            'Target_Days': days,
            'Exit_Reason': reason,
        })

        if reason == "STOP LOSS":
            cd = self.STOPLOSS_COOLDOWN.get(days, 0)
            if cd:
                self.cooldown[(ticker, days)] = date + pd.Timedelta(days=cd)

        self.active_positions.pop(i)

    # ------------------------------------------------------------------
    # Equity & Close-All
    # ------------------------------------------------------------------

    def _calculate_equity(self, date):
        equity = self.cash
        for pos in self.active_positions:
            ticker = pos['ticker']
            if ticker in self.data_store and date in self.data_store[ticker].index:
                equity += self.data_store[ticker].loc[date]['Close'] * pos['amount']
            else:
                equity += pos['entry_price'] * pos['amount']
        return equity

    def _close_all_positions(self, date):
        for pos in self.active_positions:
            ticker = pos['ticker']
            price  = pos['entry_price']
            if ticker in self.data_store and date in self.data_store[ticker].index:
                price = self.data_store[ticker].loc[date]['Close']
            revenue    = price * pos['amount']
            self.cash += revenue
            cost       = pos['entry_price'] * pos['amount']
            profit     = revenue - cost
            profit_pct = (profit / cost) * 100
            self.ledger.append({
                'Date': date, 'Ticker': ticker, 'Type': 'SELL',
                'Reason': "END OF SIM", 'Price': price,
                'Profit': profit, 'Return_Pct': profit_pct,
                'Target_Days': pos['target_days'], 'Fold_Year': pos.get('fold_year', ''),
            })
            self.completed_trades.append({
                'Ticker':      ticker,
                'Entry_Date':  pos['entry_date'],
                'Exit_Date':   date,
                'Entry_Price': pos['entry_price'],
                'Exit_Price':  price,
                'Profit':      profit,
                'Return_Pct':  profit_pct,
                'Target_Days': pos['target_days'],
                'Exit_Reason': "END OF SIM",
            })
        self.active_positions = []

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _save_ledger_csv(self):
        if not self.ledger: return
        fname = f"simulation_trades_{self.strategy}.csv"
        pd.DataFrame(self.ledger).to_csv(fname, index=False)
        print(f"  Trades saved → {fname}")

    def _generate_plots(self):
        if not self.completed_trades: return
        os.makedirs("simulation_plots", exist_ok=True)

        trades_by_ticker = {}
        for t in self.completed_trades:
            if t['Exit_Reason'] != 'END OF SIM':
                trades_by_ticker.setdefault(t['Ticker'], []).append(t)

        for ticker, trades in trades_by_ticker.items():
            if ticker not in self.data_store: continue
            df  = self.data_store[ticker]
            wins  = [t for t in trades if t['Profit'] > 0]
            loses = [t for t in trades if t['Profit'] <= 0]

            fig, ax = plt.subplots(figsize=(16, 7))
            ax.plot(df.index, df['Close'], color='#555555', linewidth=1, alpha=0.8, label='Close', zorder=1)

            # Alış-satış çizgileri
            for trade in trades:
                profit = trade['Profit']
                line_color = '#22c55e' if profit > 0 else '#ef4444'
                ax.plot(
                    [trade['Entry_Date'], trade['Exit_Date']],
                    [trade['Entry_Price'], trade['Exit_Price']],
                    color=line_color, linewidth=0.8, alpha=0.4, zorder=2
                )

            # Buy points — pink
            if trades:
                ax.scatter(
                    [t['Entry_Date']  for t in trades],
                    [t['Entry_Price'] for t in trades],
                    color='#f472b6', s=80, zorder=5, edgecolors='#9d174d',
                    linewidths=0.8, label='Buy', marker='^'
                )

            # Sell points — green (profit) / red (loss)
            if wins:
                ax.scatter(
                    [t['Exit_Date']  for t in wins],
                    [t['Exit_Price'] for t in wins],
                    color='#22c55e', s=80, zorder=5, edgecolors='#15803d',
                    linewidths=0.8, label='Sell (win)', marker='v'
                )
            if loses:
                ax.scatter(
                    [t['Exit_Date']  for t in loses],
                    [t['Exit_Price'] for t in loses],
                    color='#ef4444', s=80, zorder=5, edgecolors='#991b1b',
                    linewidths=0.8, label='Sell (loss)', marker='v'
                )

            n      = len(trades)
            n_win  = len(wins)
            wr     = n_win / n * 100 if n else 0
            total_profit = sum(t['Profit'] for t in trades)

            ax.set_title(
                f"{ticker}  —  {n} trades  |  Win Rate: {wr:.0f}%  |  "
                f"Total P&L: ${total_profit:+,.0f}",
                fontsize=13, fontweight='bold'
            )
            ax.set_xlabel("Date")
            ax.set_ylabel("Price ($)")
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            fig.savefig(f"simulation_plots/{ticker}_{self.strategy}_trades.png", dpi=150)
            plt.close(fig)

        print(f"  Plots saved → simulation_plots/  ({len(trades_by_ticker)} files)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @staticmethod
    def _trade_category(t):
        if t['Reason'] == 'TAKE PROFIT':
            return 'target_hit'
        elif t['Return_Pct'] > 0:
            return 'low_win'
        elif t['Reason'] == 'STOP LOSS':
            return 'lose'
        else:
            return 'low_loss'

    def summary(self) -> dict:
        if not self.equity_curve:
            return {}
        final_eq = self.equity_curve[-1]['Equity']
        roi      = (final_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        sells    = [t for t in self.ledger if t['Type'] == 'SELL' and t['Profit'] != 0 and t['Reason'] != 'END OF SIM']
        n        = len(sells)
        target_hits = sum(1 for t in sells if self._trade_category(t) == 'target_hit')
        low_wins    = sum(1 for t in sells if self._trade_category(t) == 'low_win')
        low_losses  = sum(1 for t in sells if self._trade_category(t) == 'low_loss')
        loses       = sum(1 for t in sells if self._trade_category(t) == 'lose')
        avg_ret  = np.mean([t['Return_Pct'] for t in sells]) if n else 0

        df_eq = pd.DataFrame(self.equity_curve).set_index('Date')
        df_eq.index = pd.to_datetime(df_eq.index)
        yearly = {}
        for yr, grp in df_eq.groupby(df_eq.index.year):
            yearly[yr] = {
                'start':      grp['Equity'].iloc[0],
                'end':        grp['Equity'].iloc[-1],
                'return_pct': (grp['Equity'].iloc[-1] / grp['Equity'].iloc[0] - 1) * 100,
            }
        return {
            'strategy':    self.strategy,
            'final_eq':    final_eq,
            'roi':         roi,
            'trades':      n,
            'win_rate':    (target_hits + low_wins) / n * 100 if n else 0,
            'avg_ret':     avg_ret,
            'target_hits': target_hits,
            'low_wins':    low_wins,
            'low_losses':  low_losses,
            'loses':       loses,
            'idle_pct':    self.idle_days / len(self.dates) * 100 if self.dates else 0,
            'yearly':      yearly,
        }

    def print_summary(self):
        s = self.summary()
        if not s:
            print("No trades.")
            return
        print(f"\n{'='*60}")
        print(f"STRATEGY: {s['strategy'].upper()}")
        print(f"{'='*60}")
        n = s['trades']
        th, lw, ll, lo = s['target_hits'], s['low_wins'], s['low_losses'], s['loses']
        th_pct = th / n * 100 if n else 0
        lw_pct = lw / n * 100 if n else 0
        ll_pct = ll / n * 100 if n else 0
        lo_pct = lo / n * 100 if n else 0
        print(f"  Final Equity : ${s['final_eq']:>10,.2f}  (ROI: {s['roi']:+.1f}%)")
        print(f"  Trades       : {n}  |  Win Rate: {s['win_rate']:.1f}%  |  Avg Return: {s['avg_ret']:+.2f}%")
        print(f"  Target Hit   : {th} ({th_pct:.1f}%)  |  Low Win: {lw} ({lw_pct:.1f}%)  |  "
              f"Low Loss: {ll} ({ll_pct:.1f}%)  |  Lose: {lo} ({lo_pct:.1f}%)")
        print(f"  Idle Days    : {s['idle_pct']:.1f}%")
        print(f"\n  Year-by-Year:")
        for yr, d in s['yearly'].items():
            bar  = '█' * int(abs(d['return_pct']) / 2)
            sign = '+' if d['return_pct'] >= 0 else ''
            print(f"    {yr}: ${d['start']:>8,.0f} → ${d['end']:>8,.0f}  ({sign}{d['return_pct']:.1f}%)  {bar}")

        sells = [t for t in self.ledger if t['Type'] == 'SELL' and t['Profit'] != 0 and t['Reason'] != 'END OF SIM']
        if not sells:
            return

        from collections import defaultdict

        def empty_counts():
            return [0, 0, 0, 0]  # [target_hit, low_win, low_loss, lose]

        def add_trade(counts, t):
            cat = self._trade_category(t)
            if cat == 'target_hit': counts[0] += 1
            elif cat == 'low_win':  counts[1] += 1
            elif cat == 'low_loss': counts[2] += 1
            else:                   counts[3] += 1

        # ── Per-ticker × per-horizon breakdown ─────────────────────────
        grid        = defaultdict(empty_counts)
        for t in sells:
            add_trade(grid[(t['Ticker'], t['Target_Days'])], t)

        all_tickers = sorted({k[0] for k in grid})
        all_days    = sorted({k[1] for k in grid})

        ticker_totals = defaultdict(empty_counts)
        day_totals    = defaultdict(empty_counts)

        print(f"\n  {'─'*80}")
        print(f"  SIGNALS BY TICKER × HORIZON  (TH=target hit  LW=low win  LL=low loss  L=lose)")
        print(f"  {'─'*80}")
        header = f"  {'Ticker':<8}" + "".join(f"  {d:>4}d" for d in all_days) + "   TOTAL"
        print(header)
        print(f"  {'─'*80}")

        for ticker in all_tickers:
            row = f"  {ticker:<8}"
            for d in all_days:
                th, lw, ll, lo = grid[(ticker, d)]
                total = th + lw + ll + lo
                if total:
                    row += f"  {th}TH/{lw}LW/{ll}LL/{lo}L"
                    for j in range(4): ticker_totals[ticker][j] += grid[(ticker, d)][j]
                    for j in range(4): day_totals[d][j]         += grid[(ticker, d)][j]
                else:
                    row += f"  {'—':>16}"
            th, lw, ll, lo = ticker_totals[ticker]
            row += f"  {th+lw+ll+lo:>3} ({th}TH/{lw}LW/{ll}LL/{lo}L)"
            print(row)

        print(f"  {'─'*80}")
        row = f"  {'TOTAL':<8}"
        g = [0, 0, 0, 0]
        for d in all_days:
            th, lw, ll, lo = day_totals[d]
            row += f"  {th}TH/{lw}LW/{ll}LL/{lo}L"
            for j in range(4): g[j] += day_totals[d][j]
        row += f"  {sum(g):>3} ({g[0]}TH/{g[1]}LW/{g[2]}LL/{g[3]}L)"
        print(row)

        # ── Per-ticker cumulative ───────────────────────────────────────
        print(f"\n  {'─'*80}")
        print(f"  CUMULATIVE BY TICKER")
        print(f"  {'─'*80}")
        print(f"  {'Ticker':<8}  {'Total':>6}  {'TH':>5}  {'TH%':>6}  {'LW':>5}  {'LW%':>6}  {'LL':>5}  {'LL%':>6}  {'L':>5}  {'L%':>6}")
        for ticker in sorted(ticker_totals, key=lambda x: sum(ticker_totals[x]), reverse=True):
            th, lw, ll, lo = ticker_totals[ticker]
            n = th + lw + ll + lo
            print(f"  {ticker:<8}  {n:>6}  {th:>5}  {th/n*100:>5.1f}%  {lw:>5}  {lw/n*100:>5.1f}%"
                  f"  {ll:>5}  {ll/n*100:>5.1f}%  {lo:>5}  {lo/n*100:>5.1f}%")

        # ── Per-horizon cumulative ──────────────────────────────────────
        print(f"\n  {'─'*80}")
        print(f"  CUMULATIVE BY HORIZON")
        print(f"  {'─'*80}")
        print(f"  {'Days':<8}  {'Total':>6}  {'TH':>5}  {'TH%':>6}  {'LW':>5}  {'LW%':>6}  {'LL':>5}  {'LL%':>6}  {'L':>5}  {'L%':>6}")
        for d in all_days:
            th, lw, ll, lo = day_totals[d]
            n = th + lw + ll + lo
            print(f"  {d}d{'':<6}  {n:>6}  {th:>5}  {th/n*100:>5.1f}%  {lw:>5}  {lw/n*100:>5.1f}%"
                  f"  {ll:>5}  {ll/n*100:>5.1f}%  {lo:>5}  {lo/n*100:>5.1f}%")

    def run_baseline_individual(self):
        results = []
        for ticker in TICKERS:
            df = self.data_store.get(ticker)
            if df is None or df.empty:
                try:
                    raw_df = dp.get_data(ticker)
                    df_f   = self._build_features(raw_df, ticker)
                    df     = df_f[df_f.index >= pd.Timestamp(TEST_START_DATE)]
                except Exception:
                    df = None
            if df is None or df.empty:
                results.append((ticker, 0.0, 0.0))
                continue
            start  = df.iloc[0]['Close']
            end    = df.iloc[-1]['Close']
            shares = 10000.0 / start
            final  = shares * end
            roi    = (final - 10000.0) / 10000.0 * 100
            results.append((ticker, final, roi))
        return results


# ============================================================================
# Comparison Runner
# ============================================================================

def run_comparison():
    print("=" * 60)
    print(f"SIMULATION  —  K={ENSEMBLE_K} Unanimous Ensemble  |  Fixed Stop-Loss")
    print(f"Period  : {TEST_START_DATE} → {TEST_END_DATE or 'today'}")
    print(f"Capital : ${INITIAL_CAPITAL:,.0f}  |  Trade size: {TRADE_SIZE_PCT*100:.0f}%")
    print(f"Threshold: 0.70  |  Voting: unanimous (all {ENSEMBLE_K} models must agree)")
    print("=" * 60)

    sim = MultiStockSimulation(strategy="fixed")
    sim.load_models()
    sim.prepare_data()
    sim.max_k = ENSEMBLE_K
    sim.run()
    sim.print_summary()

    # ── Buy & Hold Benchmark ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BUY & HOLD BENCHMARKS ($10,000 each)")
    print("=" * 60)
    for t, val, roi in sorted(sim.run_baseline_individual(), key=lambda x: x[2], reverse=True):
        print(f"  {t:<6} : ${val:>10,.2f}  (ROI: {roi:+.1f}%)")


if __name__ == "__main__":
    run_comparison()
