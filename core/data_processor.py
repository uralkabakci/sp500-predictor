import pandas as pd
import numpy as np
import yfinance as yf
import os
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD, SMAIndicator, ADXIndicator, CCIIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

# Cache directory
CACHE_DIR = "data_cache/prices"

def get_data(ticker, use_cache=True):
    """
    Downloads historical data for a given ticker.
    First tries to read from cache, otherwise downloads from yfinance.
    
    Args:
        ticker: Stock symbol
        use_cache: If True, tries to read from cache (default: True)
    
    Returns:
        DataFrame: Stock data
    """
    # Try to read from cache and incrementally update if stale
    if use_cache:
        cache_file = os.path.join(CACHE_DIR, f"{ticker}.parquet")
        if os.path.exists(cache_file):
            try:
                df_cached = pd.read_parquet(cache_file)
                last_date = df_cached.index.max()
                today = pd.Timestamp.today().normalize()
                # If cache is up to date (within 1 trading day), return as-is
                if (today - last_date).days <= 1:
                    return df_cached
                # Otherwise fetch only missing days
                fetch_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                new_data = yf.download(ticker, start=fetch_start, interval="1d",
                                       progress=False, auto_adjust=False)
                if new_data.empty:
                    return df_cached
                if isinstance(new_data.columns, pd.MultiIndex):
                    new_data.columns = new_data.columns.get_level_values(0)
                if 'Adj Close' in new_data.columns:
                    adj_factor = new_data['Adj Close'] / new_data['Close']
                    new_data['Open']  = new_data['Open']  * adj_factor
                    new_data['High']  = new_data['High']  * adj_factor
                    new_data['Low']   = new_data['Low']   * adj_factor
                    new_data['Close'] = new_data['Adj Close']
                    new_data.drop(columns=['Adj Close'], inplace=True, errors='ignore')
                df = pd.concat([df_cached, new_data[~new_data.index.isin(df_cached.index)]])
                df.sort_index(inplace=True)
                return df
            except Exception:
                pass

    # Full download (no cache or cache read failed)
    print(f"Downloading {ticker} data...")
    df = yf.download(ticker, period="max", interval="1d", progress=False, auto_adjust=False)
    
    if df.empty:
        return df
    
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)
        
    # Adjust High/Low/Open values by the same ratio, not just Close.
    # Otherwise, calculating (Raw High / Adj Close) would result in incorrect profit calculations.
    if 'Adj Close' in df.columns: 
        # Calculate adjustment factor (e.g., price drop due to dividends results in factor < 1)
        adj_factor = df['Adj Close'] / df['Close']
        
        # Apply the adjustment factor to all price columns
        df['Open'] = df['Open'] * adj_factor
        df['High'] = df['High'] * adj_factor
        df['Low'] = df['Low'] * adj_factor
        
        # The Close column is now the Adjusted Close
        df['Close'] = df['Adj Close']
        
        # Drop the original Adj Close column to avoid confusion
        df.drop(columns=['Adj Close'], inplace=True, errors='ignore')
    
    # Save downloaded data to cache (if cache directory exists)
    if use_cache and os.path.exists(CACHE_DIR):
        try:
            cache_file = os.path.join(CACHE_DIR, f"{ticker}.parquet")
            df.to_parquet(cache_file)
        except Exception:
            # Cache write error is not critical, continue
            pass
        
    return df

def calculate_indicators(df, config):
    """
    Calculates technical indicators based on the provided configuration dictionary.
    """
    df = df.copy()
    
    # Momentum
    df['RSI'] = RSIIndicator(df['Close'], window=config['rsi']).rsi()
    df['ROC'] = ROCIndicator(df['Close'], window=7).roc()

    # Trend
    df['MACD'] = MACD(df['Close']).macd()
    df['ADX'] = ADXIndicator(df['High'], df['Low'], df['Close'], window=7).adx()
    df['CCI'] = CCIIndicator(df['High'], df['Low'], df['Close'], window=7).cci()

    # Volatility
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=7).average_true_range()
    bb = BollingerBands(df['Close'], window=7)
    df['BB_Width'] = bb.bollinger_wband()

    # Volume
    df['OBV'] = OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()

    # Distances
    sma_short = SMAIndicator(df['Close'], window=config['sma_short']).sma_indicator()
    sma_long = SMAIndicator(df['Close'], window=config['sma_long']).sma_indicator()

    df['Dist_SMA_Short'] = (df['Close'] - sma_short) / sma_short
    df['Dist_SMA_Long'] = (df['Close'] - sma_long) / sma_long

    return df.dropna()

def create_super_features(raw_df, base_config=None, rsi_list=None, rsi_config=None, sma_list=None, sma_config_base=None):
    """
    Combines all indicators into a single giant dataframe.
    This is the standard feature engineering used across all files.
    
    Args:
        raw_df: Raw stock data DataFrame
        base_config: Base configuration dict (default: {'rsi': 14, 'sma_short': 20, 'sma_long': 50})
        rsi_list: List of RSI periods (default: [14])
        rsi_config: RSI configuration dict (default: {'rsi': 14, 'sma_short': 10, 'sma_long': 50})
        sma_list: List of SMA short periods (default: [14, 30])
        sma_config_base: SMA base configuration dict (default: {'rsi': 14, 'sma_long': 200})
    
    Returns:
        DataFrame: DataFrame with all super features
    """
    # Default parameters (can be overridden)
    if base_config is None:
        base_config = {'rsi': 14, 'sma_short': 20, 'sma_long': 50}
    if rsi_list is None:
        rsi_list = [14]
    if rsi_config is None:
        rsi_config = {'rsi': 14, 'sma_short': 10, 'sma_long': 50}
    if sma_list is None:
        sma_list = [14, 30]
    if sma_config_base is None:
        sma_config_base = {'rsi': 14, 'sma_long': 200}
    
    df_master = calculate_indicators(raw_df, base_config)
    
    # Drop unnecessary copies
    drop_cols = ['RSI', 'Dist_SMA_Short', 'Dist_SMA_Long']
    df_master = df_master.drop(columns=[c for c in drop_cols if c in df_master.columns], errors='ignore')
    
    # RSI Variations
    for r in rsi_list:
        cfg = rsi_config.copy()
        cfg['rsi'] = r
        temp = calculate_indicators(raw_df, cfg)
        df_master[f'RSI_{r}'] = temp['RSI']
        
    # SMA Variations
    for s in sma_list:
        cfg = sma_config_base.copy()
        cfg['sma_short'] = s
        temp = calculate_indicators(raw_df, cfg)
        df_master[f'Dist_SMA_{s}'] = temp['Dist_SMA_Short']
        
    return df_master.dropna()

def create_enhanced_features(raw_df, ticker, base_config=None, rsi_list=None, rsi_config=None, sma_list=None, sma_config_base=None):
    df = create_super_features(raw_df,
                               base_config=base_config,
                               rsi_list=rsi_list,
                               rsi_config=rsi_config,
                               sma_list=sma_list,
                               sma_config_base=sma_config_base)
    if df.empty:
        return df
    try:
        import social_data as sd
        df = sd.merge_trends(df, ticker)
    except Exception as e:
        print(f"[ENHANCED] Trends merge skipped for {ticker}: {e}")
    try:
        import earnings_data as ed
        df = ed.merge_earnings(df, ticker)
    except Exception as e:
        print(f"[ENHANCED] Earnings merge skipped for {ticker}: {e}")
    return df.dropna()

def create_full_feature_universe(raw_df, ticker=None):
    """
    Computes ALL indicators with ALL period variants.
    Used by ml_optimizer for joint feature+model random search.
    Simulation also uses this so pkg['features'] columns are always present.
    """
    df = raw_df.copy()

    # RSI variants
    for p in [7, 10, 14, 21]:
        df[f'RSI_{p}'] = RSIIndicator(df['Close'], window=p).rsi()

    # ROC variants
    for p in [7, 14, 30]:
        df[f'ROC_{p}'] = ROCIndicator(df['Close'], window=p).roc()

    # MACD variants
    for fast, slow, sig in [(12, 26, 9), (8, 21, 7)]:
        m = MACD(df['Close'], window_fast=fast, window_slow=slow, window_sign=sig)
        df[f'MACD_{fast}_{slow}_{sig}']        = m.macd()
        df[f'MACD_signal_{fast}_{slow}_{sig}'] = m.macd_signal()
        df[f'MACD_diff_{fast}_{slow}_{sig}']   = m.macd_diff()

    # ADX variants
    for p in [7, 14, 21]:
        df[f'ADX_{p}'] = ADXIndicator(df['High'], df['Low'], df['Close'], window=p).adx()

    # CCI variants
    for p in [7, 14, 20]:
        df[f'CCI_{p}'] = CCIIndicator(df['High'], df['Low'], df['Close'], window=p).cci()

    # ATR variants + backward-compat alias
    for p in [7, 14]:
        df[f'ATR_{p}'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=p).average_true_range()
    df['ATR'] = df['ATR_14']

    # Bollinger Bands variants
    for p in [10, 20, 30]:
        bb = BollingerBands(df['Close'], window=p)
        df[f'BB_Width_{p}'] = bb.bollinger_wband()

    # OBV
    df['OBV'] = OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()

    # Volume ratio: current volume / 20-day average volume
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20, min_periods=1).mean()

    # SMA distances
    for p in [10, 20, 30, 50, 100, 200]:
        sma = SMAIndicator(df['Close'], window=p).sma_indicator()
        df[f'Dist_SMA_{p}'] = (df['Close'] - sma) / sma

    df = df.dropna()

    if ticker:
        try:
            import social_data as sd
            df = sd.merge_trends(df, ticker)
            if 'search_interest_chg_14d' in df.columns:
                s14 = df['search_interest_chg_14d']
                if 'ROC_14' in df.columns:
                    df['search_price_alignment_14d'] = (s14 * df['ROC_14']).clip(-4, 4)
                    df['search_accel_alignment_14d'] = (s14.diff() * df['ROC_14']).clip(-4, 4)
            if 'search_interest_chg_30d' in df.columns:
                s30 = df['search_interest_chg_30d']
                if 'ROC_30' in df.columns:
                    df['search_price_alignment_30d'] = (s30 * df['ROC_30']).clip(-4, 4)
                    df['search_accel_alignment_30d'] = (s30.diff() * df['ROC_30']).clip(-4, 4)
            df = df.drop(columns=['search_interest_chg_14d', 'search_interest_chg_30d'],
                         errors='ignore')
        except Exception:
            pass

        # ADX × ROC: signed trend strength (ADX=strength, ROC=direction)
        for p in [14]:
            adx_col = f'ADX_{p}'
            roc_col = f'ROC_{p}'
            if adx_col in df.columns and roc_col in df.columns:
                df[f'adx_roc_{p}'] = (df[adx_col] * df[roc_col]).clip(-200, 200)

        try:
            import earnings_data as ed
            df = ed.merge_earnings(df, ticker)
        except Exception:
            pass
        try:
            import sector_data as sec
            df = sec.merge_sector_relative(df, ticker, windows=[14, 30])
            df = sec.merge_spy_relative(df, windows=[14, 30])
            df = sec.merge_gold_spy(df, windows=[14, 30])
        except Exception:
            pass

    return df.dropna()


def prepare_features_and_target(df, days, pct):
    """Creates target column and separates features."""
    df_temp = df.copy()
    df_temp['Target'] = ((df_temp['Close'].shift(-days) / df_temp['Close'] - 1) > pct).astype(int)
    df_temp.dropna(inplace=True)
    
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Target']
    feature_cols = [c for c in df_temp.columns if c not in exclude_cols]
    
    X = df_temp[feature_cols].values
    y = df_temp['Target'].values
    
    return X, y