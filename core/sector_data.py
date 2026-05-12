import os
import time
import warnings
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

CACHE_DIR = "data_cache/sectors"

TICKER_SECTOR = {
    # Technology
    "AAPL":  "XLK", "MSFT":  "XLK", "GOOGL": "XLK", "GOOG":  "XLK",
    "META":  "XLK", "NFLX":  "XLK", "ORCL":  "XLK", "CRM":   "XLK",
    "ADBE":  "XLK", "AKAM":  "XLK", "ACN":   "XLK", "ADP":   "XLK",
    "ADSK":  "XLK", "ANET":  "XLK", "APH":   "XLK", "APP":   "XLK",
    "AXON":  "XLK", "BR":    "XLK", "CDNS":  "XLK", "CDW":   "XLK",
    "CIEN":  "XLK", "CMCSA": "XLK", "CSCO":  "XLK", "CSGP":  "XLK",
    "CTSH":  "XLK", "CHTR":  "XLK", "T":     "XLK", "TMUS":  "XLK",
    "VZ":    "XLK", "WBD":   "XLK", "CPAY":  "XLK", "DDOG":  "XLK",
    "DELL":  "XLK", "EA":    "XLK", "EFX":   "XLK", "EPAM":  "XLK",
    "FFIV":  "XLK", "FICO":  "XLK", "FIS":   "XLK", "FISV":  "XLK",
    "FOX":   "XLK", "FOXA":  "XLK", "FSLR":  "XLK", "FTNT":  "XLK",
    "GDDY":  "XLK", "GEN":   "XLK", "GLW":   "XLK", "GPN":   "XLK",
    "GRMN":  "XLK", "HPE":   "XLK", "HPQ":   "XLK", "IBM":   "XLK",
    "INTU":  "XLK", "IT":    "XLK", "JBL":   "XLK", "JKHY":  "XLK",
    "KEYS":  "XLK", "LITE":  "XLK", "MSI":   "XLK", "NOW":   "XLK",
    "NTAP":  "XLK", "NWS":   "XLK", "NWSA":  "XLK", "OMC":   "XLK",
    "PANW":  "XLK", "PAYX":  "XLK", "PLTR":  "XLK", "PTC":   "XLK",
    "PYPL":  "XLK", "Q":     "XLK", "SATS":  "XLK", "SHOP":  "XLK",
    "SMCI":  "XLK", "SNOW":  "XLK", "SNPS":  "XLK", "STX":   "XLK",
    "TDY":   "XLK", "TEL":   "XLK", "TRMB":  "XLK", "TTD":   "XLK",
    "TTWO":  "XLK", "TYL":   "XLK", "VRSK":  "XLK", "VRSN":  "XLK",
    "WDAY":  "XLK", "WDC":   "XLK", "XYZ":   "XLK", "ZBRA":  "XLK",
    "ZM":    "XLK", "CRWD":  "XLK",
    # Semiconductors
    "AMD":   "SOXX", "AVGO":  "SOXX", "INTC":  "SOXX", "KLAC":  "SOXX",
    "LRCX":  "SOXX", "MCHP":  "SOXX", "MPWR":  "SOXX", "MU":    "SOXX",
    "NVDA":  "SOXX", "NXPI":  "SOXX", "ON":    "SOXX", "QCOM":  "SOXX",
    "SNDK":  "SOXX", "SWKS":  "SOXX", "TER":   "SOXX", "TXN":   "SOXX",
    "ADI":   "SOXX", "AMAT":  "SOXX", "COHR":  "SOXX",
    # Consumer Discretionary
    "AMZN":  "XLY", "TSLA":  "XLY", "HD":    "XLY", "DIS":   "XLY",
    "ABNB":  "XLY", "APTV":  "XLY", "AZO":   "XLY", "BBY":   "XLY",
    "BKNG":  "XLY", "CCL":   "XLY", "CMG":   "XLY", "CVNA":  "XLY",
    "DASH":  "XLY", "DECK":  "XLY", "DHI":   "XLY", "DPZ":   "XLY",
    "DRI":   "XLY", "EBAY":  "XLY", "EXPE":  "XLY", "F":     "XLY",
    "GM":    "XLY", "GPC":   "XLY", "HAS":   "XLY", "HLT":   "XLY",
    "LEN":   "XLY", "LOW":   "XLY", "LULU":  "XLY", "LVS":   "XLY",
    "LYV":   "XLY", "MAR":   "XLY", "MCD":   "XLY", "MGM":   "XLY",
    "NCLH":  "XLY", "NKE":   "XLY", "NVR":   "XLY", "ORLY":  "XLY",
    "PHM":   "XLY", "RCL":   "XLY", "RL":    "XLY", "ROST":  "XLY",
    "SBUX":  "XLY", "TGT":   "XLY", "TJX":   "XLY", "TKO":   "XLY",
    "TPR":   "XLY", "TSCO":  "XLY", "UBER":  "XLY", "ULTA":  "XLY",
    "WSM":   "XLY", "WYNN":  "XLY", "YUM":   "XLY",
    # Financials
    "JPM":   "XLF", "V":     "XLF", "MA":    "XLF", "GS":    "XLF",
    "BAC":   "XLF", "AFL":   "XLF", "ACGL":  "XLF", "AIG":   "XLF",
    "AIZ":   "XLF", "AJG":   "XLF", "ALL":   "XLF", "AMP":   "XLF",
    "AON":   "XLF", "APO":   "XLF", "ARES":  "XLF", "AXP":   "XLF",
    "BEN":   "XLF", "BK":    "XLF", "BLK":   "XLF", "BRK-B": "XLF",
    "BRO":   "XLF", "BX":    "XLF", "C":     "XLF", "CB":    "XLF",
    "CBOE":  "XLF", "CFG":   "XLF", "CINF":  "XLF", "CME":   "XLF",
    "COF":   "XLF", "COIN":  "XLF", "EG":    "XLF", "ERIE":  "XLF",
    "FDS":   "XLF", "FITB":  "XLF", "GL":    "XLF", "HBAN":  "XLF",
    "HIG":   "XLF", "HOOD":  "XLF", "IBKR":  "XLF", "ICE":   "XLF",
    "IVZ":   "XLF", "KEY":   "XLF", "KKR":   "XLF", "L":     "XLF",
    "MCO":   "XLF", "MET":   "XLF", "MS":    "XLF", "MRSH":  "XLF",
    "MSCI":  "XLF", "MTB":   "XLF", "NDAQ":  "XLF", "NTRS":  "XLF",
    "PFG":   "XLF", "PGR":   "XLF", "PNC":   "XLF", "PRU":   "XLF",
    "RF":    "XLF", "RJF":   "XLF", "SCHW":  "XLF", "SPGI":  "XLF",
    "STT":   "XLF", "SYF":   "XLF", "TFC":   "XLF", "TROW":  "XLF",
    "TRV":   "XLF", "USB":   "XLF", "WFC":   "XLF", "WRB":   "XLF",
    "WTW":   "XLF",
    # Healthcare
    "JNJ":   "XLV", "PFE":   "XLV", "UNH":   "XLV", "ABBV":  "XLV",
    "ABT":   "XLV", "ALGN":  "XLV", "A":     "XLV", "AMGN":  "XLV",
    "BAX":   "XLV", "BDX":   "XLV", "BIIB":  "XLV", "BMY":   "XLV",
    "BSX":   "XLV", "CAH":   "XLV", "CI":    "XLV", "CNC":   "XLV",
    "COO":   "XLV", "COR":   "XLV", "CRL":   "XLV", "CVS":   "XLV",
    "DXCM":  "XLV", "DVA":   "XLV", "EL":    "XLV", "ELV":   "XLV",
    "EW":    "XLV", "GEHC":  "XLV", "GILD":  "XLV", "HCA":   "XLV",
    "HSIC":  "XLV", "HUM":   "XLV", "IDXX":  "XLV", "INCY":  "XLV",
    "IQV":   "XLV", "ISRG":  "XLV", "LH":    "XLV", "MCK":   "XLV",
    "MDT":   "XLV", "MRNA":  "XLV", "MTD":   "XLV", "PODD":  "XLV",
    "REGN":  "XLV", "RMD":   "XLV", "RVTY":  "XLV", "SOLV":  "XLV",
    "STE":   "XLV", "SYK":   "XLV", "TECH":  "XLV", "TMO":   "XLV",
    "UHS":   "XLV", "VRTX":  "XLV", "VTRS":  "XLV", "WAT":   "XLV",
    "WST":   "XLV", "ZBH":   "XLV", "ZTS":   "XLV", "DGX":   "XLV",
    "DHR":   "XLV", "LLY":   "XLV", "MRK":   "XLV",
    # Industrials
    "BA":    "XLI", "CAT":   "XLI", "HON":   "XLI", "MMM":   "XLI",
    "ALLE":  "XLI", "AME":   "XLI", "AOS":   "XLI", "BLDR":  "XLI",
    "CARR":  "XLI", "CHRW":  "XLI", "CMI":   "XLI", "CPRT":  "XLI",
    "CSX":   "XLI", "CTAS":  "XLI", "DAL":   "XLI", "DE":    "XLI",
    "DOV":   "XLI", "EME":   "XLI", "EMR":   "XLI", "EXPD":  "XLI",
    "FAST":  "XLI", "FDX":   "XLI", "FIX":   "XLI", "FTV":   "XLI",
    "GD":    "XLI", "GE":    "XLI", "GEV":   "XLI", "GNRC":  "XLI",
    "GWW":   "XLI", "HII":   "XLI", "HUBB":  "XLI", "HWM":   "XLI",
    "IR":    "XLI", "ITW":   "XLI", "J":     "XLI", "JBHT":  "XLI",
    "JCI":   "XLI", "LDOS":  "XLI", "LHX":   "XLI", "LII":   "XLI",
    "LMT":   "XLI", "LUV":   "XLI", "MAS":   "XLI", "NDSN":  "XLI",
    "NOC":   "XLI", "NSC":   "XLI", "ODFL":  "XLI", "OTIS":  "XLI",
    "PCAR":  "XLI", "POOL":  "XLI", "PWR":   "XLI", "ROK":   "XLI",
    "ROL":   "XLI", "ROP":   "XLI", "RSG":   "XLI", "RTX":   "XLI",
    "SNA":   "XLI", "SWK":   "XLI", "TDG":   "XLI", "TT":    "XLI",
    "TXT":   "XLI", "UAL":   "XLI", "UNP":   "XLI", "UPS":   "XLI",
    "URI":   "XLI", "VRT":   "XLI", "WAB":   "XLI", "WM":    "XLI",
    "XYL":   "XLI", "ETN":   "XLI", "IEX":   "XLI", "PH":    "XLI",
    "PNR":   "XLI",
    # Materials
    "APD":   "XLB", "ALB":   "XLB", "AMCR":  "XLB", "AVY":   "XLB",
    "BALL":  "XLB", "CF":    "XLB", "CRH":   "XLB", "CTVA":  "XLB",
    "DD":    "XLB", "DOW":   "XLB", "ECL":   "XLB", "FCX":   "XLB",
    "IFF":   "XLB", "IP":    "XLB", "LIN":   "XLB", "LYB":   "XLB",
    "MLM":   "XLB", "MOS":   "XLB", "NUE":   "XLB", "PKG":   "XLB",
    "PPG":   "XLB", "SHW":   "XLB", "STLD":  "XLB", "SW":    "XLB",
    "VMC":   "XLB", "VLTO":  "XLB",
    # Consumer Staples
    "MO":    "XLP", "PG":    "XLP", "KO":    "XLP", "WMT":   "XLP",
    "ADM":   "XLP", "BF-B":  "XLP", "BG":    "XLP", "CAG":   "XLP",
    "CASY":  "XLP", "CHD":   "XLP", "CL":    "XLP", "CLX":   "XLP",
    "COST":  "XLP", "CPB":   "XLP", "DG":    "XLP", "DLTR":  "XLP",
    "GIS":   "XLP", "HRL":   "XLP", "HSY":   "XLP", "KDP":   "XLP",
    "KHC":   "XLP", "KMB":   "XLP", "KO":    "XLP", "KR":    "XLP",
    "KVUE":  "XLP", "MDLZ":  "XLP", "MKC":   "XLP", "MNST":  "XLP",
    "PEP":   "XLP", "PM":    "XLP", "SJM":   "XLP", "STZ":   "XLP",
    "SYY":   "XLP", "TAP":   "XLP", "TSN":   "XLP",
    # Energy
    "XOM":   "XLE", "CVX":   "XLE", "APA":   "XLE", "BKR":   "XLE",
    "COP":   "XLE", "CTRA":  "XLE", "DVN":   "XLE", "EOG":   "XLE",
    "EQT":   "XLE", "EXE":   "XLE", "FANG":  "XLE", "HAL":   "XLE",
    "KMI":   "XLE", "MPC":   "XLE", "OKE":   "XLE", "OXY":   "XLE",
    "PSX":   "XLE", "SLB":   "XLE", "TPL":   "XLE", "TRGP":  "XLE",
    "VLO":   "XLE", "WMB":   "XLE",
    # Utilities
    "AEE":   "XLU", "AEP":   "XLU", "AES":   "XLU", "ATO":   "XLU",
    "AWK":   "XLU", "CEG":   "XLU", "CMS":   "XLU", "CNP":   "XLU",
    "D":     "XLU", "DTE":   "XLU", "DUK":   "XLU", "ED":    "XLU",
    "EIX":   "XLU", "ES":    "XLU", "ETR":   "XLU", "EVRG":  "XLU",
    "EXC":   "XLU", "FE":    "XLU", "LNT":   "XLU", "NEE":   "XLU",
    "NI":    "XLU", "NRG":   "XLU", "PCG":   "XLU", "PEG":   "XLU",
    "PNW":   "XLU", "PPL":   "XLU", "SO":    "XLU", "SRE":   "XLU",
    "VST":   "XLU", "WEC":   "XLU", "XEL":   "XLU",
    # Real Estate
    "ARE":   "XLRE", "AMT":  "XLRE", "AVB":  "XLRE", "BXP":  "XLRE",
    "CBRE":  "XLRE", "CCI":  "XLRE", "CPT":  "XLRE", "CSGP": "XLRE",
    "DLR":   "XLRE", "DOC":  "XLRE", "EQIX": "XLRE", "EQR":  "XLRE",
    "ESS":   "XLRE", "EXR":  "XLRE", "FRT":  "XLRE", "HST":  "XLRE",
    "INVH":  "XLRE", "IRM":  "XLRE", "KIM":  "XLRE", "MAA":  "XLRE",
    "O":     "XLRE", "PLD":  "XLRE", "PSA":  "XLRE", "PSKY": "XLRE",
    "REG":   "XLRE", "SBAC": "XLRE", "SPG":  "XLRE", "UDR":  "XLRE",
    "VICI":  "XLRE", "VTR":  "XLRE", "WELL": "XLRE", "WY":   "XLRE",
    # Gold / Mining
    "NEM":   "GDX",
}


_TODAY_OVERRIDE: dict = {}  # {etf: today_close} — set by predictor for live consistency


def set_today_override(overrides: dict) -> None:
    """Inject today's intraday close for reference ETFs (SPY, GLD, XL*).

    Used by the live predictor so sector_rel/spy_rel/gold_spy features computed
    on today's bar use today's reference prices, not yesterday's.
    """
    _TODAY_OVERRIDE.clear()
    _TODAY_OVERRIDE.update(overrides)


def _apply_today_override(etf: str, close: pd.Series) -> pd.Series:
    if etf not in _TODAY_OVERRIDE or close.empty:
        return close
    today = pd.Timestamp.today().normalize()
    if today in close.index:
        return close
    today_close = pd.Series([_TODAY_OVERRIDE[etf]], index=[today], name="Close")
    return pd.concat([close, today_close])


def get_sector_etf(etf: str, force_refresh: bool = False) -> pd.Series:
    cache_file = os.path.join(CACHE_DIR, f"{etf}_sector.parquet")

    if os.path.exists(cache_file) and not force_refresh:
        try:
            df = pd.read_parquet(cache_file)
            last = df.index.max()
            if last >= pd.Timestamp.today().normalize() - pd.Timedelta(days=5):
                return _apply_today_override(etf, df["Close"])
        except Exception:
            pass

    for attempt in range(3):
        try:
            raw = yf.download(etf, start="2010-01-01", auto_adjust=True, progress=False)
            if raw.empty:
                return pd.Series(dtype=float)
            close = raw["Close"].squeeze()
            close.index = pd.to_datetime(close.index).normalize()
            close.name = "Close"
            os.makedirs(CACHE_DIR, exist_ok=True)
            close.to_frame().to_parquet(cache_file)
            print(f"[SECTOR] {etf} downloaded ({len(close)} rows).")
            return _apply_today_override(etf, close)
        except Exception as e:
            if attempt < 2:
                time.sleep(5)
            else:
                print(f"[SECTOR] Failed to download {etf}: {e}")
                return pd.Series(dtype=float)


def merge_sector_relative(df_price: pd.DataFrame, ticker: str,
                          windows: list[int] = None) -> pd.DataFrame:
    if windows is None:
        windows = [14, 30]

    etf = TICKER_SECTOR.get(ticker)
    if etf is None:
        return df_price

    etf_close = get_sector_etf(etf)
    if etf_close.empty:
        return df_price

    etf_aligned = etf_close.reindex(df_price.index).ffill()
    result = df_price.copy()

    for w in windows:
        ticker_roc = df_price["Close"].pct_change(w) * 100
        etf_roc    = etf_aligned.pct_change(w) * 100
        result[f"sector_rel_{w}d"] = (ticker_roc - etf_roc).clip(-30, 30)

    return result


def merge_spy_relative(df_price: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
    if windows is None:
        windows = [14, 30]

    spy_close = get_sector_etf("SPY")
    if spy_close.empty:
        return df_price

    spy_aligned = spy_close.reindex(df_price.index).ffill()
    result = df_price.copy()

    for w in windows:
        ticker_roc = df_price["Close"].pct_change(w) * 100
        spy_roc    = spy_aligned.pct_change(w) * 100
        result[f"spy_rel_{w}d"] = (ticker_roc - spy_roc).clip(-30, 30)

    return result


def merge_gold_spy(df_price: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
    """GLD ROC minus SPY ROC — positive = risk-off (gold outperforming), negative = risk-on."""
    if windows is None:
        windows = [14, 30]

    gld_close = get_sector_etf("GLD")
    spy_close = get_sector_etf("SPY")
    if gld_close.empty or spy_close.empty:
        return df_price

    gld_aligned = gld_close.reindex(df_price.index).ffill()
    spy_aligned = spy_close.reindex(df_price.index).ffill()
    result = df_price.copy()

    for w in windows:
        gld_roc = gld_aligned.pct_change(w) * 100
        spy_roc = spy_aligned.pct_change(w) * 100
        result[f"gold_spy_rel_{w}d"] = (gld_roc - spy_roc).clip(-30, 30)

    return result


if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "TSLA"]
    for t in tickers:
        print(f"\n=== {t} ({TICKER_SECTOR.get(t, '?')}) ===")
        import data_processor as dp
        raw = dp.get_data(t)
        df  = merge_sector_relative(raw, t, windows=[14, 30])
        print(df[["Close", "sector_rel_14d", "sector_rel_30d"]].tail(5))
