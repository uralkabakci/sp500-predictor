import os
import time
import warnings
import requests
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CACHE_DIR = "data_cache/earnings"
EDGAR_HEADERS = {"User-Agent": "metrade-research metrade@research.com"}

_CIK_MAP: dict = {}


def _load_cik_map() -> dict:
    global _CIK_MAP
    if _CIK_MAP:
        return _CIK_MAP
    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=EDGAR_HEADERS, timeout=15
        )
        resp.raise_for_status()
        _CIK_MAP = {
            v["ticker"].upper(): str(v["cik_str"]).zfill(10)
            for v in resp.json().values()
        }
    except Exception as e:
        print(f"[EARNINGS] CIK map fetch failed: {e}")
    return _CIK_MAP


def _get_cik(ticker: str) -> str | None:
    return _load_cik_map().get(ticker.upper())


def _fetch_concept(cik: str, concept: str, unit: str) -> list:
    url = (f"https://data.sec.gov/api/xbrl/companyconcept/"
           f"CIK{cik}/us-gaap/{concept}.json")
    try:
        resp = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
        if resp.status_code != 200:
            return []
        return resp.json().get("units", {}).get(unit, [])
    except Exception:
        return []


def _parse_quarterly(entries: list, value_col: str) -> pd.DataFrame:
    """
    Parse EDGAR XBRL entries into quarterly YoY % change, indexed by filed date.
    Only 10-Q filings with ~3-month periods are accepted.
    YoY is computed by matching the same fiscal quarter of the prior year.
    """
    rows = []
    for e in entries:
        if e.get("form") != "10-Q":
            continue
        filed = e.get("filed")
        start = e.get("start")
        end   = e.get("end")
        val   = e.get("val")
        if not filed or not end or val is None:
            continue
        try:
            filed_dt = pd.Timestamp(filed)
            end_dt   = pd.Timestamp(end)
            if start:
                period_days = (end_dt - pd.Timestamp(start)).days
                if not (60 <= period_days <= 105):
                    continue
        except Exception:
            continue
        rows.append({"filed": filed_dt, "period_end": end_dt, value_col: float(val)})

    if not rows:
        return pd.DataFrame()

    df = (pd.DataFrame(rows)
            .drop_duplicates("period_end", keep="last")
            .sort_values("period_end")
            .reset_index(drop=True))

    df["period_end"] = pd.to_datetime(df["period_end"])
    df["quarter"]    = df["period_end"].dt.to_period("Q")

    # YoY: compare each quarter with the same quarter 4 periods prior
    yoy_list = []
    for _, row in df.iterrows():
        prev_q = row["quarter"] - 4
        match  = df[df["quarter"] == prev_q]
        if not match.empty:
            prev_val = match[value_col].iloc[-1]
            curr_val = row[value_col]
            denom    = max(abs(prev_val), 1e-9)
            yoy_list.append(float(np.clip((curr_val - prev_val) / denom, -5, 5)))
        else:
            yoy_list.append(np.nan)

    df[f"{value_col}_yoy_pct"] = yoy_list
    result = (df[["filed", f"{value_col}_yoy_pct"]]
                .dropna()
                .set_index("filed")
                .sort_index())
    return result


def get_earnings_features(ticker: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Returns a daily-aligned DataFrame with earnings features from SEC EDGAR.
    Features: eps_yoy_pct, revenue_yoy_pct, days_since_earnings.
    Uses 'filed' date (public disclosure) to prevent lookahead bias.
    """
    cache_file = os.path.join(CACHE_DIR, f"{ticker}_earnings.parquet")

    if not force_refresh and os.path.exists(cache_file):
        try:
            df_cached = pd.read_parquet(cache_file)
            last_date = df_cached.index.max()
            if last_date >= pd.Timestamp.today().normalize() - pd.Timedelta(days=90):
                return df_cached
        except Exception:
            pass

    cik = _get_cik(ticker)
    if not cik:
        print(f"[EARNINGS] {ticker}: CIK not found, skipping.")
        return pd.DataFrame()

    print(f"[EARNINGS] {ticker} (CIK {cik}) — fetching from SEC EDGAR...")

    # EPS
    eps_entries = _fetch_concept(cik, "EarningsPerShareBasic", "USD/shares")
    if not eps_entries:
        eps_entries = _fetch_concept(cik, "EarningsPerShareDiluted", "USD/shares")
    time.sleep(0.15)

    # Revenue — try multiple GAAP concepts (companies report differently)
    rev_entries = _fetch_concept(cik, "Revenues", "USD")
    if not rev_entries:
        rev_entries = _fetch_concept(
            cik, "RevenueFromContractWithCustomerExcludingAssessedTax", "USD")
        time.sleep(0.15)
    if not rev_entries:
        rev_entries = _fetch_concept(cik, "SalesRevenueNet", "USD")
        time.sleep(0.15)
    time.sleep(0.15)

    eps_df = _parse_quarterly(eps_entries, "eps")
    rev_df = _parse_quarterly(rev_entries, "revenue")

    if eps_df.empty and rev_df.empty:
        print(f"[EARNINGS] {ticker}: no quarterly data found on EDGAR.")
        return pd.DataFrame()

    # Merge EPS and Revenue on filed date
    if not eps_df.empty and not rev_df.empty:
        quarterly = eps_df.join(rev_df, how="outer")
    else:
        quarterly = eps_df if not eps_df.empty else rev_df

    quarterly = quarterly[~quarterly.index.duplicated(keep="last")].sort_index()

    # Acceleration: change in YoY growth rate (second derivative)
    if "eps_yoy_pct" in quarterly.columns:
        quarterly["eps_yoy_acceleration"] = quarterly["eps_yoy_pct"].diff(1).clip(-5, 5)
    if "revenue_yoy_pct" in quarterly.columns:
        quarterly["revenue_yoy_acceleration"] = quarterly["revenue_yoy_pct"].diff(1).clip(-5, 5)

    feature_cols = [c for c in quarterly.columns if c.endswith("_yoy_pct") or c.endswith("_acceleration")]
    if not feature_cols:
        return pd.DataFrame()

    # Expand to daily, forward-filling from each filed date
    today     = pd.Timestamp.today().normalize()
    daily_idx = pd.date_range(start=quarterly.index.min(), end=today, freq="D")
    df_daily  = quarterly[feature_cols].reindex(daily_idx).ffill()
    df_daily.index.name = "Date"

    # days_since_earnings: normalized 0-1 (1 = 1 year or more ago)
    filing_np = np.array(quarterly.index, dtype="datetime64[D]")
    daily_np  = np.array(daily_idx,       dtype="datetime64[D]")
    pos       = np.searchsorted(filing_np, daily_np, side="right") - 1
    days_arr  = np.where(
        pos >= 0,
        (daily_np - filing_np[np.maximum(pos, 0)]).astype("timedelta64[D]").astype(int),
        365,
    )
    df_daily["days_since_earnings"] = np.clip(days_arr, 0, 365) / 365.0

    os.makedirs(CACHE_DIR, exist_ok=True)
    df_daily.to_parquet(cache_file)
    print(f"[EARNINGS] {ticker}: {len(df_daily)} daily rows "
          f"({df_daily.index.min().date()} → {df_daily.index.max().date()})")
    return df_daily


def merge_earnings(df_price: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Merge earnings features into price DataFrame. Missing data filled with neutrals."""
    df_earn = get_earnings_features(ticker)
    if df_earn.empty:
        return df_price

    aligned = df_earn.reindex(df_price.index).ffill()
    result  = df_price.copy()

    ablation = os.environ.get("ABLATION_FEATURES", "")
    if ablation:
        active_cols = ablation.split(",")
    else:
        active_cols = ["eps_yoy_pct", "revenue_yoy_pct", "eps_yoy_acceleration", "days_since_earnings"]

    for col in active_cols:
        if col not in aligned.columns:
            continue
        if col == "days_since_earnings":
            result[col] = aligned[col].fillna(1.0)
        else:
            result[col] = aligned[col].fillna(0.0)

    return result


if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "NVDA", "TSLA"]
    for t in tickers:
        print(f"\n=== {t} ===")
        df = get_earnings_features(t, force_refresh=True)
        if not df.empty:
            print(df.dropna().tail(10))
            print(f"Coverage: {df.dropna().index.min().date()} → {df.dropna().index.max().date()}")
