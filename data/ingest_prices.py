"""Daily and intraday OHLCV via yfinance, with on-disk caching."""
from __future__ import annotations
import pandas as pd
from data.store import load_df, save_df, to_naive


def _download(ticker: str, period: str, interval: str) -> pd.DataFrame | None:
    import yfinance as yf
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    except Exception as e:  # network / delisted / rate limit
        print(f"[prices] {ticker} {interval} failed: {e}")
        return None
    if df is None or df.empty:
        return None
    df = to_naive(df)
    df = df[[c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "Date"
    return df


def fetch_daily(ticker: str, period: str = "3y", use_cache: bool = True, max_age_hours: float = 6) -> pd.DataFrame | None:
    name = f"{ticker.upper()}_daily_{period}"
    df = load_df("prices", name, max_age_hours) if use_cache else None
    if df is None:
        df = _download(ticker, period, "1d")
        if df is not None:
            save_df(df, "prices", name)
    return df


def fetch_intraday(ticker: str, period: str = "5d", interval: str = "5m", use_cache: bool = True, max_age_hours: float = 1) -> pd.DataFrame | None:
    name = f"{ticker.upper()}_{interval}_{period}"
    df = load_df("prices", name, max_age_hours) if use_cache else None
    if df is None:
        df = _download(ticker, period, interval)
        if df is not None:
            save_df(df, "prices", name)
    return df


def fetch_many_daily(tickers: list[str], period: str = "3y", **kw) -> dict[str, pd.DataFrame | None]:
    out = {}
    for t in dict.fromkeys([x.upper() for x in tickers if x]):
        out[t] = fetch_daily(t, period=period, **kw)
    return out
