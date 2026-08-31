"""Analyst expectations, targets, revisions, insider and short-interest data via yfinance.
NOTE: these are CURRENT snapshots only - yfinance has no point-in-time history for them,
which is why the expectation/positioning dimensions cannot be backtested from free data."""
from __future__ import annotations
import pandas as pd
from data.store import load_json, save_json, clean

INFO_KEYS = [
    "sector", "industry", "marketCap", "currentPrice", "regularMarketPrice", "targetMeanPrice",
    "targetMedianPrice", "recommendationMean", "recommendationKey", "numberOfAnalystOpinions",
    "shortPercentOfFloat", "shortRatio", "sharesShort", "sharesShortPriorMonth", "floatShares",
    "heldPercentInstitutions", "heldPercentInsiders", "beta", "trailingPE", "forwardPE",
    "enterpriseToEbitda", "priceToSalesTrailing12Months", "revenueGrowth", "earningsGrowth",
]


def _records(df: pd.DataFrame | None, index_name: str = "index") -> list[dict]:
    if df is None or getattr(df, "empty", True):
        return []
    d = df.copy()
    if isinstance(d.index, pd.DatetimeIndex) and d.index.tz is not None:
        d.index = d.index.tz_localize(None)
    d = d.reset_index()
    d.columns = [index_name if i == 0 and c in ("index", None) else str(c) for i, c in enumerate(d.columns)]
    return clean(d.to_dict(orient="records"))


def _table(df: pd.DataFrame | None) -> dict:
    if df is None or getattr(df, "empty", True):
        return {}
    return clean({str(k): {str(c): v for c, v in row.items()} for k, row in df.to_dict(orient="index").items()})


def fetch_fundamentals(ticker: str, asof: pd.Timestamp | None = None, use_cache: bool = True, max_age_hours: float = 12) -> dict:
    import yfinance as yf
    asof = pd.Timestamp.now() if asof is None else pd.Timestamp(asof)
    name = f"{ticker.upper()}_{asof.strftime('%Y-%m-%d')}"
    if use_cache:
        cached = load_json("fundamentals", name, max_age_hours)
        if cached:
            return cached
    t = yf.Ticker(ticker)
    out = {"ticker": ticker.upper(), "fetched_at": asof.isoformat()}
    try:
        info = t.info or {}
        out["info"] = {k: info.get(k) for k in INFO_KEYS}
    except Exception as e:
        print(f"[fund] {ticker} info failed: {e}")
        out["info"] = {}
    for attr in ("eps_trend", "eps_revisions", "earnings_estimate", "revenue_estimate"):
        try:
            out[attr] = _table(getattr(t, attr))
        except Exception:
            out[attr] = {}
    try:
        out["analyst_price_targets"] = clean(dict(t.analyst_price_targets or {}))
    except Exception:
        out["analyst_price_targets"] = {}
    try:
        ud = t.upgrades_downgrades
        if ud is not None and not ud.empty:
            ud = ud.copy()
            if isinstance(ud.index, pd.DatetimeIndex) and ud.index.tz is not None:
                ud.index = ud.index.tz_localize(None)
            ud = ud[ud.index >= asof - pd.Timedelta(days=120)]
        out["upgrades_downgrades"] = _records(ud, "GradeDate")
    except Exception:
        out["upgrades_downgrades"] = []
    try:
        out["insider_transactions"] = _records(t.insider_transactions, "row")
    except Exception:
        out["insider_transactions"] = []
    save_json(out, "fundamentals", name)
    return out
