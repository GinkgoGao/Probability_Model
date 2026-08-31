"""News headlines via yfinance (free) - each fetch is cached as a dated snapshot so the
sentiment you record today is what the market could actually read today (point-in-time)."""
from __future__ import annotations
import pandas as pd
from data.store import load_json, save_json


def _norm(item: dict) -> dict | None:
    c = item.get("content") if isinstance(item.get("content"), dict) else item
    title = c.get("title")
    if not title:
        return None
    pub = c.get("pubDate") or c.get("providerPublishTime") or c.get("displayTime")
    try:
        ts = pd.to_datetime(pub, unit="s") if isinstance(pub, (int, float)) else pd.to_datetime(pub)
        if ts is not None and ts.tzinfo is not None:
            ts = ts.tz_convert("America/New_York").tz_localize(None)
    except Exception:
        ts = None
    prov = c.get("provider")
    prov = prov.get("displayName") if isinstance(prov, dict) else (c.get("publisher") or prov)
    return {"title": title, "summary": c.get("summary") or "", "published": ts.isoformat() if ts is not None else None,
            "provider": prov, "url": (c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else c.get("link")}


def fetch_news(ticker: str, asof: pd.Timestamp | None = None, use_cache: bool = True, max_age_hours: float = 3) -> list[dict]:
    import yfinance as yf
    asof = pd.Timestamp.now() if asof is None else pd.Timestamp(asof)
    name = f"{ticker.upper()}_{asof.strftime('%Y-%m-%d')}"
    if use_cache:
        cached = load_json("news", name, max_age_hours)
        if cached:
            return cached
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception as e:
        print(f"[news] {ticker} failed: {e}")
        raw = []
    items = [x for x in (_norm(i) for i in raw) if x]
    save_json(items, "news", name)
    return items
