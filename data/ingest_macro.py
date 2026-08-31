"""Macro/regime series (VIX, VIX3M, 10y yield, SPY) and the hand-maintained event calendar."""
from __future__ import annotations
import json
import pandas as pd
from config.settings import MACRO_CALENDAR_PATH
from data.ingest_prices import fetch_daily

MACRO_TICKERS = ["^VIX", "^VIX3M", "^TNX", "SPY"]


def fetch_macro(extra: list[str] | None = None, period: str = "3y", **kw) -> dict[str, pd.DataFrame | None]:
    out = {}
    for t in MACRO_TICKERS + list(extra or []):
        out[t] = fetch_daily(t, period=period, **kw)
    return out


def load_macro_calendar() -> dict[str, list[pd.Timestamp]]:
    try:
        raw = json.loads(MACRO_CALENDAR_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    cal = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        cal[k] = [pd.Timestamp(x) for x in v]
    return cal
