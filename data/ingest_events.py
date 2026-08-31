"""Earnings calendar + the single most important function in the project: compute_reaction().
Everything (dimension 3, peers, the logbook, the backtest) must define 'the reaction' the same way."""
from __future__ import annotations
import numpy as np
import pandas as pd
from data.store import load_df, save_df, to_naive


def fetch_earnings_dates(ticker: str, limit: int = 40, use_cache: bool = True, max_age_hours: float = 12) -> pd.DataFrame | None:
    import yfinance as yf
    name = f"{ticker.upper()}_earnings_{limit}"
    df = load_df("events", name, max_age_hours) if use_cache else None
    if df is not None:
        return df
    try:
        df = yf.Ticker(ticker).get_earnings_dates(limit=limit)
    except Exception as e:
        print(f"[events] {ticker} earnings dates failed: {e}")
        return None
    if df is None or df.empty:
        return None
    df = to_naive(df).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = "EarningsDate"
    save_df(df, "events", name)
    return df


def upcoming_event(ed: pd.DataFrame | None, asof: pd.Timestamp) -> tuple[pd.Timestamp | None, str | None]:
    if ed is None or ed.empty:
        return None, None
    fut = ed[ed.index.normalize() >= asof.normalize()]
    if fut.empty:
        return None, None
    ts = fut.index[0]
    return ts.normalize(), infer_timing(ts, None)


def past_events(ed: pd.DataFrame | None, before: pd.Timestamp) -> pd.DataFrame:
    if ed is None or ed.empty:
        return pd.DataFrame()
    return ed[ed.index.normalize() < pd.Timestamp(before).normalize()]


def infer_timing(ts: pd.Timestamp, daily: pd.DataFrame | None) -> str:
    """AMC (after market close) or BMO (before market open).
    1) use the clock time yfinance attaches; 2) else pick the day with the larger |move|."""
    h = ts.hour
    if 0 < h < 12:
        return "BMO"
    if h >= 12:
        return "AMC"
    if daily is not None and len(daily) > 3:
        idx = daily.index.normalize()
        d = ts.normalize()
        pos = idx.searchsorted(d)
        if 0 < pos < len(idx) - 1 and idx[pos] == d:
            c = daily["Close"].values
            r_day = abs(c[pos] / c[pos - 1] - 1)
            r_next = abs(c[pos + 1] / c[pos] - 1)
            return "AMC" if r_next > r_day else "BMO"
    return "AMC"


def estimate_beta(daily: pd.DataFrame | None, spy: pd.DataFrame | None, window: int = 120) -> float:
    if daily is None or spy is None or len(daily) < 40 or len(spy) < 40:
        return 1.0
    a = daily["Close"].pct_change().dropna().iloc[-window:]
    b = spy["Close"].pct_change().dropna()
    j = a.index.intersection(b.index)
    if len(j) < 30:
        return 1.0
    a, b = a.loc[j].values, b.loc[j].values
    var = np.var(b)
    return float(np.cov(a, b)[0, 1] / var) if var > 0 else 1.0


def compute_reaction(daily: pd.DataFrame | None, date, timing: str, spy: pd.DataFrame | None = None, beta: float = 1.0) -> dict | None:
    """Post-earnings reaction relative to the last close before the number was public.
    AMC: ref = close(D), reaction day = D+1.  BMO: ref = close(D-1), reaction day = D."""
    if daily is None or daily.empty:
        return None
    idx = daily.index.normalize()
    d = pd.Timestamp(date).normalize()
    pos = int(idx.searchsorted(d))
    if pos >= len(idx):
        return None
    if idx[pos] != d and (idx[pos] - d).days > 3:
        return None
    ref_pos, react_pos = (pos, pos + 1) if timing == "AMC" else (pos - 1, pos)
    if ref_pos < 0 or react_pos >= len(idx):
        return None
    ref_close = float(daily["Close"].iloc[ref_pos])
    ro, rc = float(daily["Open"].iloc[react_pos]), float(daily["Close"].iloc[react_pos])
    if ref_close <= 0 or ro <= 0:
        return None
    out = {
        "ref_date": str(idx[ref_pos].date()), "reaction_date": str(idx[react_pos].date()),
        "gap": ro / ref_close - 1, "close_ret": rc / ref_close - 1, "intraday": rc / ro - 1,
        "spy_ret": None, "excess": None, "beta": beta,
    }
    if spy is not None and not spy.empty:
        sidx = spy.index.normalize()
        try:
            s0 = float(spy["Close"].values[sidx.get_loc(idx[ref_pos])])
            s1 = float(spy["Close"].values[sidx.get_loc(idx[react_pos])])
            out["spy_ret"] = s1 / s0 - 1
            out["excess"] = out["close_ret"] - beta * out["spy_ret"]
        except Exception:
            pass
    return out
