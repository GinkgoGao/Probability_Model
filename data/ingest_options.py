"""Options chain snapshots via yfinance. Every fetch is written to disk as a dated snapshot,
because historical chains are not available for free - the snapshots you accumulate ARE
your future options history (needed to backtest the options dimensions)."""
from __future__ import annotations
import pandas as pd
from data.store import load_df, save_df, cache_path

COLS = ["strike", "bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility"]


def _spot(t, daily=None) -> float | None:
    for getter in (
        lambda: float(t.fast_info["last_price"]),
        lambda: float(t.fast_info["lastPrice"]),
        lambda: float(t.info["regularMarketPrice"]),
        lambda: float(t.info["currentPrice"]),
    ):
        try:
            v = getter()
            if v and v > 0:
                return v
        except Exception:
            continue
    if daily is not None and len(daily):
        return float(daily["Close"].iloc[-1])
    return None


def fetch_chains(ticker: str, asof: pd.Timestamp | None = None, max_expiries: int = 8,
                 use_cache: bool = True, max_age_hours: float = 1, daily=None) -> dict | None:
    """Return {'spot', 'asof', 'expiries', 'chains': {expiry: {'calls': df, 'puts': df}}}."""
    import yfinance as yf
    asof = pd.Timestamp.now() if asof is None else pd.Timestamp(asof)
    name = f"{ticker.upper()}_{asof.strftime('%Y-%m-%d')}"
    if use_cache:
        cached = load_snapshot(ticker, asof, max_age_hours)
        if cached is not None:
            return cached
    t = yf.Ticker(ticker)
    try:
        exps = list(t.options)[:max_expiries]
    except Exception as e:
        print(f"[options] {ticker} expiries failed: {e}")
        return None
    if not exps:
        return None
    spot = _spot(t, daily)
    frames, chains = [], {}
    for e in exps:
        try:
            oc = t.option_chain(e)
        except Exception as ex:
            print(f"[options] {ticker} {e} failed: {ex}")
            continue
        c = oc.calls[[x for x in COLS if x in oc.calls.columns]].copy()
        p = oc.puts[[x for x in COLS if x in oc.puts.columns]].copy()
        chains[e] = {"calls": c, "puts": p}
        c = c.assign(expiry=e, type="call"); p = p.assign(expiry=e, type="put")
        frames += [c, p]
    if not chains:
        return None
    snap = pd.concat(frames, ignore_index=True)
    snap["spot"] = spot
    snap["asof"] = asof.isoformat()
    save_df(snap, "options", name)
    return {"spot": spot, "asof": asof, "expiries": list(chains.keys()), "chains": chains}


def load_snapshot(ticker: str, asof: pd.Timestamp, max_age_hours: float | None = None) -> dict | None:
    name = f"{ticker.upper()}_{pd.Timestamp(asof).strftime('%Y-%m-%d')}"
    snap = load_df("options", name, max_age_hours)
    if snap is None or snap.empty:
        return None
    chains = {}
    for e, g in snap.groupby("expiry"):
        chains[e] = {"calls": g[g["type"] == "call"].drop(columns=["expiry", "type", "spot", "asof"]).reset_index(drop=True),
                     "puts": g[g["type"] == "put"].drop(columns=["expiry", "type", "spot", "asof"]).reset_index(drop=True)}
    spot = snap["spot"].dropna().iloc[0] if snap["spot"].notna().any() else None
    return {"spot": float(spot) if spot is not None else None, "asof": pd.Timestamp(snap["asof"].iloc[0]),
            "expiries": sorted(chains.keys()), "chains": chains}


def get_risk_free(fallback: float = 0.04) -> float:
    """13-week T-bill yield (^IRX, quoted in %) as the risk-free rate."""
    try:
        import yfinance as yf
        s = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
        if len(s):
            return float(s.iloc[-1]) / 100.0
    except Exception:
        pass
    return fallback
