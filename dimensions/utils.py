from __future__ import annotations
import math
import numpy as np
import pandas as pd


def clip(x, lo, hi) -> float:
    return float(min(hi, max(lo, x)))


def fnum(x, default=None):
    try:
        f = float(x)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def ret_n(close: pd.Series, n: int) -> float:
    if close is None or len(close) <= n:
        return float("nan")
    return float(close.iloc[-1] / close.iloc[-1 - n] - 1)


def rsi(close: pd.Series, n: int = 14) -> float:
    d = close.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    ru = up.ewm(alpha=1 / n, adjust=False).mean()
    rd = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = ru / rd.replace(0, np.nan)
    v = (100 - 100 / (1 + rs)).iloc[-1]
    return float(v) if math.isfinite(v) else 50.0


def atr_pct(df: pd.DataFrame, n: int = 14) -> float:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    a = tr.rolling(n).mean().iloc[-1]
    return float(a / c.iloc[-1]) if c.iloc[-1] > 0 and math.isfinite(a) else float("nan")


def upto(df: pd.DataFrame | None, ts: pd.Timestamp) -> pd.DataFrame | None:
    """Slice a DatetimeIndex frame to rows at or before ts (point-in-time guard)."""
    if df is None:
        return None
    return df[df.index <= ts]
