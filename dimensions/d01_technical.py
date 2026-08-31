"""Dimension 1 - daily K-line structure. Main signal: partial mean reversion of the pre-earnings run-up."""
from __future__ import annotations
import numpy as np
from .base import Dimension, EventContext
from .utils import clip, ret_n, rsi, atr_pct, upto


class TechnicalDimension(Dimension):
    name = "technical"
    supports_backtest = True

    def score(self, ctx: EventContext):
        d = upto(ctx.data.get("daily"), ctx.asof)
        if d is None or len(d) < 60:
            return self.abstain("need >= 60 daily bars")
        c = d["Close"].astype(float)
        runup20, runup5 = ret_n(c, 20), ret_n(c, 5)
        ma50 = float(c.rolling(50).mean().iloc[-1])
        ma200 = float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else float("nan")
        r = rsi(c)
        a = atr_pct(d)
        hi52 = float(c.iloc[-252:].max())
        dist_hi = float(c.iloc[-1] / hi52 - 1)

        pred = -0.15 * runup20                       # documented pre-announcement run-up reversal
        trend_up = c.iloc[-1] > ma50 and (np.isnan(ma200) or ma50 > ma200)
        pred += 0.01 if trend_up else -0.01
        if r > 75:
            pred -= 0.01
        elif r < 30:
            pred += 0.01
        pred = clip(pred, -0.06, 0.06)
        conf = clip(0.30 + 0.25 * min(abs(runup20) / 0.15, 1.0), 0.20, 0.60)
        smult = clip(1.0 + 0.5 * max(a / 0.03 - 1.0, 0.0), 1.0, 1.5) if np.isfinite(a) else 1.0
        ev = {"runup_20d": runup20, "runup_5d": runup5, "rsi14": r, "atr_pct": a,
              "above_ma50": bool(c.iloc[-1] > ma50), "ma50_over_ma200": None if np.isnan(ma200) else bool(ma50 > ma200),
              "dist_from_52w_high": dist_hi, "last_close": float(c.iloc[-1])}
        notes = [f"20d run-up {runup20:+.1%} -> expect partial reversal", f"RSI14 {r:.0f}, {'up' if trend_up else 'down'}-trend regime",
                 f"ATR {a:.1%}/day -> sigma x{smult:.2f}"]
        return self.make(pred, conf, ev, notes, smult)
