"""Dimension 8 - macro regime: VIX level and term structure, SPY trend, rates, and calendar collisions.
Mostly a sigma multiplier and a confidence dampener; only a small directional tilt."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .base import Dimension, EventContext
from .utils import clip, ret_n, upto


class MacroRegimeDimension(Dimension):
    name = "macro_regime"
    supports_backtest = True

    def score(self, ctx: EventContext):
        mac = ctx.data.get("macro") or {}
        asof = ctx.asof

        def last(df):
            d = upto(df, asof)
            return float(d["Close"].iloc[-1]) if d is not None and len(d) else None

        vix, vix3m = last(mac.get("^VIX")), last(mac.get("^VIX3M"))
        if vix is None:
            return self.abstain("no VIX data")
        ratio = vix / vix3m if vix3m else None
        spy = upto(mac.get("SPY"), asof)
        spy10 = ret_n(spy["Close"], 10) if spy is not None and len(spy) > 10 else 0.0
        tnx = upto(mac.get("^TNX"), asof)
        tnx_chg = (float(tnx["Close"].iloc[-1]) - float(tnx["Close"].iloc[-21])) / 100 if tnx is not None and len(tnx) > 21 else None
        collisions = []
        if ctx.event_date is not None:
            ev = pd.Timestamp(ctx.event_date).normalize()
            for name, dates in (ctx.data.get("macro_calendar") or {}).items():
                for d in dates:
                    if ev - pd.Timedelta(days=1) <= d.normalize() <= ev + pd.Timedelta(days=1):
                        collisions.append(f"{name} {d.strftime('%Y-%m-%d')}")
        stress = vix > 25 or (ratio is not None and ratio > 1.0)
        pred = -0.01 if stress else (0.005 if (vix < 15 and spy10 > 0) else 0.0)
        pred += clip(0.1 * spy10, -0.01, 0.01)
        smult = clip(1.0 + 0.4 * min(max((ratio or 1.0) - 1.0, 0.0), 0.5) + 0.3 * min(max((vix - 20) / 20, 0.0), 1.0), 0.9, 1.5)
        conf = 0.25 * (0.6 if collisions else 1.0)
        ev = {"vix": vix, "vix3m": vix3m, "vix_term_ratio": ratio, "spy_10d": spy10, "tnx_change_20d": tnx_chg,
              "stress_regime": bool(stress), "calendar_collisions": collisions}
        notes = [f"VIX {vix:.1f}" + (f", VIX/VIX3M {ratio:.2f}" if ratio else "") + (" -> STRESS regime" if stress else ""),
                 f"SPY 10d {spy10:+.1%}", (f"calendar collision: {', '.join(collisions)}" if collisions else "no macro events in the reaction window")]
        return self.make(pred, conf, ev, notes, smult)
