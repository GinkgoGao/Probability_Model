"""Dimension 2 - last 5 sessions of 5-minute bars: closing pressure, VWAP position, volume surge.
Low prior weight by design: it predicts liquidity and short-term momentum, not the earnings outcome."""
from __future__ import annotations
from .base import Dimension, EventContext
from .utils import clip, upto


class MicrostructureDimension(Dimension):
    name = "microstructure"
    supports_backtest = False

    def score(self, ctx: EventContext):
        m = upto(ctx.data.get("intraday"), ctx.asof)
        if m is None or len(m) < 50:
            return self.abstain("no intraday bars")
        m = m.copy()
        m["date"] = m.index.normalize()
        days = sorted(m["date"].unique())
        if len(days) < 2:
            return self.abstain("need >= 2 sessions")
        last = m[m["date"] == days[-1]]
        prev = m[m["date"] != days[-1]]
        if len(last) < 12:
            return self.abstain("last session too short")
        vol = last["Volume"].replace(0, 1)
        vwap = float((last["Close"] * vol).sum() / vol.sum())
        close = float(last["Close"].iloc[-1])
        close_vs_vwap = close / vwap - 1
        day_ret = close / float(last["Open"].iloc[0]) - 1
        last_hour = close / float(last["Close"].iloc[-min(12, len(last))]) - 1
        prev_vol = prev.groupby("date")["Volume"].sum().mean()
        vol_ratio = float(last["Volume"].sum() / prev_vol) if prev_vol and prev_vol > 0 else 1.0
        drift5 = float(sum(g["Close"].iloc[-1] / g["Open"].iloc[0] - 1 for _, g in m.groupby("date")))

        pred = clip(0.25 * close_vs_vwap + 0.15 * last_hour + 0.05 * drift5, -0.03, 0.03)
        conf = clip(0.20 + (0.10 if vol_ratio > 1.5 else 0.0), 0.10, 0.35)
        ev = {"close_vs_vwap": close_vs_vwap, "last_session_ret": day_ret, "last_hour_ret": last_hour,
              "volume_ratio_vs_prior": vol_ratio, "intraday_drift_5d": drift5, "sessions": len(days)}
        notes = [f"close {close_vs_vwap:+.2%} vs VWAP, last hour {last_hour:+.2%}", f"volume x{vol_ratio:.2f} vs prior sessions"]
        return self.make(pred, conf, ev, notes)
