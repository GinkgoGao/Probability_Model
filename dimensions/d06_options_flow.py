"""Dimension 6 - options flow/positioning: put-call ratios, turnover, near-money concentration.
Rules from our earlier work: extremes are contrarian; aggregate ratios are weak evidence on their own."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .base import Dimension, EventContext
from .utils import clip, fnum
from .d05_options_rnd import select_expiries


class OptionsFlowDimension(Dimension):
    name = "options_flow"
    supports_backtest = False

    def score(self, ctx: EventContext):
        ch = ctx.data.get("chains")
        if not ch or not ch.get("chains"):
            return self.abstain("no options chain")
        S = fnum(ch.get("spot"))
        e1, _ = select_expiries(list(ch["chains"].keys()), ctx.event_date, ctx.timing)
        if e1 is None or not S:
            return self.abstain("no expiry after the event")
        lo, hi = pd.Timestamp(e1), pd.Timestamp(e1) + pd.Timedelta(days=45)
        cv = pv = coi = poi = 0.0
        near_c = near_p = 0.0
        for e, d in ch["chains"].items():
            if not (lo <= pd.Timestamp(e) <= hi):
                continue
            c, p = d["calls"], d["puts"]
            cv += float(c["volume"].fillna(0).sum()); pv += float(p["volume"].fillna(0).sum())
            coi += float(c["openInterest"].fillna(0).sum()); poi += float(p["openInterest"].fillna(0).sum())
            nc = c[(c["strike"] > 0.9 * S) & (c["strike"] < 1.1 * S)]; npt = p[(p["strike"] > 0.9 * S) & (p["strike"] < 1.1 * S)]
            near_c += float(nc["openInterest"].fillna(0).sum()); near_p += float(npt["openInterest"].fillna(0).sum())
        if cv + pv < 500:
            return self.abstain("options too illiquid (volume < 500)")
        pcr_vol = pv / max(cv, 1.0)
        pcr_oi = poi / max(coi, 1.0)
        turnover = (cv + pv) / max(coi + poi, 1.0)
        near_call_share = near_c / max(near_c + near_p, 1.0)

        pred, conf, tag = 0.0, 0.15, "neutral"
        if pcr_vol > 1.5:
            pred, conf, tag = 0.015, 0.35, "extreme put volume -> contrarian bullish"
        elif pcr_vol < 0.5:
            pred, conf, tag = -0.015, 0.35, "extreme call volume -> contrarian bearish"
        pred += clip(0.02 * (near_call_share - 0.5), -0.005, 0.005)
        ev = {"pcr_volume": pcr_vol, "pcr_open_interest": pcr_oi, "turnover_vol_over_oi": turnover,
              "near_money_call_oi_share": near_call_share, "call_volume": cv, "put_volume": pv, "window_expiries": [e1, hi.strftime('%Y-%m-%d')]}
        notes = [f"P/C volume {pcr_vol:.2f}, P/C OI {pcr_oi:.2f} -> {tag}",
                 f"turnover {turnover:.2f}x OI" + (" (unusual activity)" if turnover > 1 else "")]
        return self.make(pred, conf, ev, notes)
