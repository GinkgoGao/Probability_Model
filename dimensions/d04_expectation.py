"""Dimension 4 - expectation structure: revision momentum, target upside, analyst actions,
and the 'height of the bar' (consensus vs guidance vs whisper). Manual overrides:
    --override consensus_eps=34.2 guidance_mid_eps=31.5 whisper_eps=35.0"""
from __future__ import annotations
import pandas as pd
from .base import Dimension, EventContext
from .utils import clip, fnum


class ExpectationDimension(Dimension):
    name = "expectation"
    supports_backtest = False

    def score(self, ctx: EventContext):
        f = ctx.data.get("fundamentals") or {}
        info = f.get("info") or {}
        if not f or (not info and not f.get("eps_trend")):
            return self.abstain("no fundamentals")
        ov = ctx.overrides or {}
        daily = ctx.data.get("daily")
        price = fnum(info.get("currentPrice")) or fnum(info.get("regularMarketPrice")) or (float(daily["Close"].iloc[-1]) if daily is not None and len(daily) else None)

        et = (f.get("eps_trend") or {}).get("0q") or {}
        cur, ago30 = fnum(et.get("current")), fnum(et.get("30daysAgo"))
        rev30 = (cur - ago30) / abs(ago30) if cur is not None and ago30 not in (None, 0) else None
        er = (f.get("eps_revisions") or {}).get("0q") or {}
        up30, dn30 = fnum(er.get("upLast30days"), 0) or 0, fnum(er.get("downLast30days"), 0) or 0
        net_rev = (up30 - dn30) / (up30 + dn30) if (up30 + dn30) > 0 else None
        tgt = f.get("analyst_price_targets") or {}
        mean_t = fnum(tgt.get("mean")) or fnum(info.get("targetMeanPrice"))
        upside = mean_t / price - 1 if mean_t and price else None
        ups = downs = 0
        for rec in f.get("upgrades_downgrades") or []:
            try:
                gd = pd.Timestamp(rec.get("GradeDate"))
            except Exception:
                continue
            if (ctx.asof - gd).days <= 30:
                act = str(rec.get("Action", "")).lower()
                ups += act == "up"
                downs += act == "down"
        consensus = fnum(ov.get("consensus_eps")) or cur
        guide = fnum(ov.get("guidance_mid_eps"))
        whisper = fnum(ov.get("whisper_eps"))
        high_bar = bool(consensus and guide and consensus > guide * 1.02)
        whisper_above = bool(whisper and consensus and whisper > consensus * 1.03)

        pred, signals = 0.0, 0
        if rev30 is not None:
            pred += clip(0.15 * rev30, -0.02, 0.02); signals += 1
        if net_rev is not None:
            pred += 0.01 * net_rev; signals += 1
        if upside is not None:
            pred += clip(0.04 * upside, -0.015, 0.015); signals += 1
        if ups or downs:
            pred += clip(0.005 * (ups - downs), -0.015, 0.015); signals += 1
        if high_bar:
            pred -= 0.02
        if whisper_above:
            pred -= 0.01
        pred = clip(pred, -0.05, 0.05)
        if signals == 0 and not high_bar:
            return self.abstain("no expectation signals available")
        conf = clip(0.20 + 0.07 * signals + (0.1 if guide else 0), 0.2, 0.6)
        ev = {"eps_consensus_0q": cur, "eps_30d_ago": ago30, "revision_30d": rev30, "net_revisions_30d": net_rev,
              "n_up_30d": up30, "n_down_30d": dn30, "target_mean": mean_t, "target_upside": upside,
              "upgrades_30d": ups, "downgrades_30d": downs, "recommendation_mean": fnum(info.get("recommendationMean")),
              "guidance_mid_eps": guide, "whisper_eps": whisper, "high_bar": high_bar, "whisper_above_consensus": whisper_above}
        notes = [(f"EPS consensus revised {rev30:+.1%} in 30d" if rev30 is not None else "no revision data"),
                 (f"target upside {upside:+.0%}" if upside is not None else ""),
                 f"{ups} upgrades / {downs} downgrades in 30d",
                 ("HIGH BAR: consensus above guidance midpoint" if high_bar else ""),
                 ("whisper above consensus -> even higher bar" if whisper_above else "")]
        return self.make(pred, conf, ev, [x for x in notes if x])
