"""Dimension 10 - positioning: short interest / days-to-cover (squeeze fuel, asymmetric upside),
insider net buying, institutional ownership."""
from __future__ import annotations
import pandas as pd
from .base import Dimension, EventContext
from .utils import clip, fnum


class PositioningDimension(Dimension):
    name = "positioning"
    supports_backtest = False

    def score(self, ctx: EventContext):
        f = ctx.data.get("fundamentals") or {}
        info = f.get("info") or {}
        spf = fnum(info.get("shortPercentOfFloat"))
        dtc = fnum(info.get("shortRatio"))
        inst = fnum(info.get("heldPercentInstitutions"))
        buys = sells = 0.0
        for rec in f.get("insider_transactions") or []:
            try:
                d = pd.Timestamp(rec.get("Start Date"))
                if (ctx.asof - d).days > 90:
                    continue
            except Exception:
                continue
            txt = str(rec.get("Text") or rec.get("Transaction") or "").lower()
            val = abs(fnum(rec.get("Value"), 0.0) or 0.0)
            if "purchase" in txt or "buy" in txt:
                buys += val
            elif "sale" in txt or "sell" in txt:
                sells += val
        if spf is None and dtc is None and buys == sells == 0:
            return self.abstain("no positioning data")
        pred, smult = 0.0, 1.0
        if spf is not None:
            if spf > 0.15:
                pred += 0.015; smult += 0.15
            if spf > 0.25:
                pred += 0.010; smult += 0.15
        if dtc is not None and dtc > 5:
            pred += 0.005
        net = buys - sells
        if net > 0:
            pred += 0.010
        elif sells > 0 and buys == 0:
            pred -= 0.003          # routine selling: weak evidence
        pred = clip(pred, -0.03, 0.05)
        conf = 0.20 + (0.20 if spf is not None and spf > 0.15 else 0.0) + (0.05 if net > 0 else 0.0)
        ev = {"short_pct_float": spf, "days_to_cover": dtc, "institutional_pct": inst,
              "insider_buys_90d_usd": buys, "insider_sells_90d_usd": sells, "insider_net_usd": net}
        notes = [(f"short interest {spf:.1%} of float, {dtc:.1f} days to cover" if spf is not None and dtc is not None else "short interest unavailable"),
                 (f"insiders net {'buying' if net > 0 else 'selling'} ${abs(net)/1e6:.1f}M in 90d" if (buys or sells) else "no insider activity")]
        return self.make(pred, clip(conf, 0.1, 0.5), ev, notes, smult)
