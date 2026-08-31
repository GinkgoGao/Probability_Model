"""Dimension 7 - peer read-through and the season's 'reaction function':
how did peers that already reported get treated, and how is the sector trading vs SPY."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .base import Dimension, EventContext
from .utils import clip, ret_n, upto
from data.ingest_events import infer_timing, compute_reaction


class PeerSectorDimension(Dimension):
    name = "peer_sector"
    supports_backtest = True

    def score(self, ctx: EventContext):
        peers = ctx.data.get("peers") or []
        pdaily = ctx.data.get("peer_daily") or {}
        etf = ctx.data.get("sector_etf")
        spy = upto(ctx.data.get("spy_daily"), ctx.asof)
        asof = ctx.asof

        def r_n(df, n):
            d = upto(df, asof)
            return ret_n(d["Close"], n) if d is not None and len(d) > n else float("nan")

        spy20 = r_n(spy, 20) if spy is not None else float("nan")
        spy20 = 0.0 if np.isnan(spy20) else spy20
        etf_df = pdaily.get(etf)
        sector_rs20 = r_n(etf_df, 20) - spy20 if etf_df is not None else float("nan")
        sector_rs5 = r_n(etf_df, 5) - (r_n(spy, 5) if spy is not None else 0.0) if etf_df is not None else float("nan")
        peer_rets = [r_n(pdaily[p], 20) - spy20 for p in peers if pdaily.get(p) is not None]
        peer_rets = [x for x in peer_rets if np.isfinite(x)]
        peer_mom = float(np.mean(peer_rets)) if peer_rets else None

        reacts = []
        for p, ed in (ctx.data.get("peer_events") or {}).items():
            pdf = upto(pdaily.get(p), asof)
            if ed is None or pdf is None or len(pdf) < 30:
                continue
            for ts in ed.index:
                d = ts.normalize()
                if not (asof.normalize() - pd.Timedelta(days=45) <= d < asof.normalize()):
                    continue
                rx = compute_reaction(pdf, d, infer_timing(ts, pdf), spy)
                if rx and pd.Timestamp(rx["reaction_date"]) <= asof:
                    reacts.append({"peer": p, "date": str(d.date()), "close_ret": rx["close_ret"], "excess": rx["excess"]})
        mean_react = float(np.mean([x["close_ret"] for x in reacts])) if reacts else None

        if mean_react is None and np.isnan(sector_rs20) and peer_mom is None:
            return self.abstain("no peer / sector data")
        pred = 0.0
        if mean_react is not None:
            pred += 0.35 * mean_react
        if np.isfinite(sector_rs20):
            pred += 0.10 * sector_rs20
        if peer_mom is not None:
            pred += 0.05 * peer_mom
        pred = clip(pred, -0.06, 0.06)
        conf = 0.15 + 0.08 * min(len(reacts), 3) + (0.10 if np.isfinite(sector_rs20) and abs(sector_rs20) > 0.05 else 0.0)
        ev = {"sector_etf": etf, "sector_rs_20d": None if np.isnan(sector_rs20) else sector_rs20,
              "sector_rs_5d": None if np.isnan(sector_rs5) else sector_rs5, "peer_momentum_20d_excess": peer_mom,
              "season_peer_reactions": reacts, "season_mean_reaction": mean_react, "n_peer_reports": len(reacts)}
        notes = [(f"{len(reacts)} peers reported in last 45d, mean reaction {mean_react:+.1%}" if reacts else "no peer reports in the last 45 days"),
                 (f"{etf} vs SPY 20d {sector_rs20:+.1%}" if np.isfinite(sector_rs20) else "")]
        return self.make(pred, clip(conf, 0.1, 0.6), ev, [x for x in notes if x])
