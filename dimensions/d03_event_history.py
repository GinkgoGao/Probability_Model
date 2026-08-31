"""Dimension 3 - this name's own earnings history: base rate, magnitude, 'beat-but-fell' pattern,
gap-and-go vs gap-and-fade. Uses only events strictly before ctx.asof (point-in-time)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .base import Dimension, EventContext
from .utils import clip, upto, fnum
from data.ingest_events import past_events, infer_timing, compute_reaction, estimate_beta


class EventHistoryDimension(Dimension):
    name = "event_history"
    supports_backtest = True

    def score(self, ctx: EventContext):
        ed = ctx.data.get("earnings_dates")
        daily = upto(ctx.data.get("daily"), ctx.asof)
        spy = upto(ctx.data.get("spy_daily"), ctx.asof)
        if ed is None or daily is None or len(daily) < 30:
            return self.abstain("no earnings history / prices")
        cutoff = ctx.event_date if ctx.event_date is not None else ctx.asof
        past = past_events(ed, cutoff)
        beta = estimate_beta(daily, spy)
        rows = []
        for ts, row in past.iterrows():
            timing = infer_timing(ts, daily)
            rx = compute_reaction(daily, ts, timing, spy, beta)
            if rx is None or pd.Timestamp(rx["reaction_date"]) > ctx.asof:
                continue
            sur = fnum(row.get("Surprise(%)")) if hasattr(row, "get") else None
            rows.append({"date": str(ts.date()), "timing": timing, "gap": rx["gap"], "close_ret": rx["close_ret"],
                         "intraday": rx["intraday"], "excess": rx["excess"], "surprise_pct": sur})
        if len(rows) < 3:
            return self.abstain(f"only {len(rows)} usable past events")
        rows.sort(key=lambda r: r["date"])
        rets = np.array([r["close_ret"] for r in rows])
        gaps = np.array([r["gap"] for r in rows])
        intr = np.array([r["intraday"] for r in rows])
        n = len(rets)
        mean, med, std = float(rets.mean()), float(np.median(rets)), float(rets.std(ddof=1))
        hit_up = float((rets > 0).mean())
        mean_abs = float(np.abs(rets).mean())
        last4 = float(rets[-4:].mean())
        beats = [r for r in rows if r["surprise_pct"] is not None and r["surprise_pct"] > 0]
        beat_fell = float(np.mean([r["close_ret"] < 0 for r in beats])) if beats else None
        gap_cont = float(np.mean(np.sign(intr) == np.sign(gaps))) if n else None

        shrink = n / (n + 6.0)
        pred = clip(shrink * (0.7 * mean + 0.3 * last4), -0.10, 0.10)
        conf = clip(0.15 + 0.04 * n, 0.15, 0.55)
        ev = {"n_events": n, "mean": mean, "median": med, "std": std, "hit_rate_up": hit_up, "mean_abs_move": mean_abs,
              "last4_mean": last4, "beat_but_fell_rate": beat_fell, "n_beats": len(beats),
              "gap_continuation_rate": gap_cont, "beta": beta, "recent_events": rows[-8:]}
        notes = [f"{n} past events: mean {mean:+.1%}, median {med:+.1%}, up {hit_up:.0%} of the time",
                 f"mean |move| {mean_abs:.1%}, std {std:.1%}",
                 (f"beat-but-fell in {beat_fell:.0%} of {len(beats)} beats" if beat_fell is not None else "no surprise data"),
                 (f"gap continued intraday {gap_cont:.0%} of the time" if gap_cont is not None else "")]
        return self.make(pred, conf, ev, [x for x in notes if x])
