"""Audit gates - the pipeline REFUSES to output a probability if a hard gate fails.
No degraded outputs: a number produced from bad data is worse than no number."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import pandas as pd
from dimensions.base import EventContext
from dimensions.d05_options_rnd import select_expiries


@dataclass
class AuditResult:
    passed: bool
    hard_failures: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def run_audit(ctx: EventContext, require_options: bool = True) -> AuditResult:
    hard, warn = [], []
    live = ctx.mode == "live"
    daily = ctx.data.get("daily")
    if daily is None or len(daily) < 60:
        hard.append("daily price history missing or shorter than 60 bars")
    else:
        age = (ctx.asof.normalize() - daily.index[-1].normalize()).days
        if live and age > 5:
            hard.append(f"daily prices stale by {age} days")
        if daily["Close"].iloc[-20:].isna().any() or (daily["Close"].iloc[-20:] <= 0).any():
            hard.append("bad prints (NaN / non-positive close) in the last 20 bars")
        if live and len(daily) > 2 and abs(daily["Close"].iloc[-1] / daily["Close"].iloc[-2] - 1) > 0.5:
            warn.append("last daily return > 50%: check for a split / bad print")
    if ctx.event_date is None:
        hard.append("no earnings date (pass --date YYYY-MM-DD)")
    elif live:
        gap = (pd.Timestamp(ctx.event_date).normalize() - ctx.asof.normalize()).days
        if gap < 0:
            hard.append("event date is in the past - use resolve, not predict")
        elif gap > 10:
            warn.append(f"event is {gap} days away: options/IV signals are immature")
    if ctx.timing not in ("AMC", "BMO"):
        warn.append(f"unknown timing '{ctx.timing}', assuming AMC")
    if live:
        ch = ctx.data.get("chains")
        if not ch or not ch.get("chains"):
            (hard if require_options else warn).append("no options chain (use --no-options to run without the benchmark)")
        else:
            e1, e2 = select_expiries(list(ch["chains"].keys()), ctx.event_date, ctx.timing)
            if e1 is None:
                (hard if require_options else warn).append("no option expiry after the event date")
            elif e2 is None:
                warn.append("no back expiry >= 21d after the front: event-jump vol cannot be isolated")
        if ctx.data.get("intraday") is None:
            warn.append("no intraday bars - microstructure dimension will abstain")
        if not ctx.data.get("fundamentals"):
            warn.append("no fundamentals - expectation/positioning dimensions will abstain")
        if len(ctx.data.get("news") or []) < 3:
            warn.append("fewer than 3 headlines - sentiment dimension will abstain")
        if not ctx.data.get("macro"):
            warn.append("no macro data - regime dimension will abstain")
    ed = ctx.data.get("earnings_dates")
    if ed is None or len(ed[ed.index < ctx.asof]) < 4:
        warn.append("fewer than 4 historical earnings events - base rates are weak")
    return AuditResult(passed=not hard, hard_failures=hard, warnings=warn)
