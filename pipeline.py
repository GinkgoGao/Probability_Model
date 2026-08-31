"""Orchestration: ticker -> data -> audit -> dimensions -> composite -> decision -> report -> logbook.
The same dimension code is used for live prediction and for the backtest (no dual implementations)."""
from __future__ import annotations
import pandas as pd
from config.settings import PARAMS, OUTPUT_DIRS, ensure_dirs
from config.universe import get_peers
from data.store import dumps
from data.ingest_prices import fetch_daily, fetch_intraday, fetch_many_daily
from data.ingest_options import fetch_chains, get_risk_free
from data.ingest_events import fetch_earnings_dates, upcoming_event, infer_timing
from data.ingest_fundamentals import fetch_fundamentals
from data.ingest_text import fetch_news
from data.ingest_macro import fetch_macro, load_macro_calendar
from dimensions import get_dimensions, EventContext
from quality.audit_gates import run_audit
from model.combiner import combine
from model.weights import load_weights
from decision.ev_engine import evaluate
from report.render import render_report
from report import logbook


def build_live_context(ticker: str, event_date=None, timing: str | None = None, overrides: dict | None = None,
                       asof=None, fetch_options: bool = True) -> EventContext:
    ticker = ticker.upper()
    asof = pd.Timestamp.now() if asof is None else pd.Timestamp(asof)
    print(f"[{ticker}] fetching prices ...")
    daily = fetch_daily(ticker, PARAMS["daily_period"])
    intraday = fetch_intraday(ticker, PARAMS["intraday_period"], PARAMS["intraday_interval"])
    print(f"[{ticker}] fetching earnings calendar ...")
    ed = fetch_earnings_dates(ticker, PARAMS["earnings_history_limit"])
    guess_date, guess_timing = upcoming_event(ed, asof)
    if event_date is not None:
        event_date = pd.Timestamp(event_date).normalize()
        if timing is None and ed is not None:
            hits = [ts for ts in ed.index if ts.normalize() == event_date]
            timing = infer_timing(hits[0], None) if hits else None
    else:
        event_date = guess_date
        timing = timing or guess_timing
    timing = timing or "AMC"
    print(f"[{ticker}] event {event_date.date() if event_date is not None else '?'} {timing}; fetching fundamentals, news, peers, macro ...")
    fundamentals = fetch_fundamentals(ticker, asof)
    news = fetch_news(ticker, asof)
    peers, etf = get_peers(ticker, fundamentals.get("info"))
    peer_daily = fetch_many_daily(peers + [etf, "SPY"], PARAMS["daily_period"])
    peer_events = {p: fetch_earnings_dates(p, 12) for p in peers}
    macro = fetch_macro()
    chains = None
    if fetch_options:
        print(f"[{ticker}] fetching options chains ...")
        chains = fetch_chains(ticker, asof, PARAMS["max_option_expiries"], daily=daily)
    data = {"daily": daily, "intraday": intraday, "earnings_dates": ed, "fundamentals": fundamentals, "news": news,
            "peers": peers, "sector_etf": etf, "peer_daily": peer_daily, "peer_events": peer_events,
            "spy_daily": peer_daily.get("SPY"), "macro": macro, "macro_calendar": load_macro_calendar(),
            "chains": chains, "risk_free": get_risk_free(PARAMS["risk_free_fallback"]) if fetch_options else PARAMS["risk_free_fallback"]}
    return EventContext(ticker=ticker, event_date=event_date, timing=timing, asof=asof, mode="live", data=data, overrides=overrides or {})


def score_context(ctx: EventContext, dims=None, weights: dict | None = None, ratio: float | None = None):
    """Run dimensions + combiner on an already-built context. Returns (outputs, composite)."""
    w = weights or load_weights()["weights"]
    outputs = [d.safe_score(ctx) for d in (dims or get_dimensions())]
    opt = next((o for o in outputs if o.name == "options_rnd" and not o.abstain), None)
    eh = next((o for o in outputs if o.name == "event_history" and not o.abstain), None)
    comp = combine(outputs, w, PARAMS,
                   implied_move=opt.evidence.get("implied_move") if opt else None,
                   sigma_jump=opt.evidence.get("sigma_jump") if opt else None,
                   hist_std=eh.evidence.get("std") if eh else None,
                   hist_abs_mean=eh.evidence.get("mean_abs_move") if eh else None, ratio=ratio)
    return outputs, comp


def run_prediction(ticker: str, event_date=None, timing=None, overrides=None, asof=None, require_options=True,
                   save=True, dims=None) -> dict:
    ensure_dirs()
    ticker = ticker.upper()
    ctx = build_live_context(ticker, event_date, timing, overrides, asof, fetch_options=True)
    audit = run_audit(ctx, require_options=require_options)
    if not audit.passed:
        print("\nAUDIT FAILED - refusing to output a probability:")
        for h in audit.hard_failures:
            print("  x", h)
        for w in audit.warnings:
            print("  !", w)
        return {"status": "REFUSED", "ticker": ticker, "audit": audit.to_dict()}
    wdoc = load_weights()
    outputs, comp = score_context(ctx, dims, wdoc["weights"], (wdoc.get("meta") or {}).get("realized_to_implied_ratio"))
    opt = next((o for o in outputs if o.name == "options_rnd" and not o.abstain), None)
    decision = evaluate(comp, opt.evidence.get("implied_move") if opt else None, opt.evidence.get("rn_p_up") if opt else None, PARAMS)
    report = render_report(ticker, ctx, outputs, comp, decision, audit, wdoc["weights"], PARAMS)
    result = {"status": "OK", "ticker": ticker, "event_date": str(pd.Timestamp(ctx.event_date).date()), "timing": ctx.timing,
              "asof": ctx.asof.isoformat(), "audit": audit.to_dict(), "dimensions": [o.to_dict() for o in outputs],
              "composite": comp.to_dict(), "decision": decision, "weights": wdoc["weights"]}
    if save:
        tag = f"{ticker}_{pd.Timestamp(ctx.event_date).strftime('%Y-%m-%d')}_asof{ctx.asof.strftime('%Y%m%dT%H%M')}"
        rp = OUTPUT_DIRS["reports"] / f"{tag}.md"
        jp = OUTPUT_DIRS["predictions"] / f"{tag}.json"
        rp.write_text(report, encoding="utf-8")
        jp.write_text(dumps(result), encoding="utf-8")
        logbook.append(logbook.make_record(ticker, ctx, outputs, comp, decision, audit))
        result["report_path"], result["json_path"] = str(rp), str(jp)
    print(report)
    return result
