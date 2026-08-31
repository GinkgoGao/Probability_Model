"""Event study over historical earnings: computes every backtestable dimension as-of each event
(point-in-time slices), the realized reaction, and per-dimension IC / hit rate / MAE.

Only price-derived dimensions can be backtested from free data (technical, event_history,
peer_sector, macro_regime). Options, expectations, sentiment and positioning are forward-logged."""
from __future__ import annotations
import numpy as np
import pandas as pd
from config.settings import PARAMS, OUTPUT_DIRS, ensure_dirs
from config.universe import get_peers
from data.ingest_prices import fetch_daily, fetch_many_daily
from data.ingest_events import fetch_earnings_dates, past_events, infer_timing, compute_reaction, estimate_beta
from data.ingest_macro import fetch_macro, load_macro_calendar
from dimensions import get_dimensions, EventContext
from model.calibration import spearman_ic, hit_rate


def fetch_bundle(ticker: str) -> dict | None:
    daily = fetch_daily(ticker, PARAMS["backtest_daily_period"])
    ed = fetch_earnings_dates(ticker, 60)
    if daily is None or ed is None:
        return None
    peers, etf = get_peers(ticker)
    peer_daily = fetch_many_daily(peers + [etf, "SPY"], PARAMS["backtest_daily_period"])
    peer_events = {p: fetch_earnings_dates(p, 60) for p in peers}
    return {"daily": daily, "earnings_dates": ed, "peers": peers, "sector_etf": etf, "peer_daily": peer_daily,
            "peer_events": peer_events, "spy_daily": peer_daily.get("SPY"),
            "macro": fetch_macro(period=PARAMS["backtest_daily_period"]), "macro_calendar": load_macro_calendar()}


def build_backtest_context(ticker: str, event_ts: pd.Timestamp, bundle: dict) -> tuple[EventContext, dict | None]:
    daily = bundle["daily"]
    timing = infer_timing(event_ts, daily)
    ev = event_ts.normalize()
    # information cutoff: close of event day for AMC (report comes after the close), previous close for BMO
    asof = ev + pd.Timedelta(hours=16) if timing == "AMC" else ev - pd.Timedelta(hours=8)
    data = {k: bundle[k] for k in ("earnings_dates", "peers", "sector_etf", "peer_daily", "peer_events", "spy_daily", "macro", "macro_calendar")}
    data["daily"] = daily[daily.index <= asof]
    data["intraday"] = None; data["chains"] = None; data["fundamentals"] = None; data["news"] = None
    ctx = EventContext(ticker=ticker, event_date=ev, timing=timing, asof=asof, mode="backtest", data=data)
    beta = estimate_beta(data["daily"], bundle["spy_daily"])
    realized = compute_reaction(daily, ev, timing, bundle["spy_daily"], beta)
    return ctx, realized


def run_event_study(tickers: list[str], max_events: int | None = None, verbose: bool = True) -> pd.DataFrame:
    ensure_dirs()
    max_events = max_events or PARAMS["backtest_max_events_per_ticker"]
    dims = get_dimensions(backtest_only=True)
    rows = []
    for t in [x.upper() for x in tickers]:
        b = fetch_bundle(t)
        if b is None:
            print(f"[{t}] no data, skipped"); continue
        events = past_events(b["earnings_dates"], pd.Timestamp.now() - pd.Timedelta(days=3)).index[-max_events:]
        n_ok = 0
        for ts in events:
            ctx, rx = build_backtest_context(t, ts, b)
            if rx is None or len(ctx.data["daily"]) < 120:
                continue
            n_ok += 1
            for d in dims:
                o = d.safe_score(ctx)
                rows.append({"ticker": t, "event_date": str(ts.date()), "timing": ctx.timing, "dim": o.name,
                             "pred": None if o.abstain else o.predicted_return, "conf": None if o.abstain else o.confidence,
                             "abstain": o.abstain, "sigma_mult": o.sigma_multiplier,
                             "realized_close_ret": rx["close_ret"], "realized_gap": rx["gap"], "realized_excess": rx["excess"]})
        if verbose:
            print(f"[{t}] {n_ok} events scored")
    df = pd.DataFrame(rows)
    if df.empty:
        print("no events"); return df
    df.to_csv(OUTPUT_DIRS["backtest"] / "event_study_long.csv", index=False)
    label = {"close_ret": "realized_close_ret", "excess": "realized_excess", "gap": "realized_gap"}[PARAMS["label_type"]]
    summ = []
    for name, g in df[~df["abstain"]].groupby("dim"):
        summ.append({"dim": name, "n": len(g), "ic_spearman": spearman_ic(g["pred"], g[label]), "hit_rate": hit_rate(g["pred"], g[label]),
                     "mae": float((g["pred"] - g[label]).abs().mean()), "mean_pred": float(g["pred"].mean()),
                     "mean_realized": float(g[label].mean()), "abstain_rate": 1 - len(g) / len(df[df["dim"] == name])})
    s = pd.DataFrame(summ).sort_values("ic_spearman", ascending=False)
    s.to_csv(OUTPUT_DIRS["backtest"] / "event_study_summary.csv", index=False)
    if verbose:
        print("\nPer-dimension diagnostics (label = %s):" % PARAMS["label_type"])
        print(s.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print("\nRule: a dimension with IC <= 0 over >= 100 events does not earn a place in the combiner.")
    return df
