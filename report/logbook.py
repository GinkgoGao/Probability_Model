"""Every prediction is appended to outputs/logbook.jsonl. After the event, `resolve` fills in the
realized reaction and the per-dimension errors - this is the data the weight learner feeds on."""
from __future__ import annotations
import json
import pandas as pd
from config.settings import LOGBOOK_PATH, PARAMS, ensure_dirs
from data.store import dumps, clean
from data.ingest_events import compute_reaction, estimate_beta


def make_record(ticker, ctx, outputs, comp, decision, audit) -> dict:
    opt = next((o for o in outputs if o.name == "options_rnd"), None)
    ev = (opt.evidence if opt and not opt.abstain else {}) or {}
    return clean({
        "id": f"{ticker}_{pd.Timestamp(ctx.event_date).strftime('%Y-%m-%d')}_{ctx.asof.strftime('%Y%m%dT%H%M')}",
        "ticker": ticker, "event_date": pd.Timestamp(ctx.event_date).strftime("%Y-%m-%d"), "timing": ctx.timing,
        "asof": ctx.asof.isoformat(), "status": "pending",
        "dim_preds": {o.name: (None if o.abstain else o.predicted_return) for o in outputs},
        "dim_conf": {o.name: (None if o.abstain else o.confidence) for o in outputs},
        "mu": comp.mu, "sigma": comp.sigma, "p_up": comp.p_up, "conviction": comp.conviction,
        "expected_abs_move": comp.expected_abs_move, "implied_move": ev.get("implied_move"), "rn_p_up": ev.get("rn_p_up"),
        "decision": decision.get("decision"), "audit_warnings": audit.warnings,
        "realized": None, "dim_errors": None,
    })


def append(record: dict) -> None:
    ensure_dirs()
    with LOGBOOK_PATH.open("a", encoding="utf-8") as fh:
        fh.write(dumps(record) + "\n")


def load_records() -> list[dict]:
    if not LOGBOOK_PATH.exists():
        return []
    out = []
    for line in LOGBOOK_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def save_records(records: list[dict]) -> None:
    ensure_dirs()
    LOGBOOK_PATH.write_text("".join(dumps(r) + "\n" for r in records), encoding="utf-8")


def resolve_pending(fetch_daily_fn, today: pd.Timestamp | None = None, params: dict = PARAMS) -> list[dict]:
    """Fill realized outcomes for records whose reaction day has passed. Returns the resolved records."""
    today = pd.Timestamp.now() if today is None else pd.Timestamp(today)
    records = load_records()
    resolved, cache = [], {}
    for r in records:
        if r.get("status") == "resolved":
            continue
        ev = pd.Timestamp(r["event_date"])
        if today.normalize() < ev.normalize() + pd.Timedelta(days=2 if r.get("timing") == "AMC" else 1):
            continue
        t = r["ticker"]
        for key in (t, "SPY"):
            if key not in cache:
                cache[key] = fetch_daily_fn(key)
        daily, spy = cache.get(t), cache.get("SPY")
        if daily is None:
            continue
        beta = estimate_beta(daily[daily.index <= ev], spy)
        rx = compute_reaction(daily, ev, r.get("timing", "AMC"), spy, beta)
        if rx is None:
            continue
        y = rx.get(params.get("label_type", "close_ret"))
        r["realized"] = rx
        r["dim_errors"] = {k: (None if (p is None or y is None) else float(p) - float(y)) for k, p in (r.get("dim_preds") or {}).items()}
        r["composite_error"] = (float(r["mu"]) - float(y)) if y is not None else None
        r["hit"] = (bool((y > 0) == (r["p_up"] > 0.5))) if y is not None else None
        r["status"] = "resolved"
        r["resolved_at"] = today.isoformat()
        resolved.append(r)
    if resolved:
        save_records(records)
    return resolved
