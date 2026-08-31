"""Command-line entry point.

  python run.py predict MU                       # next scheduled earnings, all dimensions
  python run.py predict MU --date 2026-09-23 --timing AMC --override consensus_eps=3.1 guidance_mid_eps=2.9
  python run.py predict MU --no-options          # run without the options benchmark (audit becomes a warning)
  python run.py audit MU                         # data quality only
  python run.py resolve                          # fill realized outcomes for past predictions
  python run.py update-weights                   # learn weights from resolved predictions
  python run.py show-weights
  python run.py backtest --tickers MU AMD NVDA --max-events 24
  python run.py walkforward
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _overrides(items):
    out = {}
    for it in items or []:
        if "=" in it:
            k, v = it.split("=", 1)
            try:
                out[k.strip()] = float(v)
            except ValueError:
                out[k.strip()] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Event-driven earnings reaction model")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict"); p.add_argument("ticker"); p.add_argument("--date"); p.add_argument("--timing", choices=["AMC", "BMO"])
    p.add_argument("--no-options", action="store_true"); p.add_argument("--override", nargs="*"); p.add_argument("--no-save", action="store_true")
    a = sub.add_parser("audit"); a.add_argument("ticker"); a.add_argument("--date"); a.add_argument("--timing", choices=["AMC", "BMO"])
    sub.add_parser("resolve")
    sub.add_parser("update-weights")
    sub.add_parser("show-weights")
    b = sub.add_parser("backtest"); b.add_argument("--tickers", nargs="+", required=True); b.add_argument("--max-events", type=int)
    w = sub.add_parser("walkforward"); w.add_argument("--min-train", type=int)
    args = ap.parse_args(argv)

    if args.cmd == "predict":
        from pipeline import run_prediction
        run_prediction(args.ticker, args.date, args.timing, _overrides(args.override), require_options=not args.no_options, save=not args.no_save)
    elif args.cmd == "audit":
        from pipeline import build_live_context
        from quality.audit_gates import run_audit
        ctx = build_live_context(args.ticker, args.date, args.timing)
        r = run_audit(ctx)
        print("PASSED" if r.passed else "FAILED")
        for h in r.hard_failures: print("  x", h)
        for x in r.warnings: print("  !", x)
    elif args.cmd == "resolve":
        from report.logbook import resolve_pending
        from data.ingest_prices import fetch_daily
        res = resolve_pending(lambda t: fetch_daily(t, "1y", use_cache=False))
        for r in res:
            y = r["realized"]
            print(f"{r['ticker']} {r['event_date']}: predicted mu {r['mu']:+.1%} P(up) {r['p_up']:.0%} | realized gap {y['gap']:+.1%} close {y['close_ret']:+.1%} | hit={r['hit']}")
        print(f"{len(res)} prediction(s) resolved")
    elif args.cmd == "update-weights":
        from report.logbook import load_records
        from model.weights import update_weights_from_logbook
        doc = update_weights_from_logbook(load_records())
        print(json.dumps(doc["weights"], indent=2))
        print("resolved events:", doc["meta"]["n_resolved"], "| learned realized/implied ratio:", doc["meta"]["realized_to_implied_ratio"])
        for k, v in doc["meta"]["dim_stats"].items():
            print(f"  {k:>16}: n={v['n']:3d} " + (f"rmse={v['rmse']:.3f} mean_err={v['mean_err']:+.3f}" if v.get("updated") else "(prior kept)"))
    elif args.cmd == "show-weights":
        from model.weights import load_weights
        doc = load_weights()
        print(json.dumps(doc, indent=2, default=str))
    elif args.cmd == "backtest":
        from backtest.event_study import run_event_study
        run_event_study(args.tickers, args.max_events)
    elif args.cmd == "walkforward":
        from backtest.walkforward import run_walkforward
        run_walkforward(min_train=args.min_train)


if __name__ == "__main__":
    main()
