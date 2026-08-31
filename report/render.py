"""Markdown report: scorecard, composite, distribution, decision, self-attack."""
from __future__ import annotations
import pandas as pd


def _bar(p: float, width: int = 30) -> str:
    return "#" * int(round(p * width))


def render_report(ticker, ctx, outputs, comp, decision, audit, weights, params) -> str:
    L = []
    L.append(f"# {ticker} - post-earnings reaction forecast")
    L.append(f"- Event: {pd.Timestamp(ctx.event_date).date() if ctx.event_date is not None else 'n/a'} ({ctx.timing})   |   as of: {ctx.asof.strftime('%Y-%m-%d %H:%M')}   |   mode: {ctx.mode}")
    L.append(f"- Audit: {'PASSED' if audit.passed else 'FAILED'}" + (f" - warnings: {len(audit.warnings)}" if audit.warnings else ""))
    for w in audit.warnings:
        L.append(f"  - warning: {w}")
    L.append("")
    L.append("## Dimension scorecard")
    L.append("| Dimension | Predicted move | Dir | Confidence | Prior w | Effective w | Key evidence |")
    L.append("|---|---:|:-:|---:|---:|---:|---|")
    for o in outputs:
        ew = comp.weights_used.get(o.name, 0.0)
        note = (o.notes[0] if o.notes else "").replace("|", "/")
        if o.abstain:
            L.append(f"| {o.name} | abstain | - | - | {weights.get(o.name, 0):.2f} | 0.00 | {note} |")
        else:
            d = {1: "UP", -1: "DOWN", 0: "flat"}[o.direction]
            L.append(f"| {o.name} | {o.predicted_return:+.1%} | {d} | {o.confidence:.2f} | {weights.get(o.name, 0):.2f} | {ew:.2f} | {note} |")
    L.append("")
    L.append("## Composite")
    L.append(f"- Expected move mu = **{comp.mu:+.2%}** (raw weighted mean {comp.mu_raw:+.2%}, shrinkage {comp.shrinkage:.0%})")
    L.append(f"- Sigma = **{comp.sigma:.1%}** from {comp.sigma_source} ({comp.sigma_base:.1%}) x ratio {comp.ratio_used:.2f} x multipliers {comp.sigma_multiplier:.2f}")
    L.append(f"- P(up) = **{comp.p_up:.1%}**   |   conviction score (-100..100) = {comp.conviction:+.0f}   |   dimension agreement = {('%.0f%%' % (100 * comp.agreement)) if comp.agreement is not None else 'n/a'}")
    L.append(f"- Model E|move| = {comp.expected_abs_move:.1%}" + (f" vs options implied {decision['implied_move']:.1%}" if decision.get('implied_move') else " (no options benchmark)"))
    L.append(f"- Active dimensions: {comp.n_active}, abstaining: {comp.n_abstain}; distribution: {comp.distribution}")
    L.append("")
    L.append("## Probability distribution of the reaction")
    L.append("| Bucket | Prob | |")
    L.append("|---|---:|---|")
    for b in comp.buckets:
        L.append(f"| {b['label']} | {b['prob']:.1%} | {_bar(b['prob'])} |")
    q = comp.quantiles
    L.append(f"\nQuantiles: 5% {q['q5']:+.1%} | 25% {q['q25']:+.1%} | 50% {q['q50']:+.1%} | 75% {q['q75']:+.1%} | 95% {q['q95']:+.1%}")
    L.append("")
    L.append("## Decision")
    L.append(f"**{decision['decision']}**" + (f" - {decision['structure']}" if decision.get("structure") else ""))
    if decision.get("size_pct_of_account"):
        L.append(f"- suggested size: {decision['size_pct_of_account']:.1%} of account (quarter-Kelly, capped)")
    for r in decision.get("reasons", []):
        L.append(f"- {r}")
    L.append("")
    L.append("## Self-attack (how this forecast could be wrong)")
    for o in outputs:
        if o.abstain:
            L.append(f"- {o.name} abstained: {o.notes[0] if o.notes else ''}")
    if comp.agreement is not None and comp.agreement < 0.6:
        L.append(f"- Only {comp.agreement:.0%} of active dimensions agree on direction: the composite sign is fragile.")
    for o in outputs:
        ev = o.evidence or {}
        if ev.get("calendar_collisions"):
            L.append(f"- Macro calendar collision {ev['calendar_collisions']}: the reaction will be contaminated by macro flows.")
        if ev.get("stress_regime"):
            L.append("- Stress regime (VIX elevated / inverted term structure): historical base rates may not apply.")
        if ev.get("high_bar"):
            L.append("- Consensus sits above management guidance: a beat may already be required just to hold the price.")
        if ev.get("beat_but_fell_rate") is not None and ev["beat_but_fell_rate"] >= 0.4:
            L.append(f"- This name fell after {ev['beat_but_fell_rate']:.0%} of its beats: 'good numbers' is not the same as 'up'.")
        if ev.get("n_events") is not None and ev["n_events"] < 8:
            L.append(f"- Only {ev['n_events']} historical events: the base rate is statistically weak.")
    L.append("- The options market already embeds most of these dimensions; the composite is shrunk toward zero for that reason. Your real benchmark is the risk-neutral P(up), not 50/50.")
    L.append("- Weights are uncalibrated until the logbook has enough resolved events (see `python run.py show-weights`).")
    L.append("- Not investment advice. This is a measurement instrument whose first job is to say NO TRADE.")
    return "\n".join(L)
