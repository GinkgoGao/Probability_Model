"""Trade decision layer. The model's number is compared with the market's number:
    directional edge = P(up)_model - P(up)_risk-neutral
    volatility edge  = E|move|_model / implied_move - 1
Below thresholds -> NO TRADE. Sizing is quarter-Kelly capped at max_position_pct."""
from __future__ import annotations


def evaluate(comp, implied_move: float | None, rn_p_up: float | None, params: dict) -> dict:
    th_d, th_v, cap = params["no_trade_dir_edge"], params["no_trade_vol_edge"], params["max_position_pct"]
    rn = rn_p_up if rn_p_up is not None else 0.5
    dir_edge = comp.p_up - rn
    vol_edge = (comp.expected_abs_move / implied_move - 1.0) if implied_move else None
    reasons, structure, size = [], None, 0.0
    if implied_move is None:
        decision = "NO TRADE"
        reasons.append("no options benchmark: the model cannot be compared with the market price of the event")
    else:
        strong_dir = abs(dir_edge) >= th_d
        strong_vol = vol_edge is not None and abs(vol_edge) >= th_v
        if not strong_dir and not strong_vol:
            decision = "NO TRADE"
            reasons.append(f"directional edge {dir_edge:+.1%} < {th_d:.0%} and vol edge {vol_edge:+.0%} < {th_v:.0%}: the ticket price already reflects the view")
        else:
            decision = "TRADE CANDIDATE"
            if strong_dir:
                structure = "debit call vertical (buy ATM / sell +1σ)" if dir_edge > 0 else "debit put vertical (buy ATM / sell -1σ)"
                size = min(cap, 0.5 * abs(dir_edge))
                reasons.append(f"directional edge {dir_edge:+.1%} vs market {rn:.0%} up")
            if strong_vol:
                vs = "long straddle / strangle (model expects a bigger move than implied)" if vol_edge > 0 else \
                     "iron condor / short strangle with defined risk (model expects a smaller move)"
                structure = f"{structure} + {vs}" if structure else vs
                size = max(size, min(cap, 0.05 * abs(vol_edge)))
                reasons.append(f"model E|move| {comp.expected_abs_move:.1%} vs implied {implied_move:.1%} ({vol_edge:+.0%})")
    if comp.agreement is not None and comp.agreement < 0.5 and decision != "NO TRADE":
        reasons.append(f"WARNING: only {comp.agreement:.0%} of active dimensions agree with the composite sign")
    return {"decision": decision, "structure": structure, "size_pct_of_account": size, "dir_edge": dir_edge,
            "vol_edge": vol_edge, "model_abs_move": comp.expected_abs_move, "implied_move": implied_move,
            "rn_p_up": rn_p_up, "reasons": reasons}
