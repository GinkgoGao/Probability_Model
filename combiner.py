"""Turn ten dimension outputs into one distribution.

    mu     = shrink * weighted mean of predicted returns   (weight_i = prior_i * confidence_i, abstainers excluded)
    sigma  = options-implied event vol * (realized/implied ratio) * regime/positioning multipliers
    P(up)  = 1 - CDF(0)   under Normal(mu, sigma) (or Student-t)

The shrinkage is the explicit 'market already knows' penalty for double-counted information.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
import numpy as np
from scipy import stats as st
from data.store import clean


def bucket_labels(edges):
    e = [f"{x:+.0%}" for x in edges]
    return [f"< {e[0]}"] + [f"{e[i]} to {e[i + 1]}" for i in range(len(e) - 1)] + [f"> {e[-1]}"]


def make_dist(mu: float, sigma: float, kind: str = "normal", df: int = 4):
    if kind == "t":
        scale = sigma * math.sqrt((df - 2) / df)
        return st.t(df, loc=mu, scale=scale)
    return st.norm(loc=mu, scale=sigma)


def expected_abs_move(dist, sigma: float) -> float:
    x = np.linspace(-10 * sigma, 10 * sigma, 8001)
    y = np.abs(x) * dist.pdf(x)
    f = getattr(np, "trapezoid", None) or np.trapz
    return float(f(y, x))


@dataclass
class Composite:
    mu_raw: float
    mu: float
    sigma: float
    sigma_base: float
    sigma_source: str
    ratio_used: float
    sigma_multiplier: float
    p_up: float
    conviction: float
    expected_abs_move: float
    agreement: float | None
    buckets: list = field(default_factory=list)
    quantiles: dict = field(default_factory=dict)
    weights_used: dict = field(default_factory=dict)
    n_active: int = 0
    n_abstain: int = 0
    distribution: str = "normal"
    shrinkage: float = 0.0

    def to_dict(self) -> dict:
        return clean(asdict(self))


def combine(outputs, weights: dict, params: dict, implied_move=None, sigma_jump=None,
            hist_std=None, hist_abs_mean=None, ratio=None) -> Composite:
    active = [o for o in outputs if not o.abstain]
    eff = {o.name: max(float(weights.get(o.name, 0.0)), 0.0) * o.confidence for o in active}
    tot = sum(eff.values())
    mu_raw = sum(eff[o.name] * o.predicted_return for o in active) / tot if tot > 0 else 0.0
    shrink = float(params.get("shrinkage", 0.0))
    mu = (1.0 - shrink) * mu_raw

    ratio_used = float(ratio or params.get("realized_to_implied_ratio", 1.0))
    if sigma_jump and sigma_jump > 0.005:
        base, src = float(sigma_jump), "term-structure event-jump vol"
    elif implied_move and implied_move > 0.005:
        base, src = float(implied_move) / 0.8, "ATM straddle implied move"
    elif hist_std and hist_std > 0.005:
        base, src, ratio_used = float(hist_std), "historical reaction std (no options)", 1.0
    elif hist_abs_mean and hist_abs_mean > 0.005:
        base, src, ratio_used = float(hist_abs_mean) / 0.8, "historical mean |move| (no options)", 1.0
    else:
        base, src, ratio_used = float(params.get("sigma_fallback", 0.08)), "fallback constant", 1.0
    # Regime / positioning / ATR multipliers are combined additively, not multiplied, and are damped
    # to 30% when sigma already comes from the options market (which prices those regimes itself).
    options_based = src.startswith(("term-structure", "ATM straddle"))
    damp = 0.3 if options_based else 1.0
    excess = sum(o.sigma_multiplier - 1.0 for o in active)
    mult = min(max(1.0 + damp * excess, 0.8), float(params.get("sigma_multiplier_cap", 1.8)))
    sigma = max(base * ratio_used * mult, 0.005)

    dist = make_dist(mu, sigma, params.get("distribution", "normal"), int(params.get("t_df", 4)))
    p_up = float(1.0 - dist.cdf(0.0))
    edges = list(params.get("bucket_edges", [-0.15, -0.08, -0.03, 0.03, 0.08, 0.15]))
    cuts = [-np.inf] + edges + [np.inf]
    cdf = dist.cdf(np.array(cuts[1:-1]))
    probs = np.diff(np.concatenate([[0.0], cdf, [1.0]]))
    buckets = [{"label": lab, "lo": (None if np.isinf(cuts[i]) else cuts[i]), "hi": (None if np.isinf(cuts[i + 1]) else cuts[i + 1]),
                "prob": float(p)} for i, (lab, p) in enumerate(zip(bucket_labels(edges), probs))]
    quantiles = {f"q{int(q * 100)}": float(dist.ppf(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
    signs = [o.direction for o in active if o.direction != 0]
    agreement = (float(np.mean([s == np.sign(mu) for s in signs])) if signs and abs(mu) > 1e-9 else None)
    conviction = float(100.0 * math.tanh(mu / sigma))
    return Composite(mu_raw=mu_raw, mu=mu, sigma=sigma, sigma_base=base, sigma_source=src, ratio_used=ratio_used,
                     sigma_multiplier=mult, p_up=p_up, conviction=conviction, expected_abs_move=expected_abs_move(dist, sigma),
                     agreement=agreement, buckets=buckets, quantiles=quantiles,
                     weights_used={k: v / tot for k, v in eff.items()} if tot > 0 else {},
                     n_active=len(active), n_abstain=len(outputs) - len(active),
                     distribution=params.get("distribution", "normal"), shrinkage=shrink)
