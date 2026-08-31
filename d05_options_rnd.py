"""Dimension 5 - the options market. The anchor of the whole system.

Level 1  ATM straddle -> implied |move|; term structure -> pure event-jump vol (Dubinsky-Johannes)
Level 2  smile fit (quadratic in log-moneyness) -> Breeden-Litzenberger risk-neutral density
Level 3  25-delta risk reversal -> directional tilt (low confidence: RN mean is ~0 by construction)
Level 4  Q->P correction happens in the combiner (realized/implied ratio learned from the logbook)

Outputs used downstream: evidence['implied_move'], ['sigma_jump'], ['rn_p_up'].
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
from .base import Dimension, EventContext
from .utils import clip, fnum

TYPICAL_RR_REL = -0.08   # 25d puts usually ~8% of ATM vol richer than 25d calls in single stocks


def bs_call(S, K, T, r, sig):
    sig = np.maximum(np.asarray(sig, dtype=float), 1e-6)
    T = max(float(T), 1e-6)
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in ("bid", "ask", "strike", "impliedVolatility", "lastPrice"):
        if c not in d.columns:
            d[c] = np.nan
    ok = (d["bid"] > 0) & (d["ask"] > 0)
    d["mid"] = np.where(ok, (d["bid"] + d["ask"]) / 2, np.nan)
    d["spread_rel"] = np.where(ok, (d["ask"] - d["bid"]) / d["mid"], np.nan)
    d["iv"] = d["impliedVolatility"].where((d["impliedVolatility"] > 0.03) & (d["impliedVolatility"] < 4.0))
    return d.sort_values("strike")


def _atm_iv(calls, puts, F) -> float | None:
    vals = []
    for df in (calls, puts):
        d = df.dropna(subset=["iv"])
        if d.empty:
            continue
        near = d.iloc[(d["strike"] - F).abs().argsort()[:2]]
        vals += list(near["iv"].values)
    return float(np.mean(vals)) if vals else None


def _straddle(calls, puts, S) -> float | None:
    ks = np.intersect1d(calls["strike"].values, puts["strike"].values)
    if len(ks) == 0:
        return None
    K = ks[np.argmin(np.abs(ks - S))]
    c = calls.loc[calls["strike"] == K, "mid"].dropna()
    p = puts.loc[puts["strike"] == K, "mid"].dropna()
    if c.empty or p.empty:
        return None
    return float(c.iloc[0] + p.iloc[0])


def _smile_points(calls, puts, F):
    pts = []
    for df, side in ((puts, "p"), (calls, "c")):
        d = df.dropna(subset=["iv", "mid"])
        d = d[(d["spread_rel"] < 0.6) & (d["mid"] > 0.02)]
        d = d[d["strike"] < F] if side == "p" else d[d["strike"] >= F]
        for _, row in d.iterrows():
            k = np.log(row["strike"] / F)
            if abs(k) < 0.6:
                pts.append((k, row["iv"], 1.0 / (0.05 + row["spread_rel"])))
    if not pts:
        return np.array([]), np.array([]), np.array([])
    a = np.array(pts)
    return a[:, 0], a[:, 1], a[:, 2]


def _fit_smile(k, iv, w):
    if len(k) >= 5:
        poly = np.polyfit(k, iv, 2, w=w)
        if poly[0] < 0:                      # concave fit is unphysical -> fall back to linear
            poly = np.concatenate([[0.0], np.polyfit(k, iv, 1, w=w)])
        return poly
    if len(k) >= 2:
        return np.concatenate([[0.0], np.polyfit(k, iv, 1, w=w)])
    return None


def _trapz(y, x):
    f = getattr(np, "trapezoid", None) or np.trapz
    return float(f(y, x))


def _rnd(S, F, T, r, poly, sig_atm, n=801) -> dict | None:
    span = 4.0 * sig_atm * np.sqrt(T) + 0.05
    K = np.linspace(F * np.exp(-span), F * np.exp(span), n)
    k = np.log(K / F)
    iv = np.clip(np.polyval(poly, k), 0.4 * sig_atm, 2.5 * sig_atm)
    iv = np.clip(iv, 0.03, 4.0)
    C = bs_call(S, K, T, r, iv)
    dK = K[1] - K[0]
    f = np.exp(r * T) * (C[2:] - 2 * C[1:-1] + C[:-2]) / dK ** 2
    Kf = K[1:-1]
    f = np.clip(f, 0, None)
    area = _trapz(f, Kf)
    if area <= 0:
        return None
    f = f / area
    ret = Kf / S - 1
    p_up = _trapz(np.where(Kf > S, f, 0.0), Kf)
    mean_ret = _trapz(ret * f, Kf)
    sd = float(np.sqrt(max(_trapz((ret - mean_ret) ** 2 * f, Kf), 0)))
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(Kf))])
    cdf = cdf / cdf[-1]
    qs = {f"q{int(q * 100)}": float(np.interp(q, cdf, ret)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
    return {"p_up": float(p_up), "mean_ret": float(mean_ret), "sd": sd, "quantiles": qs}


def select_expiries(expiries, event_date, timing: str):
    """Front expiry = first one after the reaction; back expiry = first one >= 21 days later."""
    if event_date is None or not expiries:
        return None, None
    ev = pd.Timestamp(event_date).normalize()
    exps = sorted(pd.Timestamp(e).normalize() for e in expiries)
    front = [e for e in exps if (e > ev if timing == "AMC" else e >= ev)]
    if not front:
        return None, None
    e1 = front[0]
    back = [e for e in exps if e >= e1 + pd.Timedelta(days=21)]
    fmt = lambda e: e.strftime("%Y-%m-%d")
    return fmt(e1), (fmt(back[0]) if back else None)


def _T(exp: str, asof: pd.Timestamp) -> float:
    days = (pd.Timestamp(exp) + pd.Timedelta(hours=16) - asof).total_seconds() / 86400.0
    return max(days, 0.5) / 365.25


class OptionsRNDDimension(Dimension):
    name = "options_rnd"
    supports_backtest = False

    def score(self, ctx: EventContext):
        ch = ctx.data.get("chains")
        if not ch or not ch.get("chains"):
            return self.abstain("no options chain")
        S = fnum(ch.get("spot"))
        if not S:
            return self.abstain("no spot price")
        r = fnum(ctx.data.get("risk_free"), 0.04)
        e1, e2 = select_expiries(list(ch["chains"].keys()), ctx.event_date, ctx.timing)
        if e1 is None:
            return self.abstain("no expiry after the event")
        c1, p1 = _prep(ch["chains"][e1]["calls"]), _prep(ch["chains"][e1]["puts"])
        T1 = _T(e1, ctx.asof)
        F1 = S * np.exp(r * T1)
        iv1 = _atm_iv(c1, p1, F1)
        straddle = _straddle(c1, p1, S)
        implied_move = straddle / S if straddle else (0.8 * iv1 * np.sqrt(T1) if iv1 else None)
        if implied_move is None:
            return self.abstain("cannot price the ATM straddle")

        sigma_j = sigma_d = iv2 = T2 = None
        if e2:
            c2, p2 = _prep(ch["chains"][e2]["calls"]), _prep(ch["chains"][e2]["puts"])
            T2 = _T(e2, ctx.asof)
            iv2 = _atm_iv(c2, p2, S * np.exp(r * T2))
            if iv1 and iv2 and T2 > T1 + 0.01:
                var_d = (iv2 ** 2 * T2 - iv1 ** 2 * T1) / (T2 - T1)
                if var_d > 0:
                    sigma_d = float(np.sqrt(var_d))
                    var_j = iv1 ** 2 * T1 - var_d * T1
                    sigma_j = float(np.sqrt(var_j)) if var_j > 0 else None

        k, iv, w = _smile_points(c1, p1, F1)
        poly = _fit_smile(k, iv, w) if len(k) else None
        rnd = _rnd(S, F1, T1, r, poly, iv1) if (poly is not None and iv1) else None
        rr_rel = None
        if poly is not None and iv1:
            k25 = 0.6745 * iv1 * np.sqrt(T1)
            rr_rel = float((np.polyval(poly, k25) - np.polyval(poly, -k25)) / iv1)

        pred = 0.0
        if rr_rel is not None:
            pred += clip(0.15 * (rr_rel - TYPICAL_RR_REL), -0.03, 0.03)
        if rnd:
            pred += clip(0.3 * rnd["mean_ret"], -0.01, 0.01)
        conf = 0.30 if rr_rel is not None else 0.15
        ev = {"expiry_front": e1, "expiry_back": e2, "T1_days": T1 * 365.25, "T2_days": (T2 * 365.25 if T2 else None),
              "atm_iv_front": iv1, "atm_iv_back": iv2, "straddle_price": straddle, "implied_move": float(implied_move),
              "sigma_jump": sigma_j, "sigma_diffusive": sigma_d, "rr25_rel": rr_rel,
              "rn_p_up": rnd["p_up"] if rnd else None, "rn_mean": rnd["mean_ret"] if rnd else None,
              "rn_sd": rnd["sd"] if rnd else None, "rn_quantiles": rnd["quantiles"] if rnd else None,
              "smile_points": int(len(k)), "spot": S, "risk_free": r}
        notes = [f"implied move (straddle/spot) {implied_move:.1%} to {e1}",
                 (f"event-jump vol {sigma_j:.1%} (diffusive {sigma_d:.0%} ann.)" if sigma_j else "term-structure decomposition unavailable"),
                 (f"risk-neutral P(up) {rnd['p_up']:.0%}, RN sd {rnd['sd']:.1%}" if rnd else "RND unavailable"),
                 (f"25d risk reversal {rr_rel:+.1%} of ATM vol (typical {TYPICAL_RR_REL:+.0%})" if rr_rel is not None else "")]
        return self.make(pred, conf, ev, [x for x in notes if x])
