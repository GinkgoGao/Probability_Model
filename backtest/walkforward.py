"""Walk-forward evaluation of the combiner on the event-study table:
weights are re-fit (inverse-MSE, same code as production) on all PAST events only, then applied
to the next event. Reports MAE / hit rate / Brier of the composite vs naive baselines."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
from config.settings import PARAMS, OUTPUT_DIRS
from model.weights import load_prior, inverse_mse_weights
from model.calibration import brier, log_loss, hit_rate, reliability_table


def run_walkforward(long_csv=None, min_train: int | None = None, verbose: bool = True) -> pd.DataFrame:
    long_csv = long_csv or OUTPUT_DIRS["backtest"] / "event_study_long.csv"
    df = pd.read_csv(long_csv)
    label = {"close_ret": "realized_close_ret", "excess": "realized_excess", "gap": "realized_gap"}[PARAMS["label_type"]]
    piv = df.pivot_table(index=["event_date", "ticker"], columns="dim", values="pred", aggfunc="first")
    conf = df.pivot_table(index=["event_date", "ticker"], columns="dim", values="conf", aggfunc="first")
    y = df.groupby(["event_date", "ticker"])[label].first()
    piv, conf = piv.loc[y.index], conf.loc[y.index]
    order = np.argsort(piv.index.get_level_values(0).values, kind="stable")
    piv, conf, y = piv.iloc[order], conf.iloc[order], y.iloc[order]
    prior = load_prior()
    dims = [d for d in piv.columns if d in prior]
    min_train = min_train or PARAMS["walkforward_min_train"]
    out = []
    for i in range(min_train, len(piv)):
        past_p, past_y = piv.iloc[:i], y.iloc[:i]
        errs = {d: (past_p[d] - past_y).dropna().tolist() for d in dims}
        w, _ = inverse_mse_weights(errs, {d: prior[d] for d in dims}, PARAMS["weights_alpha"], PARAMS["weights_half_life"],
                                   PARAMS["weights_floor"], PARAMS["weights_cap"], PARAMS["weights_min_events"])
        row_p, row_c = piv.iloc[i], conf.iloc[i]
        eff = {d: w[d] * (row_c[d] if np.isfinite(row_c[d]) else 0) for d in dims if np.isfinite(row_p[d])}
        tot = sum(eff.values())
        mu = (1 - PARAMS["shrinkage"]) * (sum(eff[d] * row_p[d] for d in eff) / tot if tot > 0 else 0.0)
        sigma = float(np.std(past_y.values[-40:])) if i >= 10 else PARAMS["sigma_fallback"]   # proxy: no historical options
        sigma = max(sigma, 0.02)
        eh = row_p.get("event_history", np.nan)
        out.append({"event_date": piv.index[i][0], "ticker": piv.index[i][1], "mu": mu, "sigma": sigma,
                    "p_up": float(norm.cdf(mu / sigma)), "realized": float(y.iloc[i]),
                    "baseline_event_history": (float(eh) if np.isfinite(eh) else 0.0), "weights": w})
    res = pd.DataFrame(out)
    if res.empty:
        print("not enough events for walk-forward (need > %d)" % min_train); return res
    res.drop(columns=["weights"]).to_csv(OUTPUT_DIRS["backtest"] / "walkforward.csv", index=False)
    up = (res["realized"] > 0).astype(float)
    summary = {
        "n": len(res),
        "composite_mae": float((res["mu"] - res["realized"]).abs().mean()),
        "zero_mae": float(res["realized"].abs().mean()),
        "composite_hit_rate": hit_rate(res["mu"], res["realized"]),
        "event_history_hit_rate": hit_rate(res["baseline_event_history"], res["realized"]),
        "composite_brier": brier(res["p_up"], up), "coin_flip_brier": 0.25,
        "composite_logloss": log_loss(res["p_up"], up),
        "reliability": reliability_table(res["p_up"], up),
        "final_weights": out[-1]["weights"],
    }
    if verbose:
        for k, v in summary.items():
            if k not in ("reliability", "final_weights"):
                print(f"{k:>26}: {v:.4f}" if isinstance(v, float) else f"{k:>26}: {v}")
        print("reliability:")
        for r in summary["reliability"]:
            print(f"   {r['bin']}  n={r['n']:3d}  pred={r['mean_pred']:.2f}  realized={r['realized_up']:.2f}")
        print("final walk-forward weights:", {k: round(v, 3) for k, v in summary["final_weights"].items()})
        print("\nRead: composite_brier must beat 0.25 and composite_mae must beat zero_mae, or the combiner adds nothing.")
    return res
