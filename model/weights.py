"""Weight learning from the logbook: exponentially-weighted inverse-MSE, blended with the prior.

    w_i  ∝  prior_i^(1-alpha) * (1/MSE_i)^alpha      then floor / cap / renormalise
Dimensions with fewer than `weights_min_events` resolved predictions keep their prior weight.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from config.settings import PRIOR_WEIGHTS_PATH, WEIGHTS_PATH, WEIGHTS_HISTORY_PATH, PARAMS, ensure_dirs
from data.store import dumps


def load_prior() -> dict:
    return json.loads(PRIOR_WEIGHTS_PATH.read_text(encoding="utf-8"))


def load_weights() -> dict:
    """Returns {'weights': {...}, 'meta': {...}}."""
    if WEIGHTS_PATH.exists():
        try:
            return json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"weights": load_prior(), "meta": {"source": "prior", "n_resolved": 0, "realized_to_implied_ratio": None}}


def save_weights(weights: dict, meta: dict) -> None:
    ensure_dirs()
    doc = {"weights": weights, "meta": {**meta, "saved_at": pd.Timestamp.now().isoformat()}}
    WEIGHTS_PATH.write_text(dumps(doc), encoding="utf-8")
    with WEIGHTS_HISTORY_PATH.open("a", encoding="utf-8") as fh:
        fh.write(dumps(doc) + "\n")


def inverse_mse_weights(errors_by_dim: dict[str, list[float]], prior: dict, alpha=0.5, half_life=12,
                        floor=0.02, cap=0.35, min_n=8) -> tuple[dict, dict]:
    """errors_by_dim[name] = list of (pred - realized), oldest first."""
    stats, raw = {}, {}
    updated = {}
    for name, p in prior.items():
        errs = np.array(errors_by_dim.get(name, []), dtype=float)
        errs = errs[np.isfinite(errs)]
        n = len(errs)
        if n < min_n:
            stats[name] = {"n": int(n), "mse": None, "updated": False}
            continue
        w = 0.5 ** (np.arange(n)[::-1] / float(half_life))
        mse = float(np.sum(w * errs ** 2) / np.sum(w))
        updated[name] = p ** (1 - alpha) * (1.0 / (mse + 1e-4)) ** alpha
        stats[name] = {"n": int(n), "mse": mse, "rmse": mse ** 0.5, "mean_err": float(errs.mean()), "updated": True}
    new = dict(prior)
    if updated:
        prior_mass = sum(prior[k] for k in updated)
        tot = sum(updated.values())
        for k, v in updated.items():
            new[k] = prior_mass * v / tot          # updated dims share the same total mass as before
    new = {k: min(max(v, floor), cap) for k, v in new.items()}
    s = sum(new.values())
    new = {k: v / s for k, v in new.items()}
    return new, stats


def errors_from_records(records: list[dict], label_type: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for r in sorted(records, key=lambda x: x.get("event_date", "")):
        if r.get("status") != "resolved":
            continue
        y = (r.get("realized") or {}).get(label_type)
        if y is None:
            continue
        for name, pred in (r.get("dim_preds") or {}).items():
            if pred is None:
                continue
            out.setdefault(name, []).append(float(pred) - float(y))
    return out


def learn_ratio(records: list[dict], label_type: str = "close_ret", min_n: int = 8) -> float | None:
    vals = []
    for r in records:
        if r.get("status") != "resolved":
            continue
        im = r.get("implied_move")
        y = (r.get("realized") or {}).get(label_type)
        if im and y is not None and im > 0.005:
            vals.append(abs(float(y)) / float(im))
    return float(np.median(vals)) if len(vals) >= min_n else None


def update_weights_from_logbook(records: list[dict], params: dict = PARAMS) -> dict:
    prior = load_prior()
    label = params.get("label_type", "close_ret")
    errs = errors_from_records(records, label)
    new, stats = inverse_mse_weights(errs, prior, params["weights_alpha"], params["weights_half_life"],
                                     params["weights_floor"], params["weights_cap"], params["weights_min_events"])
    n_res = sum(1 for r in records if r.get("status") == "resolved")
    ratio = learn_ratio(records, label, params["weights_min_events"])
    meta = {"source": "logbook", "n_resolved": n_res, "label_type": label, "dim_stats": stats,
            "realized_to_implied_ratio": ratio, "prior": prior}
    save_weights(new, meta)
    return {"weights": new, "meta": meta}
