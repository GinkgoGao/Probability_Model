"""Scoring rules and diagnostics: Brier, log-loss, reliability table, Spearman IC, hit rate."""
from __future__ import annotations
import numpy as np
from scipy import stats as st


def brier(p_up, up) -> float:
    p, y = np.asarray(p_up, float), np.asarray(up, float)
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def log_loss(p_up, up, eps=1e-6) -> float:
    p = np.clip(np.asarray(p_up, float), eps, 1 - eps); y = np.asarray(up, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(p) else float("nan")


def reliability_table(p_up, up, bins=(0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0)) -> list[dict]:
    p, y = np.asarray(p_up, float), np.asarray(up, float)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum():
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "mean_pred": float(p[m].mean()), "realized_up": float(y[m].mean())})
    return rows


def spearman_ic(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5 or np.std(x[m]) == 0:
        return float("nan")
    return float(st.spearmanr(x[m], y[m]).correlation)


def hit_rate(pred, realized) -> float:
    x, y = np.asarray(pred, float), np.asarray(realized, float)
    m = np.isfinite(x) & np.isfinite(y) & (np.abs(x) > 1e-9)
    return float(np.mean(np.sign(x[m]) == np.sign(y[m]))) if m.sum() else float("nan")
