"""Dimension 9 - news/text sentiment. Default scorer is a small finance lexicon (no downloads).
Plug an LLM by passing overrides={'llm_scorer': fn} where fn(list[str]) -> list[float in -1..1].

LEAKAGE WARNING: never use an LLM to score HISTORICAL headlines for a backtest - the model already
knows how the stock reacted. This dimension is forward-logged only (supports_backtest=False)."""
from __future__ import annotations
import re
import pandas as pd
from .base import Dimension, EventContext
from .utils import clip

POS = {"beat", "beats", "record", "strong", "surge", "surges", "soar", "soars", "upgrade", "upgrades", "raises", "raised",
       "outperform", "bullish", "growth", "accelerat", "exceed", "exceeds", "momentum", "buy", "rally", "rallies", "boost",
       "demand", "expands", "profit", "optimistic", "top", "tops", "jump", "jumps", "gain", "gains", "higher", "upside"}
NEG = {"miss", "misses", "weak", "plunge", "plunges", "fall", "falls", "drop", "drops", "downgrade", "downgrades", "cuts", "cut",
       "lowers", "lowered", "underperform", "bearish", "decline", "declines", "slump", "warning", "warns", "lawsuit", "probe",
       "recall", "layoffs", "loss", "losses", "concern", "concerns", "risk", "risks", "sell", "selloff", "tumble", "tumbles",
       "slide", "slides", "lower", "downside", "delay", "delays", "investigation"}


def lexicon_score(text: str) -> float:
    toks = re.findall(r"[a-z]+", (text or "").lower())
    p = sum(1 for t in toks if t in POS or any(t.startswith(x) for x in ("accelerat",)))
    n = sum(1 for t in toks if t in NEG)
    return (p - n) / (p + n + 1.0)


class SentimentDimension(Dimension):
    name = "sentiment"
    supports_backtest = False

    def score(self, ctx: EventContext):
        news = ctx.data.get("news") or []
        if len(news) < 3:
            return self.abstain(f"only {len(news)} headlines")
        texts = [f"{n.get('title') or ''}. {n.get('summary') or ''}" for n in news]
        scorer = (ctx.overrides or {}).get("llm_scorer")
        scores = list(scorer(texts)) if callable(scorer) else [lexicon_score(t) for t in texts]
        ws = []
        for n in news:
            try:
                age = max((ctx.asof - pd.Timestamp(n.get("published"))).days, 0)
            except Exception:
                age = 3
            ws.append(0.5 ** (age / 3.0))
        net = sum(w * s for w, s in zip(ws, scores)) / max(sum(ws), 1e-9)
        pos_share = sum(1 for s in scores if s > 0.1) / len(scores)
        neg_share = sum(1 for s in scores if s < -0.1) / len(scores)
        pred = clip(0.04 * net, -0.03, 0.03)
        conf = clip(0.12 + 0.02 * min(len(news), 10), 0.10, 0.35)
        ev = {"n_headlines": len(news), "net_sentiment": net, "positive_share": pos_share, "negative_share": neg_share,
              "scorer": "llm" if callable(scorer) else "lexicon",
              "top_headlines": [{"title": n.get("title"), "score": s, "published": n.get("published")} for n, s in list(zip(news, scores))[:8]]}
        notes = [f"{len(news)} headlines, recency-weighted net sentiment {net:+.2f} ({pos_share:.0%} pos / {neg_share:.0%} neg)",
                 "forward-logged only: never backtested with an LLM (hindsight leakage)"]
        return self.make(pred, conf, ev, notes)
