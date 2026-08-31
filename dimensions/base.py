"""Contract shared by every dimension.

Each dimension answers ONE question with ONE output type:
    predicted_return  - its own estimate of the post-earnings move (decimal, e.g. +0.05 = +5%)
    confidence        - how much it trusts that estimate (0..1); multiplies the prior weight
    abstain           - True when it has no usable information ("I don't know" is a valid answer)
    sigma_multiplier  - >1 widens the final distribution (used by regime/positioning/vol dims)
    evidence, notes   - everything needed to audit the number afterwards
"""
from __future__ import annotations
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
import pandas as pd
from data.store import clean


@dataclass
class EventContext:
    ticker: str
    event_date: pd.Timestamp | None       # calendar day of the report
    timing: str                           # "AMC" or "BMO"
    asof: pd.Timestamp                    # last moment of information the model may use
    mode: str = "live"                    # "live" | "backtest"
    data: dict = field(default_factory=dict)
    overrides: dict = field(default_factory=dict)


@dataclass
class DimOutput:
    name: str
    predicted_return: float = 0.0
    confidence: float = 0.0
    abstain: bool = False
    direction: int = 0
    sigma_multiplier: float = 1.0
    evidence: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return clean(asdict(self))


class Dimension(ABC):
    name: str = "base"
    supports_backtest: bool = False       # True only if computable from point-in-time free data
    max_abs_return: float = 0.5           # the user's (-50%, +50%) contract

    def make(self, pred: float, conf: float, evidence: dict | None = None, notes: list | None = None,
             sigma_multiplier: float = 1.0) -> DimOutput:
        pred = float(pred) if pred is not None and math.isfinite(float(pred)) else 0.0
        pred = max(-self.max_abs_return, min(self.max_abs_return, pred))
        conf = max(0.0, min(1.0, float(conf)))
        direction = 1 if pred > 0.002 else (-1 if pred < -0.002 else 0)
        sm = float(sigma_multiplier) if math.isfinite(float(sigma_multiplier)) else 1.0
        return DimOutput(self.name, pred, conf, False, direction, sm, clean(evidence or {}), list(notes or []))

    def abstain(self, reason: str) -> DimOutput:
        return DimOutput(self.name, 0.0, 0.0, True, 0, 1.0, {}, [f"abstain: {reason}"])

    def safe_score(self, ctx: EventContext) -> DimOutput:
        try:
            return self.score(ctx)
        except Exception as e:  # a broken dimension must never break the pipeline
            return self.abstain(f"error {type(e).__name__}: {e}")

    @abstractmethod
    def score(self, ctx: EventContext) -> DimOutput: ...
