"""Unified cache read/write helpers (parquet for tables, JSON for documents)."""
from __future__ import annotations
import json, math, time
from datetime import datetime, date
from pathlib import Path
import numpy as np
import pandas as pd
from config.settings import CACHE_DIRS, ensure_dirs


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return None if not math.isfinite(float(o)) else float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp, datetime, date)):
            return o.isoformat()
        if isinstance(o, pd.Series):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return str(o)


def clean(obj):
    """Recursively make an object JSON-safe (NaN -> None, numpy -> python)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    return obj


def dumps(obj) -> str:
    return json.dumps(clean(obj), cls=JSONEncoder, ensure_ascii=False)


def cache_path(kind: str, name: str, ext: str = "parquet") -> Path:
    ensure_dirs()
    return CACHE_DIRS[kind] / f"{name}.{ext}"


def is_fresh(path: Path, max_age_hours: float | None) -> bool:
    if not path.exists():
        return False
    if max_age_hours is None:
        return True
    return (time.time() - path.stat().st_mtime) < max_age_hours * 3600


def save_df(df: pd.DataFrame, kind: str, name: str) -> None:
    df.to_parquet(cache_path(kind, name))


def load_df(kind: str, name: str, max_age_hours: float | None = None) -> pd.DataFrame | None:
    p = cache_path(kind, name)
    if not is_fresh(p, max_age_hours):
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def save_json(obj, kind: str, name: str) -> None:
    cache_path(kind, name, "json").write_text(dumps(obj), encoding="utf-8")


def load_json(kind: str, name: str, max_age_hours: float | None = None):
    p = cache_path(kind, name, "json")
    if not is_fresh(p, max_age_hours):
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def to_naive(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Convert a tz-aware DatetimeIndex to naive New-York wall time."""
    if df is None or df.empty:
        return df
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        df = df.copy()
        df.index = idx.tz_convert("America/New_York").tz_localize(None)
    return df
