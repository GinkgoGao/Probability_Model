"""Global paths and tunable parameters for the event-alpha project."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_CACHE = ROOT / "data_cache"
OUTPUTS = ROOT / "outputs"

CACHE_DIRS = {k: DATA_CACHE / k for k in ("prices", "options", "fundamentals", "news", "macro", "events")}
OUTPUT_DIRS = {k: OUTPUTS / k for k in ("reports", "predictions", "weights", "backtest")}

LOGBOOK_PATH = OUTPUTS / "logbook.jsonl"
WEIGHTS_PATH = OUTPUT_DIRS["weights"] / "weights_current.json"
WEIGHTS_HISTORY_PATH = OUTPUT_DIRS["weights"] / "weights_history.jsonl"
PRIOR_WEIGHTS_PATH = ROOT / "config" / "weights_prior.json"
MACRO_CALENDAR_PATH = ROOT / "config" / "macro_calendar.json"


def ensure_dirs() -> None:
    for p in list(CACHE_DIRS.values()) + list(OUTPUT_DIRS.values()):
        p.mkdir(parents=True, exist_ok=True)


PARAMS = {
    # ---- data ----
    "daily_period": "3y",
    "backtest_daily_period": "10y",
    "intraday_period": "5d",
    "intraday_interval": "5m",
    "earnings_history_limit": 40,
    "max_option_expiries": 8,
    "risk_free_fallback": 0.04,
    # ---- combiner ----
    "shrinkage": 0.4,               # pull the composite mean toward 0 (market-efficiency prior)
    "distribution": "normal",       # "normal" or "t"
    "t_df": 4,
    "realized_to_implied_ratio": 0.9,   # Q -> P correction until the logbook learns a better one
    "sigma_fallback": 0.08,
    "sigma_multiplier_cap": 1.5,
    "bucket_edges": [-0.15, -0.08, -0.03, 0.03, 0.08, 0.15],
    "label_type": "close_ret",      # realized quantity used to train weights: close_ret | excess | gap
    # ---- decision ----
    "no_trade_dir_edge": 0.06,      # |P(up)_model - P(up)_market| below this -> no directional trade
    "no_trade_vol_edge": 0.15,      # |E|move|_model / implied - 1| below this -> no vol trade
    "max_position_pct": 0.02,
    # ---- weight learning ----
    "weights_alpha": 0.5,           # 0 = keep prior, 1 = pure inverse-MSE
    "weights_half_life": 12,        # events
    "weights_min_events": 8,
    "weights_floor": 0.02,
    "weights_cap": 0.35,
    # ---- backtest ----
    "backtest_max_events_per_ticker": 24,
    "walkforward_min_train": 30,
}
