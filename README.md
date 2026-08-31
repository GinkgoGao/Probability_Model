# event_alpha — post-earnings reaction model

Input a ticker → fetch prices / options / expectations / news / peers / macro → ten dimensions each
predict the post-earnings move **in the same unit (−50%…+50%)** → confidence-weighted composite →
Normal(μ, σ) distribution, P(up), 7 buckets → compared with the options-implied price of the event →
TRADE / NO TRADE → every prediction logged → weights learned from prediction-vs-actual gaps.

## Install
    pip install -r requirements.txt          # yfinance, pandas, numpy, scipy, pyarrow

## Daily workflow
    python run.py predict MU                       # next scheduled earnings (auto-detected)
    python run.py predict MU --date 2026-09-23 --timing AMC \
        --override consensus_eps=3.10 guidance_mid_eps=2.90 whisper_eps=3.20
    python run.py predict MU --no-options          # if the chain is unavailable (benchmark missing -> NO TRADE)
    python run.py audit MU                         # data-quality gates only
    python run.py resolve                          # after the event: fill realized outcomes, per-dimension errors
    python run.py update-weights                   # learn weights (needs >= 8 resolved predictions)
    python run.py show-weights

## Research workflow (offline)
    python run.py backtest --tickers MU AMD NVDA AVGO AMAT LRCX WDC STX --max-events 24
    python run.py walkforward
Only price-derived dimensions (technical, event_history, peer_sector, macro_regime) are backtestable
from free data. Options, expectations, sentiment and positioning have no free point-in-time history:
they are forward-logged and earn their weights from the logbook.

## Source layout
    run.py                     CLI
    pipeline.py                ticker -> context -> audit -> dimensions -> composite -> decision -> report -> logbook
    config/settings.py         paths + all tunable parameters (shrinkage, thresholds, weight-learning)
    config/universe.py         peer groups, sector ETF mapping
    config/weights_prior.json  starting weights (prior)
    config/macro_calendar.json FOMC / CPI / NFP dates (maintain by hand, verify!)
    data/                      ingest_prices, ingest_options (daily chain snapshots), ingest_events (compute_reaction),
                               ingest_fundamentals, ingest_text, ingest_macro, store (cache)
    quality/audit_gates.py     hard gates refuse output; soft gates warn
    dimensions/base.py         Dimension contract: DimOutput(predicted_return, confidence, abstain, sigma_multiplier, evidence)
    dimensions/d01..d10        technical, microstructure, event_history, expectation, options_rnd, options_flow,
                               peer_sector, macro_regime, sentiment_nlp, positioning
    model/combiner.py          weighted mean + shrinkage -> mu; options-derived sigma; Normal/t; P(up); buckets
    model/weights.py           inverse-MSE weight learning from the logbook; learned realized/implied ratio
    model/calibration.py       Brier, log-loss, reliability, Spearman IC, hit rate
    decision/ev_engine.py      directional edge vs risk-neutral P(up), vol edge vs implied move, quarter-Kelly, NO TRADE
    report/render.py           markdown report incl. mandatory self-attack section
    report/logbook.py          outputs/logbook.jsonl append + resolve (realized reaction, per-dimension errors)
    backtest/event_study.py    point-in-time event study, per-dimension IC / hit rate / MAE
    backtest/walkforward.py    weights re-fit on the past only; composite vs baselines; reliability

## Generated files
    data_cache/prices/{TICKER}_daily_3y.parquet, {TICKER}_5m_5d.parquet, SPY_daily_3y.parquet ...
    data_cache/options/{TICKER}_{YYYY-MM-DD}.parquet        one snapshot per day = your future options history
    data_cache/events/{TICKER}_earnings_40.parquet
    data_cache/fundamentals/{TICKER}_{YYYY-MM-DD}.json
    data_cache/news/{TICKER}_{YYYY-MM-DD}.json
    data_cache/macro/  (VIX, VIX3M, TNX via prices/)
    outputs/reports/{TICKER}_{EVENT}_asof{TIMESTAMP}.md      human-readable report
    outputs/predictions/{TICKER}_{EVENT}_asof{TIMESTAMP}.json full machine-readable prediction
    outputs/logbook.jsonl                                    one line per prediction; resolve fills realized + errors
    outputs/weights/weights_current.json, weights_history.jsonl
    outputs/backtest/event_study_long.csv, event_study_summary.csv, walkforward.csv

## Definitions that must never drift
    reaction (AMC): close(D) -> close(D+1)     reaction (BMO): close(D-1) -> close(D)
    gap = open(reaction day)/ref close - 1 ;   excess = close_ret - beta * SPY return over the same window
    label_type in settings.py chooses which one trains the weights (default close_ret; excess is cleaner)
