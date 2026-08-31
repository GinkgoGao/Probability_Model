# MU - post-earnings reaction forecast
- Event: 2026-09-23 (AMC)   |   as of: 2026-08-31 10:00   |   mode: live
- Audit: PASSED - warnings: 1
  - warning: event is 23 days away: options/IV signals are immature

## Dimension scorecard
| Dimension | Predicted move | Dir | Confidence | Prior w | Effective w | Key evidence |
|---|---:|:-:|---:|---:|---:|---|
| options_rnd | -0.2% | flat | 0.30 | 0.20 | 0.15 | implied move (straddle/spot) 19.9% to 2026-09-25 |
| event_history | +1.5% | UP | 0.55 | 0.15 | 0.21 | 12 past events: mean +1.7%, median -0.4%, up 42% of the time |
| expectation | +0.6% | UP | 0.58 | 0.15 | 0.22 | EPS consensus revised +5.1% in 30d |
| peer_sector | -0.1% | flat | 0.31 | 0.12 | 0.10 | 2 peers reported in last 45d, mean reaction -0.7% |
| technical | +0.7% | UP | 0.49 | 0.10 | 0.13 | 20d run-up -11.6% -> expect partial reversal |
| positioning | +1.7% | UP | 0.40 | 0.08 | 0.08 | short interest 18.0% of float, 6.2 days to cover |
| options_flow | +0.0% | flat | 0.15 | 0.06 | 0.02 | P/C volume 1.02, P/C OI 0.83 -> neutral |
| macro_regime | -0.7% | DOWN | 0.25 | 0.06 | 0.04 | VIX 27.8, VIX/VIX3M 2.05 -> STRESS regime |
| sentiment | +1.2% | UP | 0.20 | 0.04 | 0.02 | 4 headlines, recency-weighted net sentiment +0.29 (50% pos / 25% neg) |
| microstructure | +1.1% | UP | 0.20 | 0.04 | 0.02 | close +1.47% vs VWAP, last hour +0.90% |

## Composite
- Expected move mu = **+0.40%** (raw weighted mean +0.67%, shrinkage 40%)
- Sigma = **26.7%** from ATM straddle implied move (24.8%) x ratio 0.90 x multipliers 1.19
- P(up) = **50.6%**   |   conviction score (-100..100) = +2   |   dimension agreement = 86%
- Model E|move| = 21.3% vs options implied 19.9%
- Active dimensions: 10, abstaining: 0; distribution: normal

## Probability distribution of the reaction
| Bucket | Prob | |
|---|---:|---|
| < -15% | 28.2% | ######## |
| -15% to -8% | 9.5% | ### |
| -8% to -3% | 7.3% | ## |
| -3% to +3% | 9.0% | ### |
| +3% to +8% | 7.3% | ## |
| +8% to +15% | 9.6% | ### |
| > +15% | 29.2% | ######### |

Quantiles: 5% -43.4% | 25% -17.6% | 50% +0.4% | 75% +18.4% | 95% +44.3%

## Decision
**NO TRADE**
- directional edge +2.4% < 6% and vol edge +7% < 15%: the ticket price already reflects the view

## Self-attack (how this forecast could be wrong)
- This name fell after 62% of its beats: 'good numbers' is not the same as 'up'.
- Consensus sits above management guidance: a beat may already be required just to hold the price.
- Stress regime (VIX elevated / inverted term structure): historical base rates may not apply.
- The options market already embeds most of these dimensions; the composite is shrunk toward zero for that reason. Your real benchmark is the risk-neutral P(up), not 50/50.
- Weights are uncalibrated until the logbook has enough resolved events (see `python run.py show-weights`).
- Not investment advice. This is a measurement instrument whose first job is to say NO TRADE.