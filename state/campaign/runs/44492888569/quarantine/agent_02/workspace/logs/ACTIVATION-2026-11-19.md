# Activation Log — 2026-11-19

## Activation — 2026-11-19T09:07:22Z (campaign slot 2, run 44492888569, task R-001)

Start timestamp: 2026-11-19T09:07:22Z (observed session message time).
Finish timestamp: not captured; the `date` shell form is denied in this runner.
Run identity: campaign slot 2, global agent 2, run_id 44492888569,
event_sha 618539a4a9627acf46f68bc72b855293fd64c871.

### Objective

Run a bounded cost-sensitivity check for lookback 3/5/10 momentum under the
repository's existing realistic cost model and a matched null, and classify
cost robustness on the existing 10-asset collected universe.

### Method (predeclared before execution)

- Universe: existing 10-asset adjusted-close universe
  (`yf-ohlcv-universe-2009-to-2026-10-03`); manifest 10/10 OK; preflight
  PASSED; leakage PASS on AMZN/JPM.
- Signal: `momentum_signals` (long the previous N-day return, hold 1 day,
  daily rebalance), lookbacks 3/5/10.
- Walk-forward per segment: train=252d/test=84d/warmup=60d/overlap=60d; 4
  contiguous volatility blocks; min 400 bars/segment; matched coin-flip null
  (`noise_benchmark`) on the same segments and under the same cost
  configuration.
- Cost grid: {0.0, 0.5, 1.0, 2.0} x the existing realistic cost model
  (commission_per_trade=2.0, commission_per_share=0.003,
  slippage_cents=2.0, slippage_proportional=0.0005); 2.0x is the conservative
  stress.
- Classification (predeclared): `edge(m)` = `stress_segments` verdict
  REGIME_STABLE and all segment medians > 0; COST_ROBUST = edge(1.0) and
  edge(2.0); COST_RESILIENT_BASE_ONLY = edge(1.0) and not edge(2.0);
  COST_FRAGILE = edge(0.0) and not edge(1.0); NO_PRE_COST_EDGE = not edge(0.0).
- A-priori prediction (declared before execution): COST_ROBUST <= 3/10 at
  lookback 5 (and <= 3/10 at lookbacks 3 and 10); >= 6/10 COST_ROBUST at any
  lookback falsifies it.

### Results

- Lookback 3: COST_ROBUST=6 (AAPL, MSFT, GOOGL, AMZN, META, TSLA),
  NO_PRE_COST_EDGE=4 (NVDA, JPM, JNJ, XOM).
- Lookback 5: COST_ROBUST=5 (AAPL, GOOGL, AMZN, META, TSLA), COST_FRAGILE=1
  (MSFT), NO_PRE_COST_EDGE=4 (NVDA, JPM, JNJ, XOM).
- Lookback 10: COST_ROBUST=5 (MSFT, GOOGL, AMZN, META, TSLA),
  NO_PRE_COST_EDGE=5 (AAPL, NVDA, JPM, JNJ, XOM).
- Overall: 16/30 COST_ROBUST, 13 NO_PRE_COST_EDGE, 1 COST_FRAGILE.
- Lookback 5 at 1.0x: median candidate +0.069, median matched null -0.008.
- The a-priori prediction is falsified: 5/10 COST_ROBUST at lookback 5, not
  <= 3/10.

### Independent verification

- `python3 research/checks/momentum_cost_sensitivity.py`: exit 0; manifest
  10/10 OK; preflight PASSED; leakage PASS; 0.0x cross-consistency vs
  `momentum_lookback_sweep_results.json` MATCH on all 10 tickers x 3
  lookbacks (segments, medians, null medians, verdicts); determinism identical.
- `python3 research/checks/verify_momentum_cost_sensitivity.py`: all gates
  MATCH — fresh AMZN/JPM `walk_forward` + `noise_benchmark` recomputation at
  every lookback/cost; 0.0x cross-consistency 30/30; classification and
  break-even reproduce 30/30.
- `python3 -m unittest discover -s tests`: 168 tests, all passing.

### Durable finding: the existing cost model is not a conservative stress

The engine adds slippage to the fill price on both buys and sells
(`research/backtest/engine.py:154-167`), so slippage is a gain on sells. As a
result, candidate medians are essentially invariant to the cost multiplier:
AAPL lookback 5 +0.070 at 0.0x and 2.0x; MSFT +0.062/+0.062; AMZN
+0.076/+0.076; TSLA +0.107/+0.107; NVDA +0.013/+0.011. The matched null is
also unstable at positive costs for NVDA: bankruptcy-scale fold returns
(-27.631 at 0.5x lookback 10) and large positive median nulls after clipping
(0.396 at 1.0x lookback 5); null-dispersion changes flip framework verdicts
(MSFT lookback 5 REGIME_STABLE at 0.5x -> REGIME_DEPENDENT at 1.0x while
candidate medians remain [0.052, 0.072]).

The COST_ROBUST classification is therefore conditional on the existing cost
model, not evidence that the momentum edge survives a valid conservative cost
stress. Cost robustness is UNVERIFIED.

### Decision and next action

Decision: UNVERIFIED (cost robustness); process decision RETAIN.
Next bounded action: repair the existing cost model (symmetric slippage on both
sides; bound commission/position sizing so the matched null cannot go
bankrupt), add regression tests, and rerun the cost-sensitivity check and
verifier.

Research/simulation only. No live trading or production execution.