#!/usr/bin/env python3
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.checks.momentum_lookback_sweep_cost_sensitivity import segment_result

bars, dates = bt.load_ticker('AMZN')
bars_list = list(bars)

cfg = bt.BacktestConfig(
    commission_per_share=0.003,
    slippage_cents=5.0,
    slippage_proportional=0.001,
    warmup_periods=0,
)

TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)

m1 = []
for lb in LOOKBACKS:
    m, _, _ = segment_result('AMZN', bars_list, lb, cfg)
    m1.append([round(v, 3) for v in m])

m2 = []
for lb in LOOKBACKS:
    m, _, _ = segment_result('AMZN', bars_list, lb, cfg)
    m2.append([round(v, 3) for v in m])

print('m1:', m1)
print('m2:', m2)
print('Identical:', m1 == m2)