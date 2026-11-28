import sys
sys.path.insert(0, '/home/runner/work/X/X')
import numpy as np
import research.backtest as bt

TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW, N_BLOCKS, MIN_SEG = 60, 4, 400

cfg0 = bt.BacktestConfig(initial_capital=1e6)
cfgC = bt.BacktestConfig(initial_capital=1e6, commission_per_trade=4.0,
                         commission_per_share=0.006, slippage_cents=4.0,
                         slippage_proportional=0.001)

for ticker in ['AAPL', 'NVDA']:
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    print('====', ticker, 'close range', round(float(closes.min()),2), round(float(closes.max()),2),
          'first', round(float(closes[0]),2), 'last', round(float(closes[-1]),2))
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    runs = []
    cur = labels[0]; start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - start >= MIN_SEG:
                runs.append((cur, start, i))
            cur = labels[i]; start = i
    if len(labels) - start >= MIN_SEG:
        runs.append((cur, start, len(labels)))
    lab, s, e = runs[0]
    seg_bars = list(bars)[s:e]
    sig = bt.momentum_signals(closes, lookback=5)[s:e]
    # signal flip count in segment
    w = np.array([x.weight for x in sig])
    flips = int(np.sum(np.abs(np.diff(w)) > 0))
    print('  seg', lab, 'len', e-s, 'signal flips in seg', flips, 'unique weights', set(np.round(w,3)))
    for name, cfg in [('zero', cfg0), ('cons', cfgC)]:
        res = bt.walk_forward(seg_bars, sig, train_window=TRAIN, test_window=TEST,
                              warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
        # total fills across all folds, and fills in OOS windows
        # Reconstruct: run one full-sample backtest to count fills/commission
        full = bt.run_bars(seg_bars, sig, cfg)
        tot_comm = sum(f.commission for f in full.trades)
        print('  ', name, 'full-sample fills', len(full.trades), 'total commission',
              round(tot_comm,1), 'final equity', round(float(full.equity_curve[-1]),1))