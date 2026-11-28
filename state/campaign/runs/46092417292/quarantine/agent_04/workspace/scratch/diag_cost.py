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
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    # reconstruct segments
    runs = []
    cur = labels[0]; start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - start >= MIN_SEG:
                runs.append((cur, start, i))
            cur = labels[i]; start = i
    if len(labels) - start >= MIN_SEG:
        runs.append((cur, start, len(labels)))
    print('====', ticker, 'segments', [(r[0], r[2]-r[1]) for r in runs])
    for lab, s, e in runs[:1]:
        seg_bars = list(bars)[s:e]
        for lb in [5]:
            sig = bt.momentum_signals(closes, lookback=lb)[s:e]
            for name, cfg in [('zero', cfg0), ('cons', cfgC)]:
                res = bt.walk_forward(seg_bars, sig, train_window=TRAIN, test_window=TEST,
                                      warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
                rets = [f.metrics['total_return'] for f in res.folds]
                logrets = np.log1p(np.clip(rets, -1+1e-12, None))
                print('  seg', lab, 'lb', lb, name, 'nfolds', len(rets),
                      'median_logret', round(float(np.median(logrets)), 4),
                      'mean', round(float(np.mean(logrets)), 4),
                      'first3', [round(x,4) for x in logrets[:3]])
                # count trades in first fold OOS
                f0 = res.folds[0]
                print('     fold0 n_metrics_trades? total_return', round(rets[0],4))