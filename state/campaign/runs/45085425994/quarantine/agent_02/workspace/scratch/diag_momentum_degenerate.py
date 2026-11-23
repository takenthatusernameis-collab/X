"""Diagnostic: is `momentum_signals` an always-long (buy-and-hold) bet, and
does the existing lookback-sweep artifact reflect that?

Evidence to establish:
  A. The signal is ~constant +1 on every collected asset for every lookback.
  B. The zero-cost medians reported in momentum_results.json equal the medians
     of an explicit always-long (+1 every bar) bet on the same segments.
  C. The same equivalence holds for the lookback-sweep artifact at 3/5/10.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "state" / "check_artifacts"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW, N_BLOCKS, MIN_SEG = 60, 4, 400


def always_long(closes):
    n = len(closes)
    return [bt.Signal(date=i + 1, weight=1.0) for i in range(n)]


def seg_result(bars, sig, lookback):
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    res = bt.stress_segments(
        signals_fn=lambda c, **p: sig(c),
        bars=bars, signals=sig(closes),
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN, test_window=TEST, warmup=WARM,
        overlap_window=OVERLAP, periods_per_year=252,
        min_segment_bars=MIN_SEG,
    )
    return [round(s.baseline_median_log_return, 3) for s in res.scenarios]


manifest = json.loads((ROOT / "research" / "data" / "manifest.json").read_text())
tickers = {t: bt.load_ticker(t)[0] for t in manifest["universe"]}

print("=== A. weight distribution per asset/lookback (current implementation) ===")
for t in manifest["universe"]:
    closes = tickers[t].closes_array()
    parts = []
    for lb in (3, 5, 10):
        sig = bt.momentum_signals(closes, lookback=lb)
        flips = sum(1 for i in range(1, len(sig)) if sig[i].weight != sig[i - 1].weight)
        frac_pos = sum(1 for s in sig if s.weight > 0) / len(sig)
        parts.append(f"lb{lb}: flips={flips:3d} frac_long={frac_pos:.3f}")
    print(f"  {t:6s} " + " | ".join(parts))

print()
print("=== B. momentum_results.json medians vs explicit always-long medians ===")
mom = json.loads((ART / "momentum_results.json").read_text())
all_match = True
for t in manifest["universe"]:
    bars = tickers[t]
    al = seg_result(bars, always_long, 5)
    pub = mom["universe"]["per_asset"][t]["medians"]
    match = al == pub
    all_match = all_match and match
    print(f"  {t:6s} published={pub} always_long={al} -> {'MATCH' if match else 'DIFFER'}")
print("  all match:", all_match)

print()
print("=== C. lookback sweep artifact vs always-long, lookbacks 3/5/10 ===")
swp = json.loads((ART / "momentum_lookback_sweep_results.json").read_text())
swp_ok = True
for t in manifest["universe"]:
    bars = tickers[t]
    al = {lb: seg_result(bars, always_long, lb) for lb in (3, 5, 10)}
    row = swp["universe"]["per_asset"][t]["lookback_medians"]
    for lb in (3, 5, 10):
        if al[lb] != row[str(lb)]:
            swp_ok = False
            print(f"  DIFFER {t} lb{lb}: published={row[str(lb)]} always_long={al[lb]}")
print("  all lookbacks all assets match always-long:", swp_ok)

print()
print("=== D. does the existing momentum cell contain any short exposure? ===")
for t in manifest["universe"]:
    closes = tickers[t].closes_array()
    sig = bt.momentum_signals(closes, lookback=5)
    short_frac = sum(1 for s in sig if s.weight < 0) / len(sig)
    print(f"  {t:6s} short_fraction={short_frac:.4f}")