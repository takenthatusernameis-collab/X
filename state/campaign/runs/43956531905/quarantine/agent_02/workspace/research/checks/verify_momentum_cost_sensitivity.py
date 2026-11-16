"""Independent verification of the momentum cost-sensitivity check
(``research/checks/momentum_cost_sensitivity.py``).

Separate implementation path (does not call ``stress_segments``):
- Gate A: recompute the signal-definition diagnostic for representative
  tickers (AAPL, AMZN, NVDA, JPM, XOM) across lookbacks -- weight counts,
  active-region sign flips, price-level rule match, and agreement with the
  intended lookback-return sign -- and compare to the artifact.
- Gate B: for AMZN and JPM, recompute every segment's candidate median via a
  fresh ``walk_forward`` (with the same ``BacktestConfig`` per cost level) and
  the coin-flip null via a fresh ``noise_benchmark``; compare both to the
  artifact.
- Gate C: regenerate each asset's regime verdict from the recomputed candidate
  and null medians using the documented verdict logic; compare to the artifact.
- Gate D: recompute the universe median and positive-edge count from the
  artifact's per-asset medians and compare.
- Gate E: determinism of the verification path.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"

COST_LEVELS = {
    "zero": dict(commission_per_trade=0.0, commission_per_share=0.0,
                 slippage_cents=0.0, slippage_proportional=0.0),
    "realistic": dict(commission_per_trade=2.0, commission_per_share=0.003,
                      slippage_cents=2.0, slippage_proportional=0.0005),
    "stress2x": dict(commission_per_trade=4.0, commission_per_share=0.006,
                     slippage_cents=4.0, slippage_proportional=0.001),
}


def cfg(level):
    return bt.BacktestConfig(**COST_LEVELS[level])


def vol_blocks(closes, n_blocks, window):
    n = len(closes)
    block_size = n // n_blocks
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(float(np.std(np.log(closes[i - window:i]), ddof=1)) * np.sqrt(252.0))
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            b = i // block_size
            s, e = b * block_size, (b + 1) * block_size
            bmed = float(np.median([v for v in bar_vols[s:e] if v > 0]) if e - s > window else 0.0)
            labels.append("turbulent" if bmed > series_med else "calm")
    return labels


def segments_from_labels(labels, min_segment_bars):
    runs = []
    cur = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - start >= min_segment_bars:
                runs.append((cur, start, i))
            cur = labels[i]
            start = i
    if len(labels) - start >= min_segment_bars:
        runs.append((cur, start, len(labels)))
    return runs


def momentum_signals(closes, lookback):
    """Fresh re-implementation of the check's momentum signal."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        r = float(np.mean(np.log(closes[i - lookback + 1:i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if r > 0 else -1.0)
    return out


def segment_candidate_median(ticker, s, e, lookback, level):
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, lookback)
    res = bt.walk_forward(list(bars)[s:e], signals[s:e], train_window=TRAIN,
                          test_window=TEST, warmup=WARM, overlap_window=OVERLAP,
                          cfg=cfg(level))
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log))


def segment_null_median(ticker, s, e, lookback, level):
    """Fresh noise_benchmark on the segment, matching stress_segments' null."""
    bars, _ = bt.load_ticker(ticker)
    seg_bars = list(bars)[s:e]
    noise = bt.noise_benchmark(
        seg_bars, param_grid=[{"lookback": lookback}],
        train_window=TRAIN, test_window=TEST, warmup=0, overlap_window=0,
        cfg=cfg(level), periods_per_year=252)
    return float(noise.baseline_median_log_return)


def regen_verdict(medians, nulls):
    cand = np.array(medians, dtype=np.float64)
    null = np.array(nulls, dtype=np.float64)
    cand_disp = float(np.std(cand))
    null_disp = float(np.std(null))
    if all(abs(m) <= 0.05 for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID
    all_ok = True

    print("=== Gate A: signal-definition recomputation ===")
    sig = artifact["signal_definition"]["per_ticker"]
    for ticker in ["AAPL", "AMZN", "NVDA", "JPM", "XOM"]:
        bars, _ = bt.load_ticker(ticker)
        closes = bars.closes_array()
        n = len(closes)
        for lb in LOOKBACKS:
            w = np.array([s.weight for s in momentum_signals(closes, lb)], dtype=np.float64)
            idx = np.arange(lb, n)
            wa = w[lb:]
            active_flips = int(np.sum(np.abs(np.diff(wa)) > 0))
            mean_log_price = np.array(
                [float(np.mean(np.log(closes[i - lb + 1:i + 1]))) for i in idx])
            price_rule = np.where(mean_log_price > 0.0, 1.0, -1.0)
            match_pl = bool(np.array_equal(wa, price_rule))
            lb_ret = np.array([float(np.log(closes[i]) - np.log(closes[i - lb])) for i in idx])
            return_rule = np.where(lb_ret > 0.0, 1.0, -1.0)
            agreement = round(float(np.mean(wa == return_rule)), 4)
            pub = sig[ticker][str(lb)]
            ok = (active_flips == pub["active_flips"]
                  and match_pl == pub["match_price_level_rule"]
                  and agreement == pub["agreement_with_return_sign"])
            all_ok = all_ok and ok
            print(f"  {ticker} lb{lb}: flips={active_flips} (pub {pub['active_flips']}) "
                  f"price_rule={match_pl} (pub {pub['match_price_level_rule']}) "
                  f"agree={agreement} (pub {pub['agreement_with_return_sign']}) -> "
                  f"{'MATCH' if ok else 'MISMATCH'}")

    print("\n=== Gate B/C: AMZN/JPM segment medians + nulls + verdicts (fresh path) ===")
    for ticker in ["AMZN", "JPM"]:
        bars, _ = bt.load_ticker(ticker)
        closes = bars.closes_array()
        runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
        for lb in LOOKBACKS:
            for level in COST_LEVELS:
                pub = artifact["cost_grid"][str(lb)][level]["per_asset"][ticker]
                cand = [round(segment_candidate_median(ticker, s, e, lb, level), 3)
                        for _, s, e in runs]
                null = [round(segment_null_median(ticker, s, e, lb, level), 3)
                        for _, s, e in runs]
                cand_ok = cand == pub["medians"]
                null_ok = null == pub["null_medians"]
                regen = regen_verdict(cand, null)
                verd_ok = regen == pub["verdict"]
                ok = cand_ok and null_ok and verd_ok
                all_ok = all_ok and ok
                print(f"  {ticker} lb{lb} {level}: cand={cand} pub={pub['medians']} "
                      f"[{ 'OK' if cand_ok else 'MISMATCH'}] | "
                      f"null={null} [{'OK' if null_ok else 'MISMATCH'}] | "
                      f"verdict={regen} (pub {pub['verdict']}) "
                      f"[{'OK' if verd_ok else 'MISMATCH'}]")

    print("\n=== Gate D: universe median + positive-edge count from artifact medians ===")
    for lb in LOOKBACKS:
        for level in COST_LEVELS:
            cell = artifact["cost_grid"][str(lb)][level]
            reps = []
            pos = 0
            for t in artifact["ticker_order"]:
                pa = cell["per_asset"][t]
                reps.append(float(np.median(pa["medians"])))
                if pa["verdict"] == "REGIME_STABLE" and all(m > 0 for m in pa["medians"]):
                    pos += 1
            umed = round(float(np.median(reps)), 3)
            ok = (umed == cell["universe"]["universe_median"]
                  and pos == cell["universe"]["n_pos_edge"])
            all_ok = all_ok and ok
            print(f"  lb{lb} {level}: universe_median={umed} "
                  f"(pub {cell['universe']['universe_median']}) "
                  f"n_pos_edge={pos} (pub {cell['universe']['n_pos_edge']}) -> "
                  f"{'MATCH' if ok else 'MISMATCH'}")

    print("\n=== Gate E: determinism of the verification path ===")
    m1 = round(segment_candidate_median("AMZN", *segments_from_labels(
        vol_blocks(bt.load_ticker("AMZN")[0].closes_array(), N_BLOCKS, WINDOW),
        MIN_SEGMENT_BARS)[0][1:], 5, "realistic"), 3)
    m2 = round(segment_candidate_median("AMZN", *segments_from_labels(
        vol_blocks(bt.load_ticker("AMZN")[0].closes_array(), N_BLOCKS, WINDOW),
        MIN_SEGMENT_BARS)[0][1:], 5, "realistic"), 3)
    det = (m1 == m2)
    all_ok = all_ok and det
    print(f"  AMZN lb5 realistic segment0: r1={m1} r2={m2} -> {'identical' if det else 'DIFFERENT'}")

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact.")
    else:
        print("MISMATCH FOUND: the artifact does not reproduce via the independent path.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production execution.")


if __name__ == "__main__":
    main()