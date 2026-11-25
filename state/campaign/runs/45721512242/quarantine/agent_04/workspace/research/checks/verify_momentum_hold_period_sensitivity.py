"""Independent verification of `research/checks/momentum_hold_period_sensitivity.py`.

Gates (all use a fresh code path, never `stress_segments`):

1. Fresh recomputation of the AMZN/JPM per-segment candidate medians and the
   coin-flip null medians for every hold in {1,2,3,5}, using an independently
   re-implemented hold transform, `volatility_blocks`, segment reconstruction,
   momentum signals, `walk_forward`, and `noise_benchmark`. Also asserts the
   hold transform is non-trivial (hold=5 effective signals differ from hold=1),
   so an accidental no-op transform cannot masquerade as "hold-invariant".

2. Fresh recomputation of two universe-only assets (AAPL, TSLA) candidate
   medians across all holds against the artifact's universe block.

3. Reproducibility of the universe-level classification (per-hold positive-edge
   counts, stable_edge_assets, singular_assets, effect_classification) re-derived
   from the artifact medians/verdicts via the documented rule, independently of
   the check's own classification code.

4. Determinism: the AMZN hold=2 recomputation is run twice and must be equal.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.backtest.perturbation import noise_benchmark

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACK = 5
HOLDS = (1, 2, 3, 5)
ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_hold_period_sensitivity_results.json"


def vol_blocks(closes, n_blocks, window):
    n = len(closes)
    block_size = n // n_blocks
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(
                float(np.std(np.log(closes[i - window : i]), ddof=1)) * np.sqrt(252.0))
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            b = i // block_size
            s, e = b * block_size, (b + 1) * block_size
            bmed = float(
                np.median([v for v in bar_vols[s:e] if v > 0])
                if e - s > window else 0.0)
            labels.append("turbulent" if bmed > series_med else "calm")
    return labels


def segments_from_labels(labels, min_segment_bars):
    runs = []
    cur = labels[0]
    run_start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - run_start >= min_segment_bars:
                runs.append((cur, run_start, i))
            cur = labels[i]
            run_start = i
    if len(labels) - run_start >= min_segment_bars:
        runs.append((cur, run_start, len(labels)))
    return runs


def momentum_signals(closes, lookback):
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def hold_transform(signals, hold, lookback):
    """Independent re-implementation of the check's holding-period transform."""
    n = len(signals)
    out = list(signals)
    for i in range(n):
        if i < lookback:
            out[i] = bt.Signal(date=i + 1, weight=0.0)
        else:
            idx = i - (i - lookback) % hold
            out[i] = bt.Signal(date=i + 1, weight=signals[idx].weight)
    return out


def seg_candidate_and_null(ticker, s, e, hold):
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    seg_closes = closes[s:e]
    sig = hold_transform(momentum_signals(seg_closes, LOOKBACK), hold, LOOKBACK)
    res = bt.walk_forward(
        seg_bars, sig, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=bt.BacktestConfig())
    cand = float(np.median(
        np.log1p(np.clip([f.metrics["total_return"] for f in res.folds], -1.0 + 1e-12, None))))
    noise = noise_benchmark(
        seg_bars, param_grid=[{"hold": hold}],
        train_window=TRAIN, test_window=TEST,
        warmup=0, overlap_window=0, periods_per_year=252)
    return round(cand, 3), round(noise.baseline_median_log_return, 3), len(res.folds)


def load_artifact():
    with open(ARTIFACT) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    art = load_artifact()
    assert art["dataset_id"] == DATASET_ID
    assert art["lookback"] == LOOKBACK
    assert art["hold_periods"] == list(HOLDS)
    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM, all holds) ===")
    for asset in ["AMZN", "JPM"]:
        bars, _ = bt.load_ticker(asset)
        closes = bars.closes_array()
        runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        for hold in HOLDS:
            cands, nulls, nfolds = [], [], None
            for lab, s, e in runs:
                c, n, nf = seg_candidate_and_null(asset, s, e, hold)
                cands.append(c)
                nulls.append(n)
                nfolds = nf
            pub = art["per_asset"][asset][str(hold)]
            ok = (cands == [round(v, 3) for v in pub["medians"]]
                  and nulls == [round(v, 3) for v in pub["null_medians"]]
                  and lbls == pub["segments"]
                  and nfolds == pub["n_folds"])
            all_ok = all_ok and ok
            print("  {} hold={} segments={} folds={} -> {}".format(
                asset, hold, lbls, nfolds, "MATCH" if ok else "MISMATCH"))
            if not ok:
                print("    cand {} vs {}".format(cands, pub["medians"]))
                print("    null {} vs {}".format(nulls, pub["null_medians"]))

    print("\n=== Gate 1b: hold transform is non-trivial (hold=5 != hold=1) ===")
    bars, _ = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    base = hold_transform(momentum_signals(closes, LOOKBACK), 1, LOOKBACK)
    held = hold_transform(momentum_signals(closes, LOOKBACK), 5, LOOKBACK)
    n_diff = sum(1 for a, b in zip(base, held) if a.weight != b.weight)
    nontrivial = n_diff > 0
    all_ok = all_ok and nontrivial
    print("  AMZN bars where hold=5 effective weight differs from hold=1: {} -> {}".format(
        n_diff, "NON-TRIVIAL" if nontrivial else "NO-OP (defect)"))

    print("\n=== Gate 2: fresh recomputation (AAPL, TSLA universe-only, all holds) ===")
    for asset in ["AAPL", "TSLA"]:
        bars, _ = bt.load_ticker(asset)
        closes = bars.closes_array()
        runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
        for hold in HOLDS:
            cands = [seg_candidate_and_null(asset, s, e, hold)[0] for lab, s, e in runs]
            pub = art["universe"]["hold_sums"][str(hold)]["per_asset"][asset]
            ok = cands == [round(v, 3) for v in pub["medians"]]
            all_ok = all_ok and ok
            print("  {} hold={} -> {}".format(asset, hold, "MATCH" if ok else "MISMATCH"))
            if not ok:
                print("    cand {} vs {}".format(cands, pub["medians"]))

    print("\n=== Gate 3: universe classification reproducibility ===")
    ticker_order = art["universe"]["ticker_order"]
    hold_sums = art["universe"]["hold_sums"]

    def is_edge(prof):
        return prof["verdict"] == "REGIME_STABLE" and all(m > 0 for m in prof["medians"])

    edge_counts = {str(h): 0 for h in HOLDS}
    for t in ticker_order:
        for h in HOLDS:
            if is_edge(hold_sums[str(h)]["per_asset"][t]):
                edge_counts[str(h)] += 1
    stable_assets = [t for t in ticker_order
                     if all(is_edge(hold_sums[str(h)]["per_asset"][t]) for h in HOLDS)]
    singular_assets = []
    for t in ticker_order:
        n_pos = sum(1 for h in HOLDS if is_edge(hold_sums[str(h)]["per_asset"][t]))
        if n_pos == 1:
            singular_assets.append(t)
    n_h1 = edge_counts[str(HOLDS[0])]
    n_min = min(edge_counts.values())
    if n_min >= 7 and not singular_assets and n_h1 >= 7:
        effect = "HORIZON_STABLE"
    elif singular_assets:
        effect = "HORIZON_SINGULAR"
    elif n_h1 == 0:
        effect = "HORIZON_DESTROYED"
    else:
        effect = "HORIZON_SENSITIVE"

    checks = [
        ("edge_counts", edge_counts,
         {str(h): hold_sums[str(h)]["verdict_counts"]["REGIME_STABLE"] for h in HOLDS}),
        ("stable_edge_assets", stable_assets, art["universe"]["stable_edge_assets"]),
        ("singular_assets", singular_assets, art["universe"]["singular_assets"]),
        ("effect_classification", effect, art["universe"]["effect_classification"]),
        ("n_edge_at_h1", n_h1, art["universe"]["n_edge_at_h1"]),
        ("n_edge_min_across_grid", n_min, art["universe"]["n_edge_min_across_grid"]),
    ]
    for name, recomputed, published in checks:
        ok = recomputed == published
        all_ok = all_ok and ok
        print("  {}: {} vs {} -> {}".format(name, recomputed, published,
                                            "MATCH" if ok else "MISMATCH"))

    print("\n=== Gate 4: determinism (AMZN hold=2 recomputed twice) ===")
    bars, _ = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
    r1 = [seg_candidate_and_null("AMZN", s, e, 2)[0] for lab, s, e in runs]
    r2 = [seg_candidate_and_null("AMZN", s, e, 2)[0] for lab, s, e in runs]
    det = r1 == r2
    all_ok = all_ok and det
    print("  r1={} r2={} -> {}".format(r1, r2, "identical" if det else "DIFFERENT"))

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact; the hold "
              "transform is non-trivial; the universe classification reproduces; "
              "determinism holds.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Holding-period sensitivity independent verification: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()