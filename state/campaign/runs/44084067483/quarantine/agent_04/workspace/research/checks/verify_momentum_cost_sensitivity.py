"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Gates (each recomputes through a separate code path from the check):

1. Fresh recomputation. AMZN and JPM are re-run through ``walk_forward`` +
   ``noise_benchmark`` directly (NOT ``stress_segments``) for every
   lookback in {3, 5, 10} and every cost scale in {0.0, 1.0, 2.0, 5.0}, with
   freshly reimplemented volatility blocks and segment reconstruction. Every
   per-asset segment median and null median is compared to the artifact.

2. Zero-cost cross-consistency. The 0.0x lookback-5 column is the same zero-cost
   computation as `momentum_lookback_sweep_results.json` (the engine default
   config), so its per-asset medians must match that artifact exactly.

3. Classification reproducibility. The candidate-only uniform-positive counts
   and the predeclared cost-robustness classification are re-derived from the
   artifact's own medians, independently of the check's code, and compared.

4. Signal-activity defect check (decisive). The momentum signal's annualized
   sign changes are recomputed from raw closes for AMZN/AAPL/JPM. A genuine
   short-horizon timing signal must change position frequently; if the signal is
   a static position, cost sensitivity is vacuous and the prior momentum
   evidence cannot be read as a momentum edge.

5. Determinism of the verification path itself.
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
LOOKBACKS = (3, 5, 10)
COST_SCALES = (0.0, 1.0, 2.0, 5.0)
REALISTIC_COST = dict(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
SWEEP_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"


def cost_config(scale):
    return bt.BacktestConfig(
        initial_capital=1e6,
        target_exposure=1.0,
        commission_per_trade=REALISTIC_COST["commission_per_trade"] * scale,
        commission_per_share=REALISTIC_COST["commission_per_share"] * scale,
        slippage_cents=REALISTIC_COST["slippage_cents"] * scale,
        slippage_proportional=REALISTIC_COST["slippage_proportional"] * scale,
    )


def vol_blocks(closes, n_blocks, window):
    n = len(closes)
    block_size = n // n_blocks
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(
                float(np.std(np.log(closes[i - window: i]), ddof=1)) * np.sqrt(252.0))
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            b = i // block_size
            s, e = b * block_size, (b + 1) * block_size
            bmed = float(np.median([v for v in bar_vols[s:e] if v > 0])
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
    """Fresh implementation of the as-shipped momentum signal (independent path)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        q = float(np.mean(np.log(closes[i - lookback + 1: i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if q > 0 else -1.0)
    return out


def segment_result(bars, s, e, lookback, scale):
    """Direct walk_forward + noise_benchmark for one segment (no stress_segments)."""
    seg_bars = list(bars)[s:e]
    # The segment slice restarts the lookback window; pad with the pre-segment
    # closes so the first post-warmup signal is identical to the full-series one.
    full_closes = np.array([float(b.close) for b in bars], dtype=np.float64)
    full_sig = momentum_signals(full_closes, lookback)[s:e]
    res = bt.walk_forward(
        seg_bars, full_sig, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cost_config(scale))
    cand = float(np.median(
        np.log1p(np.clip([f.metrics["total_return"] for f in res.folds],
                         -1.0 + 1e-12, None))))
    noise = noise_benchmark(
        bars=seg_bars, param_grid=[{"lookback": lookback}],
        train_window=TRAIN, test_window=TEST, warmup=0, overlap_window=0,
        cfg=cost_config(scale), periods_per_year=252)
    return cand, float(noise.baseline_median_log_return)


def load(path):
    with open(path) as f:
        return json.load(f)


def classify(base, real, cons):
    if real == 0:
        return "COST_DESTROYED"
    if real >= base and cons >= base:
        return "COST_ROBUST"
    if real >= base and cons >= base - 1:
        return "COST_TOLERANT"
    return "COST_SEVERE"


def main() -> int:
    art = load(ARTIFACT)
    assert art["dataset_id"] == DATASET_ID
    assert art["lookbacks"] == list(LOOKBACKS)
    assert art["cost_scales"] == list(COST_SCALES)
    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM; all lookbacks x costs) ===")
    for ticker in ["AMZN", "JPM"]:
        bars, _ = bt.load_ticker(ticker)
        labels = vol_blocks(bars.closes_array(), N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]
        for lookback in LOOKBACKS:
            for scale in COST_SCALES:
                meds, nulls = [], []
                for lab, s, e in runs:
                    m, n = segment_result(bars, s, e, lookback, scale)
                    meds.append(round(m, 3))
                    nulls.append(round(n, 3))
                pub = art["grid"][str(lookback)][str(scale)]["per_asset"][ticker]
                ok = (meds == pub["medians"] and nulls == pub["null_medians"]
                      and lbls == pub["segments"])
                if not ok:
                    all_ok = False
                print("  {} lb={} cost={:.1f}x -> {} | cand {} vs {} | null {} vs {}".format(
                    ticker, lookback, scale, "MATCH" if ok else "MISMATCH",
                    meds, pub["medians"], nulls, pub["null_medians"]))
    print("")

    print("=== Gate 2: zero-cost lookback-5 column vs momentum_lookback_sweep_results.json ===")
    sweep = load(SWEEP_ARTIFACT)
    cmismatch = False
    for ticker in sweep["universe"]["per_asset"]:
        pub5 = art["grid"]["5"]["0.0"]["per_asset"][ticker]["medians"]
        ref5 = sweep["universe"]["per_asset"][ticker]["lookback_medians"]["5"]
        ok = pub5 == ref5
        if not ok:
            cmismatch = True
            all_ok = False
        print("  {}: cost-check {} vs lookback-sweep {} -> {}".format(
            ticker, pub5, ref5, "MATCH" if ok else "MISMATCH"))
    if not cmismatch:
        print("  all 10 tickers cross-checked -> MATCH")
    print("")

    print("=== Gate 3: classification reproducibility from artifact medians ===")
    rmismatch = False
    for lookback in LOOKBACKS:
        counts = {}
        for scale in COST_SCALES:
            per = art["grid"][str(lookback)][str(scale)]["per_asset"]
            up = sum(1 for t in per if all(m > 0 for m in per[t]["medians"]))
            counts[str(scale)] = up
        base = counts["0.0"]
        real = counts["1.0"]
        cons = counts["2.0"]
        cls = classify(base, real, cons)
        pub = art["per_lookback"][str(lookback)]
        ok = (cls == pub["classification"]
              and base == pub["base_uniform_positive"]
              and real == pub["realistic_uniform_positive"]
              and cons == pub["conservative_uniform_positive"])
        if not ok:
            rmismatch = True
            all_ok = False
        print("  lb={}: base={} real={} cons={} -> {} vs published {} -> {}".format(
            lookback, base, real, cons, cls, pub["classification"],
            "MATCH" if ok else "MISMATCH"))
    if not rmismatch:
        print("  all lookback classifications reproduce -> MATCH")
    print("")

    print("=== Gate 4: signal-activity defect check (decisive) ===")
    defect_cells = 0
    total_cells = 0
    for ticker in ["AAPL", "AMZN", "JPM"]:
        bars, _ = bt.load_ticker(ticker)
        closes = bars.closes_array()
        for lookback in LOOKBACKS:
            w = np.array([s.weight for s in momentum_signals(closes, lookback)])
            changes = int(np.sum(w[1:] != w[:-1]))
            total_cells += 1
            if changes <= 2:
                defect_cells += 1
            pub = art["turnover"][ticker][str(lookback)]
            ok = (pub["changes"] == changes)
            if not ok:
                all_ok = False
            print("  {} lb={}: sign changes={} (annualized {:.1f}/yr) | artifact changes={} "
                  "-> {}".format(ticker, lookback, changes, changes / (len(w) / 252.0),
                                 pub["changes"], "MATCH" if ok else "MISMATCH"))
    print("  static-position cells: {}/{} -> {}".format(
        defect_cells, total_cells,
        "DEFECT CONFIRMED: signal is a static position; cost sensitivity is vacuous"
        if defect_cells == total_cells else "signal has genuine turnover"))
    if defect_cells == total_cells:
        # Not a pass/fail of the cost artifact, but a recorded decisive finding.
        print("  NOTE: this is not a mismatch of the cost artifact; it is an "
              "independent confirmation that the as-implemented momentum signal "
              "never changes direction, so the cost-robustness result is "
              "vacuous as evidence about a momentum edge.")
    print("")

    print("=== Gate 5: verification-path determinism ===")
    bars, _ = bt.load_ticker("AMZN")
    labels = vol_blocks(bars.closes_array(), N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    a = [round(segment_result(bars, s, e, 5, 1.0)[0], 3) for _, s, e in runs]
    b = [round(segment_result(bars, s, e, 5, 1.0)[0], 3) for _, s, e in runs]
    ok = a == b
    if not ok:
        all_ok = False
    print("  AMZN lb=5 cost=1.0x r1={} r2={} -> {}".format(a, b, "identical" if ok else "DIFFERENT"))
    print("")

    if all_ok:
        print("All gates MATCH: the cost artifact reproduces via an independent "
              "walk_forward + noise_benchmark path, the zero-cost column "
              "cross-checks against momentum_lookback_sweep_results.json, and the "
              "classification reproduces.")
    else:
        print("MISMATCH FOUND: the cost artifact does not reproduce. Investigate "
              "before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification.")
    return 0


if __name__ == "__main__":
    sys.exit(main())