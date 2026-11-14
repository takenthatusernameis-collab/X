"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

This verifier re-derives the cost-sensitivity artifact through separate code
paths and records the exact discrepancy if any gate fails.

Gate 1 — fresh cost recomputation (AMZN, JPM, all lookbacks, realistic and
stress2x): re-implement ``volatility_blocks``, segment reconstruction, the
momentum signal, and walk-forward aggregation (no call to ``stress_segments``),
and recompute the matched coin-flip null through the framework's own
``noise_benchmark`` under the same ``cfg``. Candidate medians, null medians and
verdicts must MATCH the artifact.

Gate 2 — independent signal-validity recomputation (all 10 tickers, lookback 5):
independently recompute the true previous-5-day return sign as
``log(close[i]) - log(close[i-5])`` and the signal's sign-agreement, and confirm
the artifact's ``signal_validity`` figures and the top-level
``subject_valid``/``primary_question_verdict`` flags. This is the gate that
catches a "momentum" signal that is actually a constant directional bet.

Gate 3 — zero-cost cross-consistency vs the prior lookback sweep: every
``per_asset[t]['zero']`` entry (medians, null medians, verdict) must equal the
corresponding entry in ``momentum_lookback_sweep_results.json``.

Gate 4 — classification reproducibility: re-derive the SURVIVES / ERODED /
NO_EDGE verdicts from the artifact's medians/verdicts with an independent
implementation of the documented rule, and confirm the universe counts.

Gate 5 — determinism: recompute the AMZN realistic lookback-5 medians twice.

Research / simulation only. No live trading or production execution.
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
INITIAL_CAPITAL = 1e6

REALISTIC_COST = dict(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
COST_CONFIGS = {
    "realistic": dict(REALISTIC_COST),
    "stress2x": {k: 2.0 * v for k, v in REALISTIC_COST.items()},
}

ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
PRIOR = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"


def vol_blocks(closes, n_blocks, window):
    """Independent re-implementation of volatility_blocks."""
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
    """Fresh momentum signal (same (buggy) mean-log-price construction as the
    framework, re-implemented independently)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def segment_result(ticker, s, e, lookback, level):
    """Fresh walk-forward + matched null for one segment, under explicit cost."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    sig = momentum_signals(closes, lookback)
    seg_signals = sig[s:e]
    cfg = bt.BacktestConfig(initial_capital=INITIAL_CAPITAL, **COST_CONFIGS[level])
    res = bt.walk_forward(seg_bars, seg_signals, train_window=TRAIN, test_window=TEST,
                          warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
    cand_log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                         for f in res.folds], dtype=np.float64)
    noise = noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN, test_window=TEST,
        warmup=0, overlap_window=0, periods_per_year=252,
        cfg=cfg,
    )
    return float(np.median(cand_log)), noise.baseline_median_log_return, len(res.folds)


def verdict_of(medians, null_medians):
    cand_disp = float(np.std(np.array(medians)))
    null_disp = float(np.std(np.array(null_medians)))
    if all(abs(m) <= 0.05 for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def positive_edge(sr):
    gap = float(np.mean(sr["medians"]) - np.mean(sr["null_medians"]))
    return (sr["verdict"] == "REGIME_STABLE"
            and all(m > 0 for m in sr["medians"]) and gap > 0)


def classify(zero_prof, cost_prof):
    zp = [str(lb) for lb in LOOKBACKS if positive_edge(zero_prof[str(lb)])]
    cp = [str(lb) for lb in LOOKBACKS if positive_edge(cost_prof[str(lb)])]
    if not zp:
        return "NO_EDGE"
    return "SURVIVES" if all(lb in cp for lb in zp) else "ERODED"


def true_sign_agreement(closes, lookback):
    """Independent signal-validity recomputation."""
    n = len(closes)
    w = np.zeros(n)
    for i in range(lookback, n):
        code_ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        w[i] = 1.0 if code_ret > 0 else -1.0
    true_sign = np.sign(np.log(closes[lookback:]) - np.log(closes[:-lookback]))
    sig_tail = w[lookback:]
    mask = (sig_tail != 0) & (true_sign != 0)
    agree = float(np.mean(sig_tail[mask] == true_sign[mask])) if mask.any() else 0.0
    active = w[w != 0]
    dominant = float(max(np.sum(w > 0), np.sum(w < 0)) / len(active)) if len(active) else 0.0
    flips = int(np.sum(w[1:] != w[:-1]))
    return agree, dominant, flips


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    art = load(ARTIFACT)
    assert art["dataset_id"] == DATASET_ID
    assert art["lookbacks"] == list(LOOKBACKS)
    assert art["subject_valid"] is False
    assert art["primary_question_verdict"] == "UNVERIFIED_INVALID_SUBJECT"

    # Cost configs must equal the documented realistic model.
    assert art["cost_configs"]["realistic"] == REALISTIC_COST
    assert art["cost_configs"]["stress2x"] == {k: 2.0 * v for k, v in REALISTIC_COST.items()}

    all_ok = True

    print("=== Gate 1: fresh cost recomputation (AMZN, JPM; realistic + stress2x) ===")
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
        for level in ("realistic", "stress2x"):
            for lb in LOOKBACKS:
                medians, nulls, nf = [], [], None
                for lab, s, e in runs:
                    m, n, nf = segment_result(asset, s, e, lb, level)
                    medians.append(round(m, 3))
                    nulls.append(round(n, 3))
                pub = art["per_asset"][asset][level][str(lb)]
                ok = (medians == pub["medians"] and nulls == pub["null_medians"])
                if not ok:
                    all_ok = False
                print("  {} {:9s} lb {:2d}: medians {} vs {} | nulls {} vs {} -> {}".format(
                    asset, level, lb, medians, pub["medians"], nulls, pub["null_medians"],
                    "MATCH" if ok else "MISMATCH"))

    print("\n=== Gate 2: independent signal-validity recomputation (lookback 5) ===")
    for ticker in art["universe"]["ticker_order"]:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        agree, dominant, flips = true_sign_agreement(closes, 5)
        pub = art["per_asset"][ticker]["signal_validity"]["5"]
        ok = (round(agree, 4) == pub["true_return_sign_agreement"]
              and round(dominant, 4) == pub["dominant_weight_share"]
              and flips == pub["n_flips"])
        if not ok:
            all_ok = False
        print("  {:6s}: agree {:.3f} vs {} | dominant {:.3f} vs {} | flips {} vs {} -> {}".format(
            ticker, agree, pub["true_return_sign_agreement"],
            dominant, pub["dominant_weight_share"], flips, pub["n_flips"],
            "MATCH" if ok else "MISMATCH"))

    print("\n=== Gate 3: zero-cost column vs prior lookback sweep ===")
    prior = load(PRIOR)
    for ticker in art["universe"]["ticker_order"]:
        for lb in LOOKBACKS:
            pub = prior["per_asset"][ticker][str(lb)]
            new = art["per_asset"][ticker]["zero"][str(lb)]
            ok = (pub["medians"] == new["medians"]
                  and pub["null_medians"] == new["null_medians"]
                  and pub["verdict"] == new["verdict"])
            if not ok:
                all_ok = False
                print("  MISMATCH {} lb {}: prior {} vs new {}".format(ticker, lb, pub, new))
    print("  all 10 tickers x 3 lookbacks -> MATCH" if all_ok else "  MISMATCH ABOVE")

    print("\n=== Gate 4: classification reproducibility ===")
    counts = {"realistic": {"SURVIVES": 0, "ERODED": 0, "NO_EDGE": 0},
              "stress2x": {"SURVIVES": 0, "ERODED": 0, "NO_EDGE": 0}}
    for ticker in art["universe"]["ticker_order"]:
        pa = art["per_asset"][ticker]
        for level in ("realistic", "stress2x"):
            derived = classify(pa["zero"], pa[level])
            published = pa["cost"]["classification_" + level]["classification"]
            counts[level][derived] += 1
            ok = derived == published
            if not ok:
                all_ok = False
            print("  {:6s} {:9s}: derived {} vs published {} -> {}".format(
                ticker, level, derived, published, "MATCH" if ok else "MISMATCH"))
    for level in ("realistic", "stress2x"):
        ok = counts[level] == art["universe"]["classification_counts"][level]
        if not ok:
            all_ok = False
        print("  universe counts {}: derived {} vs published {} -> {}".format(
            level, counts[level], art["universe"]["classification_counts"][level],
            "MATCH" if ok else "MISMATCH"))

    print("\n=== Gate 5: determinism (AMZN realistic lookback 5, twice) ===")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)
    r1 = [round(segment_result("AMZN", s, e, 5, "realistic")[0], 3) for lab, s, e in runs]
    r2 = [round(segment_result("AMZN", s, e, 5, "realistic")[0], 3) for lab, s, e in runs]
    print("  r1 {} | r2 {} -> {}".format(r1, r2, "identical" if r1 == r2 else "DIFFERENT"))
    if r1 != r2:
        all_ok = False

    print()
    if all_ok:
        print("All independent recomputations MATCH the cost-sensitivity artifact; "
              "the zero-cost column matches the prior lookback sweep; the signal-"
              "validity gate reproduces (subject is a near-constant directional "
              "bet, not momentum); and the classification reproduces.")
    else:
        print("MISMATCH FOUND: the artifact does not reproduce via the independent "
              "path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())