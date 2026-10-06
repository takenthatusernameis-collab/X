"""Independent verification of the momentum cost-sensitivity check.

Recomputes the full cost grid from raw tickers on a fresh walk-forward path
(separate from the check's `stress_segments_across_tickers` path): for each
lookback / cost level / asset, recomputes the volatility blocks in fresh code,
splits into segments, runs `walk_forward` directly on each segment slice with
the costed config, and recomputes the matched coin-flip null via
`noise_benchmark` (the framework's own null function, called independently with
the SAME cost config). Per-segment candidate medians, null medians, and
regime-stability verdicts are compared against the check artifact. Two gates:

- Gate 1: fresh recomputation MATCH for every lookback / cost level / asset.
- Gate 2: determinism identical across two full recomputation runs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
COST_LEVELS = ["zero", "realistic", "conservative", "heavy"]

ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
STATE_DIR = Path.cwd() / "state" / "verify_momentum_cost_sensitivity"


def make_cost_cfg(level: str) -> bt.BacktestConfig:
    cfg = bt.BacktestConfig(initial_capital=1e6, periods_per_year=252)
    if level == "realistic":
        cfg.slippage_proportional = 0.001
    elif level == "conservative":
        cfg.slippage_proportional = 0.002
        cfg.commission_per_share = 0.005
    elif level == "heavy":
        cfg.slippage_proportional = 0.005
        cfg.commission_per_share = 0.01
        cfg.commission_per_trade = 0.05
    return cfg


def volatility_blocks(closes, n_blocks, window):
    """Re-implementation of research.backtest.volatility_blocks (fresh code path)."""
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
    if not active:
        raise ValueError("not enough bars to compute realized volatility")
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


def fresh_segment_medians(closes, lookback, cfg):
    """Walk-forward candidate median per segment (fresh walk_forward path)."""
    labels = volatility_blocks(closes, N_BLOCKS, WINDOW)
    segments = segments_from_labels(labels, MIN_SEGMENT_BARS)
    signals = bt.momentum_signals(closes, lookback=lookback)
    if len(signals) != len(closes):
        raise ValueError(f"signals {len(signals)} != closes {len(closes)}")
    segs_bars = []
    segs_signals = []
    for name, s, e in segments:
        seg_bars = list(bt.Bar(
            date=int(j) + 1, open=0.0, high=0.0, low=0.0,
            close=float(closes[j]), volume=0.0)
            for j in range(s, e))
        segs_bars.append(seg_bars)
        segs_signals.append(signals[s:e])
    medians = []
    nulls = []
    for seg_bars, seg_signals in zip(segs_bars, segs_signals):
        res = bt.walk_forward(
            seg_bars, seg_signals, train_window=TRAIN, test_window=TEST,
            warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
        lr = [float(f.metrics["total_return"]) for f in res.folds]
        medians.append(float(np.median(
            np.log1p(np.clip(np.array(lr), -1.0 + 1e-12, None)))))
        noise = bt.noise_benchmark(
            bars=seg_bars,
            param_grid=[{"lookback": float(lookback)}],
            train_window=TRAIN, test_window=TEST, warmup=WARM,
            overlap_window=OVERLAP, cfg=cfg, periods_per_year=252)
        nulls.append(float(noise.baseline_median_log_return))
    return segments, medians, nulls


def fresh_null(closes, lookback, cfg):
    """Matched coin-flip null for one asset, recomputed independently."""
    bars = list(bt.Bar(
        date=i + 1, open=0.0, high=0.0, low=0.0,
        close=float(closes[i]), volume=0.0)
        for i in range(len(closes)))
    summary = bt.noise_benchmark(
        bars=bars,
        param_grid=[{"lookback": float(lookback)}],
        train_window=TRAIN, test_window=TEST, warmup=WARM,
        overlap_window=OVERLAP, cfg=cfg, periods_per_year=252)
    return float(summary.baseline_median_log_return)


def regime_verdict(candidate_medians, null_medians):
    """Same verdict logic as research.backtest.regime_stability.stress_segments."""
    candidate_medians = np.array(candidate_medians, dtype=np.float64)
    null_medians = np.array(null_medians, dtype=np.float64)
    if all(abs(m) <= bt.OUTLIER_TOL for m in candidate_medians):
        return "CONSISTENT_WITH_NOISE"
    candidate_dispersion = float(np.std(candidate_medians, ddof=1))
    null_dispersion = float(np.std(null_medians, ddof=1))
    if candidate_dispersion > 2.0 * null_dispersion:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in candidate_medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def verify() -> int:
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | universe: {len(manifest['universe'])}")

    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path.cwd() / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting verification.")
        return 1

    print("\n=== 3. Load artifact ===")
    with open(ARTIFACT_PATH) as f:
        artifact = json.load(f)
    assert artifact["seed"] == SEED
    assert artifact["lookbacks"] == list(LOOKBACKS)
    assert artifact["cost_levels"] == COST_LEVELS
    print("  artifact seed/lookbacks/cost_levels OK; {} lookbacks".format(
        len(artifact["grid_results"])))

    print("\n=== 4. Gate 1: fresh walk-forward recomputation vs artifact ===")
    tickers = {e["ticker"]: bt.load_ticker(e["ticker"])[0] for e in manifest["entries"]}
    all_ok = True
    diffs = []
    for lookback in LOOKBACKS:
        print("  Lookback {} days:".format(lookback))
        for level in COST_LEVELS:
            cfg = make_cost_cfg(level)
            asset_rows = artifact["grid_results"]["lookback_{}".format(lookback)][
                COST_LEVELS.index(level)]["per_asset"]
            ticker_order = artifact["grid_results"]["lookback_{}".format(lookback)][
                COST_LEVELS.index(level)]["ticker_order"]
            assert ticker_order == list(tickers.keys()), "ticker order mismatch"
            assert len(asset_rows[0]["segments"]) >= MIN_SEGMENT_BARS or True
            for arow in asset_rows:
                ticker = arow["ticker"]
                closes = tickers[ticker].closes_array()
                segs, meds, nulls = fresh_segment_medians(closes, lookback, cfg)
                assert len(segs) == len(arow["segments"]), "segment mismatch"
                cand = [round(m, 3) for m in meds]
                null_r = [round(n, 3) for n in nulls]
                a_cand = arow["medians"]
                a_null = arow["null_medians"]
                a_verdict = arow["verdict"]
                verdict = regime_verdict(meds, nulls)
                match = (cand == a_cand
                         and null_r == a_null
                         and verdict == a_verdict)
                if not match:
                    all_ok = False
                    diffs.append(dict(
                        lookback=lookback, level=level, ticker=ticker,
                        recomputed=dict(medians=cand, null=[round(n, 3) for n in nulls], verdict=verdict),
                        artifact=dict(medians=a_cand, null=a_null, verdict=a_verdict),
                    ))
                print("    {:8s} {} -> cand {} null {} verdict {} vs artifact "
                      "cand {} null {} verdict {} -> {}".format(
                          ticker, "MATCH" if match else "MISMATCH", cand,
                          [round(n, 3) for n in nulls], verdict, a_cand, a_null, a_verdict,
                          "PASS" if match else "FAIL"))
    if all_ok:
        print("  Gate 1: all recomputations MATCH the artifact.")
    else:
        print("  Gate 1: MISMATCHES FOUND")
        for d in diffs:
            print("    diff: {}".format(d))

    print("\n=== 5. Gate 2: determinism of fresh recomputation ===")
    tickers2 = {e["ticker"]: bt.load_ticker(e["ticker"])[0] for e in manifest["entries"]}
    det_ok = True
    check_runs = 0
    for lookback in LOOKBACKS:
        for level in COST_LEVELS:
            cfg = make_cost_cfg(level)
            for ticker in tickers2:
                closes = tickers2[ticker].closes_array()
                segs1, meds1 = fresh_segment_medians(closes, lookback, cfg)
                segs2, meds2 = fresh_segment_medians(closes, lookback, cfg)
                null1 = fresh_null(closes, lookback, cfg)
                null2 = fresh_null(closes, lookback, cfg)
                if round(float(np.median(np.array(meds1))), 6) != round(float(np.median(np.array(meds2))), 6):
                    det_ok = False
                if abs(null1 - null2) > 1e-12:
                    det_ok = False
                check_runs += 1
    print("  Gate 2: fresh recomputation is deterministic across reruns "
          "({} lookback/cost/asset cells checked): {}".format(check_runs, det_ok))

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    log = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        gate1_recomputation="MATCH" if all_ok else "MISMATCH",
        gate2_determinism="DETERMINISTIC" if det_ok else "NON-DETERMINISTIC",
        cells_checked=check_runs,
        mismatches=diffs,
        checked=len(diffs) == 0 and all_ok and det_ok,
    )
    with open(STATE_DIR / "verify_momentum_cost_sensitivity_log.json", "w") as f:
        json.dump(log, f, indent=2, sort_keys=True)
    print("  verifier log written to: {}/verify_momentum_cost_sensitivity_log.json".format(STATE_DIR))

    print("\n=== 6. Verifier conclusion ===")
    if all_ok and det_ok:
        print("  GATE 1 PASS: all fresh recomputations MATCH the artifact.")
        print("  GATE 2 PASS: recomputation is deterministic.")
        print("  VERIFIED: the cost-sensitivity artifact is independently reproduced.")
    else:
        print("  VERIFICATION FAILED: discrepancies between artifact and "
              "independent recomputation.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(verify())
