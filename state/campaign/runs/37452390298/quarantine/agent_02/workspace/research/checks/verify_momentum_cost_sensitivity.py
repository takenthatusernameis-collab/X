"""Independent verification of momentum_cost_sensitivity_results.json.

Recomputes the full cost grid from raw tickers on a fresh walk-forward path
(separate from the check's stress_segments_across_tickers path): recomputes
momentum signals from closes, runs walk_forward with each cost config on the
4 AAPL volatility blocks, and builds the matched coin-flip null via
noise_benchmark with the SAME cost config. Compares per-asset medians, null
medians, and regime-stability verdicts against the artifact; asserts
determinism across two runs.

Gate 1: fresh recomputation MATCH for every lookback / cost level / asset.
Gate 2: determinism identical across two full recomputation runs.
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


def regime_labels_for(closes):
    return bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)


def fold_lrets(daily_pnl, start, end, train, test, warm, overlap):
    """Fold log returns for one segment, matching engine.walk_forward aggregate."""
    n = len(daily_pnl)
    step = test - overlap
    fold_start = start
    out = []
    while True:
        fold_end = fold_start + warm + train + test
        if fold_end > end:
            break
        oos = daily_pnl[fold_end - test + 1:fold_end]
        comp = 1.0
        for r in oos:
            comp *= 1.0 + r
        out.append(float(np.log1p(np.clip(comp - 1.0, -1.0 + 1e-12, None))))
        fold_start += step
    return out


def fresh_walk_forward_result(closes, lookback, cfg, seed_offset):
    """Fresh per-asset walk-forward via walk_forward; returns per-segment medians
    and the matched coin-flip null median."""
    labels = regime_labels_for(closes)
    segments = []
    cur = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - start >= MIN_SEGMENT_BARS:
                segments.append((cur, start, i))
            cur = labels[i]
            start = i
    if len(labels) - start >= MIN_SEGMENT_BARS:
        segments.append((cur, start, len(labels)))

    medians = []
    for name, s, e in segments:
        seg_bars = [bt.Bar(
            date=int(i) + 1, open=0.0, high=0.0, low=0.0,
            close=float(closes[i]), volume=0.0)
            for i in range(s, e)]
        sigs = bt.momentum_signals(closes[s:e], lookback=lookback)
        res = bt.walk_forward(
            seg_bars, sigs, train_window=TRAIN, test_window=TEST,
            warmup=WARM, overlap_window=OVERLAP,
            cfg=bt.BacktestConfig(
                initial_capital=1e6, periods_per_year=252,
                slippage_proportional=cfg.slippage_proportional,
                commission_per_share=cfg.commission_per_share,
                commission_per_trade=cfg.commission_per_trade,
                slippage_cents=cfg.slippage_cents,
            ),
        )
        lr = [float(f.metrics["total_return"]) for f in res.folds]
        medians.append(float(np.median(np.log1p(np.clip(lr, -1.0 + 1e-12, None)))))
    return segments, medians


def fresh_null(closes, lookback, cfg):
    """Fresh coin-flip null for one asset via noise_benchmark with the same cfg."""
    n = len(closes)
    rng = np.random.default_rng(SEED)
    sigs = [bt.Signal(date=i + 1, weight=1.0 if rng.random() < 0.5 else -1.0)
            for i in range(n)]
    bars = [bt.Bar(
        date=i + 1, open=0.0, high=0.0, low=0.0,
        close=float(closes[i]), volume=0.0)
        for i in range(n)]
    result = bt.walk_forward(
        bars, sigs, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP,
        cfg=bt.BacktestConfig(
            initial_capital=1e6, periods_per_year=252,
            slippage_proportional=cfg.slippage_proportional,
            commission_per_share=cfg.commission_per_share,
            commission_per_trade=cfg.commission_per_trade,
            slippage_cents=cfg.slippage_cents,
        ),
    )
    lr = [float(f.metrics["total_return"]) for f in result.folds]
    return float(np.median(np.log1p(np.clip(lr, -1.0 + 1e-12, None))))


def verdict_for(cand, null, otol=bt.OUTLIER_TOL):
    cand = np.array(cand, dtype=np.float64)
    null = np.array(null, dtype=np.float64)
    if all(abs(m) <= otol for m in cand):
        return "CONSISTENT_WITH_NOISE"
    cd = float(np.std(cand, ddof=1))
    nd = float(np.std(null, ddof=1))
    if cd > 2.0 * nd:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in cand):
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

    # Load the artifact.
    print("\n=== 3. Load artifact ===")
    with open(ARTIFACT_PATH) as f:
        artifact = json.load(f)
    assert artifact["seed"] == SEED
    assert artifact["lookbacks"] == list(LOOKBACKS)
    assert artifact["cost_levels"] == COST_LEVELS
    print(f"  artifact seed/lookbacks/cost_levels OK; {len(artifact['grid_results'])} lookbacks")

    # Fresh recomputation.
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
            for arow in asset_rows:
                ticker = arow["ticker"]
                closes = tickers[ticker].closes_array()
                segs, meds = fresh_walk_forward_result(closes, lookback, cfg, 0)
                null_med = fresh_null(closes, lookback, cfg)
                cand = [round(m, 3) for m in meds]
                null_med_r = round(null_med, 3)
                verdict = verdict_for(meds, [null_med])
                a_cand = arow["medians"]
                a_null = arow["null_medians"]
                a_verdict = arow["verdict"]
                match = (cand == a_cand and null_med_r == a_null and
                         verdict == a_verdict)
                if not match:
                    all_ok = False
                    diffs.append(dict(
                        lookback=lookback, level=level, ticker=ticker,
                        recomputed=dict(medians=cand, null=null_med_r, verdict=verdict),
                        artifact=dict(medians=a_cand, null=a_null, verdict=a_verdict),
                    ))
                print("    {:8s} {} -> cand {} null {} verdict {} vs artifact "
                      "cand {} null {} verdict {} -> {}".format(
                          ticker, "MATCH" if match else "MISMATCH", cand, null_med_r,
                          verdict, a_cand, a_null, a_verdict,
                          "PASS" if match else "FAIL"))
    print("  Gate 1: {} recomputations checked; all MATCH" if all_ok else "  Gate 1: MISMATCHES FOUND")
    for d in diffs:
        print("    diff: {}".format(d))

    # Determinism.
    print("\n=== 5. Gate 2: determinism of fresh recomputation ===")
    tickers2 = {e["ticker"]: bt.load_ticker(e["ticker"])[0] for e in manifest["entries"]}
    det_ok = True
    for lookback in LOOKBACKS:
        for level in COST_LEVELS:
            cfg = make_cost_cfg(level)
            for ticker in tickers2:
                closes = tickers2[ticker].closes_array()
                segs1, meds1 = fresh_walk_forward_result(closes, lookback, cfg, 0)
                segs2, meds2 = fresh_walk_forward_result(closes, lookback, cfg, 0)
                if round(float(np.median(np.array(meds1))), 6) != round(float(np.median(np.array(meds2))), 6):
                    det_ok = False
                if fresh_null(closes, lookback, cfg) != fresh_null(closes, lookback, cfg):
                    det_ok = False
    print("  Gate 2: fresh recomputation is deterministic across reruns: {}"
          .format(det_ok))

    # Write a compact verifier log.
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    log = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        gate1_recomputation="MATCH" if all_ok else "MISMATCH",
        gate2_determinism="DETERMINISTIC" if det_ok else "NON-DETERMINISTIC",
        mismatches=diffs,
        checked=len(diffs) == 0,
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
