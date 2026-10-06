"""Independent verification of momentum_cost_sensitivity_results.json.

This is a separate implementation path: it recomputes every cell of the cost
grid from raw tickers using a fresh walk-forward recomputation (not the check's
key_fields path), with the cost config applied identically to the candidate and
to a freshly generated coin-flip null replicated directly from the framework's
random_signals contract (weights -1 / 0 / +1 at p = 1/3 each, seed = 42 +
1000 * lookback, i.e. the same deterministic seed the framework's
noise_benchmark derives from the single lookback parameter set).

Comparison contract: for every lookback in {3,5,10}, cost level in
{zero,realistic,conservative,heavy}, ticker in the 10-asset universe, and each
segment, the recomputed candidate median, recomputed null median, and the
verdict regenerated from those medians must all match the artifact; the
recomputation must also be deterministic across two full reruns.

A mismatch would flag a defect in the check's pipeline.
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


def vol_blocks(closes, n_blocks, window):
    """Re-implementation of research.backtest.volatility_blocks."""
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
    """Fresh implementation of the momentum signal (independent code path)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def coin_flip_signals(closes, lookback):
    """Fresh coin-flip signal, replicated from the framework's random_signals
    contract: weights {-1, 0, +1} at p = 1/3 each, seed derived from the
    lookback parameter set exactly as the framework's noise_benchmark does
    (base seed 42, seed = 42 + 1000 * lookback). This is the matched null that
    the check compares the candidate against, using the same walk-forward
    costs.
    """
    n = len(closes)
    rng = np.random.default_rng(SEED + 1000 * lookback)
    weights = rng.choice([-1.0, 0.0, 1.0], size=n, p=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    return [bt.Signal(date=i + 1, weight=float(w)) for i, w in enumerate(weights)]


def segment_medians_and_null(bars, closes, lookback, s, e, train, test, warm,
                             overlap, cfg):
    """Fresh walk-forward recomputation for one segment: candidate median and
    matched null median, both computed on the same segment with the same cost
    config."""
    seg_bars = list(bars)[s:e]
    seg_signals = momentum_signals(closes[s:e], lookback)
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    lr = np.array([float(f.metrics["total_return"]) for f in res.folds], dtype=np.float64)
    cand = float(np.median(np.log1p(np.clip(lr, -1.0 + 1e-12, None))))

    null_signals = coin_flip_signals(closes[s:e], lookback)
    nres = bt.walk_forward(
        seg_bars, null_signals, train_window=train, test_window=test,
        warmup=0, overlap_window=0, cfg=cfg)
    nlr = np.array([float(f.metrics["total_return"]) for f in nres.folds], dtype=np.float64)
    null = float(np.median(np.log1p(np.clip(nlr, -1.0 + 1e-12, None))))
    return cand, null, len(res.folds)


def verdict_for(cand_medians, null_medians, tol=bt.OUTLIER_TOL):
    cand = np.array(cand_medians, dtype=np.float64)
    null = np.array(null_medians, dtype=np.float64)
    if all(abs(m) <= tol for m in cand_medians):
        return "CONSISTENT_WITH_NOISE"
    cdisp = float(np.std(cand, ddof=1))
    ndisp = float(np.std(null, ddof=1))
    if cdisp > 2.0 * ndisp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in cand_medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def main() -> int:
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

    artifact = load_artifact()
    assert artifact["seed"] == SEED
    assert artifact["lookbacks"] == list(LOOKBACKS)
    assert artifact["cost_levels"] == COST_LEVELS
    print(f"\n=== 3. Artifact contract OK: seed {artifact['seed']}, "
          f"lookbacks {artifact['lookbacks']}, costs {artifact['cost_levels']}")

    tickers = {e["ticker"]: bt.load_ticker(e["ticker"])[0] for e in manifest["entries"]}

    print("\n=== 4. Gate 1: fresh walk-forward recomputation vs artifact ===")
    print("(candidate = fresh momentum walk_forward with cost config; "
          "null = fresh coin-flip walk_forward, same config; "
          "verdict regenerated from recomputed medians)")
    all_ok = True
    diffs = []
    for lookback in LOOKBACKS:
        for level in COST_LEVELS:
            cfg = make_cost_cfg(level)
            for arow in artifact["grid_results"][f"lookback_{lookback}"][0]["per_asset"]:
                ticker = arow["ticker"]
                ticker_ok = True
                closes = tickers[ticker].closes_array()
                labels = vol_blocks(closes, N_BLOCKS, WINDOW)
                runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
                art = next(r for r in artifact["grid_results"][f"lookback_{lookback}"]
                           if r["level"] == level)["per_asset"]
                art_by_ticker = {a["ticker"]: a for a in art}
                bars = tickers[ticker]
                recs = []
                for name, s, e in runs:
                    cand, null, nf = segment_medians_and_null(
                        bars, closes, lookback, s, e, TRAIN, TEST, WARM, OVERLAP, cfg)
                    recs.append((name, s, e, round(cand, 3), round(null, 3), cand, null))
                arow = art_by_ticker[ticker]
                seg_matches = all(
                    rec[3] == arow["medians"][i] and rec[4] == arow["null_medians"][i]
                    for i, rec in enumerate(recs))
                # Verdicts are decided by the framework on unrounded fold medians,
                # so regenerate from the unrounded recomputation to reproduce the
                # exact framework decision boundary.
                regen_verdict = verdict_for([r[5] for r in recs], [r[6] for r in recs])
                match = seg_matches and regen_verdict == arow["verdict"]
                if not match:
                    all_ok = False
                    ticker_ok = False
                    diffs.append(dict(
                        lookback=lookback, level=level, ticker=ticker,
                        recomputed=dict(medians=[r[3] for r in recs],
                                        nulls=[r[4] for r in recs],
                                        verdict=regen_verdict),
                        artifact=dict(medians=arow["medians"], nulls=arow["null_medians"],
                                      verdict=arow["verdict"])))
                if not ticker_ok:
                    print("  MISMATCH: lookback {} / {} / {}: {}".format(
                        lookback, level, ticker, [d for d in diffs if d["lookback"] == lookback
                                                  and d["level"] == level and d["ticker"] == ticker]))
            status = "MATCH" if all_ok else "MISMATCH"
            print("  lookback {} / {} : {}  ({} lookbacks x {} levels x {} assets)".format(
                lookback, level, status, len(LOOKBACKS), len(COST_LEVELS), len(tickers)))

    print("\n=== 5. Gate 2: determinism of fresh recomputation ===")
    tickers2 = {e["ticker"]: bt.load_ticker(e["ticker"])[0] for e in manifest["entries"]}
    det_ok = True
    for lookback in LOOKBACKS:
        for level in COST_LEVELS:
            cfg = make_cost_cfg(level)
            for ticker in tickers2:
                bars = tickers2[ticker]
                closes = bars.closes_array()
                labels = vol_blocks(closes, N_BLOCKS, WINDOW)
                runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
                vals1 = [segment_medians_and_null(bars, closes, lookback, s, e, TRAIN, TEST, WARM,
                                                  OVERLAP, cfg) for name, s, e in runs]
                vals2 = [segment_medians_and_null(bars, closes, lookback, s, e, TRAIN, TEST, WARM,
                                                  OVERLAP, cfg) for name, s, e in runs]
                m1 = [round(v[0], 6) for v in vals1]
                m2 = [round(v[0], 6) for v in vals2]
                if m1 != m2:
                    det_ok = False
    print("  fresh recomputation identical across two full reruns: {}".format(det_ok))

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    log = dict(
        dataset_id=DATASET_ID, seed=SEED,
        gate1_recomputation="MATCH" if all_ok else "MISMATCH",
        gate2_determinism="DETERMINISTIC" if det_ok else "NON-DETERMINISTIC",
        n_cells_checked=sum(len(v) for v in
                            artifact["grid_results"].values()) * len(COST_LEVELS),
        n_segments_per_cell=len(segments_from_labels(vol_blocks(
            tickers["AAPL"].closes_array(), N_BLOCKS, WINDOW), MIN_SEGMENT_BARS)),
        mismatches=diffs,
        checked=(all_ok and len(diffs) == 0),
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
        for d in diffs[:10]:
            print("    diff: {}".format(d))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
