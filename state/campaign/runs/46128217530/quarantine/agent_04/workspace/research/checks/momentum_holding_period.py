"""Holding-period robustness test for the lookback-5 momentum signal.

Background: the momentum signal class (go LONG the previous 5-day return) is
the only candidate positive evidence in the durable record. It has survived
lookback robustness (lookback 3/5/10; activation 37361000967) and
longer-horizon robustness in magnitude (lookback 20/60; activation 37394344201).
The remaining unresolved robustness dimension is the HOLDING PERIOD: the
admitted evidence is strongest at a one-day hold, so a critic can argue the edge
is a single-point timing artifact that only pays at exactly one day.

This check tests that objection with a PREDECLARED holding-period grid. The grid
and the lookback are fixed here, before any result is inspected, and are not
tuned afterward:

    lookback = 5 (fixed, NOT swept)
    hold grid = [1, 2, 5, 10] bars  (declared a-priori)

For each hold H the candidate is the same lookback-5 momentum signal with
positions held H bars (``bt.momentum_signals(closes, lookback=5, hold=H)``).
Hold=1 reproduces the already-admitted daily-rebalance result exactly and serves
as the regression anchor against ``momentum_results.json``.

Method (fixed a-priori, existing infrastructure):
- 10-asset collected universe, manifest checksums, preflight gate, leakage
  review on AMZN/JPM.
- Regime gate: ``bt.stress_segments_across_tickers`` over the 4 volatility
  blocks per asset (framework path; authoritative candidate medians + verdict).
- Matched null: a hold-aware coin-flip null (random +/-1/0, neutral before the
  lookback, carried forward every H bars, same walk-forward settings and the
  SAME warmup as the candidate) run on the identical segments. This is the
  methodologically correct reference for a hold-H candidate: the framework's
  internal null rebalances daily (warmup=0), which is anti-conservative for
  H>1 because it changes the number of independent bets. Both the framework
  verdict and the matched-null verdict are reported.
- Determinism: the entire numeric pipeline is run twice and asserted equal.

Research / simulation only. No live trading or production execution.
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
LOOKBACK = 5                     # fixed; NOT swept (out of scope)
HOLD_GRID = [1, 2, 5, 10]        # PREDECLARED holding-period grid (bars)
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_holding_period_results.json"
BASELINE_PATH = ARTIFACT_DIR / "momentum_results.json"


def _segments_from_labels(labels, min_segment_bars):
    """Contiguous runs of equal labels with length >= min_segment_bars.

    Mirrors the segment construction inside ``bt.stress_segments`` so the
    matched-null walk-forward uses exactly the candidate's segments.
    """
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


def _matched_null_weights(n, lookback, hold, seed):
    """Hold-aware coin-flip null: random +/-1/0, neutral before lookback,
    carried forward every ``hold`` bars (same structure as the candidate)."""
    rng = np.random.default_rng(seed)
    raw = rng.choice([-1.0, 0.0, 1.0], size=n, p=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    out = [0.0] * n
    last = 0.0
    for i in range(n):
        if i < lookback:
            out[i] = 0.0
            continue
        if (i - lookback) % hold == 0:
            last = float(raw[i])
        out[i] = last
    return out


def _segment_median(bars_list, weights, s, e, train, test, warm, overlap):
    """Median fold log return of a weight series over one segment, via a fresh
    ``walk_forward`` (no call to stress_segments)."""
    seg_bars = bars_list[s:e]
    seg_w = weights[s:e]
    signals = [bt.Signal(date=b.date, weight=float(w))
               for b, w in zip(seg_bars, seg_w)]
    cfg = bt.BacktestConfig(warmup_periods=warm)
    res = bt.walk_forward(
        seg_bars, signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    log = np.array(
        [np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
         for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def _verdict(cand_medians, null_medians):
    """Regime verdict from candidate/null segment medians.

    Identical decision logic to ``bt.stress_segments`` (CONSISTENT_WITH_NOISE /
    REGIME_DEPENDENT / REGIME_STABLE_LOSS / REGIME_STABLE) so the matched-null
    verdict is directly comparable to the framework verdict.
    """
    cand = np.asarray(cand_medians, dtype=np.float64)
    null = np.asarray(null_medians, dtype=np.float64)
    if all(abs(m) <= bt.OUTLIER_TOL for m in cand):
        return "CONSISTENT_WITH_NOISE"
    if float(np.std(cand)) > 2.0 * float(np.std(null)):
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in cand):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def _positive_stable_assets(per_asset):
    """Tickers whose matched-null verdict is REGIME_STABLE with all medians > 0
    (a positive, regime-stable edge)."""
    out = set()
    for ticker, r in per_asset.items():
        if r["matched_verdict"] == "REGIME_STABLE" and all(
                m > 0 for m in r["candidate_medians"]):
            out.add(ticker)
    return out


def compute_all(tickers, ticker_order):
    """Run the full holding-period grid. Pure/deterministic in its inputs."""
    per_hold = {}
    for hold in HOLD_GRID:
        # --- Candidate: framework regime gate (authoritative path) ---
        summary = bt.stress_segments_across_tickers(
            tickers=tickers,
            signals_fn=lambda c, **kw: bt.momentum_signals(
                c, lookback=LOOKBACK, hold=hold),
            regime_labels_fn=lambda c: bt.volatility_blocks(
                c, n_blocks=N_BLOCKS, window=WINDOW),
            baseline=(("lookback", LOOKBACK), ("hold", hold)),
            param_grid=[{"lookback": LOOKBACK, "hold": hold}],
            train_window=TRAIN, test_window=TEST, warmup=WARM,
            overlap_window=OVERLAP, periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
        )

        per_asset = {}
        for idx, ticker in enumerate(ticker_order):
            bars = tickers[ticker]
            bars_list = list(bars)
            closes = bars.closes_array()
            labels = bt.volatility_blocks(
                closes, n_blocks=N_BLOCKS, window=WINDOW)
            segments = _segments_from_labels(labels, MIN_SEGMENT_BARS)

            res = summary.assets[ticker]
            cand_raw = [float(s.baseline_median_log_return) for s in res.scenarios]
            framework_verdict = res.overall_verdict

            # --- Matched hold-aware null on the identical segments ---
            null_seed = SEED + 100 * hold + idx
            null_w = _matched_null_weights(len(closes), LOOKBACK, hold, null_seed)
            null_medians = []
            for _lab, s, e in segments:
                m, _nf = _segment_median(
                    bars_list, null_w, s, e, TRAIN, TEST, WARM, OVERLAP)
                null_medians.append(m)
            matched_verdict = _verdict(cand_raw, null_medians)

            per_asset[ticker] = dict(
                segments=[lab for lab, _s, _e in segments],
                candidate_medians=[round(m, 3) for m in cand_raw],
                null_medians=[round(m, 3) for m in null_medians],
                candidate_dispersion=round(res.candidate_dispersion, 3),
                null_dispersion=round(float(np.std(null_medians)), 3),
                framework_verdict=framework_verdict,
                matched_verdict=matched_verdict,
                n_folds=res.n_folds,
            )

        framework_counts = {"CONSISTENT_WITH_NOISE": 0, "REGIME_STABLE": 0,
                            "REGIME_STABLE_LOSS": 0, "REGIME_DEPENDENT": 0}
        matched_counts = dict(framework_counts)
        for r in per_asset.values():
            framework_counts[r["framework_verdict"]] += 1
            matched_counts[r["matched_verdict"]] += 1

        per_hold[hold] = dict(
            hold=hold,
            per_asset=per_asset,
            framework_verdict_counts=framework_counts,
            matched_verdict_counts=matched_counts,
            positive_stable_assets=sorted(_positive_stable_assets(per_asset)),
        )

    # Holding-period classification (descriptive, from the grid above).
    pos_sets = {h: set(per_hold[h]["positive_stable_assets"]) for h in HOLD_GRID}
    base = pos_sets[HOLD_GRID[0]]
    classification = {
        "positive_stable_assets_by_hold": {
            h: sorted(pos_sets[h]) for h in HOLD_GRID},
        "hold1_positive_stable": sorted(base),
        "holds_retaining_all_hold1_assets": [
            h for h in HOLD_GRID if base.issubset(pos_sets[h])],
        "holds_retaining_any_hold1_asset": [
            h for h in HOLD_GRID if base & pos_sets[h]],
        "n_holds": len(HOLD_GRID),
    }
    return dict(per_hold=per_hold, classification=classification)


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")

    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 3. Leakage review (AMZN/JPM, lookback=5, hold in grid) ===")
    for ticker in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        for hold in HOLD_GRID:
            signals = bt.momentum_signals(closes, lookback=LOOKBACK, hold=hold)
            bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
            res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
            bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        print(f"  {ticker}: {bars.n_bars} bars, holds {HOLD_GRID} "
              f"leakage [PASS] (signal at t uses closes[:t] only; "
              f"carried weights were computed at earlier bars)")
    print("  no-look-ahead: hold>1 carries a weight computed at an earlier bar "
          "from earlier closes only.")

    print("\n=== 4. Load universe (manifest order) ===")
    tickers = {}
    ticker_order = []
    for entry in manifest["entries"]:
        t = entry["ticker"]
        tickers[t] = bt.load_ticker(t)[0]
        ticker_order.append(t)
    print(f"  {len(ticker_order)} assets: {', '.join(ticker_order)}")
    print(f"  lookback={LOOKBACK} (fixed) | hold grid={HOLD_GRID} (predeclared) | "
          f"train={TRAIN}/test={TEST}/warmup={WARM}/overlap={OVERLAP}")

    print("\n=== 5. Holding-period grid (candidate + matched hold-aware null) ===")
    art1 = compute_all(tickers, ticker_order)
    art2 = compute_all(tickers, ticker_order)
    deter = json.dumps(art1, sort_keys=True) == json.dumps(art2, sort_keys=True)
    print(f"  determinism: r1 == r2: {deter}")
    assert deter, "non-deterministic output"

    for hold in HOLD_GRID:
        ph = art1["per_hold"][hold]
        print(f"\n  hold={hold}:")
        print(f"    framework verdict counts: {ph['framework_verdict_counts']}")
        print(f"    matched-null verdict counts: {ph['matched_verdict_counts']}")
        print(f"    positive REGIME_STABLE assets: {ph['positive_stable_assets']}")
        for t in ticker_order:
            r = ph["per_asset"][t]
            meds = "[" + ",".join(str(m) for m in r["candidate_medians"]) + "]"
            print(f"      {t:6s} cand={meds} "
                  f"fw={r['framework_verdict']} matched={r['matched_verdict']}")

    print("\n=== 6. Holding-period classification ===")
    cls = art1["classification"]
    print(f"  positive REGIME_STABLE assets by hold: {cls['positive_stable_assets_by_hold']}")
    print(f"  hold=1 positive-stable set: {cls['hold1_positive_stable']}")
    print(f"  holds retaining ALL hold-1 assets: {cls['holds_retaining_all_hold1_assets']}")
    print(f"  holds retaining ANY hold-1 asset: {cls['holds_retaining_any_hold1_asset']}")

    # --- Regression anchor: hold=1 candidate medians must reproduce the
    # already-admitted momentum_results.json universe medians exactly. ---
    print("\n=== 7. Regression anchor vs momentum_results.json (hold=1) ===")
    anchor_ok = True
    if BASELINE_PATH.exists():
        with open(BASELINE_PATH) as f:
            base = json.load(f)
        base_u = base["universe"]["per_asset"]
        for t in ticker_order:
            pub = base_u[t]["medians"]
            new = art1["per_hold"][1]["per_asset"][t]["candidate_medians"]
            ok = pub == new
            anchor_ok = anchor_ok and ok
            print(f"  {t:6s} published={pub} new={new} -> {'MATCH' if ok else 'MISMATCH'}")
        print(f"  anchor: {'MATCH' if anchor_ok else 'MISMATCH'}")
    else:
        anchor_ok = False
        print(f"  baseline artifact missing: {BASELINE_PATH}")
    assert anchor_ok, "hold=1 candidate medians do not reproduce the admitted result"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookback=LOOKBACK,
        hold_grid=HOLD_GRID,
        predeclared=True,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        ticker_order=ticker_order,
        determinism=deter,
        regression_anchor_hold1=anchor_ok,
        per_hold=art1["per_hold"],
        classification=art1["classification"],
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Holding-period robustness of lookback-5 momentum: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())