"""Real-data transaction-cost sensitivity of the momentum signal class.

Objective (focused task R-001): test whether the short-horizon momentum edge
that was ADMITTED AS CANDIDATE POSITIVE EVIDENCE at zero cost
(`research/checks/momentum.py`, activation 37318950814; lookback sweep
`momentum_lookback_sweep.py`, activation 37361000967) survives conservative
transaction-cost stress on the collected adjusted-close universe.

This is a *qualification* of the existing candidate, not a new signal search:
the momentum rule, lookbacks {3, 5, 10}, walk-forward windows, volatility
blocks, and coin-flip null are all unchanged from the prior momentum cells.
The only new dimension is a predeclared, bounded cost grid applied to the
repository's existing realistic cost model
(`examples/ma_crossover.py`, `examples/ma_crossover_real_data.py`,
`examples/volatility_regime_filter.py`):

    commission_per_trade = 2.0
    commission_per_share = 0.003
    slippage_cents       = 2.0
    slippage_proportional = 0.0005   (5 bps)

The cost grid multiplies every cost field by a fixed factor
m in {0.0, 0.5, 1.0, 2.0}:

    0.0  zero        -- reproduces the existing zero-cost momentum evidence
    0.5  half        -- half the realistic cost model
    1.0  realistic   -- the repository's existing realistic cost model (primary)
    2.0  conservative-- twice the realistic model (extra stress)

A-priori design (fixed BEFORE execution; NOT tuned to results):

* For each lookback L in {3, 5, 10} and each cost level m, momentum is
  walk-forward validated inside each of the 4 contiguous volatility blocks of
  the real-data regime family for every one of the 10 collected tickers
  (train=252d / test=84d / warmup=60d / overlap=60d, min 400 bars/segment),
  exactly as in `momentum_lookback_sweep.py`.
* The coin-flip null is run on the SAME segments with the SAME cost config
  (``stress_segments`` threads ``cfg`` into both the candidate sweep and
  ``noise_benchmark``), so the null is cost-matched.
* An asset has a *positive edge* at (L, m) iff its overall segment verdict is
  REGIME_STABLE and every segment median log return is > 0 -- the identical
  "EDGE" definition used by `momentum_lookback_sweep.py`.
* Let E(L, m) be the number of assets with a positive edge at (L, m). The
  per-lookback cost class is:
    - NO_EDGE_AT_ZERO_COST : E(L, 0.0) == 0 (no edge to qualify)
    - COST_FRAGILE         : E(L, 0.0) >= 1 but E(L, 1.0) == 0
                             (edge present at zero cost, gone at realistic cost)
    - COST_ERODING         : E(L, 1.0) >= 1 but E(L, 2.0) == 0
                             (survives realistic cost, lost at 2x conservative)
    - COST_ROBUST          : E(L, 1.0) >= 1 and E(L, 2.0) >= 1
                             (a positive edge survives realistic AND 2x costs)
* Overall primary verdict is the most fragile class among lookbacks with a
  zero-cost edge: SURVIVES_REALISTIC_COST unless some active lookback is
  COST_FRAGILE (then FAILS_REALISTIC_COST); a separate flag records whether
  the edge also survives the 2x conservative level.

The absolute-positive-edge definition is deliberately conservative: it asks
whether the *net-of-cost* strategy is still profitable, which is the
decision-relevant question. The artifact additionally reports the
cost-adjusted edge (candidate median minus matched null median) so a positive
gross edge that is merely eaten by costs is distinguishable from a gross edge
that disappears.

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
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)

# Repository's existing realistic cost model (examples/ma_crossover.py).
REALISTIC_COSTS = dict(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
# Predeclared bounded cost grid (multipliers on the realistic model).
COST_MULTIPLIERS = (0.0, 0.5, 1.0, 2.0)
COST_LABELS = {0.0: "zero", 0.5: "half", 1.0: "realistic", 2.0: "conservative"}

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
SWEEP_ARTIFACT = ARTIFACT_DIR / "momentum_lookback_sweep_results.json"

# A-priori classification thresholds live in classify_lookback / overall_verdict.


def cost_cfg(mult):
    """BacktestConfig with every realistic cost field scaled by ``mult``."""
    return bt.BacktestConfig(**{k: v * float(mult) for k, v in REALISTIC_COSTS.items()})


def segment_result(ticker, bars, lookback, mult):
    """Walk-forward regime-stability result for one ticker, lookback and cost
    level, with a cost-matched coin-flip null on the same segments."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=bars,
        signals=sig,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cost_cfg(mult),
    )
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


def is_edge(sr):
    """Positive REGIME_STABLE edge (identical rule to momentum_lookback_sweep)."""
    return sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"])


def mean_segment_median(cell):
    """Mean over assets of each asset's mean segment median log return."""
    vals = []
    for sr in cell.values():
        if sr["medians"]:
            vals.append(float(np.mean(sr["medians"])))
    return round(float(np.mean(vals)), 4) if vals else None


def mean_null_median(cell):
    vals = []
    for sr in cell.values():
        if sr["null_medians"]:
            vals.append(float(np.mean(sr["null_medians"])))
    return round(float(np.mean(vals)), 4) if vals else None


def classify_lookback(cells_by_mult):
    """A-priori per-lookback cost class and positive-edge counts."""
    E = {m: sum(1 for t in cells_by_mult[m] if is_edge(cells_by_mult[m][t]))
         for m in COST_MULTIPLIERS}
    if E[0.0] == 0:
        cls = "NO_EDGE_AT_ZERO_COST"
    elif E[1.0] == 0:
        cls = "COST_FRAGILE"
    elif E[2.0] >= 1:
        cls = "COST_ROBUST"
    else:
        cls = "COST_ERODING"
    return cls, E


def overall_verdict(classes):
    """Most fragile class among lookbacks with a zero-cost edge."""
    active = {lb: cls for lb, cls in classes.items()
              if cls != "NO_EDGE_AT_ZERO_COST"}
    if not active:
        return "NO_ZERO_COST_EDGE", False
    if any(cls == "COST_FRAGILE" for cls in active.values()):
        primary = "FAILS_REALISTIC_COST"
    else:
        primary = "SURVIVES_REALISTIC_COST"
    survives_2x = all(cls == "COST_ROBUST" for cls in active.values())
    return primary, survives_2x


def _segments(closes):
    """Reconstruct contiguous volatility-block segments (same rule as
    stress_segments: contiguous equal labels, min length)."""
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    segs, cur, start = [], labels[0], 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - start >= MIN_SEGMENT_BARS:
                segs.append((cur, start, i))
            cur, start = labels[i], i
    if len(labels) - start >= MIN_SEGMENT_BARS:
        segs.append((cur, start, len(labels)))
    return segs


def instrument_audit():
    """Independently expose whether the existing cost model drives the matched
    null into insolvency (negative equity), which makes ``total_return =
    end/start - 1`` a meaningless negative/negative ratio.

    Reconstructs the first walk-forward fold of the NVDA lookback-5 null at
    m=0.5 (the cell with the largest reported null median) from a fixed
    coin-flip signal and reports the OOS equity endpoints.
    """
    bars, _ = bt.load_ticker("NVDA")
    closes = bars.closes_array()
    segs = _segments(closes)
    lab, s, e = segs[0]
    seg_bars = list(bars)[s:e]
    lookback, mult = 5, 0.5
    sig = bt.random_signals(len(seg_bars), seed=42 + lookback * 1000)
    fold_bars = seg_bars[: TRAIN + TEST]
    fold_signals = sig[: TRAIN + TEST]
    res = bt.Backtest(cost_cfg(mult)).run(fold_bars, fold_signals)
    eq = res.equity_curve
    oos = eq[-TEST:]
    start_eq, end_eq = float(oos[0]), float(oos[-1])
    total_return = end_eq / start_eq - 1.0
    cum_shares = float(sum(f.shares for f in res.trades))
    final_cash = float(eq[-1] - cum_shares * fold_bars[-1].close)
    return dict(
        ticker="NVDA", lookback=lookback, cost_multiplier=mult, segment=lab,
        oos_start_equity=round(start_eq, 1),
        oos_end_equity=round(end_eq, 1),
        eq_min=round(float(eq.min()), 1),
        total_return=round(total_return, 3),
        final_cash=round(final_cash, 1),
        final_shares=round(cum_shares, 1),
        n_fills=len(res.trades),
        insolvent=bool(start_eq < 0.0 or float(eq.min()) < 0.0),
        note=("negative OOS equity (insolvency) makes total_return=end/start-1 a "
              "ratio of two negative numbers; the reported positive return is an "
              "accounting artifact, not a profit"),
    )


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        print(f"  {entry['ticker']}: {'OK' if actual == entry['checksum_sha256'] else 'MISMATCH'}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")

    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path.cwd() / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 3. Leakage review per asset (lookback=5, realistic costs) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, cost_cfg(1.0))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], realistic-cost total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "costs are applied to every fill in engine._run_bars.")

    print("\n=== 4. Momentum cost grid (lookback 3/5/10 x cost 0.0/0.5/1.0/2.0) ===")
    print("  Realistic cost model: {}".format(REALISTIC_COSTS))
    print("  Cost multipliers: {} -> {}\n".format(
        list(COST_MULTIPLIERS), [COST_LABELS[m] for m in COST_MULTIPLIERS]))

    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(tickers_univ.keys())

    # cells[lb][mult][ticker] = segment_result dict
    cells = {}
    for lb in LOOKBACKS:
        cells[lb] = {}
        for mult in COST_MULTIPLIERS:
            cells[lb][mult] = {}
            for ticker in ticker_order:
                cells[lb][mult][ticker] = segment_result(
                    ticker, tickers_univ[ticker], lb, mult)
        # concise per-lookback summary
        for mult in COST_MULTIPLIERS:
            cell = cells[lb][mult]
            E = sum(1 for t in cell if is_edge(cell[t]))
            mc = mean_segment_median(cell)
            mn = mean_null_median(cell)
            gap = (round(mc - mn, 4) if mc is not None and mn is not None else None)
            print("  lookback {:>2} | {:11s} (m={:<3}) | edge_assets={:>2}/10 | "
                  "mean_cand={:>7} | mean_null={:>7} | gap={:>7}".format(
                      lb, COST_LABELS[mult], mult, E, mc, mn, gap))
        print("")

    # Cross-consistency: zero-cost column must reproduce the prior lookback sweep.
    print("=== 5. Cross-consistency: zero-cost column vs momentum_lookback_sweep_results.json ===")
    prior = json.loads(SWEEP_ARTIFACT.read_text())
    prior_assets = prior["universe"]["per_asset"]
    xchecks = {}
    all_match = True
    for ticker in ticker_order:
        for lb in LOOKBACKS:
            sr = cells[lb][0.0][ticker]
            pref = prior_assets[ticker]
            ok = (sr["medians"] == pref["lookback_medians"][str(lb)]
                  and sr["null_medians"] == pref["lookback_null_medians"][str(lb)])
            # The prior artifact stores the lookback-5 verdict explicitly
            # (verdicts_5); at other lookbacks it stores only medians/nulls in
            # the universe view, so compare the verdict only at lookback 5.
            if lb == 5:
                ok = ok and sr["verdict"] == pref["verdicts_5"]
            xchecks[f"{ticker}|{lb}"] = "MATCH" if ok else "MISMATCH"
            if not ok:
                all_match = False
                print("  MISMATCH {} lookback {}: cand {} vs {} | null {} vs {}".format(
                    ticker, lb, sr["medians"], pref["lookback_medians"][str(lb)],
                    sr["null_medians"], pref["lookback_null_medians"][str(lb)]))
    print("  cross-checked {} (ticker, lookback) cells -> {}".format(
        len(xchecks), "ALL MATCH" if all_match else "MISMATCH FOUND"))

    # Classification.
    classes = {}
    edge_counts = {}
    for lb in LOOKBACKS:
        cells_by_mult = {m: cells[lb][m] for m in COST_MULTIPLIERS}
        cls, E = classify_lookback(cells_by_mult)
        classes[str(lb)] = cls
        edge_counts[str(lb)] = {str(m): E[m] for m in COST_MULTIPLIERS}
    primary, survives_2x = overall_verdict(classes)

    # Instrument audit: does the existing cost model drive the matched null
    # into insolvency, invalidating the total_return metric for this sweep?
    audit = instrument_audit()
    if audit["insolvent"]:
        decision = "UNVERIFIED"
        decision_reason = (
            "existing cost model is scale-dependent (per-share commission_per_share="
            "0.003 and slippage_cents=2.0); under constant-notional sizing it drives "
            "the matched coin-flip null into insolvency for low-priced split-adjusted "
            "series (NVDA OOS start equity < 0), after which total_return=end/start-1 "
            "is a negative/negative ratio whose reported positive value is an "
            "accounting artifact; this inflates null dispersion and makes REGIME_STABLE "
            "trivially achievable, so the predeclared cost-robustness classification is "
            "not decision-grade.")
    else:
        decision = primary
        decision_reason = "instrument audit found no null insolvency"

    print("\n=== 6. A-priori cost classification (predeclared rule) ===")
    for lb in LOOKBACKS:
        print("  lookback {:>2}: class={:<20} edge_counts={}".format(
            lb, classes[str(lb)], edge_counts[str(lb)]))
    print("  OVERALL primary verdict: {}".format(primary))
    print("  OVERALL survives 2x conservative cost: {}".format(survives_2x))

    print("\n=== 7. Instrument audit (negative-equity / metric validity) ===")
    print("  case: {} lookback {} m={} segment {}".format(
        audit["ticker"], audit["lookback"], audit["cost_multiplier"], audit["segment"]))
    print("  OOS start equity={} end equity={} eq_min={} total_return={}".format(
        audit["oos_start_equity"], audit["oos_end_equity"], audit["eq_min"],
        audit["total_return"]))
    print("  final_cash={} final_shares={} n_fills={}".format(
        audit["final_cash"], audit["final_shares"], audit["n_fills"]))
    print("  insolvent={} -> decision_relevant_classification={}".format(
        audit["insolvent"], decision))
    print("  reason: {}".format(decision_reason))

    # Determinism: rerun two cells and assert identical output.
    det_checks = [("AMZN", 5, 1.0), ("JPM", 10, 2.0)]
    det_ok = True
    for ticker, lb, mult in det_checks:
        r1 = segment_result(ticker, tickers_univ[ticker], lb, mult)
        r2 = segment_result(ticker, tickers_univ[ticker], lb, mult)
        same = json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
        det_ok = det_ok and same
        print("determinism {} lookback {} m={}: {}".format(
            ticker, lb, mult, "identical" if same else "DIFFERENT"))
    assert det_ok, "non-deterministic cost-sensitivity output"

    # Build the artifact (cells keyed by string cost levels).
    per_cell = {}
    for lb in LOOKBACKS:
        per_cell[str(lb)] = {}
        for mult in COST_MULTIPLIERS:
            cell = cells[lb][mult]
            mc = mean_segment_median(cell)
            mn = mean_null_median(cell)
            per_cell[str(lb)][str(mult)] = dict(
                cost_label=COST_LABELS[mult],
                mean_candidate_median=mc,
                mean_null_median=mn,
                mean_cost_adjusted_edge=(round(mc - mn, 4)
                                         if mc is not None and mn is not None else None),
                edge_asset_count=sum(1 for t in cell if is_edge(cell[t])),
                extreme_null_assets=sorted(
                    t for t in cell if cell[t]["null_dispersion"] > 0.1),
                per_asset=cell,
            )

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        cost_multipliers=list(COST_MULTIPLIERS),
        cost_labels={str(m): COST_LABELS[m] for m in COST_MULTIPLIERS},
        cost_model=REALISTIC_COSTS,
        windows=dict(train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
                     window=WINDOW, n_blocks=N_BLOCKS,
                     min_segment_bars=MIN_SEGMENT_BARS),
        ticker_order=ticker_order,
        per_cell=per_cell,
        classification=dict(
            predeclared_rule=dict(
                per_lookback=classes,
                edge_counts=edge_counts,
                overall_primary=primary,
                overall_survives_2x_conservative=survives_2x,
            ),
            instrument_audit=audit,
            decision_relevant=dict(
                classification=decision,
                reason=decision_reason,
            ),
        ),
        cross_consistency_zero_vs_prior=dict(
            artifact=str(SWEEP_ARTIFACT.relative_to(Path.cwd())),
            all_match=all_match,
            cells=xchecks,
        ),
        determinism=dict(checks=[f"{t}|{lb}|{m}" for t, lb, m in det_checks],
                         all_identical=det_ok),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity sweep across the collected "
          "universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())