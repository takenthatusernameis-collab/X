"""Bounded transaction-cost sensitivity for the momentum family (lookback 3/5/10).

Contract R-001 (campaign slot 02): run a bounded cost-sensitivity check for
lookback 3/5/10 momentum under the repository's existing realistic cost model and
a matched coin-flip null, and classify cost robustness. Research / simulation
only; no live trading or production execution.

During inspection of the existing framework a CRITICAL implementation defect was
found in ``research/backtest/regime_stability.py``:

  * ``momentum_signals`` computes ``mean(log(closes[i-lookback+1:i+1]))`` -- the
    mean log PRICE over the window -- and takes its sign. For every collected
    large-cap (adjusted close > 1) this is always positive, so the signal is a
    CONSTANT LONG on every bar (4460/4460 non-neutral bars long, 0 short),
    independent of the lookback. It does NOT implement "long the previous
    lookback-day return".
  * ``mean_reversion_signals`` has the same defect with the opposite sign and is
    a CONSTANT SHORT.

Consequence: the previously admitted "momentum SUPPORTED (7/10 REGIME_STABLE)"
result is really a constant-long (equity beta / drift) bet, and the "lookback
3/5/10 robustness" is vacuous (the signal is identical at every lookback -- the
prior artifact's identical medians across lookbacks are the signature of this).
The "reversal FALSIFIED" result is a constant-short, and the reported -0.99
momentum/reversal correlation is trivially -1.

Because the contract's primary question is whether the *short-horizon momentum
edge* survives costs, this check reports BOTH:
  (A) ``existing_momentum``  -- the admitted candidate exactly as implemented
      (constant long), i.e. the contracted object.
  (B) ``corrected_momentum`` -- a defect-repair diagnostic implementing the
      documented hypothesis (long the previous lookback-day RETURN, short
      otherwise). This is the same momentum hypothesis, not a new signal class,
      and it is required to answer the primary question truthfully.

A-priori prediction (fixed before running):
  * (A) constant-long is low turnover (constant notional, small daily
    rebalance), so realistic costs are a modest drag and the positive median is
    expected to SURVIVE (COST_ROBUST) -- this is a beta/drift result, not
    momentum.
  * (B) corrected momentum flips the position frequently (hundreds of sign
    flips over ~4.5k bars), so under the same per-trade cost model the edge is
    expected to be COST_FRAGILE (realistic or stress cost removes it).

Cost grid (bounded, predeclared): {zero, realistic, 2x, 4x} x the repository's
existing realistic cost model
(commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0,
slippage_proportional=0.0005), with a matched coin-flip null at each level.

Deliverable: ``state/check_artifacts/momentum_cost_sensitivity_results.json``,
recomputed independently by ``verify_momentum_cost_sensitivity.py``.
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

# The repository's existing realistic cost model (examples/ma_crossover_real_data.py).
REALISTIC_COST = dict(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

# Bounded cost grid: (label, multiplier on the realistic cost model).
COST_LEVELS = (
    ("zero", 0.0),
    ("realistic", 1.0),
    ("double", 2.0),
    ("stress", 4.0),
)
REFERENCE_LEVEL = "realistic"
STRESS_LEVEL = "stress"

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def cost_config(multiplier: float) -> bt.BacktestConfig:
    """Scale the repository's realistic cost model by ``multiplier``."""
    return bt.BacktestConfig(
        initial_capital=1e6,
        commission_per_trade=REALISTIC_COST["commission_per_trade"] * multiplier,
        commission_per_share=REALISTIC_COST["commission_per_share"] * multiplier,
        slippage_cents=REALISTIC_COST["slippage_cents"] * multiplier,
        slippage_proportional=REALISTIC_COST["slippage_proportional"] * multiplier,
    )


def corrected_momentum_signals(closes, lookback: int = 5):
    """Defect-repair diagnostic: the documented momentum hypothesis.

    Long the previous ``lookback``-day return (go long if it was up, short if it
    was down), hold one day, daily rebalance. Uses the actual RETURN
    ``log(close[i]) - log(close[i-lookback])`` (the existing implementation used
    the mean log PRICE, which is always positive and degenerates to a constant
    long). Past-only.
    """
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.log(closes[i]) - np.log(closes[i - lookback]))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


VARIANTS = (
    ("existing_momentum", bt.momentum_signals),
    ("corrected_momentum", corrected_momentum_signals),
)


def segment_result(ticker, bars, lookback, signals_fn, cfg):
    """Regime-stability result for one ticker/lookback/signal/cost level."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = signals_fn(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    res = bt.stress_segments(
        signals_fn=signals_fn,
        bars=bars,
        signals=sig,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=cfg,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
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


def framework_positive_edge(sr) -> bool:
    """The framework's strict positive-edge rule (REGIME_STABLE + all positive)."""
    return sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"])


def net_positive_edge(sr) -> bool:
    """All per-segment net-of-cost medians > 0 (raw sign check)."""
    return len(sr["medians"]) > 0 and all(m > 0 for m in sr["medians"])


def edge_present_at(sr) -> bool:
    """Primary cost-robustness signal: a distinguishable positive edge at a level.

    Edge present iff the framework does NOT call the result noise or a stable
    loss (i.e. verdict in {REGIME_STABLE, REGIME_DEPENDENT}) AND every
    per-segment net-of-cost median is positive. Treating REGIME_DEPENDENT with
    positive medians as "edge present" neutralizes the null-dispersion confound
    (the REGIME_STABLE vs REGIME_DEPENDENT label flips under cost because the
    coin-flip null's own dispersion changes, even when the candidate's positive
    medians are unchanged). CONSISTENT_WITH_NOISE (all |median| <= 0.05) and
    REGIME_STABLE_LOSS correctly exclude in-noise and negative results.
    """
    if sr["verdict"] not in ("REGIME_STABLE", "REGIME_DEPENDENT"):
        return False
    return net_positive_edge(sr)


def classify_cost_robustness(per_level):
    """Classify an asset's cost robustness from its per-level edge status.

    COST_ROBUST                 : edge present at zero, realistic, and 4x stress.
    COST_SENSITIVE_AT_STRESS    : edge at zero and realistic, lost at 4x.
    COST_SENSITIVE_AT_REALISTIC : edge at zero, lost at realistic.
    NO_EDGE_AT_ZERO_COST        : no edge even at zero cost.
    """
    edge_zero = edge_present_at(per_level["zero"])
    edge_real = edge_present_at(per_level["realistic"])
    edge_stress = edge_present_at(per_level["stress"])
    if not edge_zero:
        return "NO_EDGE_AT_ZERO_COST"
    if edge_real and edge_stress:
        return "COST_ROBUST"
    if edge_real and not edge_stress:
        return "COST_SENSITIVE_AT_STRESS"
    return "COST_SENSITIVE_AT_REALISTIC"


def turnover_profile(ticker, lookback, signals_fn):
    """Full-sample turnover / cost diagnostic on one representative ticker."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    sig = signals_fn(closes, lookback=lookback)
    res = bt.run_bars(list(bars), sig, cost_config(1.0))
    years = len(closes) / 252.0
    traded_notional = float(sum(abs(f.shares) * f.price for f in res.trades))
    total_commission = float(sum(f.commission for f in res.trades))
    return dict(
        n_fills=len(res.trades),
        annualized_turnover=round(traded_notional / 1e6 / years, 2),
        total_commission_realistic=round(total_commission, 2),
        total_return_realistic=round((res.equity_curve[-1] - 1e6) / 1e6, 3),
    )


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
    sys.path.insert(0, str(Path.cwd() / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 3. Leakage review per asset (existing implementation, lookback=5) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, leakage [PASS]")

    all_tickers = {}
    for entry in manifest["entries"]:
        all_tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]
    ticker_order = list(all_tickers.keys())

    print("\n=== 4. Cost grid: {} x {}".format(
        [lbl for lbl, _ in COST_LEVELS], list(REALISTIC_COST.keys())))
    print("  variants: existing_momentum (as implemented) | corrected_momentum (diagnostic)")
    print("  params: walk-forward per segment train={}d/test={}d/warmup={}d/overlap={}d, "
          "min {} bars/segment, 4 volatility blocks, each level vs a matched "
          "coin-flip null on the same segments\n".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # results[variant][lookback][ticker][cost_label] = segment_result dict
    results = {}
    for vname, vfn in VARIANTS:
        results[vname] = {}
        for lb in LOOKBACKS:
            results[vname][str(lb)] = {}
            for ticker in ticker_order:
                per_level = {}
                for clabel, cmult in COST_LEVELS:
                    per_level[clabel] = segment_result(
                        ticker, all_tickers[ticker], lb, vfn, cost_config(cmult))
                results[vname][str(lb)][ticker] = per_level
            # per-asset robustness classification for this variant/lookback
            cls_counts = {}
            for ticker in ticker_order:
                per_level = results[vname][str(lb)][ticker]
                cls = classify_cost_robustness(per_level)
                per_level["_cost_robustness"] = cls
                per_level["_net_positive_by_level"] = {
                    cl: net_positive_edge(per_level[cl]) for cl, _ in COST_LEVELS}
                per_level["_edge_present_by_level"] = {
                    cl: edge_present_at(per_level[cl]) for cl, _ in COST_LEVELS}
                per_level["_framework_positive_by_level"] = {
                    cl: framework_positive_edge(per_level[cl]) for cl, _ in COST_LEVELS}
                cls_counts[cls] = cls_counts.get(cls, 0) + 1
            print("  variant={} lookback={} cost-robustness counts: {}".format(
                vname, lb, cls_counts))

    print("\n=== 5. Per-asset cost-robustness (lookback 5 reference) ===")
    for vname, _ in VARIANTS:
        print("  variant: {}".format(vname))
        for ticker in ticker_order:
            row = results[vname]["5"][ticker]
            meds = {cl: row[cl]["medians"][0] for cl, _ in COST_LEVELS}
            print("    {:6s} {:28s} first-segment medians(z/r/2x/4x)="
                  "{:+.3f}/{:+.3f}/{:+.3f}/{:+.3f}".format(
                      ticker, row["_cost_robustness"],
                      meds["zero"], meds["realistic"], meds["double"], meds["stress"]))
        print("")

    # Turnover diagnostic: why the two variants differ under cost.
    print("=== 6. Turnover / cost diagnostic (AMZN, lookback 5, full sample) ===")
    turnover = {}
    for vname, vfn in VARIANTS:
        prof = turnover_profile("AMZN", 5, vfn)
        turnover[vname] = prof
        print("  {:20s} fills={} annualized_turnover={}x total_commission(realistic)={} "
              "total_return(realistic)={:+.3f}".format(
                  vname, prof["n_fills"], prof["annualized_turnover"],
                  prof["total_commission_realistic"], prof["total_return_realistic"]))
    print("")

    # Determinism: recompute one representative cell twice.
    r1 = segment_result("AMZN", all_tickers["AMZN"], 5, bt.momentum_signals, cost_config(1.0))
    r2 = segment_result("AMZN", all_tickers["AMZN"], 5, bt.momentum_signals, cost_config(1.0))
    print("determinism (existing, AMZN, lb5, realistic): r1 == r2: {}".format(r1 == r2))
    assert r1 == r2, "non-deterministic output"

    universe = {}
    for vname, _ in VARIANTS:
        universe[vname] = {}
        for lb in LOOKBACKS:
            counts = {}
            for ticker in ticker_order:
                cls = results[vname][str(lb)][ticker]["_cost_robustness"]
                counts[cls] = counts.get(cls, 0) + 1
            universe[vname][str(lb)] = dict(counts=counts, ticker_order=ticker_order)

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        realistic_cost=REALISTIC_COST,
        cost_levels=[lbl for lbl, _ in COST_LEVELS],
        classification_rule=(
            "edge_present_at(level) = verdict in {REGIME_STABLE, REGIME_DEPENDENT} "
            "and all per-segment net-of-cost medians > 0; COST_ROBUST = edge at "
            "zero+realistic+stress; COST_SENSITIVE_AT_STRESS = edge at "
            "zero+realistic, lost at stress; COST_SENSITIVE_AT_REALISTIC = edge at "
            "zero, lost at realistic; NO_EDGE_AT_ZERO_COST = no edge at zero cost."
        ),
        variants=[v for v, _ in VARIANTS],
        implementation_defect=dict(
            component="research/backtest/regime_stability.py::momentum_signals "
                      "and ::mean_reversion_signals",
            description="momentum_signals computes mean(log(price)) over the lookback "
                        "window (always positive for adjusted close > 1) and takes its "
                        "sign, so it emits a constant LONG on every bar independent of "
                        "lookback; mean_reversion_signals emits a constant SHORT. Neither "
                        "implements the documented 'previous N-day return' hypothesis.",
            evidence="probe: existing_momentum non-neutral bars all weight +1 "
                     "(0 short) on AAPL/AMZN/JPM/XOM; corrected return-based momentum "
                     "flips ~800-900 times over ~4.5k bars.",
            implication="The prior 'momentum SUPPORTED' and 'reversal FALSIFIED' verdicts "
                        "describe constant long / constant short (equity beta / drift), "
                        "not a momentum/reversal signal; lookback 3/5/10 robustness is "
                        "vacuous because the signal is identical at every lookback.",
        ),
        per_variant=results,
        universe=universe,
        turnover_profile=turnover,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production execution. "
          "Cost-sensitivity of the momentum family: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())