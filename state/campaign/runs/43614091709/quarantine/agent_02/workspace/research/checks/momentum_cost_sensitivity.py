"""Transaction-cost sensitivity of short-horizon momentum on the collected universe.

Background: short-horizon per-asset momentum (long the previous N-day return,
hold 1 day, daily rebalance) was admitted as candidate positive evidence
(`research/checks/momentum.py`, activation 37318950814) and survived a real-data
lookback sweep 3/5/10 (`research/checks/momentum_lookback_sweep.py`, activation
37361000967). Every one of those results used the engine's default
``BacktestConfig`` — i.e. ZERO commission, ZERO slippage. The remaining
decision-relevant qualification is therefore execution-cost robustness: does the
admitted edge survive conservative transaction costs, or is it a gross-only
artifact of a zero-cost fill?

Objective (R-001): run a bounded cost-sensitivity check for lookback 3/5/10
momentum under the repository's existing realistic cost model and matched null.

Predeclared design (fixed before execution, not tuned to results):
- Universe: the 10 collected tickers (unchanged).
- Signal: ``bt.momentum_signals`` (unchanged); lookbacks {3, 5, 10}.
- Gates: the same per-asset contiguous volatility-block regime split
  (``bt.volatility_blocks``), walk-forward windows (train=252 / test=84 /
  warmup=60 / overlap=60), and matched coin-flip null
  (``bt.stress_segments`` -> ``noise_benchmark``) used by the existing momentum
  checks.
- Cost grid (a-priori, bounded, three levels):
    * ``zero``      : BacktestConfig() — zero cost anchor; must reproduce the
                      existing gross lookback sweep exactly.
    * ``canonical`` : the repository's realistic cost model
                      (examples/ma_crossover_real_data.py):
                      $2/trade + $0.003/share commission, 2c/share fixed
                      slippage, 5 bps proportional slippage.
    * ``stress2x``  : double every canonical cost component, a conservative
                      stress level.
  The cost ``cfg`` is threaded into BOTH the candidate walk-forward and the
  matched coin-flip null by ``stress_segments``, so the null carries the same
  cost structure as the candidate.

Predeclared classification (based on the lookback-5 reference edge count E_c =
number of assets whose lookback-5 result is REGIME_STABLE with uniformly
positive segment medians at cost level c; E_ref = E at ``zero``):
    COST_ROBUST        if E_canonical >= ceil(0.8 * E_ref)
                         and E_stress2x >= ceil(0.5 * E_ref)
    COST_ERASED        if E_canonical <= floor(0.2 * E_ref)
    PARTIALLY_ROBUST   otherwise
    NO_EDGE_BASELINE   if E_ref == 0 (no admitted edge to erode)

The framework verdict (``REGIME_STABLE`` positive = EDGE) is the primary
classifier because it is the existing, frozen evidence standard. The
candidate-minus-null "relative edge" (candidate median minus matched-null median)
is reported as supporting evidence: it shows whether the signal's advantage over
random trading with the same cost drag survives, separately from whether the
absolute net-of-cost strategy is profitable.

Out of scope (contract): no signal tuning after seeing cost results, no universe
expansion, no leverage. Determinism is asserted via an internal recomputation.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import math
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
REFERENCE_LOOKBACK = 5
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"

# Predeclared cost grid. The BacktestConfig objects are not JSON-serialisable,
# so they are defined here and flattened to a serialisable descriptor for the
# artifact.
COST_GRID = [
    ("zero", bt.BacktestConfig()),
    ("canonical", bt.BacktestConfig(
        commission_per_trade=2.0, commission_per_share=0.003,
        slippage_cents=2.0, slippage_proportional=0.0005,
    )),
    ("stress2x", bt.BacktestConfig(
        commission_per_trade=4.0, commission_per_share=0.006,
        slippage_cents=4.0, slippage_proportional=0.0010,
    )),
]


def cfg_descriptor(cfg: bt.BacktestConfig) -> dict:
    return dict(
        commission_per_trade=cfg.commission_per_trade,
        commission_per_share=cfg.commission_per_share,
        slippage_cents=cfg.slippage_cents,
        slippage_proportional=cfg.slippage_proportional,
    )


def segment_result(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker/lookback/cost."""
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
        cfg=cfg,
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


def lookback_verdict(sr):
    """EDGE / LOSS / NO_EDGE for one lookback's scenario result."""
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def robustness_verdict(lookback_results):
    """ROBUST / SENSITIVE / NO_EDGE for one asset across lookbacks.

    ROBUST    : positive REGIME_STABLE edge at >= 2 lookbacks.
    SENSITIVE : positive REGIME_STABLE edge at exactly one lookback.
    NO_EDGE   : no lookback shows a positive REGIME_STABLE edge.
    """
    counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    for lb in LOOKBACKS:
        counts[lookback_verdict(lookback_results[str(lb)])] += 1
    if counts["EDGE"] >= 2:
        return "ROBUST", counts
    if counts["EDGE"] == 1:
        return "SENSITIVE", counts
    return "NO_EDGE", counts


def classify(E_ref, E_canon, E_stress):
    """Predeclared cost-robustness classification (see module docstring)."""
    if E_ref == 0:
        return "NO_EDGE_BASELINE"
    if E_canon >= math.ceil(0.8 * E_ref) and E_stress >= math.ceil(0.5 * E_ref):
        return "COST_ROBUST"
    if E_canon <= math.floor(0.2 * E_ref):
        return "COST_ERASED"
    return "PARTIALLY_ROBUST"


def signal_mechanism(ticker_order, all_tickers):
    """Characterize the signal actually produced by bt.momentum_signals.

    The momentum docstring promises "long the previous lookback-day return, else
    short". This audit independently measures whether the emitted weights match
    that mechanism: a genuine momentum signal must take BOTH signs and flip
    frequently, and its weights must change when the lookback changes. A
    constant-sign, lookback-invariant signal is not momentum regardless of the
    name. Turnover (fills, dollar notional, commission) is measured on AMZN at
    zero vs canonical cost to explain the cost-sensitivity result.
    """
    per_ticker = {}
    all_constant_long = True
    for ticker in ticker_order:
        bars = all_tickers[ticker]
        closes = bars.closes_array()
        w5 = np.array([s.weight for s in bt.momentum_signals(closes, lookback=5)])
        w3 = np.array([s.weight for s in bt.momentum_signals(closes, lookback=3)])
        w10 = np.array([s.weight for s in bt.momentum_signals(closes, lookback=10)])
        impl_ret = np.array(
            [float(np.mean(np.log(closes[i - 5 + 1:i + 1]))) for i in range(5, len(closes))])
        n_pos = int(np.sum(w5 == 1.0))
        n_neg = int(np.sum(w5 == -1.0))
        constant_long = (n_neg == 0 and n_pos > 0)
        if not constant_long:
            all_constant_long = False
        per_ticker[ticker] = dict(
            n_pos=n_pos,
            n_neg=n_neg,
            n_zero=int(np.sum(w5 == 0.0)),
            sign_flips=int(np.sum(np.diff(np.sign(w5)) != 0)),
            weights_identical_across_lookbacks=bool(
                np.array_equal(w3, w5) and np.array_equal(w5, w10)),
            impl_ret_min=round(float(impl_ret.min()), 4),
            impl_ret_max=round(float(impl_ret.max()), 4),
            constant_long_only=constant_long,
        )

    bars = all_tickers["AMZN"]
    sig = bt.momentum_signals(bars.closes_array(), lookback=5)
    turnover = {}
    for name, cfg in COST_GRID:
        res = bt.run_bars(list(bars), sig, cfg)
        fills = res.trades
        turnover[name] = dict(
            n_fills=len(fills),
            dollar_notional_traded=int(round(float(np.sum(
                [abs(f.shares) * f.price for f in fills])))),
            total_commission=round(float(np.sum([f.commission for f in fills])), 2),
            equity_final=int(round(float(res.equity_curve[-1]))),
        )

    return dict(
        per_ticker=per_ticker,
        turnover_amzn=turnover,
        all_constant_long_only=all_constant_long,
        verdict="CONSTANT_LONG_ONLY" if all_constant_long else "FLUCTUATING",
        interpretation=(
            "bt.momentum_signals computes mean(log(closes)) over the lookback "
            "window (a log PRICE level, always positive for prices > 1), not the "
            "lookback log RETURN, so it emits +1 on every bar after warmup "
            "regardless of the lookback. The emitted 'momentum' signal is a "
            "constant long-only (buy-and-hold) position. Its near-zero "
            "transaction-cost drag is therefore a turnover artifact (one "
            "directional position, low traded notional), not evidence that a "
            "momentum edge survives costs."),
    )


def run_cost_level(ticker_order, all_tickers, cfg):
    """Full per-asset + universe result for one cost level."""
    lookback_results = {
        lb: {t: segment_result(t, all_tickers[t], lb, cfg) for t in ticker_order}
        for lb in LOOKBACKS
    }

    per_asset = {}
    robustness_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    per_lookback_edge_counts = {lb: 0 for lb in LOOKBACKS}
    reference_edge_tickers = []

    for ticker in ticker_order:
        profile = {str(lb): lookback_results[lb][ticker] for lb in LOOKBACKS}
        v, counts = robustness_verdict(profile)
        robustness_counts[v] += 1
        profile["edge_counts"] = counts
        profile["robustness_verdict"] = v
        # Relative edge over the matched null at the reference lookback.
        ref = profile[str(REFERENCE_LOOKBACK)]
        profile["relative_edge_5"] = round(
            float(np.mean(ref["medians"])) - float(np.mean(ref["null_medians"])), 3)
        per_asset[ticker] = profile
        for lb in LOOKBACKS:
            if lookback_verdict(profile[str(lb)]) == "EDGE":
                per_lookback_edge_counts[lb] += 1
        if lookback_verdict(profile[str(REFERENCE_LOOKBACK)]) == "EDGE":
            reference_edge_tickers.append(ticker)

    universe = dict(
        per_lookback_edge_counts=per_lookback_edge_counts,
        reference_edge_count=len(reference_edge_tickers),
        reference_edge_tickers=reference_edge_tickers,
        robustness_counts=robustness_counts,
        ticker_order=ticker_order,
    )
    return per_asset, universe


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

    print("\n=== 3. Leakage review per asset (lookback=5 reference) ===")
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
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "costs are applied by the engine on realized fills only.")

    print("\n=== 4. Loading full universe ===")
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())
    print(f"  tickers: {ticker_order}")

    mechanism = signal_mechanism(ticker_order, all_tickers)
    print("\n=== 5. Signal mechanism audit (what bt.momentum_signals actually emits) ===")
    print("  verdict: {}".format(mechanism["verdict"]))
    print("  {:6s} {:>8s} {:>8s} {:>8s} {:>8s} {:>10s}  {}".format(
        "ticker", "n_pos", "n_neg", "n_zero", "flips", "lb_invariant", "impl_ret[min,max]"))
    for ticker in ticker_order:
        m = mechanism["per_ticker"][ticker]
        print("  {:6s} {:>8d} {:>8d} {:>8d} {:>8d} {:>10s}  [{:.3f},{:.3f}]".format(
            ticker, m["n_pos"], m["n_neg"], m["n_zero"], m["sign_flips"],
            str(m["weights_identical_across_lookbacks"]),
            m["impl_ret_min"], m["impl_ret_max"]))
    print("  AMZN turnover: " + json.dumps(mechanism["turnover_amzn"], sort_keys=True))
    print("  " + mechanism["interpretation"])
    print("")

    print("\n=== 6. Cost-sensitivity sweep (lookback 3/5/10 x cost grid) ===")
    print("  cost grid: " + ", ".join(name for name, _ in COST_GRID))
    print("  per level: 4 contiguous volatility blocks, walk-forward "
          "train={}d/test={}d/warmup={}d/overlap={}d, matched coin-flip null\n"
          .format(TRAIN, TEST, WARM, OVERLAP))

    per_cost = {}
    universe_by_cost = {}
    for name, cfg in COST_GRID:
        per_asset, universe = run_cost_level(ticker_order, all_tickers, cfg)
        per_cost[name] = per_asset
        universe_by_cost[name] = universe
        print("  cost={:10s} lookback-5 EDGE assets={:2d}/{}  "
              "per-lookback EDGE counts={}  robustness={}".format(
                  name, universe["reference_edge_count"], len(ticker_order),
                  universe["per_lookback_edge_counts"], universe["robustness_counts"]))

    E_ref = universe_by_cost["zero"]["reference_edge_count"]
    E_canon = universe_by_cost["canonical"]["reference_edge_count"]
    E_stress = universe_by_cost["stress2x"]["reference_edge_count"]
    cost_class = classify(E_ref, E_canon, E_stress)
    # The mechanical classification only describes the signal under test. When
    # the mechanism audit shows a degenerate (constant) position, the mechanical
    # cost-robustness does NOT establish that a momentum edge survives costs.
    if mechanism["all_constant_long_only"]:
        interpretation = (
            "UNVERIFIED as a momentum-edge property: the signal under test is a "
            "constant long-only position (mean log price, not return; "
            "lookback-invariant), so the near-zero cost drag reflects ~zero "
            "directional turnover, not a robust momentum edge. The mechanical "
            "COST_ROBUST label is a turnover artifact and must not be read as "
            "evidence that the admitted momentum candidate survives costs.")
    else:
        interpretation = (
            "The signal takes both signs and is lookback-sensitive; the "
            "mechanical cost classification describes a genuine momentum signal.")

    print("\n=== 7. Per-asset lookback-5 segment medians by cost level ===")
    print("  {:6s} {:>22s} {:>22s} {:>22s}  {}".format(
        "ticker", "zero", "canonical", "stress2x", "rel_edge_5(canonical)"))
    for ticker in ticker_order:
        row = [per_cost[name][ticker][str(REFERENCE_LOOKBACK)]["medians"]
               for name, _ in COST_GRID]
        rel = per_cost["canonical"][ticker]["relative_edge_5"]
        print("  {:6s} {:>22s} {:>22s} {:>22s}  {:+8.3f}".format(
            ticker, str(row[0]), str(row[1]), str(row[2]), rel))
    print("")

    # Determinism: recompute the AMZN/JPM target subset and assert identical.
    res2 = {name: {t: segment_result(t, tickers[t], lb, cfg)
                   for t in target for lb in [REFERENCE_LOOKBACK]}
            for name, cfg in COST_GRID}
    out1 = json.dumps({name: {t: per_cost[name][t][str(REFERENCE_LOOKBACK)]
                              for t in target} for name, _ in COST_GRID},
                      sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        reference_lookback=REFERENCE_LOOKBACK,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_grid=[dict(name=name, cfg=cfg_descriptor(cfg)) for name, cfg in COST_GRID],
        per_cost=per_cost,
        universe=universe_by_cost,
        classification=dict(
            E_zero=E_ref, E_canonical=E_canon, E_stress2x=E_stress,
            cost_robustness=cost_class,
            rule=("COST_ROBUST if E_canonical >= ceil(0.8*E_zero) and "
                  "E_stress2x >= ceil(0.5*E_zero); COST_ERASED if "
                  "E_canonical <= floor(0.2*E_zero); else PARTIALLY_ROBUST; "
                  "NO_EDGE_BASELINE if E_zero == 0"),
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("classification: {} (E_zero={} E_canonical={} E_stress2x={})".format(
        cost_class, E_ref, E_canon, E_stress))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity on the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())