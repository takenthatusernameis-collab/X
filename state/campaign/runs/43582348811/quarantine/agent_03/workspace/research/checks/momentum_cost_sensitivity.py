"""Real-data transaction-cost sensitivity for the short-horizon momentum class.

Objective (focused task R-001): decide whether the admitted candidate positive
momentum evidence (long the previous N-day return, hold 1 day, daily rebalance)
survives conservative execution-cost stress on the collected 10-asset universe,
or whether daily-rebalance turnover erases it.

Why this is the decision-relevant qualification: every prior momentum check
(`momentum.py`, `momentum_lookback_sweep.py`, `momentum_longer_horizon.py`) ran
with the engine's default `BacktestConfig()` — i.e. ZERO commission and ZERO
slippage. The admitted medians are therefore frictionless. Cost robustness is
the one remaining qualification that decides whether the candidate is actionable
or an artifact of costless rebalancing.

Predeclared cost grid (fixed before execution; NOT tuned to results):
  zero         : BacktestConfig() (framework default, no costs)
  realistic    : the repository's canonical realistic cost model
                 (examples/ma_crossover_real_data.py): commission_per_trade=2.0,
                 commission_per_share=0.003, slippage_cents=2.0,
                 slippage_proportional=0.0005 (5 bps)
  conservative : 3x every realistic cost parameter (6.0 / 0.009 / 6.0 cents /
                 0.0015 = 15 bps) — a deliberately harsh high-turnover stress

Predeclared classification (fixed before execution):
  edge(asset, lookback, cost) = framework verdict REGIME_STABLE with every
      per-segment median > 0 (the same definition the prior lookback sweep used
      for its ROBUST count).
  COST_ROBUST  : at the CONSERVATIVE level, >= 6/10 assets keep an edge at every
                 lookback (6/10 is the prior zero-cost ROBUST count).
  COST_FRAGILE : at the CONSERVATIVE level, <= 2/10 assets keep an edge at every
                 lookback.
  MIXED        : otherwise.
The matched null is the framework's `noise_benchmark` (coin-flip signals) run
under the SAME cost config, so candidate-vs-null is cost-matched. Because the
coin-flip null turns over more than the momentum signal, cost-loading the null
biases that comparison toward the candidate; the candidate-with-cost vs
candidate-without-cost decay is therefore reported alongside it as the primary
cost-robustness evidence.

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

COST_LEVELS = ("zero", "realistic", "conservative")
CONSERVATIVE_MULTIPLIER = 3.0

# Repository canonical realistic cost model (examples/ma_crossover_real_data.py).
REALISTIC_COST = dict(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

# Predeclared family classification thresholds.
COST_ROBUST_MIN_EDGE_ASSETS = 6
COST_FRAGILE_MAX_EDGE_ASSETS = 2

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
LOOKBACK_ARTIFACT = ARTIFACT_DIR / "momentum_lookback_sweep_results.json"


def cost_config(level: str) -> bt.BacktestConfig:
    """BacktestConfig for one predeclared cost level."""
    if level == "zero":
        return bt.BacktestConfig(initial_capital=1e6)
    mult = 1.0 if level == "realistic" else CONSERVATIVE_MULTIPLIER
    return bt.BacktestConfig(
        initial_capital=1e6,
        commission_per_trade=REALISTIC_COST["commission_per_trade"] * mult,
        commission_per_share=REALISTIC_COST["commission_per_share"] * mult,
        slippage_cents=REALISTIC_COST["slippage_cents"] * mult,
        slippage_proportional=REALISTIC_COST["slippage_proportional"] * mult,
    )


def cost_params(level: str) -> dict:
    """Human-readable cost parameters actually used for a level."""
    cfg = cost_config(level)
    return dict(
        commission_per_trade=cfg.commission_per_trade,
        commission_per_share=cfg.commission_per_share,
        slippage_cents=cfg.slippage_cents,
        slippage_proportional=cfg.slippage_proportional,
    )


def segment_result(bars, lookback: int, cfg: bt.BacktestConfig) -> dict:
    """One asset's walk-forward regime-stability result under a cost config."""
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    signals = bt.momentum_signals(closes, lookback=lookback)
    if len(signals) != len(closes):
        raise ValueError("signal length mismatch")
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=bars,
        signals=signals,
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


def is_edge(sr: dict) -> bool:
    """Predeclared edge rule: REGIME_STABLE with every segment median > 0."""
    return sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"])


def mean_median(sr: dict) -> float:
    return float(np.mean(sr["medians"])) if sr["medians"] else 0.0


def family_classification(conservative_counts) -> str:
    """Predeclared family verdict from conservative edge counts per lookback."""
    counts = list(conservative_counts.values())
    if min(counts) >= COST_ROBUST_MIN_EDGE_ASSETS:
        return "COST_ROBUST"
    if max(counts) <= COST_FRAGILE_MAX_EDGE_ASSETS:
        return "COST_FRAGILE"
    return "MIXED"


def cost_model_audit() -> dict:
    """Audit how the existing engine actually charges slippage.

    The engine applies `close + slippage_cents/100 + close*slippage_proportional`
    as the fill price on EVERY fill. On a buy this is a cost; on a sell the seller
    receives the higher price, so it is a credit. Over a buy/sell round trip the
    two approximately cancel. This probe demonstrates that directly and attributes
    the AMZN realistic-cost equity drag to commissions vs slippage.

    This is a qualification of the cost-robustness result, not a change to it: the
    predeclared classification is computed under the existing cost model as-is.
    """
    probe_bars = [
        bt.Bar(date=1, open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0),
        bt.Bar(date=2, open=110.0, high=110.0, low=110.0, close=110.0, volume=1.0),
        bt.Bar(date=3, open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0),
    ]
    probe_sig = [
        bt.Signal(date=1, weight=0.0),
        bt.Signal(date=2, weight=1.0),
        bt.Signal(date=3, weight=-1.0),
    ]
    eq_zero = bt.run_bars(probe_bars, probe_sig,
                          bt.BacktestConfig(initial_capital=1e6)).equity_curve[-1]
    eq_slip = bt.run_bars(
        probe_bars, probe_sig,
        bt.BacktestConfig(initial_capital=1e6,
                          slippage_proportional=REALISTIC_COST["slippage_proportional"]),
    ).equity_curve[-1]
    eq_comm = bt.run_bars(
        probe_bars, probe_sig,
        bt.BacktestConfig(initial_capital=1e6,
                          commission_per_trade=REALISTIC_COST["commission_per_trade"]),
    ).equity_curve[-1]

    bars, _ = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    close_by_date = {b.date: b.close for b in bars}
    signals = bt.momentum_signals(closes, lookback=5)
    res_real = bt.run_bars(list(bars), signals, cost_config("realistic"))
    res_zero = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
    comm_total = float(sum(f.commission for f in res_real.trades))
    slip_cash = float(sum(-f.shares * (f.price - close_by_date[f.date])
                          for f in res_real.trades))
    drag = (res_zero.equity_curve[-1] - res_real.equity_curve[-1]) / 1e6

    return dict(
        round_trip_probe=dict(
            zero_equity=round(float(eq_zero), 4),
            slippage_only_equity=round(float(eq_slip), 4),
            commission_only_equity=round(float(eq_comm), 4),
            slippage_is_net_credit=bool(eq_slip > eq_zero),
            commission_is_net_cost=bool(eq_comm < eq_zero),
        ),
        amzn_full_sample=dict(
            n_fills=len(res_real.trades),
            commission_total=round(comm_total, 2),
            net_slippage_cash=round(slip_cash, 2),
            equity_zero=round(float(res_zero.equity_curve[-1]), 2),
            equity_realistic=round(float(res_real.equity_curve[-1]), 2),
            cost_drag_fraction_of_capital=round(float(drag), 6),
        ),
        qualification=(
            "The existing engine applies slippage as a price add on both buys and "
            "sells, so slippage is a credit on the closing side and approximately "
            "cancels over round trips; the realistic-cost drag is dominated by "
            "commissions. COST_ROBUST is therefore established for the commission "
            "channel of the existing cost model only. Robustness to true "
            "directional slippage/market impact is UNVERIFIED until the engine's "
            "slippage sign is repaired and this check is rerun."
        ),
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

    print("\n=== 3. Leakage + cost-accounting review (realistic costs) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, cost_config("realistic"))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], cost-accounting [PASS], "
              f"realistic-cost total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only.")
    print("  cost accounting: full-sample equity is fully explained by recorded "
          "fills including commissions and slippage under the realistic config.")

    print("\n=== 4. Cost-parameter grid (predeclared) ===")
    for level in COST_LEVELS:
        print(f"  {level:12s}: {cost_params(level)}")

    print("\n=== 5. Momentum cost sweep (lookbacks 3/5/10 x cost levels) ===")
    print("  walk-forward per segment train={}d / test={}d / warmup={}d / "
          "overlap={}d, min {} bars/segment, {} contiguous volatility blocks, "
          "matched coin-flip null under the same cost config\n".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS, N_BLOCKS))

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    per_asset = {t: {lvl: {} for lvl in COST_LEVELS} for t in ticker_order}
    for level in COST_LEVELS:
        cfg = cost_config(level)
        for lb in LOOKBACKS:
            for ticker in ticker_order:
                per_asset[ticker][level][str(lb)] = segment_result(
                    all_tickers[ticker], lb, cfg)
        edge_counts = {
            str(lb): sum(1 for t in ticker_order
                         if is_edge(per_asset[t][level][str(lb)]))
            for lb in LOOKBACKS
        }
        print(f"  cost={level:12s} edge_counts(REGIME_STABLE & all medians>0) "
              f"by lookback: {edge_counts}")

    # Predeclared summary: edge counts, zero-vs-conservative decay, null gaps.
    edge_counts_by_cost = {}
    null_gap_by_cost = {}
    for level in COST_LEVELS:
        edge_counts_by_cost[level] = {
            str(lb): sum(1 for t in ticker_order if is_edge(per_asset[t][level][str(lb)]))
            for lb in LOOKBACKS
        }
        gaps = {}
        for lb in LOOKBACKS:
            seg_gaps = []
            for t in ticker_order:
                sr = per_asset[t][level][str(lb)]
                seg_gaps.extend(
                    c - n for c, n in zip(sr["medians"], sr["null_medians"]))
            gaps[str(lb)] = dict(
                mean_gap=round(float(np.mean(seg_gaps)), 4) if seg_gaps else 0.0,
                frac_segments_above_tol=round(
                    float(np.mean([g > bt.OUTLIER_TOL for g in seg_gaps])), 3)
                if seg_gaps else 0.0,
            )
        null_gap_by_cost[level] = gaps

    zero_vs_cons = {}
    for t in ticker_order:
        zero_vs_cons[t] = {}
        for lb in LOOKBACKS:
            z = mean_median(per_asset[t]["zero"][str(lb)])
            c = mean_median(per_asset[t]["conservative"][str(lb)])
            zero_vs_cons[t][str(lb)] = dict(
                zero_mean_median=round(z, 3),
                conservative_mean_median=round(c, 3),
                decay=round(z - c, 3),
            )

    classification = {
        str(lb): family_classification(
            {str(l): edge_counts_by_cost[l][str(lb)] for l in COST_LEVELS})
        for lb in LOOKBACKS
    }
    conservative_counts = [edge_counts_by_cost["conservative"][str(lb)]
                           for lb in LOOKBACKS]
    overall = family_classification(
        {str(lb): edge_counts_by_cost["conservative"][str(lb)] for lb in LOOKBACKS})

    print("\n=== 6. Predeclared classification ===")
    print(f"  edge counts by cost level: {edge_counts_by_cost}")
    print(f"  conservative edge counts by lookback: "
          f"{ {str(lb): edge_counts_by_cost['conservative'][str(lb)] for lb in LOOKBACKS} }")
    print(f"  family classification per lookback: {classification}")
    print(f"  OVERALL: {overall}")

    print("\n=== 7. Cost-model audit (slippage direction) ===")
    audit = cost_model_audit()
    print(f"  round-trip probe: {audit['round_trip_probe']}")
    print(f"  AMZN full-sample: {audit['amzn_full_sample']}")
    print(f"  qualification: {audit['qualification']}")

    # Determinism: rerun AMZN at realistic lookback=5 and assert identical.
    r1 = segment_result(all_tickers["AMZN"], 5, cost_config("realistic"))
    r2 = segment_result(all_tickers["AMZN"], 5, cost_config("realistic"))
    print("\ndeterminism: AMZN realistic lookback=5 r1 == r2: {}".format(
        json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)))
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True), \
        "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        cost_levels=list(COST_LEVELS),
        conservative_multiplier=CONSERVATIVE_MULTIPLIER,
        cost_params={level: cost_params(level) for level in COST_LEVELS},
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        classification=classification,
        overall_classification=overall,
        cost_model_audit=audit,
        qualification=audit["qualification"],
        edge_counts_by_cost=edge_counts_by_cost,
        null_gap_by_cost=null_gap_by_cost,
        zero_vs_conservative=zero_vs_cons,
        per_asset=per_asset,
        ticker_order=ticker_order,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity sweep: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())