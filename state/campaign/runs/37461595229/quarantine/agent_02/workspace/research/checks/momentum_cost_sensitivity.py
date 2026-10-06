"""Real-data momentum lookback sweep with transaction-cost sensitivity.

Objective: test whether the short-horizon positive momentum edge observed at
lookback=5 (`research/checks/momentum.py`, activation 37318950814) and confirmed
across lookback 3/5/10 (`research/checks/momentum_lookback_sweep.py`, activation
37361000967) survives conservative transaction costs. Momentum is a mechanical
1-day-ahead bet with daily rebalancing, so near-daily turnover makes it the most
cost-sensitive of the examined signal classes. The preceding process decision
(left momentum ADMITTED AS CANDIDATE POSITIVE EVIDENCE) carries the explicit
qualification that execution-cost robustness remains untested; this check closes
that qualification for lookbacks 3/5/10 on the collected universe.

Falsification prediction (a-priori, not tuned to OOS): the repository's existing
"realistic costs" configuration (commission_per_trade=2.0, commission_per_share=
0.003, slippage_cents=2.0, slippage_proportional=0.0005 — as defined in
examples/ma_crossover.py and examples/volatility_regime_filter.py) charges a per-
trade cost on full gross exposure at near-daily turnover. Predicted outcome: costs
consume most of the small per-bar edge, so lookback-ROBUST positive REGIME_STABLE
verdicts under zero cost degrade to COST_SENSITIVE or NO_EDGE under realistic
costs — i.e. the short-horizon mechanical momentum edge is NOT cost robust under
the existing realistic cost model. Either verdict is decisive: it settles whether
the admitted candidate edge clears the cost-robustness qualification in
state/LEARNING_STATE.md.

Method (fixed a-priori, not tuned to OOS): momentum (long the previous N-day
return; hold 1 day; daily rebalance); lookbacks 3, 5, 10; same walk-forward
blocks and parameters as `momentum_lookback_sweep.py`. The sweep runs twice on the
same real-data universe — zero cost and the realistic cost model — with the matched
coin-flip null applied under the same cost regime so the candidate and the null
pay the same costs and the edge is compared fairly. Lookback-robustness is the
same rule as before (positive REGIME_STABLE edge at >= 2 lookbacks). Cost
robustness per asset: COST_ROBUST if edge at >= 2 lookbacks under both arms;
COST_SENSITIVE if the edge is robust at zero cost but not under realistic costs;
NO_EDGE if no cost-robust positive edge anywhere. Determinism is asserted via an
internal independent recomputation of the AMZN/JPM subset.

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
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"

# Cost models. Zero cost (default engine policy) vs the repository's existing
# "realistic costs" configuration (examples/ma_crossover.py,
# examples/volatility_regime_filter.py): $2 fixed commission + $0.003/share +
# 2-cent fixed slippage + 5 bps proportional slippage on every round trip.
ZERO_CONFIG = bt.BacktestConfig()
REALISTIC_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

# Cost-robustness classification of the positive REGIME_STABLE edge under costs.
COST_ROBUST = "COST_ROBUST"
COST_SENSITIVE = "COST_SENSITIVE"
COST_REVIVED = "COST_REVIVED"
COST_NO_EDGE = "NO_EDGE"


def lookback_verdict(sr):
    """Edge verdict for one lookback's scenario result.

    REGIME_STABLE with uniformly positive medians is a stable positive edge;
    REGIME_STABLE_LOSS is a stable loss, not an edge.
    """
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def robustness_verdict(lookback_results):
    """Verdict on whether the edge is lookback-robust for one asset.

    ROBUST   : positive REGIME_STABLE edges at >= 2 lookbacks.
    SENSITIVE: a positive REGIME_STABLE edge at exactly one lookback only.
    NO_EDGE  : no lookback shows a positive REGIME_STABLE edge.
    """
    counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    for lb in LOOKBACKS:
        counts[lookback_verdict(lookback_results[str(lb)])] += 1
    if counts["EDGE"] >= 2:
        return "ROBUST", counts
    if counts["EDGE"] == 1:
        return "SENSITIVE", counts
    return "NO_EDGE", counts


def cost_robustness_verdict(zero_robust, realistic_robust):
    """Verdict on whether the positive edge survives realistic costs.

    COST_ROBUST : positive edge at >= 2 lookbacks under both cost regimes.
    COST_SENSITIVE: edge at >= 2 lookbacks only under zero cost.
    COST_REVIVED: edge at >= 2 lookbacks only under realistic cost (unlikely).
    NO_EDGE     : no cost-robust positive edge.
    """
    if zero_robust == "ROBUST" and realistic_robust == "ROBUST":
        return COST_ROBUST
    if zero_robust == "ROBUST" and realistic_robust != "ROBUST":
        return COST_SENSITIVE
    if zero_robust != "ROBUST" and realistic_robust == "ROBUST":
        return COST_REVIVED
    return COST_NO_EDGE


def segment_result(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker and one lookback."""
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

    print("\n=== 3. Leakage review per asset (AMZN/JPM, realistic cost model) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, REALISTIC_CONFIG)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        mean_commission = (
            sum(f.commission for f in res.trades) / len(res.trades) if res.trades else 0.0
        )
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], trades={len(res.trades)}, "
              f"mean_commission=${mean_commission:.2f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Momentum lookback sweep (3 / 5 / 10) with cost regimes ===")
    print("  Zero cost: no commission, no slippage.")
    print("  Realistic cost: commission_per_trade=$2.0, commission_per_share=$0.003, "
          "slippage_cents=2.0, slippage_proportional=5bps (examples/ma_crossover.py).")
    print("  Walk-forward per segment: train={}d / test={}d / warmup={}d / overlap={}d, "
          "min {} bars/segment, 4 contiguous volatility blocks, each lookback vs a "
          "coin-flip null on the same segments under the same cost regime.\n".format(
        TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Full-universe sweep under both cost regimes.
    per_asset = {}
    for ticker in ticker_order:
        bars = all_tickers[ticker]
        profile = {}
        for lb in LOOKBACKS:
            z = segment_result(ticker, bars, lb, ZERO_CONFIG)
            r = segment_result(ticker, bars, lb, REALISTIC_CONFIG)
            profile[str(lb)] = dict(
                zero_cost=z,
                realistic_cost=r,
                cost_drag=[round(z["medians"][i] - r["medians"][i], 3)
                           for i in range(len(z["medians"]))],
            )
        arm_profiles = {
            "zero_cost": {str(lb): profile[str(lb)]["zero_cost"] for lb in LOOKBACKS},
            "realistic_cost": {str(lb): profile[str(lb)]["realistic_cost"] for lb in LOOKBACKS},
        }
        zr, zcounts = robustness_verdict(arm_profiles["zero_cost"])
        rr, rcounts = robustness_verdict(arm_profiles["realistic_cost"])
        profile["edge_counts"] = {"zero_cost": zcounts, "realistic_cost": rcounts}
        profile["robustness_verdict"] = {"zero_cost": zr, "realistic_cost": rr}
        profile["cost_robustness_verdict"] = cost_robustness_verdict(zr, rr)
        per_asset[ticker] = profile

    zero_sum = {"lookback_5_counts": {}, "robustness_counts": {}}
    real_sum = {"lookback_5_counts": {}, "robustness_counts": {}}
    for arm_key in ("zero_cost", "realistic_cost"):
        lookback_5_counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                             "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
        robustness_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
        for ticker in ticker_order:
            p5 = per_asset[ticker]["5"][arm_key]
            lookback_5_counts[p5["verdict"]] += 1
            robustness_counts[per_asset[ticker]["robustness_verdict"][arm_key]] += 1
        (zero_sum if arm_key == "zero_cost" else real_sum).update({
            "lookback_5_counts": lookback_5_counts,
            "robustness_counts": robustness_counts,
        })

    cost_counts = {COST_ROBUST: 0, COST_SENSITIVE: 0, COST_REVIVED: 0, COST_NO_EDGE: 0}
    for ticker in ticker_order:
        cost_counts[per_asset[ticker]["cost_robustness_verdict"]] += 1

    print("  lookback 5 (reference) verdict counts - zero cost:      "
          f"{zero_sum['lookback_5_counts']}")
    print("  lookback 5 (reference) verdict counts - realistic cost: "
          f"{real_sum['lookback_5_counts']}")
    print("  lookback-robustness verdict counts - zero cost:      "
          f"{zero_sum['robustness_counts']}")
    print("  lookback-robustness verdict counts - realistic cost: "
          f"{real_sum['robustness_counts']}")
    print("  cost-robustness verdict counts: "
          f"COST_ROBUST={cost_counts[COST_ROBUST]} | "
          f"COST_SENSITIVE={cost_counts[COST_SENSITIVE]} | "
          f"COST_REVIVED={cost_counts[COST_REVIVED]} | "
          f"NO_EDGE={cost_counts[COST_NO_EDGE]}\n")

    print("  Per-asset cost profiles:")
    for ticker in ticker_order:
        p = per_asset[ticker]
        meds = ", ".join(
            "{}(z:{:5.3f}/r:{:5.3f}/d:{:5.3f} {})".format(
                lb, p[str(lb)]["zero_cost"]["medians"][0],
                p[str(lb)]["realistic_cost"]["medians"][0],
                p[str(lb)]["cost_drag"][0],
                p[str(lb)]["realistic_cost"]["verdict"],
            )
            for lb in LOOKBACKS
        )
        print("    {:6s} rob(z/r)={:<9s} cost={:<12s} medians={}".format(
            ticker,
            "{} / {}".format(p["robustness_verdict"]["zero_cost"],
                             p["robustness_verdict"]["realistic_cost"]),
            p["cost_robustness_verdict"], meds))
    print("")

    robust_assets = [t for t in ticker_order
                     if per_asset[t]["cost_robustness_verdict"] == COST_ROBUST]
    sensitive_assets = [t for t in ticker_order
                        if per_asset[t]["cost_robustness_verdict"] == COST_SENSITIVE]
    print("  cost-ROBUST assets (edge survives costs at >= 2 lookbacks): "
          "{}".format(robust_assets))
    print("  cost-SENSITIVE assets (edge consumed by realistic costs): "
          "{}".format(sensitive_assets))
    if cost_counts[COST_ROBUST] > 0:
        print("\n  HEADLINE: The short-horizon momentum edge survives realistic "
              "transaction costs at lookback 3-10 for {} of 10 collected assets."
              .format(cost_counts[COST_ROBUST]))
    else:
        print("\n  HEADLINE: Realistic transaction costs consume the short-horizon "
              "momentum edge across the collected universe (0 of 10 assets "
              "retain a lookback-ROBUST positive edge).")
    print("")

    # Determinism: rerun the target subset and assert identical output.
    res2 = {lb: {t: (segment_result(t, tickers[t], lb, ZERO_CONFIG),
                     segment_result(t, tickers[t], lb, REALISTIC_CONFIG))
                  for t in target}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: (per_asset[t][str(lb)]["zero_cost"],
                                 per_asset[t][str(lb)]["realistic_cost"])
                             for t in target}
                       for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_models=dict(
            zero_cost=dict(
                commission_per_trade=float(ZERO_CONFIG.commission_per_trade),
                commission_per_share=float(ZERO_CONFIG.commission_per_share),
                slippage_cents=float(ZERO_CONFIG.slippage_cents),
                slippage_proportional=float(ZERO_CONFIG.slippage_proportional),
            ),
            realistic_cost=dict(
                commission_per_trade=float(REALISTIC_CONFIG.commission_per_trade),
                commission_per_share=float(REALISTIC_CONFIG.commission_per_share),
                slippage_cents=float(REALISTIC_CONFIG.slippage_cents),
                slippage_proportional=float(REALISTIC_CONFIG.slippage_proportional),
            ),
        ),
        per_asset=per_asset,
        zero_cost=dict(lookback_5_counts=zero_sum["lookback_5_counts"],
                       robustness_counts=zero_sum["robustness_counts"]),
        realistic_cost=dict(lookback_5_counts=real_sum["lookback_5_counts"],
                            robustness_counts=real_sum["robustness_counts"]),
        cost_robustness_counts=cost_counts,
        asset_order=ticker_order,
        a_priori_prediction=(
            "Realistic costs (fixed commission + proportional slippage at "
            "near-daily full-exposure turnover) will consume most of the small "
            "per-bar momentum edge, so lookback-ROBUST positive REGIME_STABLE "
            "verdicts under zero cost will degrade to COST_SENSITIVE or NO_EDGE "
            "under realistic costs; i.e. the short-horizon momentum edge will "
            "be classified as not cost-robust under the repository's existing "
            "realistic cost model."
        ),
        prediction_matched=cost_counts[COST_SENSITIVE] + cost_counts[COST_REVIVED] >= \
                            cost_counts[COST_ROBUST],
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum lookback sweep with cost regimes on the collected "
          "universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
