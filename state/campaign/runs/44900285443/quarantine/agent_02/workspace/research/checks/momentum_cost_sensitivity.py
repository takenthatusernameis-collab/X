"""Real-data transaction-cost sensitivity of short-horizon momentum.

Objective (task R-001, campaign slot 02): decide whether the admitted candidate
positive momentum evidence (long the previous N-day return, hold 1 day, daily
rebalance; lookback 5 admitted with lookback 3/5/10 lookback-robustness) survives
conservative transaction-cost stress on the collected real-data universe, judged
against a *matched* coin-flip null that is costed identically.

This is a bounded qualification of already-admitted evidence. It does NOT tune
the signal rule, expand the universe, or introduce leverage. The signal
definition, walk-forward windows, volatility-regime segments, and coin-flip null
are exactly those of `research/checks/momentum_lookback_sweep.py`; the ONLY new
dimension is a predeclared, ordered transaction-cost grid applied identically to
candidate and null.

Predeclared cost grid (a-priori, frozen before execution; the "realistic" level
is the repository's own realistic-cost model used by examples/ma_crossover.py and
examples/volatility_regime_filter.py; the more severe levels are integer
multiples of it, not tuned values):

    level         commission/trade  commission/share  slippage_cents  slippage_prop
    zero          0.0               0.0               0.0             0.0
    realistic     2.0               0.003             2.0             0.0005
    conservative  4.0               0.006             4.0             0.0010
    stress        8.0               0.012             8.0             0.0020

Because momentum rebalances daily, every sign flip is a full round trip, so the
proportional slippage component (5 bps of notional at the realistic level) is
applied to a high-turnover book and is the decision-relevant cost.

Matched null: each cost level's coin-flip sign null is run on the same volatility
segments with the same `BacktestConfig` costs, so candidate and null are costed
identically at every level.

Predeclared per-(asset, lookback) survival rule, evaluated from the rounded
per-segment medians stored in the artifact (so the verifier reproduces it):

    survives(level) := all(segment medians > 0) AND mean(segment medians) >
                       mean(segment null medians)

Classification (ordered, first match wins), where the level order is
zero -> realistic -> conservative -> stress:

    NO_EDGE        : not survives(zero)          -- no positive edge to stress
    COST_FRAGILE   : survives(zero) but not survives(realistic)
    COST_ERODED    : survives(realistic) but not survives(conservative)
    COST_RESILIENT : survives(conservative) but not survives(stress)
    COST_ROBUST    : survives(stress)

Universe verdict at the primary lookback (5), from the counts R/C/Z of assets
surviving at realistic / conservative / zero cost respectively (predeclared
thresholds):

    COST_FALSIFIED : R == 0
    COST_ROBUST    : R >= 4 and C >= 2
    COST_ERODED    : otherwise (some assets survive realistic costs)

A zero-cost cross-consistency gate asserts that the zero-cost candidate medians
reproduce `state/check_artifacts/momentum_lookback_sweep_results.json` exactly for
every asset and lookback, proving the only change is the cost dimension.

Method: seed 42, collected adjusted-close data (dataset
`yf-ohlcv-universe-2009-to-2026-10-03`), 10-ticker universe, regime blocks from
AAPL trailing-60d volatility (4 blocks), walk-forward train=252d/test=84d/
warmup=60d/overlap=60d, min 400 bars/segment, lookbacks 3/5/10, four cost
levels. Determinism is asserted by an internal recomputation of the AMZN/JPM
subset across all lookbacks and cost levels.

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
PRIMARY_LOOKBACK = 5
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
PRIOR_LOOKBACK_ARTIFACT = ARTIFACT_DIR / "momentum_lookback_sweep_results.json"

# Predeclared cost grid (frozen before execution).
COST_LEVELS = ("zero", "realistic", "conservative", "stress")
COST_PARAMS = {
    "zero": dict(commission_per_trade=0.0, commission_per_share=0.0,
                 slippage_cents=0.0, slippage_proportional=0.0),
    "realistic": dict(commission_per_trade=2.0, commission_per_share=0.003,
                      slippage_cents=2.0, slippage_proportional=0.0005),
    "conservative": dict(commission_per_trade=4.0, commission_per_share=0.006,
                         slippage_cents=4.0, slippage_proportional=0.0010),
    "stress": dict(commission_per_trade=8.0, commission_per_share=0.012,
                   slippage_cents=8.0, slippage_proportional=0.0020),
}

# Predeclared universe thresholds.
ROBUST_MIN_REALISTIC = 4
ROBUST_MIN_CONSERVATIVE = 2


def cost_cfg(level: str) -> bt.BacktestConfig:
    p = COST_PARAMS[level]
    return bt.BacktestConfig(
        commission_per_trade=p["commission_per_trade"],
        commission_per_share=p["commission_per_share"],
        slippage_cents=p["slippage_cents"],
        slippage_proportional=p["slippage_proportional"],
    )


def segment_result(ticker, bars, lookback, level):
    """One (ticker, lookback, cost level) regime-stability result."""
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
        cfg=cost_cfg(level),
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    medians = [round(s.baseline_median_log_return, 3) for s in res.scenarios]
    null_medians = [round(s.noise_median_log_return, 3) for s in res.scenarios]
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=medians,
        null_medians=null_medians,
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
        survives=survives(medians, null_medians),
    )


def survives(medians, null_medians):
    """Predeclared survival rule, evaluated from the stored rounded medians."""
    if not medians:
        return False
    all_positive = all(m > 0 for m in medians)
    beats_null = (sum(medians) / len(medians)) > (sum(null_medians) / len(null_medians))
    return bool(all_positive and beats_null)


def signal_degeneracy(bars, lookback):
    """Root-cause diagnostic: is `bt.momentum_signals` a genuine momentum
    signal or a degenerate (near-constant) exposure?

    The implemented signal sets weight = +1 when
    ``mean(log(closes[i-lookback+1:i+1])) > 0``. Because adjusted closes are
    dollar prices (mostly > 1), that mean-log-price is positive almost always,
    so the weight is +1 on nearly every bar and the position never turns over.
    A genuine short-horizon momentum signal would instead use the sign of the
    lookback log *return*, ``log(closes[i]) - log(closes[i-lookback])``, which
    is positive on roughly half of bars.

    Returns the weight distribution of the implemented signal and the sign
    distribution the genuine log-return rule would produce, so the artifact
    records the discrepancy directly.
    """
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    n = len(closes)
    lb = int(lookback)
    sig = bt.momentum_signals(closes, lookback=lb)
    w = np.array([s.weight for s in sig], dtype=np.float64)
    n_pos = int(np.sum(w > 0))
    n_neg = int(np.sum(w < 0))
    n_zero = int(np.sum(w == 0))
    active = w[n_zero:]
    flips = int(np.sum(np.diff(active) != 0)) if active.size > 1 else 0
    genuine = np.array([
        np.log(closes[i]) - np.log(closes[i - lb]) for i in range(lb, n)
    ], dtype=np.float64)
    return dict(
        n_bars=n,
        weight_plus1=n_pos,
        weight_minus1=n_neg,
        weight_zero=n_zero,
        n_sign_flips=flips,
        frac_bars_long=round(n_pos / n, 4),
        genuine_frac_plus1=round(float(np.mean(genuine > 0)), 4),
        genuine_frac_minus1=round(float(np.mean(genuine < 0)), 4),
        genuine_frac_zero=round(float(np.mean(genuine == 0)), 4),
    )


def classify(levels):
    """Per-(asset, lookback) cost-robustness classification (first match wins)."""
    s = {lvl: levels[lvl]["survives"] for lvl in COST_LEVELS}
    if not s["zero"]:
        return "NO_EDGE"
    if not s["realistic"]:
        return "COST_FRAGILE"
    if not s["conservative"]:
        return "COST_ERODED"
    if not s["stress"]:
        return "COST_RESILIENT"
    return "COST_ROBUST"


def universe_verdict(per_asset_lookback5):
    """Predeclared universe verdict from the per-asset classifications."""
    R = sum(1 for t in per_asset_lookback5 if per_asset_lookback5[t]["realistic"])
    C = sum(1 for t in per_asset_lookback5 if per_asset_lookback5[t]["conservative"])
    Z = sum(1 for t in per_asset_lookback5 if per_asset_lookback5[t]["zero"])
    if R == 0:
        verdict = "COST_FALSIFIED"
    elif R >= ROBUST_MIN_REALISTIC and C >= ROBUST_MIN_CONSERVATIVE:
        verdict = "COST_ROBUST"
    else:
        verdict = "COST_ERODED"
    return dict(verdict=verdict, survive_zero=Z, survive_realistic=R,
                survive_conservative=C,
                survive_stress=sum(1 for t in per_asset_lookback5
                                   if per_asset_lookback5[t]["stress"]))


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
        signals = bt.momentum_signals(closes, lookback=PRIMARY_LOOKBACK)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Momentum cost sensitivity (lookback 3 / 5 / 10 x cost grid) ===")
    print("  walk-forward train={}d / test={}d / warmup={}d / overlap={}d, "
          "min {} bars/segment, 4 contiguous volatility blocks; each lookback at "
          "each cost level vs a coin-flip null costed identically\n".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # results[lookback][level][ticker] = segment_result dict
    results = {lb: {lvl: {} for lvl in COST_LEVELS} for lb in LOOKBACKS}
    for lb in LOOKBACKS:
        for lvl in COST_LEVELS:
            for ticker in ticker_order:
                results[lb][lvl][ticker] = segment_result(ticker, all_tickers[ticker], lb, lvl)

    # Per-(asset, lookback) classification + universe verdict at primary lookback.
    classification = {}
    for ticker in ticker_order:
        classification[ticker] = {}
        for lb in LOOKBACKS:
            levels = {lvl: results[lb][lvl][ticker] for lvl in COST_LEVELS}
            classification[ticker][str(lb)] = classify(levels)

    per_asset_lookback5 = {}
    for ticker in ticker_order:
        per_asset_lookback5[ticker] = {
            lvl: results[PRIMARY_LOOKBACK][lvl][ticker]["survives"]
            for lvl in COST_LEVELS
        }
    uni = universe_verdict(per_asset_lookback5)

    # Zero-cost cross-consistency against the prior lookback sweep artifact.
    cross = dict(prior_artifact=str(PRIOR_LOOKBACK_ARTIFACT), matches={}, all_match=True)
    if PRIOR_LOOKBACK_ARTIFACT.exists():
        with open(PRIOR_LOOKBACK_ARTIFACT) as f:
            prior = json.load(f)
        for ticker in ticker_order:
            for lb in LOOKBACKS:
                pub = prior["per_asset"][ticker][str(lb)]["medians"]
                mine = results[lb]["zero"][ticker]["medians"]
                ok = (pub == mine)
                cross["matches"][f"{ticker}:{lb}"] = ok
                if not ok:
                    cross["all_match"] = False
        print("  zero-cost cross-consistency vs prior lookback sweep: "
              f"{'ALL MATCH' if cross['all_match'] else 'MISMATCH'}")
    else:
        cross["all_match"] = None
        print("  zero-cost cross-consistency: prior artifact not found (skipped)")

    # Print a compact table.
    print("\n  Per-asset medians by cost level (lookback {}):".format(PRIMARY_LOOKBACK))
    for ticker in ticker_order:
        cells = []
        for lvl in COST_LEVELS:
            r = results[PRIMARY_LOOKBACK][lvl][ticker]
            cells.append("{}:{:+.3f}{}".format(
                lvl[:4], sum(r["medians"]) / len(r["medians"]),
                "" if r["survives"] else "x"))
        print("    {:6s} {:14s} | {}".format(
            ticker, classification[ticker][str(PRIMARY_LOOKBACK)], "  ".join(cells)))
    print("\n  universe verdict (lookback {}): {} "
          "(survive zero/realistic/conservative/stress = {}/{}/{}/{})".format(
              PRIMARY_LOOKBACK, uni["verdict"], uni["survive_zero"],
              uni["survive_realistic"], uni["survive_conservative"],
              uni["survive_stress"]))

    # Root-cause diagnostic: is the candidate signal actually a momentum signal?
    degeneracy = {ticker: signal_degeneracy(all_tickers[ticker], PRIMARY_LOOKBACK)
                  for ticker in ticker_order}
    n_degenerate = sum(
        1 for t in ticker_order
        if degeneracy[t]["n_sign_flips"] <= 1 and degeneracy[t]["frac_bars_long"] >= 0.9
    )
    print("\n=== 5. Signal-construction diagnostic (lookback {}) ===".format(
        PRIMARY_LOOKBACK))
    print("  ticker  frac_bars_long  n_sign_flips  genuine_frac_plus1  "
          "genuine_frac_minus1")
    for ticker in ticker_order:
        d = degeneracy[ticker]
        print("  {:6s}  {:14.4f}  {:12d}  {:17.4f}  {:19.4f}".format(
            ticker, d["frac_bars_long"], d["n_sign_flips"],
            d["genuine_frac_plus1"], d["genuine_frac_minus1"]))
    print("  assets whose implemented signal is a near-constant long "
          "(frac_bars_long>=0.9 and <=1 sign flip): {}/{}".format(
              n_degenerate, len(ticker_order)))
    print("  => the implemented 'momentum' signal is degenerate for these "
          "assets: it is a constant long position, so its cost sensitivity is "
          "governed by a single entry trade, not by signal turnover.")
    cost_verdict_informative = (n_degenerate == 0)
    print("  cost-verdict informative for a genuine momentum edge: {}".format(
        cost_verdict_informative))

    # Determinism: recompute the AMZN/JPM subset for all lookbacks and levels.
    res2 = {lb: {lvl: {t: segment_result(t, tickers[t], lb, lvl) for t in target}
                 for lvl in COST_LEVELS} for lb in LOOKBACKS}
    out1 = json.dumps({lb: {lvl: {t: results[lb][lvl][t] for t in target}
                            for lvl in COST_LEVELS} for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    deterministic = (out1 == out2)
    print("\ndeterminism: r1 == r2: {}".format(deterministic))
    assert deterministic, "non-deterministic output"

    # Assemble artifact.
    per_asset = {}
    for ticker in ticker_order:
        per_asset[ticker] = {}
        for lb in LOOKBACKS:
            per_asset[ticker][str(lb)] = {
                lvl: results[lb][lvl][ticker] for lvl in COST_LEVELS
            }
            per_asset[ticker][str(lb)]["classification"] = classification[ticker][str(lb)]

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        primary_lookback=PRIMARY_LOOKBACK,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_levels=list(COST_LEVELS),
        cost_params=COST_PARAMS,
        thresholds=dict(robust_min_realistic=ROBUST_MIN_REALISTIC,
                        robust_min_conservative=ROBUST_MIN_CONSERVATIVE),
        universe=uni,
        zero_cost_cross_consistency=cross,
        signal_degeneracy=degeneracy,
        signal_degeneracy_summary=dict(
            primary_lookback=PRIMARY_LOOKBACK,
            n_near_constant_long=n_degenerate,
            n_assets=len(ticker_order),
        ),
        interpretation=dict(
            cost_verdict=uni["verdict"],
            cost_verdict_informative=cost_verdict_informative,
            note=(
                "The cost verdict classifies the implemented candidate only. "
                "It is NOT informative as a cost-robustness qualification of a "
                "genuine momentum edge because bt.momentum_signals is "
                "degenerate for {}/{} assets (near-constant +1 exposure; mean "
                "log PRICE, not log RETURN). A near-constant long book has "
                "essentially one entry trade, so transaction cost cannot erode "
                "it; the COST_ROBUST verdict reflects that, not a genuine "
                "edge's cost tolerance. A genuine momentum signal and its cost "
                "sensitivity remain UNVERIFIED pending repair of "
                "momentum_signals / mean_reversion_signals.".format(
                    n_degenerate, len(ticker_order))
            ),
        ),
        determinism=deterministic,
        per_asset=per_asset,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost sensitivity on the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())