"""Bounded transaction-cost sensitivity for short-horizon momentum (R-001).

Objective (campaign slot 02, task R-001): decide whether the admitted
candidate positive momentum evidence (lookback 3/5/10, long the previous
N-day return, hold 1 day, daily rebalance) survives conservative
execution-cost stress on the collected 10-asset universe, under the
repository's *existing* realistic cost model and a matched coin-flip null.

Two orthogonal questions are reported, because the cost result cannot be
interpreted without the first:

A. What does the signal actually trade?  `bt.momentum_signals(closes,
   lookback)` computes ``mean(log(closes[i-lookback+1:i+1]))`` and goes long
   when that is > 0. For any price above 1 this is *always* positive, so the
   signal is a static long (buy-and-hold), not a momentum rule: the lookback
   is inert. This check measures per-asset turnover and long-fraction and
   labels each signal STATIC vs TRADING. For a STATIC signal the cost test is
   not informative about a trading strategy.

B. If/when the signal trades, does its own cost-inclusive performance survive
   the cost grid?  The decision-relevant quantity is the candidate's net
   (cost-inclusive) per-segment median log return at each cost level, not the
   candidate-minus-null gap. The matched coin-flip null (iid weights in
   {-1,0,1}) churns ~3000 times over the series, so cost stress penalises the
   null far more than the candidate; the candidate-minus-null gap therefore
   *widens* with cost and is reported only as a confounded secondary
   diagnostic, never as evidence that the edge strengthens.

Predeclared, fixed a-priori (not tuned to any observed result):

Cost grid (three levels, applied identically to candidate and null):
  zero       : no commission, no slippage (reference; reproduces the prior
               momentum lookback sweep exactly).
  realistic  : the repository's existing realistic cost model
               (examples/ma_crossover_real_data.py):
               commission_per_trade=2.0, commission_per_share=0.003,
               slippage_cents=2.0, slippage_proportional=0.0005.
  stress2x   : 2x the realistic model (a conservative stress, not a fit):
               commission_per_trade=4.0, commission_per_share=0.006,
               slippage_cents=4.0, slippage_proportional=0.001.

Lookbacks: 3, 5, 10 (same grid as the prior sweep). Reference lookback: 5.

Cost verdict per (asset, lookback) from the candidate's net medians
(net = median over segments of the candidate's per-segment median log return):
  NO_EDGE_BASE : net(zero) <= 0            (no positive zero-cost edge).
  COST_FAILS   : net(realistic) <= 0       (edge gone at the repo cost model).
  COST_ERODED  : net(realistic) > 0 but net(stress2x) <= 0 or
                 net(stress2x)/net(zero) < 0.5.
  COST_ROBUST  : net(realistic) > 0 and net(stress2x) > 0 and
                 net(stress2x)/net(zero) >= 0.5.

A-priori prediction (recorded before execution): the lookback-5 positive
candidate-vs-null gap is expected to survive the repo realistic cost model for
most previously-admitted momentum assets but to degrade to COST_ERODED for
higher-turnover assets at 2x stress. This prediction was made assuming the
signal trades; the STATIC finding (question A) falsifies that premise and must
be read first.

Method: per ticker, past-only volatility blocks (window=60, 4 blocks) define
contiguous segments; each (lookback, cost level) is walk-forward validated per
segment (train=252/test=84/warmup=60/overlap=60) through `bt.stress_segments`
with `cfg` set to the cost model, so the internal coin-flip null
(`noise_benchmark`) is costed identically (matched null). Determinism is
asserted via an internal independent recomputation of the AMZN/JPM subset.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.backtest.perturbation import random_signals
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
REFERENCE_LOOKBACK = 5
STATIC_TURNOVER_MAX = 3
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
SWEEP_ARTIFACT = ARTIFACT_DIR / "momentum_lookback_sweep_results.json"

COST_MODELS = {
    "zero": dict(commission_per_trade=0.0, commission_per_share=0.0,
                 slippage_cents=0.0, slippage_proportional=0.0),
    "realistic": dict(commission_per_trade=2.0, commission_per_share=0.003,
                      slippage_cents=2.0, slippage_proportional=0.0005),
    "stress2x": dict(commission_per_trade=4.0, commission_per_share=0.006,
                     slippage_cents=4.0, slippage_proportional=0.001),
}
COST_LEVELS = ("zero", "realistic", "stress2x")
RETENTION_FLOOR = 0.5

A_PRIORI_PREDICTION = (
    "The lookback-5 positive candidate-vs-null gap is expected to survive the "
    "repo realistic cost model for most previously-admitted momentum assets but "
    "to degrade to COST_ERODED for higher-turnover assets at 2x stress. This "
    "prediction assumed the signal trades; the STATIC finding falsifies that "
    "premise and must be read first."
)

INTERPRETATION = (
    "Cost robustness is only meaningful for a signal that trades. "
    "bt.momentum_signals computes mean(log(price)) over the lookback window "
    "(always positive for prices > 1), so it is a static long: the lookback is "
    "inert and per-segment medians are identical across lookbacks. The admitted "
    "'momentum edge' is therefore a buy-and-hold drift, not a momentum "
    "strategy; its cost robustness is trivially high because turnover is ~1. "
    "The matched-null gap widens with cost only because the iid null churns "
    "~3000 times and is penalised far more, so the gap is confounded and must "
    "not be read as edge strengthening."
)


def cost_cfg(level):
    return bt.BacktestConfig(initial_capital=1e6, **COST_MODELS[level])


def signal_character(ticker, bars):
    """Turnover and long-fraction of the repo momentum signal at ref lookback."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    sig = bt.momentum_signals(closes, lookback=REFERENCE_LOOKBACK)
    turnover = sum(1 for i in range(1, len(sig))
                   if abs(sig[i].weight - sig[i - 1].weight) > 1e-12)
    frac_long = float(np.mean([s.weight > 0 for s in sig]))
    character = "STATIC" if turnover <= STATIC_TURNOVER_MAX else "TRADING"
    return dict(signal_turnover=turnover, signal_frac_long=round(frac_long, 4),
                signal_character=character)


def null_turnover(n, lookback):
    """Turnover of the matched coin-flip null (framework's random_signals)."""
    # noise_benchmark seeds per param set via base_seed + round(sum(params))*1000.
    seed = SEED + int(round(float(lookback))) * 1000
    sig = random_signals(n, seed=seed)
    return sum(1 for i in range(1, len(sig))
               if abs(sig[i].weight - sig[i - 1].weight) > 1e-12)


def segment_result(ticker, bars, lookback, cfg):
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals, bars=bars, signals=sig,
        param_grid=[{"lookback": lookback}], baseline=(("lookback", lookback),),
        segment_fn=seg_fn, train_window=TRAIN, test_window=TEST, warmup=WARM,
        overlap_window=OVERLAP, periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS, cfg=cfg)
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        medians_raw=[float(s.baseline_median_log_return) for s in res.scenarios],
        null_medians_raw=[float(s.noise_median_log_return) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


def classify(net_zero, net_real, net_2x):
    if net_zero <= 0.0:
        return "NO_EDGE_BASE"
    if net_real <= 0.0:
        return "COST_FAILS"
    retention_2x = net_2x / net_zero
    if net_2x <= 0.0 or retention_2x < RETENTION_FLOOR:
        return "COST_ERODED"
    return "COST_ROBUST"


def run_asset(ticker, bars):
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    char = signal_character(ticker, bars)
    null_to = {str(lb): null_turnover(len(closes), lb) for lb in LOOKBACKS}

    raw, derived = {}, {}
    for lb in LOOKBACKS:
        per_cost, gap_by_level = {}, {}
        for level in COST_LEVELS:
            rec = segment_result(ticker, bars, lb, cost_cfg(level))
            g = float(np.median(np.array(rec["medians_raw"], dtype=np.float64)
                                - np.array(rec["null_medians_raw"], dtype=np.float64)))
            rec.pop("medians_raw")
            rec.pop("null_medians_raw")
            rec["gap_candidate_minus_null"] = round(g, 4)
            per_cost[level] = rec
            gap_by_level[level] = g
        net = {lvl: float(np.median(per_cost[lvl]["medians"])) for lvl in COST_LEVELS}
        verdict = classify(net["zero"], net["realistic"], net["stress2x"])
        ret_real = (net["realistic"] / net["zero"]) if net["zero"] > 0 else None
        ret_2x = (net["stress2x"] / net["zero"]) if net["zero"] > 0 else None
        raw[str(lb)] = per_cost
        derived[str(lb)] = dict(
            net_zero=round(net["zero"], 4),
            net_realistic=round(net["realistic"], 4),
            net_stress2x=round(net["stress2x"], 4),
            net_retention_realistic=(round(ret_real, 4) if ret_real is not None else None),
            net_retention_stress2x=(round(ret_2x, 4) if ret_2x is not None else None),
            gap_zero=round(gap_by_level["zero"], 4),
            gap_realistic=round(gap_by_level["realistic"], 4),
            gap_stress2x=round(gap_by_level["stress2x"], 4),
            null_turnover=null_to[str(lb)],
            cost_verdict=verdict,
            cost_test_informative=(char["signal_character"] == "TRADING"
                                   and net["zero"] > 0),
        )
    return dict(signal=char, by_lookback=raw, derived=derived)


def load_admitted_assets():
    if not SWEEP_ARTIFACT.exists():
        return None
    with open(SWEEP_ARTIFACT) as f:
        sweep = json.load(f)
    per = sweep["universe"]["per_asset"]
    return [t for t in sweep["universe"]["ticker_order"]
            if per[t]["robustness_verdict"] == "ROBUST"]


def cross_check_zero_column(per_asset):
    if not SWEEP_ARTIFACT.exists():
        return dict(status="SKIPPED", detail="prior sweep artifact absent")
    with open(SWEEP_ARTIFACT) as f:
        sweep = json.load(f)
    sw = sweep["per_asset"]
    mism = []
    for ticker, prof in per_asset.items():
        for lb in LOOKBACKS:
            z = prof["by_lookback"][str(lb)]["zero"]
            pub = sw[ticker][str(lb)]
            if (z["medians"] != pub["medians"]
                    or z["null_medians"] != pub["null_medians"]
                    or z["verdict"] != pub["verdict"]):
                mism.append(dict(ticker=ticker, lookback=lb,
                                 medians=z["medians"], pub_medians=pub["medians"],
                                 verdict=z["verdict"], pub_verdict=pub["verdict"]))
    return dict(status=("MATCH" if not mism else "MISMATCH"),
                n_checked=len(per_asset) * len(LOOKBACKS), mismatches=mism)


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

    print("\n=== 3. Leakage review per asset (lookback=5, realistic cost) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, cost_cfg("realistic"))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], fills {len(res.trades)}, "
              f"realistic total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Signal character (turnover / long-fraction, ref lookback 5) ===")
    print("  STATIC if signal turnover <= {} over the full series".format(
        STATIC_TURNOVER_MAX))
    for ticker in manifest["universe"]:
        bars = bt.load_ticker(ticker)[0]
        c = signal_character(ticker, bars)
        print("    {:6s} turnover={:<5d} frac_long={:.4f} -> {}".format(
            ticker, c["signal_turnover"], c["signal_frac_long"], c["signal_character"]))

    print("\n=== 5. Cost-sensitivity grid across collected universe ===")
    print("  Cost levels: {} | lookbacks {} | reference lookback {}".format(
        list(COST_LEVELS), list(LOOKBACKS), REFERENCE_LOOKBACK))
    for level in COST_LEVELS:
        print("    {:9s}: {}".format(level, COST_MODELS[level]))
    print("  Walk-forward per segment: train={}d / test={}d / warmup={}d / "
          "overlap={}d, min {} bars/segment, {} volatility blocks; matched "
          "coin-flip null costed identically\n".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS, N_BLOCKS))

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    per_asset = {t: run_asset(t, all_tickers[t]) for t in ticker_order}

    ref = str(REFERENCE_LOOKBACK)
    ref_counts = {"COST_ROBUST": 0, "COST_ERODED": 0, "COST_FAILS": 0, "NO_EDGE_BASE": 0}
    char_counts = {"STATIC": 0, "TRADING": 0}
    informative = []
    print("  Per-asset reference-lookback ({}) net medians and cost verdict:".format(ref))
    for ticker in ticker_order:
        d = per_asset[ticker]["derived"][ref]
        ch = per_asset[ticker]["signal"]["signal_character"]
        ref_counts[d["cost_verdict"]] += 1
        char_counts[ch] += 1
        if d["cost_test_informative"]:
            informative.append(ticker)
        print("    {:6s} net zero/real/2x = {:+.4f}/{:+.4f}/{:+.4f}  "
              "ret2x={}  null_turn={}  -> {} [{}]".format(
                  ticker, d["net_zero"], d["net_realistic"], d["net_stress2x"],
                  ("n/a" if d["net_retention_stress2x"] is None
                   else "{:.2f}".format(d["net_retention_stress2x"])),
                  d["null_turnover"], d["cost_verdict"], ch))
    print("  signal character counts: {}".format(char_counts))
    print("  reference-lookback cost counts: {}".format(ref_counts))
    print("  informative (TRADING & positive zero-cost) assets: {}".format(informative))

    admitted = load_admitted_assets()
    if admitted is not None:
        adm_counts = {"COST_ROBUST": 0, "COST_ERODED": 0, "COST_FAILS": 0, "NO_EDGE_BASE": 0}
        for ticker in admitted:
            adm_counts[per_asset[ticker]["derived"][ref]["cost_verdict"]] += 1
        print("  previously-admitted momentum assets ({}): {}".format(len(admitted), admitted))
        print("  their reference-lookback cost counts: {}".format(adm_counts))

    print("\n=== 6. Determinism (AMZN/JPM subset recomputed) ===")
    res2 = {t: run_asset(t, tickers[t]) for t in target}
    out1 = json.dumps({t: per_asset[t] for t in target}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("  determinism r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    print("\n=== 7. Zero-cost column cross-check vs prior momentum lookback sweep ===")
    xcheck = cross_check_zero_column(per_asset)
    print("  {}".format(xcheck["status"]))
    for m in xcheck.get("mismatches", []):
        print("    MISMATCH {}".format(m))

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        reference_lookback=REFERENCE_LOOKBACK,
        cost_levels=list(COST_LEVELS),
        cost_models=COST_MODELS,
        retention_floor=RETENTION_FLOOR,
        static_turnover_max=STATIC_TURNOVER_MAX,
        a_priori_prediction=A_PRIORI_PREDICTION,
        interpretation=INTERPRETATION,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=per_asset,
        universe=dict(
            ticker_order=ticker_order,
            signal_character_counts=char_counts,
            reference_lookback_cost_counts=ref_counts,
            informative_assets=informative,
            admitted_momentum_assets=admitted,
            admitted_reference_lookback_cost_counts=(
                None if admitted is None else {
                    k: sum(1 for t in admitted
                           if per_asset[t]["derived"][ref]["cost_verdict"] == k)
                    for k in ref_counts
                }
            ),
            per_asset=dict(
                (t, dict(
                    signal_character=per_asset[t]["signal"]["signal_character"],
                    signal_turnover=per_asset[t]["signal"]["signal_turnover"],
                    signal_frac_long=per_asset[t]["signal"]["signal_frac_long"],
                    reference_lookback_cost_verdict=per_asset[t]["derived"][ref]["cost_verdict"],
                    reference_lookback_net_zero=per_asset[t]["derived"][ref]["net_zero"],
                    reference_lookback_net_realistic=per_asset[t]["derived"][ref]["net_realistic"],
                    reference_lookback_net_stress2x=per_asset[t]["derived"][ref]["net_stress2x"],
                    reference_lookback_null_turnover=per_asset[t]["derived"][ref]["null_turnover"],
                ))
                for t in ticker_order
            ),
        ),
        zero_cost_cross_check=xcheck,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity on the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())