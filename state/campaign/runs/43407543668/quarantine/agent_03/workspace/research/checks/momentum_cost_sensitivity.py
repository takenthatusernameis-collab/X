"""Real-data transaction-cost sensitivity for short-horizon momentum.

Objective (task R-001): classify whether the qualified short-horizon momentum
edge (LEARNING_STATE: ADMITTED AS CANDIDATE POSITIVE EVIDENCE, lookback 3/5/10,
6/10 lookback-ROBUST assets) survives conservative execution costs on the
collected 10-asset real-data universe.

Design contract (fixed a-priori, before any cost result is seen):

* Signal: `research.backtest.momentum_signals` (long the previous ``lookback``
  day return, hold 1 day, daily rebalance). No signal rule is tuned here.
* Universe: the existing 10 collected tickers. No expansion, no leverage.
* Walk-forward and regime segmentation: identical to
  `research/checks/momentum_lookback_sweep.py` (train=252d / test=84d /
  warmup=60d / overlap=60d, 4 contiguous volatility blocks, min 400 bars).
* Cost grid (predeclared): four levels, applied identically to the candidate
  and to the matched coin-flip null.

    zero         : BacktestConfig defaults (no commission, no slippage)
    realistic    : commission $2/trade + $0.003/share, 2c fixed slippage,
                   5 bps proportional  (the repository's documented
                   "realistic costs" model; see
                   examples/ma_crossover_real_data.py and
                   logs/ACTIVATION-2026-10-04.md)
    conservative : same commission, 4c fixed slippage, 10 bps proportional
                   (2x the realistic proportional/fixed slippage)
    stress       : same commission, 8c fixed slippage, 20 bps proportional
                   (4x the realistic slippage)

* Classification rules (predeclared, unchanged after seeing results):
    per (asset, lookback, cost): EDGE if the segment verdict is REGIME_STABLE
        and every segment median is positive; LOSS if REGIME_STABLE_LOSS;
        otherwise NO_EDGE.
    per (asset, cost): COST_ROBUST if >= 2 of the 3 lookbacks show EDGE;
        COST_FRAGILE if exactly 1; COST_ERASED if 0. This is the same rule the
        existing lookback sweep uses for lookback robustness.
    universe: let B be the set of assets that are COST_ROBUST at zero cost
        (the lookback-robust set from the prior artifact). At each cost level,
        SURVIVING = assets in B still COST_ROBUST. Verdict is PERSISTS if
        SURVIVING == B, PARTIAL if SURVIVING is a non-empty proper subset of B,
        ERASED if SURVIVING is empty.

* Gate 0 (internal consistency): the zero-cost column must reproduce
  `state/check_artifacts/momentum_lookback_sweep_results.json` exactly
  (medians, null medians, verdicts, per lookback) for all 10 assets. A
  mismatch aborts the run, because it would mean this check is not the same
  experiment.

* Gates: manifest checksums, data preflight, leakage review (signal integrity
  + fill-equity audit with costs), determinism (AMZN/JPM rerun twice).

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
REFERENCE_LOOKBACK = 5

# Predeclared cost grid. Each entry is (name, BacktestConfig kwargs).
COST_LEVELS = (
    ("zero", {}),
    ("realistic", dict(
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    )),
    ("conservative", dict(
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=4.0,
        slippage_proportional=0.0010,
    )),
    ("stress", dict(
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=8.0,
        slippage_proportional=0.0020,
    )),
)
COST_NAMES = tuple(name for name, _ in COST_LEVELS)

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
LOOKBACK_ARTIFACT = ARTIFACT_DIR / "momentum_lookback_sweep_results.json"


def cost_config(name):
    for n, kwargs in COST_LEVELS:
        if n == name:
            return bt.BacktestConfig(**kwargs)
    raise KeyError(name)


def segment_result(bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker, lookback, cost level.

    Same segmentation, windows and null as `momentum_lookback_sweep.py`; the
    only difference is that `cfg` is forwarded to the engine so commission and
    slippage apply to both the candidate and the matched coin-flip null.
    """
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"signals {len(sig)} != bars {len(closes)}")
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
    scen = res.scenarios
    return dict(
        segments=[s.name for s in scen],
        medians=[round(s.baseline_median_log_return, 3) for s in scen],
        null_medians=[round(s.noise_median_log_return, 3) for s in scen],
        medians_raw=[round(s.baseline_median_log_return, 9) for s in scen],
        null_medians_raw=[round(s.noise_median_log_return, 9) for s in scen],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        candidate_dispersion_raw=round(res.candidate_dispersion, 9),
        null_dispersion_raw=round(res.null_dispersion, 9),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


def lookback_verdict(sr):
    """EDGE / LOSS / NO_EDGE for one lookback result at one cost level."""
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def robustness_verdict(lookback_results):
    """COST_ROBUST (>=2 EDGE lookbacks) / COST_FRAGILE (1) / COST_ERASED (0)."""
    counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    for lb in LOOKBACKS:
        counts[lookback_verdict(lookback_results[str(lb)])] += 1
    if counts["EDGE"] >= 2:
        return "COST_ROBUST", counts
    if counts["EDGE"] == 1:
        return "COST_FRAGILE", counts
    return "COST_ERASED", counts


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
        if status != "OK":
            print("MANIFEST MISMATCH; aborting.")
            sys.exit(1)
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

    print("\n=== 3. Leakage review with costs (lookback=5, realistic) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=REFERENCE_LOOKBACK)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        cfg = cost_config("realistic")
        res = bt.run_bars(list(bars), signals, cfg)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        n_fills = len(res.trades)
        print(f"  {ticker}: {bars.n_bars} bars, {n_fills} fills, leakage [PASS], "
              f"total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "commission + slippage are charged on every fill.")

    print("\n=== 4. Predeclared cost grid ===")
    for name, kwargs in COST_LEVELS:
        print(f"  {name:12s} {kwargs if kwargs else 'BacktestConfig defaults (zero cost)'}")
    print("  Same grid applied to candidate and matched coin-flip null.\n")

    # Full universe: [cost][lookback][ticker] results.
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    results = {}
    for cname in COST_NAMES:
        cfg = cost_config(cname)
        results[cname] = {
            lb: {t: segment_result(all_tickers[t], lb, cfg) for t in ticker_order}
            for lb in LOOKBACKS
        }

    print("=== 5. Cost-sensitivity results across the collected universe ===")
    per_asset = {}
    for ticker in ticker_order:
        profile = {}
        for cname in COST_NAMES:
            lb_res = {str(lb): results[cname][lb][ticker] for lb in LOOKBACKS}
            v, counts = robustness_verdict(lb_res)
            profile[cname] = dict(
                lookback=lb_res,
                robustness_verdict=v,
                edge_counts=counts,
                medians_ref=lb_res[str(REFERENCE_LOOKBACK)]["medians"],
                null_medians_ref=lb_res[str(REFERENCE_LOOKBACK)]["null_medians"],
                verdict_ref=lb_res[str(REFERENCE_LOOKBACK)]["verdict"],
            )
        per_asset[ticker] = profile

    # Baseline (zero-cost) lookback-robust set B, and cost survival.
    baseline_robust = [t for t in ticker_order
                       if per_asset[t]["zero"]["robustness_verdict"] == "COST_ROBUST"]
    cost_universe = {}
    for cname in COST_NAMES:
        counts = {"COST_ROBUST": 0, "COST_FRAGILE": 0, "COST_ERASED": 0}
        for t in ticker_order:
            counts[per_asset[t][cname]["robustness_verdict"]] += 1
        surviving = [t for t in baseline_robust
                     if per_asset[t][cname]["robustness_verdict"] == "COST_ROBUST"]
        if set(surviving) == set(baseline_robust):
            verdict = "PERSISTS"
        elif surviving:
            verdict = "PARTIAL"
        else:
            verdict = "ERASED"
        cost_universe[cname] = dict(
            robustness_counts=counts,
            baseline_robust=baseline_robust,
            surviving=surviving,
            lost=[t for t in baseline_robust if t not in surviving],
            verdict=verdict,
        )

    print(f"  baseline (zero cost) COST_ROBUST assets: {baseline_robust}")
    for cname in COST_NAMES:
        cu = cost_universe[cname]
        print(f"  {cname:12s} counts={cu['robustness_counts']} "
              f"surviving={cu['surviving']} lost={cu['lost']} -> {cu['verdict']}")
    print("\n  Per-asset lookback-5 candidate medians by cost level:")
    for ticker in ticker_order:
        med = ", ".join(
            "{}:{:+.3f}({})".format(
                cname,
                float(np.median(per_asset[ticker][cname]["medians_ref"])),
                per_asset[ticker][cname]["verdict_ref"],
            )
            for cname in COST_NAMES
        )
        print(f"    {ticker:6s} {med}")
    print("")

    # Cost-drag diagnostic: how much candidate return each cost level actually
    # removes relative to zero cost, per asset (median over segments).
    cost_drag = {}
    for ticker in ticker_order:
        base = float(np.median(per_asset[ticker]["zero"]["medians_ref"]))
        cost_drag[ticker] = {
            cname: round(float(np.median(per_asset[ticker][cname]["medians_ref"])) - base, 4)
            for cname in COST_NAMES
        }
    print("  Cost drag vs zero cost (median segment return delta):")
    for cname in COST_NAMES[1:]:
        drags = [cost_drag[t][cname] for t in ticker_order]
        print(f"    {cname:12s} min={min(drags):+.4f} median={float(np.median(drags)):+.4f} "
              f"max={max(drags):+.4f}")
    print("")

    # Gate 0: zero-cost column must reproduce the existing lookback sweep.
    print("=== 6. Gate 0: zero-cost cross-consistency vs momentum_lookback_sweep ===")
    with open(LOOKBACK_ARTIFACT) as f:
        prior = json.load(f)
    prior_assets = prior["per_asset"]
    g0_ok = True
    for ticker in ticker_order:
        for lb in LOOKBACKS:
            pub = prior_assets[ticker][str(lb)]
            got = per_asset[ticker]["zero"]["lookback"][str(lb)]
            same = (
                got["medians"] == pub["medians"]
                and got["null_medians"] == pub["null_medians"]
                and got["verdict"] == pub["verdict"]
            )
            if not same:
                g0_ok = False
                print(f"  MISMATCH {ticker} lookback {lb}: "
                      f"medians {got['medians']} vs {pub['medians']}; "
                      f"nulls {got['null_medians']} vs {pub['null_medians']}")
    if not g0_ok:
        print("  GATE 0 FAILED: this check is not the same experiment as the "
              "prior lookback sweep. Aborting before writing an artifact.")
        sys.exit(1)
    print("  all 10 tickers x lookbacks 3/5/10 reproduce the prior artifact "
          "medians/nulls -> MATCH")

    # Determinism: rerun AMZN/JPM over all cost levels and lookbacks twice.
    print("\n=== 7. Determinism (AMZN/JPM, all cost levels and lookbacks) ===")
    det = {}
    for cname in COST_NAMES:
        cfg = cost_config(cname)
        det[cname] = {
            lb: {t: segment_result(tickers[t], lb, cfg) for t in target}
            for lb in LOOKBACKS
        }
    out1 = json.dumps(
        {c: {lb: {t: results[c][lb][t] for t in target} for lb in LOOKBACKS}
         for c in COST_NAMES}, sort_keys=True)
    out2 = json.dumps(det, sort_keys=True)
    deterministic = out1 == out2
    print(f"  r1 == r2: {deterministic}")
    assert deterministic, "non-deterministic output"

    # Cost-model behavior diagnostic: isolate commission from slippage on a
    # single full-sample run. If additive slippage were a genuine round-trip
    # cost, the slippage-only variant would underperform zero cost.
    print("\n=== 8. Cost-model behavior diagnostic (AMZN, full sample, lookback 5) ===")
    amzn_bars = all_tickers["AMZN"]
    amzn_sig = bt.momentum_signals(amzn_bars.closes_array(), lookback=REFERENCE_LOOKBACK)
    variants = (
        ("zero", bt.BacktestConfig()),
        ("commission_only", bt.BacktestConfig(
            commission_per_trade=2.0, commission_per_share=0.003)),
        ("slippage_only_stress", bt.BacktestConfig(
            slippage_cents=8.0, slippage_proportional=0.0020)),
        ("full_stress", cost_config("stress")),
    )
    behavior = {}
    for vname, cfg in variants:
        r = bt.run_bars(list(amzn_bars), amzn_sig, cfg)
        total = float((r.equity_curve[-1] - 1e6) / 1e6)
        comm = float(sum(f.commission for f in r.trades))
        behavior[vname] = dict(
            total_return=round(total, 4),
            commission_paid=round(comm, 2),
            n_fills=len(r.trades),
        )
        print(f"  {vname:22s} total_return={total:+.4f} "
              f"commission_paid=${comm:,.0f} fills={len(r.trades)}")
    print("  Interpretation: slippage-only must not reduce (and may raise) the")
    print("  return if the engine's additive fill-price slippage cancels over a")
    print("  round trip; a genuine cost would make it worse than zero cost.")

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        reference_lookback=REFERENCE_LOOKBACK,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_levels=[dict(name=n, config=k) for n, k in COST_LEVELS],
        classification=dict(
            per_lookback="EDGE if REGIME_STABLE and all medians > 0; LOSS if "
                         "REGIME_STABLE_LOSS; else NO_EDGE",
            per_asset="COST_ROBUST if >=2 EDGE lookbacks; COST_FRAGILE if 1; "
                      "COST_ERASED if 0",
            universe="PERSISTS if all zero-cost COST_ROBUST assets stay "
                     "COST_ROBUST; PARTIAL if a non-empty subset; ERASED if none",
        ),
        per_asset=per_asset,
        cost_drag=cost_drag,
        cost_model_behavior=behavior,
        universe=dict(
            ticker_order=ticker_order,
            baseline_robust=baseline_robust,
            cost_levels=cost_universe,
        ),
        determinism=dict(r1_eq_r2=deterministic, assets=target),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Bounded cost-sensitivity check on the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())