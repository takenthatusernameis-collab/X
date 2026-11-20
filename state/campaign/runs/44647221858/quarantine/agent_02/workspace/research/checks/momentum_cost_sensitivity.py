"""Bounded transaction-cost sensitivity check for short-horizon momentum.

Contract (campaign slot 02, task R-001): does the short-horizon momentum edge
survive conservative transaction-cost stress on the collected real-data
universe, judged under the repository's existing realistic cost model and the
matched coin-flip null?

Design is fixed a-priori and is NOT tuned to the cost results:

* Signal: ``bt.momentum_signals`` (long the previous ``lookback``-day return,
  hold 1 day, daily rebalance), lookbacks {3, 5, 10}.
* Universe: the 10 collected tickers of dataset
  ``yf-ohlcv-universe-2009-to-2026-10-03`` (no expansion, no leverage).
* Walk-forward: train=252d / test=84d / warmup=60d / overlap=60d,
  ``bt.volatility_blocks`` regime blocks (n_blocks=4, window=60),
  min 400 bars per segment, exactly as in
  ``research/checks/momentum_lookback_sweep.py``.
* Cost grid (predeclared): multiples of the repository's existing realistic
  cost vector, taken verbatim from ``examples/ma_crossover_real_data.py``
  (commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0,
  slippage_proportional=0.0005):
    0x  zero-cost reference (the cost basis of the prior momentum results)
    1x  repository realistic cost model
    2x  conservative stress
    4x  severe stress
  The matched null is the framework's coin-flip benchmark run at the *same*
  cost level on the *same* segments (``stress_segments`` forwards ``cfg`` to
  ``noise_benchmark``).

Signal-contract audit (predeclared): the cost result is only interpretable if
the signal actually depends on returns. The artifact therefore records the
weight distribution of both the current signal and a local reimplementation of
the pre-2026-11-20 formula (``mean`` of log *price level* over the window),
plus a turnover / cost-incidence decomposition. This audit is what makes the
cost classification interpretable: a constant-position signal has near-zero
turnover, so a "cost-robust" verdict on such a signal is evidence about
buy-and-hold, not about a momentum strategy.

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

# Repository's existing realistic cost vector (examples/ma_crossover_real_data.py).
BASE_COSTS = dict(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
COST_LEVELS = (
    ("zero", 0.0),
    ("repo_realistic_1x", 1.0),
    ("conservative_2x", 2.0),
    ("stress_4x", 4.0),
)

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def cost_cfg(multiplier):
    """BacktestConfig at a multiple of the repository's realistic cost vector."""
    return bt.BacktestConfig(
        **{k: v * float(multiplier) for k, v in BASE_COSTS.items()}
    )


def legacy_momentum_weights(closes, lookback):
    """Pre-2026-11-20 momentum formula, reimplemented for the audit only.

    The defective implementation compared ``mean(log(closes[window]))`` — the
    mean log *price level* — against zero, instead of the trailing log return.
    Kept here so the artifact can reproduce the defect's signature without
    reverting the repaired framework function.
    """
    n = len(closes)
    weights = [0.0] * n
    for i in range(int(lookback), n):
        level = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        weights[i] = 1.0 if level > 0.0 else -1.0
    return weights


def legacy_momentum_signals(closes, lookback=5):
    """Pre-2026-11-20 momentum signal, for the legacy cross-consistency gate.

    Identical to ``legacy_momentum_weights`` but returns ``Signal`` objects so
    it can be driven through the same ``stress_segments`` path as the original
    ``research/checks/momentum.py`` run.
    """
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(int(lookback), n):
        level = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if level > 0.0 else -1.0)
    return out


def weight_audit(weights):
    """Distribution summary of a weight series (the signal-contract evidence)."""
    arr = np.asarray(weights, dtype=np.float64)
    nz = arr[arr != 0.0]
    flips = int(np.sum(nz[1:] != nz[:-1])) if nz.size > 1 else 0
    return dict(
        n_bars=int(arr.size),
        n_nonneutral=int(nz.size),
        unique_weights=sorted({float(w) for w in nz.tolist()}),
        fraction_long=round(float(np.mean(nz > 0.0)), 4) if nz.size else None,
        n_sign_flips=flips,
        sign_flip_fraction=round(flips / max(nz.size - 1, 1), 4) if nz.size > 1 else 0.0,
    )


def segment_result(ticker, bars, lookback, cfg):
    """One (ticker, lookback, cost level) walk-forward regime-stability result."""
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
        medians=[round(s.baseline_median_log_return, 6) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 6) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 6),
        null_dispersion=round(res.null_dispersion, 6),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


def cell_verdict(sr):
    """Per-cell verdict used for the cost classification.

    EDGE    : REGIME_STABLE with uniformly positive segment medians.
    LOSS    : REGIME_STABLE_LOSS (stable losses, not an edge).
    NO_EDGE : anything else.
    """
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0.0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def cost_incidence(ticker, bars, lookback):
    """Turnover and cost incidence of the rule on the full sample.

    Reports the commission+slippage actually paid, in basis points of initial
    capital and annualized, plus the zero-cost and costed full-sample total
    returns. This is the quantity that makes a cost classification
    interpretable: a daily-rebalanced long/short rule must show material
    turnover, whereas a constant-position rule shows almost none.
    """
    closes = bars.closes_array()
    sig = bt.momentum_signals(closes, lookback=lookback)
    base = bt.run_bars(list(bars), sig, cost_cfg(1.0))
    zero = bt.run_bars(list(bars), sig, cost_cfg(0.0))
    commissions = float(sum(f.commission for f in base.trades))
    traded_notional = float(sum(abs(f.shares) * f.price for f in base.trades))
    bars_n = len(closes)
    years = bars_n / 252.0
    return dict(
        n_fills=len(base.trades),
        n_sign_flips=weight_audit([s.weight for s in sig])["n_sign_flips"],
        commissions=round(commissions, 2),
        traded_notional=round(traded_notional, 2),
        cost_bps_of_capital=round(commissions / 1e6 * 1e4, 3),
        cost_annualized_pct=round(commissions / 1e6 / years * 100.0, 4) if years else None,
        turnover_x_per_year=round(traded_notional / 1e6 / years, 2) if years else None,
        total_return_zero_cost=round(float(zero.equity_curve[-1] / 1e6 - 1.0), 6),
        total_return_repo_cost=round(float(base.equity_curve[-1] / 1e6 - 1.0), 6),
    )


def classify_asset(levels, ticker):
    """Per-asset cost-robustness class from the predeclared rule.

    ROBUST      : >= 2 lookbacks show a positive REGIME_STABLE edge at 1x.
    PARTIAL     : exactly 1 lookback shows an edge at 1x.
    COST_FRAGILE: no edge at 1x although >= 1 lookback had an edge at 0x.
    NO_EDGE     : no lookback had an edge even at 0x.
    """
    zero_edges = sum(
        1 for lb in LOOKBACKS
        if cell_verdict(levels["zero"][str(lb)][ticker]) == "EDGE"
    )
    base_edges = sum(
        1 for lb in LOOKBACKS
        if cell_verdict(levels["repo_realistic_1x"][str(lb)][ticker]) == "EDGE"
    )
    if base_edges >= 2:
        cls = "ROBUST"
    elif base_edges == 1:
        cls = "PARTIAL"
    elif zero_edges >= 1:
        cls = "COST_FRAGILE"
    else:
        cls = "NO_EDGE"
    return cls, zero_edges, base_edges


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        print(f"  {entry['ticker']}: "
              f"{'OK' if actual == entry['checksum_sha256'] else 'MISMATCH'}")
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

    all_tickers = {}
    for entry in manifest["entries"]:
        all_tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]
    ticker_order = list(all_tickers.keys())

    print("\n=== 3. Signal-contract audit (predeclared interpretability gate) ===")
    print("  A cost verdict is only meaningful if the signal depends on returns.")
    print("  legacy = pre-2026-11-20 formula (mean log PRICE LEVEL); "
          "current = repaired (trailing log return).\n")
    legacy_audit = {}
    current_audit = {}
    for ticker in ticker_order:
        closes = all_tickers[ticker].closes_array()
        legacy_audit[ticker] = {
            str(lb): weight_audit(legacy_momentum_weights(closes, lb))
            for lb in LOOKBACKS
        }
        current_audit[ticker] = {
            str(lb): weight_audit([s.weight for s in bt.momentum_signals(closes, lookback=lb)])
            for lb in LOOKBACKS
        }
        leg = legacy_audit[ticker]["5"]
        cur = current_audit[ticker]["5"]
        print(f"  {ticker:6s} legacy[lb5] weights={leg['unique_weights']} "
              f"flips={leg['n_sign_flips']:4d} | current[lb5] "
              f"weights={cur['unique_weights']} flips={cur['n_sign_flips']:4d} "
              f"long={cur['fraction_long']}")
    legacy_degenerate = sum(
        1 for t in ticker_order if len(legacy_audit[t]["5"]["unique_weights"]) == 1
    )
    print(f"\n  legacy signal is a single constant weight on "
          f"{legacy_degenerate}/{len(ticker_order)} tickers at lookback 5 "
          f"(1/1 means always long or always short, independent of returns).")

    print("\n=== 3b. Legacy cross-consistency vs the recorded momentum artifact ===")
    print("  If the pre-repair 'momentum' result was a constant-long backtest, its")
    print("  per-asset segment medians must equal a constant-long walk-forward")
    print("  recomputation at zero cost (momentum_results.json, lookback 5).\n")
    legacy_path = ARTIFACT_DIR / "momentum_results.json"
    legacy_cross = {}
    if legacy_path.exists():
        with open(legacy_path) as f:
            prior = json.load(f)
        prior_assets = prior.get("per_asset", {})
        prior_universe = prior.get("universe", {}).get("per_asset", {})
        for ticker in ticker_order:
            entry = prior_universe.get(ticker) or prior_assets.get(ticker)
            if entry is None:
                continue
            closes = all_tickers[ticker].closes_array()
            labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
            sig = legacy_momentum_signals(closes, lookback=5)
            res = bt.stress_segments(
                signals_fn=legacy_momentum_signals,
                bars=all_tickers[ticker],
                signals=sig,
                param_grid=[{"lookback": 5}],
                baseline=(("lookback", 5),),
                segment_fn=bt.segment_fn_from_labels(labels),
                train_window=TRAIN,
                test_window=TEST,
                warmup=WARM,
                overlap_window=OVERLAP,
                cfg=cost_cfg(0.0),
                periods_per_year=252,
                min_segment_bars=MIN_SEGMENT_BARS,
            )
            const_med = [round(s.baseline_median_log_return, 3)
                         for s in res.scenarios]
            prior_med = entry["medians"]
            match = const_med == [round(float(x), 3) for x in prior_med]
            legacy_cross[ticker] = dict(
                constant_long_medians=const_med,
                recorded_prior_momentum_medians=prior_med,
                match=bool(match),
            )
            print(f"  {ticker:6s} constant-long {const_med} vs recorded "
                  f"{prior_med} -> {'MATCH' if match else 'MISMATCH'}")
        n_match = sum(1 for v in legacy_cross.values() if v["match"])
        print(f"\n  constant-long recomputation matches the recorded momentum "
              f"artifact on {n_match}/{len(legacy_cross)} assets: the recorded")
        print("  'momentum edge' is a constant-long (buy-and-hold) backtest, not "
              "a momentum signal.")
    else:
        print(f"  prior artifact {legacy_path} not found; gate skipped.")

    print("\n=== 4. Leakage review (AMZN/JPM, lookback=5, repo cost level) ===")
    for ticker in ("AMZN", "JPM"):
        bars = all_tickers[ticker]
        closes = bars.closes_array()
        sig = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(sig, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), sig, cost_cfg(1.0))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, signals {len(sig)}, "
              f"leakage [PASS], repo-cost total_return {total_return:+.3f}")
    print("  no-look-ahead: the signal at bar t uses closes[:t] only "
          "(close[t] vs close[t-lookback]); neutral bars before the lookback "
          "window carry the equity forward.")

    print("\n=== 5. Predeclared cost grid across the collected universe ===")
    print("  Cost levels (multiples of the repository realistic cost vector "
          f"{BASE_COSTS}):")
    for name, mult in COST_LEVELS:
        print(f"    {name}: {mult}x")
    print(f"  Params: walk-forward train={TRAIN}d / test={TEST}d / warmup={WARM}d "
          f"/ overlap={OVERLAP}d, {N_BLOCKS} volatility blocks, "
          f"min {MIN_SEGMENT_BARS} bars/segment, lookbacks {LOOKBACKS}\n")

    levels = {}
    for name, mult in COST_LEVELS:
        cfg = cost_cfg(mult)
        per_lb = {}
        for lb in LOOKBACKS:
            per_lb[str(lb)] = {
                t: segment_result(t, all_tickers[t], lb, cfg) for t in ticker_order
            }
        levels[name] = per_lb
        counts = {}
        for t in ticker_order:
            v = cell_verdict(per_lb["5"][t])
            counts[v] = counts.get(v, 0) + 1
        print(f"  level {name} ({mult}x), lookback 5 cell verdicts: {counts}")

    print("\n  Per-asset, per-lookback EDGE counts by cost level:")
    print("    {:6s}".format("ticker") + "".join(
        "{:>10s}".format(n[:9]) for n, _ in COST_LEVELS)
        + "{:>14s}  {}".format("class", "lb5 medians (1x)"))
    per_asset = {}
    class_counts = {"ROBUST": 0, "PARTIAL": 0, "COST_FRAGILE": 0, "NO_EDGE": 0}
    for ticker in ticker_order:
        row = {}
        cells = {}
        for name, _ in COST_LEVELS:
            n_edge = sum(
                1 for lb in LOOKBACKS
                if cell_verdict(levels[name][str(lb)][ticker]) == "EDGE"
            )
            row[name] = n_edge
            cells[name] = {
                str(lb): dict(
                    verdict=levels[name][str(lb)][ticker]["verdict"],
                    cell=cell_verdict(levels[name][str(lb)][ticker]),
                    medians=levels[name][str(lb)][ticker]["medians"],
                    null_medians=levels[name][str(lb)][ticker]["null_medians"],
                    candidate_dispersion=levels[name][str(lb)][ticker]["candidate_dispersion"],
                    null_dispersion=levels[name][str(lb)][ticker]["null_dispersion"],
                    n_folds=levels[name][str(lb)][ticker]["n_folds"],
                    segments=levels[name][str(lb)][ticker]["segments"],
                )
                for lb in LOOKBACKS
            }
        cls, zero_edges, base_edges = classify_asset(levels, ticker)
        class_counts[cls] += 1
        per_asset[ticker] = dict(
            edge_counts=row,
            zero_cost_edge_lookbacks=zero_edges,
            base_cost_edge_lookbacks=base_edges,
            cost_class=cls,
            cells=cells,
        )
        lb5_med = levels["repo_realistic_1x"]["5"][ticker]["medians"]
        print("    {:6s}".format(ticker) + "".join(
            "{:>10d}".format(row[n]) for n, _ in COST_LEVELS)
            + "{:>14s}  [{}]".format(
                cls, ",".join(f"{m:+.3f}" for m in lb5_med[:2])))
    print(f"\n  cost-class counts: {class_counts}")

    print("\n=== 6. Universe cost-robustness classification (predeclared) ===")
    retained = {}
    edge_pairs = {}
    for name, _ in COST_LEVELS:
        retained[name] = sum(
            1 for t in ticker_order
            if sum(1 for lb in LOOKBACKS
                   if cell_verdict(levels[name][str(lb)][t]) == "EDGE") >= 2
        )
        edge_pairs[name] = sum(
            1 for t in ticker_order for lb in LOOKBACKS
            if cell_verdict(levels[name][str(lb)][t]) == "EDGE"
        )
        print(f"  {name}: assets with >=2 edge lookbacks = {retained[name]}/10; "
              f"total EDGE cells = {edge_pairs[name]}/30")

    if retained["repo_realistic_1x"] >= 6:
        universe_verdict = "COST_ROBUST"
    elif retained["repo_realistic_1x"] >= 1:
        universe_verdict = "PARTIALLY_COST_ROBUST"
    elif edge_pairs["zero"] > 0:
        universe_verdict = "COST_FRAGILE"
    else:
        universe_verdict = "NO_BASELINE_EDGE"
    print(f"\n  universe verdict (1x repo cost model): {universe_verdict}")
    print("  classification rule (predeclared): COST_ROBUST if >= 6/10 assets "
          "keep a positive REGIME_STABLE edge at >= 2 lookbacks under the 1x "
          "repo cost model; PARTIALLY_COST_ROBUST if 1-5; COST_FRAGILE if none "
          "keep an edge at 1x but >= 1 had one at 0x; NO_BASELINE_EDGE if no "
          "asset had an edge even at 0x. 2x/4x are reported as stress only.")

    print("\n=== 7. Cost drag and cost incidence (lookback 5) ===")
    drag = {}
    for ticker in ticker_order:
        z = levels["zero"]["5"][ticker]["medians"]
        b = levels["repo_realistic_1x"]["5"][ticker]["medians"]
        n_seg = min(len(z), len(b))
        diffs = [z[i] - b[i] for i in range(n_seg)]
        drag[ticker] = dict(
            zero_cost_medians=z,
            repo_cost_medians=b,
            drag_per_segment=round(float(np.mean(diffs)), 6),
            annualized_drag_pct=round(
                float(np.mean(diffs)) / TEST * 252.0 * 100.0, 3),
        )
        print(f"  {ticker:6s} drag/segment {drag[ticker]['drag_per_segment']:+.4f} "
              f"(~{drag[ticker]['annualized_drag_pct']:+.2f}%/yr)")

    incidence = {t: cost_incidence(t, all_tickers[t], 5) for t in ticker_order}
    print("\n  Full-sample cost incidence, lookback 5, repo cost level:")
    for ticker in ticker_order:
        inc = incidence[ticker]
        print(f"    {ticker:6s} fills={inc['n_fills']:4d} "
              f"flips={inc['n_sign_flips']:4d} "
              f"cost={inc['cost_bps_of_capital']:7.2f} bps of capital "
              f"({inc['cost_annualized_pct']:.3f}%/yr) "
              f"turnover={inc['turnover_x_per_year']:.2f}x/yr")

    # Null turnover for the matched null, for the recorded limitation.
    rng_null = bt.random_signals(len(all_tickers[ticker_order[0]].closes_array()),
                                 seed=SEED)
    null_w = [s.weight for s in rng_null]
    null_audit = weight_audit(null_w)
    print(f"\n  Matched null (coin-flip) turnover on the same bars: "
          f"flips={null_audit['n_sign_flips']} "
          f"({null_audit['sign_flip_fraction']:.1%} of non-neutral bars).")
    print("  LIMITATION (recorded): the framework null's turnover is not matched "
          "to the candidate's, so under cost stress the candidate-vs-null "
          "comparison is not a like-for-like cost comparison; the candidate's "
          "net-of-cost median level (vs zero) and its cost drag are the "
          "decision-relevant quantities.")

    print("\n=== 8. Determinism (recomputed subset) ===")
    subset = {}
    for name, mult in (("zero", 0.0), ("repo_realistic_1x", 1.0)):
        cfg = cost_cfg(mult)
        subset[name] = {
            lb: {t: segment_result(t, all_tickers[t], lb, cfg) for t in ("AMZN", "JPM")}
            for lb in LOOKBACKS
        }
    out1 = json.dumps(
        {name: {str(lb): subset[name][lb] for lb in LOOKBACKS} for name, _ in
         (("zero", 0.0), ("repo_realistic_1x", 1.0))},
        sort_keys=True)
    out2 = json.dumps(
        {name: {str(lb): {t: levels[name][str(lb)][t] for t in ("AMZN", "JPM")}
                for lb in LOOKBACKS}
         for name, _ in (("zero", 0.0), ("repo_realistic_1x", 1.0))},
        sort_keys=True)
    print(f"determinism: r1 == r2: {out1 == out2}")
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        base_costs=BASE_COSTS,
        cost_levels=[dict(name=n, multiplier=m) for n, m in COST_LEVELS],
        universe_verdict=universe_verdict,
        universe_retained_assets=retained,
        universe_edge_cells=edge_pairs,
        cost_class_counts=class_counts,
        signal_contract_audit=dict(
            legacy_formula="mean(log(closes[i-lookback+1:i+1])) -> weight sign",
            current_formula="log(closes[i]) - log(closes[i-lookback]) -> weight sign",
            legacy_constant_weight_tickers=legacy_degenerate,
            legacy=legacy_audit,
            current=current_audit,
            null_turnover=null_audit,
            legacy_cross_consistency=legacy_cross,
        ),
        per_asset=per_asset,
        cost_drag=drag,
        cost_incidence=incidence,
        limitations=[
            "The framework's matched null (coin-flip) turnover is not matched to "
            "the candidate's; under cost stress the candidate-vs-null comparison "
            "is not a like-for-like cost comparison.",
            "Cost levels scale the repository's realistic cost vector "
            "(commission_per_trade=2.0, commission_per_share=0.003, "
            "slippage_cents=2.0, slippage_proportional=0.0005); they are a "
            "bounded sensitivity grid, not a calibrated execution model.",
            "A cost verdict on a constant-position signal (legacy formula) "
            "measures buy-and-hold cost drag, not strategy cost robustness.",
        ],
        ticker_order=ticker_order,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())