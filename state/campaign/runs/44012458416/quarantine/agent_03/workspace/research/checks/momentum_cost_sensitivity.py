"""Cost-sensitivity check for the short-horizon momentum candidate.

Objective (campaign task R-001): decide whether the admitted short-horizon
momentum candidate (long the previous N-day return, hold 1 day, daily
rebalance) survives conservative transaction-cost stress on the collected
real-data universe.

Why this is decision-relevant: every prior momentum artifact
(`research/checks/momentum.py`, `momentum_lookback_sweep.py`,
`momentum_longer_horizon.py`, `momentum_horizon_gap_analysis.py`) was produced
with the framework's DEFAULT ``BacktestConfig`` -- i.e. ZERO commission and
ZERO slippage. A mechanically daily-rebalanced 1-day directional bet is the
exact strategy class whose verdict is decided by transaction costs, so the
zero-cost verdict is not yet an execution-realistic verdict.

Predeclared cost grid (fixed a-priori, NOT tuned to results):

- ``zero``            : ``BacktestConfig()`` defaults (all cost components 0).
                        This is the reference column; it must reproduce the
                        prior zero-cost lookback sweep.
- ``realistic``       : the repository's existing realistic cost model, copied
                        verbatim from ``examples/ma_crossover_real_data.py``
                        (commission_per_trade=2.0, commission_per_share=0.003,
                        slippage_cents=2.0, slippage_proportional=0.0005).
- ``conservative_2x`` : every realistic cost component doubled. A predeclared
                        stress level, not a fitted parameter.

Lookbacks {3, 5, 10} and the 10-asset collected universe are held fixed from
the prior lookback sweep. The matched null is the framework's own coin-flip
benchmark run on the SAME segments with the SAME cost config, so the null pays
the same cost model as the candidate (``stress_segments`` forwards ``cfg`` to
both ``parameter_sweep`` and ``noise_benchmark``).

A-priori falsification prediction (fixed before execution): the momentum signal
flips exposure whenever the lookback return changes sign, so it churns; the
``realistic`` level charges a fixed 2-cents-per-share slippage that is large
relative to split-adjusted historical prices (e.g. AAPL's 2009 adjusted close
is ~2.7). The prediction is therefore that the candidate's absolute per-segment
median log returns fall sharply at ``realistic`` (potentially uniformly
negative) and that FEWER THAN HALF of the zero-cost EDGE cells remain EDGE at
``realistic`` -> classification COST_FRAGILE. The opposite outcome (COST_ROBUST)
would refute this prediction.

Predeclared headline classification rule (over all 10 assets x 3 lookbacks = 30
cells), where a cell is EDGE when its verdict is REGIME_STABLE and every segment
median is positive:

- ``NO_EDGE_AT_ZERO`` : no zero-cost EDGE cell exists (degenerate; nothing to
  stress).
- ``COST_ROBUST``     : >= 50% of the zero-cost EDGE cells are still EDGE at
  ``realistic``.
- ``COST_FRAGILE``    : some zero-cost EDGE cells exist but < 50% survive
  ``realistic``.

``candidate_minus_null`` (mean candidate median minus mean null median, per cell
and cost level) is reported as a diagnostic: the verdict's fixed 0.05 band is
applied to absolute medians, which under high costs is dominated by the cost
drag common to candidate and null; the gap is the cleaner "does the signal add
value beyond a cost-paying coin flip" measure.

Turnover diagnostic (interpretability, not a rule change): a cost verdict is
only meaningful if the strategy takes directional positions. The check therefore
also reports, per asset and lookback, the momentum signal's weight counts and
sign flips, plus the full-sample cost drag (zero-cost minus realistic terminal
return) and fill count. The engine rebalances to a constant notional every bar,
so ``n_fills`` counts rebalancing fills, not directional trades; the signal's
``n_sign_flips`` is the directional-turnover measure. If directional turnover is
near zero, the cost test is DEGENERATE and the headline must be read as "costs
are not the binding constraint", not as validation of an execution-realistic
trading edge. This diagnostic is reported because the predeclared cost
comparison alone cannot distinguish a cost-robust strategy from a signal that
never changes direction.

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
LOOKBACK_REFERENCE = 5
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"

# The repository's existing realistic cost model (examples/ma_crossover_real_data.py).
REALISTIC_COSTS = dict(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

# Predeclared cost grid. ``zero`` uses framework defaults (no cost components).
COST_LEVELS = {
    "zero": dict(),
    "realistic": dict(REALISTIC_COSTS),
    "conservative_2x": {k: 2.0 * v for k, v in REALISTIC_COSTS.items()},
}
COST_ORDER = ("zero", "realistic", "conservative_2x")


def cost_config(level):
    """BacktestConfig for a predeclared cost level."""
    return bt.BacktestConfig(**COST_LEVELS[level])


def edge_status(medians, verdict):
    """Classify one cell's regime-stability result as EDGE / LOSS / NO_EDGE.

    EDGE requires REGIME_STABLE with every segment median positive; a
    REGIME_STABLE_LOSS verdict is a stable loss; everything else is NO_EDGE.
    """
    if verdict == "REGIME_STABLE" and all(m > 0 for m in medians):
        return "EDGE"
    if verdict == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def classify_cell(edge_zero, edge_realistic, edge_conservative):
    """Per-cell cost-robustness class across the three cost levels."""
    if not edge_zero:
        return "NO_EDGE_AT_ZERO"
    if edge_realistic and edge_conservative:
        return "SURVIVES"
    if edge_realistic:
        return "SURVIVES_REALISTIC"
    return "COST_FRAGILE"


def cell_result(ticker, bars, lookback, level):
    """Regime-stability result for one ticker / lookback / cost level.

    Uses the same walk-forward windows and volatility segments as the prior
    momentum lookback sweep; the only change is the cost config, which is
    forwarded to both the candidate sweep and the matched coin-flip null.
    """
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    signals = bt.momentum_signals(closes, lookback=lookback)
    if len(signals) != len(closes):
        raise ValueError(f"{ticker}: signals {len(signals)} != bars {len(closes)}")
    cfg = cost_config(level)
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
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cfg,
    )
    medians = [s.baseline_median_log_return for s in res.scenarios]
    nulls = [s.noise_median_log_return for s in res.scenarios]
    gap = float(np.mean(medians) - np.mean(nulls)) if medians else 0.0
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(m, 3) for m in medians],
        null_medians=[round(n, 3) for n in nulls],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        verdict=res.overall_verdict,
        n_folds=res.n_folds,
        edge_status=edge_status(medians, res.overall_verdict),
        gap=round(gap, 3),
    )


def turnover_profile(bars, lookback):
    """Signal turnover diagnostic: weight counts and sign flips for one lookback.

    A cost-sensitivity verdict is only informative if the signal actually
    trades. This reports how often the momentum signal is long, short, or flat
    and how many times its sign flips (each flip is a full position reversal in
    the engine). It is a diagnostic on the signal, not a change to the
    predeclared cost grid or classification rule.
    """
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    weights = np.array(
        [s.weight for s in bt.momentum_signals(closes, lookback=lookback)],
        dtype=np.float64)
    flips = int(np.sum(np.diff(weights) != 0.0))
    return dict(
        n_bars=int(len(weights)),
        n_long=int(np.sum(weights > 0)),
        n_short=int(np.sum(weights < 0)),
        n_flat=int(np.sum(weights == 0)),
        n_sign_flips=flips,
        flips_per_bar=round(flips / max(1, len(weights)), 4),
        mean_weight=round(float(np.mean(weights)), 4),
    )


def cost_incidence(bars, lookback):
    """Full-sample cost incidence at one lookback.

    Reports how much realistic costs reduce the terminal equity and how many
    fills the signal generates. If the signal barely trades, the cost drag is
    near zero and the cost test cannot discriminate a trading edge.
    """
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    signals = bt.momentum_signals(closes, lookback=lookback)
    zero = bt.run_bars(list(bars), signals, cost_config("zero"))
    real = bt.run_bars(list(bars), signals, cost_config("realistic"))
    zero_ret = (zero.equity_curve[-1] - 1e6) / 1e6
    real_ret = (real.equity_curve[-1] - 1e6) / 1e6
    total_comm = float(sum(f.commission for f in real.trades))
    return dict(
        zero_return=round(float(zero_ret), 4),
        realistic_return=round(float(real_ret), 4),
        cost_drag=round(float(zero_ret - real_ret), 4),
        n_fills_realistic=len(real.trades),
        total_commission_realistic=round(total_comm, 2),
        fills_per_bar=round(len(real.trades) / max(1, len(closes)), 4),
    )


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

    print("\n=== 3. Leakage review + cost-accounting audit (AMZN, JPM) ===")
    target = ["AMZN", "JPM"]
    leakage = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=LOOKBACK_REFERENCE)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        entry = {}
        for level in COST_ORDER:
            res = bt.run_bars(list(bars), signals, cost_config(level))
            bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
            total_return = (res.equity_curve[-1] - 1e6) / 1e6
            entry[level] = round(total_return, 4)
        leakage[ticker] = dict(bars=bars.n_bars, signals=len(signals),
                               total_return_by_cost=entry)
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], equity-fill audit [PASS], "
              f"total_return by cost {entry}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Predeclared cost grid ===")
    for level in COST_ORDER:
        print(f"  {level:16s}: {COST_LEVELS[level]}")
    print("  (realistic copied verbatim from examples/ma_crossover_real_data.py; "
          "conservative_2x doubles every component)")

    all_tickers = {}
    for entry in manifest["entries"]:
        all_tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]
    ticker_order = list(all_tickers.keys())

    print("\n=== 5. Cost x lookback sweep across collected universe ===")
    print("  Params: walk-forward per segment train={}d / test={}d / warmup={}d / "
          "overlap={}d, min {} bars/segment, {} contiguous volatility blocks, "
          "matched coin-flip null per cost level\n".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS, N_BLOCKS))

    cells = {t: {} for t in ticker_order}
    for lookback in LOOKBACKS:
        for ticker in ticker_order:
            cells[ticker][str(lookback)] = {}
            for level in COST_ORDER:
                cells[ticker][str(lookback)][level] = cell_result(
                    ticker, all_tickers[ticker], lookback, level)

    # Per-cell and headline cost-robustness classification.
    robustness = {t: {} for t in ticker_order}
    edge_counts = {level: {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0} for level in COST_ORDER}
    verdict_counts = {level: {} for level in COST_ORDER}
    class_counts = {"SURVIVES": 0, "SURVIVES_REALISTIC": 0,
                    "COST_FRAGILE": 0, "NO_EDGE_AT_ZERO": 0}
    gap_sum = {level: [] for level in COST_ORDER}
    for ticker in ticker_order:
        for lookback in LOOKBACKS:
            key = str(lookback)
            per_level = cells[ticker][key]
            for level in COST_ORDER:
                edge_counts[level][per_level[level]["edge_status"]] += 1
                v = per_level[level]["verdict"]
                verdict_counts[level][v] = verdict_counts[level].get(v, 0) + 1
                gap_sum[level].append(per_level[level]["gap"])
            e0 = per_level["zero"]["edge_status"] == "EDGE"
            er = per_level["realistic"]["edge_status"] == "EDGE"
            ec = per_level["conservative_2x"]["edge_status"] == "EDGE"
            cls = classify_cell(e0, er, ec)
            class_counts[cls] += 1
            robustness[ticker][key] = dict(
                edge_zero=e0, edge_realistic=er, edge_conservative=ec,
                cost_robustness=cls,
            )

    zero_edge_cells = edge_counts["zero"]["EDGE"]
    realistic_edge_cells = edge_counts["realistic"]["EDGE"]
    survived = sum(
        1 for t in ticker_order for lb in LOOKBACKS
        if robustness[t][str(lb)]["edge_zero"] and robustness[t][str(lb)]["edge_realistic"]
    )
    survival_fraction = (survived / zero_edge_cells) if zero_edge_cells else None
    if zero_edge_cells == 0:
        headline = "NO_EDGE_AT_ZERO"
    elif survival_fraction is not None and survival_fraction >= 0.5:
        headline = "COST_ROBUST"
    else:
        headline = "COST_FRAGILE"

    print("  Per-cell edge counts by cost level (30 cells = 10 assets x 3 lookbacks):")
    for level in COST_ORDER:
        ec = edge_counts[level]
        print(f"    {level:16s}: EDGE={ec['EDGE']:2d} LOSS={ec['LOSS']:2d} "
              f"NO_EDGE={ec['NO_EDGE']:2d}  mean_gap={np.mean(gap_sum[level]):+.3f}")
    print("  Verdict counts by cost level:")
    for level in COST_ORDER:
        print(f"    {level:16s}: {verdict_counts[level]}")
    print(f"  Cost-robustness classes (30 cells): {class_counts}")
    print(f"  Zero-cost EDGE cells: {zero_edge_cells} | still EDGE at realistic: "
          f"{realistic_edge_cells} | survived: {survived} | "
          f"survival_fraction={survival_fraction if survival_fraction is None else round(survival_fraction, 3)}")
    print(f"  HEADLINE CLASSIFICATION: {headline}")

    print("\n  Per-asset detail (medians at each lookback: zero / realistic / 2x):")
    for ticker in ticker_order:
        parts = []
        for lookback in LOOKBACKS:
            p = cells[ticker][str(lookback)]
            parts.append("lb{}:{}/{}/{}".format(
                lookback,
                p["zero"]["medians"], p["realistic"]["medians"],
                p["conservative_2x"]["medians"]))
        rob = robustness[ticker]
        rob_s = ",".join("{}:{}".format(lb, rob[str(lb)]["cost_robustness"])
                         for lb in LOOKBACKS)
        print(f"    {ticker:6s} {rob_s}")
        for part in parts:
            print(f"        {part}")

    print("\n=== 6. Turnover / cost-incidence diagnostic ===")
    print("  A cost verdict is informative only if the signal trades. This reports")
    print("  signal weight counts, sign flips, and full-sample cost drag.")
    turnover = {}
    for t in ticker_order:
        bars_t = all_tickers[t]
        turnover[t] = {str(lb): turnover_profile(bars_t, lb) for lb in LOOKBACKS}
        turnover[t]["cost_incidence"] = cost_incidence(bars_t, LOOKBACK_REFERENCE)
        inc = turnover[t]["cost_incidence"]
        flips = [turnover[t][str(lb)]["n_sign_flips"] for lb in LOOKBACKS]
        tb5 = turnover[t]["5"]
        print(f"    {t:6s} flips(lb3/5/10)={flips} "
              f"cost_drag={inc['cost_drag']:+.4f} n_fills={inc['n_fills_realistic']} "
              f"frac_long(lb5)={tb5['n_long'] / max(1, tb5['n_bars']):.2f} "
              f"min_close_note=mean_weight {tb5['mean_weight']:+.3f}")
    all_flips = [turnover[t][str(lb)]["n_sign_flips"]
                 for t in ticker_order for lb in LOOKBACKS]
    median_flips = float(np.median(all_flips))
    mean_cost_drag = float(np.mean(
        [turnover[t]["cost_incidence"]["cost_drag"] for t in ticker_order]))
    print(f"  median sign flips per asset/lookback = {median_flips:.0f} over ~4000 bars; "
          f"mean cost drag = {mean_cost_drag:+.4f}")
    degenerate = median_flips < 0.05 * float(np.median(
        [turnover[t]["5"]["n_bars"] for t in ticker_order]))
    if degenerate:
        print("  CAVEAT: near-zero directional signal turnover -> the cost test is")
        print("  DEGENERATE. Costs cannot discriminate a signal that never changes")
        print("  direction. Read the COST_ROBUST headline as 'costs are not the")
        print("  binding constraint', NOT as validation of an execution-realistic")
        print("  trading edge.")
    else:
        print("  Signal directional turnover is non-trivial; the cost test is informative.")

    # Determinism: rerun the AMZN/JPM subset across all lookbacks/cost levels.
    res2 = {t: {str(lb): {level: cell_result(t, bt.load_ticker(t)[0], lb, level)
                          for level in COST_ORDER}
                for lb in LOOKBACKS}
            for t in target}
    out1 = json.dumps({t: {lb: cells[t][lb] for lb in cells[t]} for t in target},
                      sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("\ndeterminism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        lookback_reference=LOOKBACK_REFERENCE,
        cost_levels={level: COST_LEVELS[level] for level in COST_ORDER},
        cost_order=list(COST_ORDER),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        leakage=leakage,
        cells=cells,
        robustness=robustness,
        turnover=turnover,
        summary=dict(
            edge_counts=edge_counts,
            verdict_counts=verdict_counts,
            cost_robustness_counts=class_counts,
            gap_mean={level: round(float(np.mean(gap_sum[level])), 3)
                      for level in COST_ORDER},
            zero_edge_cells=zero_edge_cells,
            realistic_edge_cells=realistic_edge_cells,
            survived_realistic=survived,
            survival_fraction=(None if survival_fraction is None
                               else round(survival_fraction, 3)),
            headline=headline,
            median_sign_flips=median_flips,
            mean_cost_drag=round(mean_cost_drag, 4),
            degenerate_cost_test=bool(degenerate),
        ),
        determinism=True,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity sweep across the collected "
          "universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())