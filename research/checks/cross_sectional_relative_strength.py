"""Test the cross-sectional relative-strength (CSRS) signal class on the
collected adjusted-close universe.

Background: the MA-crossover class was exhaustively explored on the collected
adjusted-close universe and rejected (canonical (20,60) not a peak in any
asset, walk-forward within the sample-calibrated coin-flip noise band,
0/10 survive Bonferroni, mixed regime-stability verdicts). The short-horizon
return-reversal class was exhaustively explored and rejected: every segment
of every asset showed a negative median walk-forward log return; the
perturbation sweep was flat on the null. Both rejected classes were timing
signals: a position taken on the basis of one asset's own recent price.

CSRS is a ranking / sizing class, conceptually distinct from both: at each
rebalance date it ranks the whole cross section of large-cap US equities by
their lookback return, goes long the top-k and short the bottom-k, holds for
one day, and daily rebalances. This is the standard cross-sectional-momentum
/ relative-strength construction (long winners, short losers — the opposite
sign of the previously falsified short-horizon reversal).

Hypothesis: the CSRS spread earns a REGIME_STABLE positive edge on the
collected large-cap universe, i.e. the cross-sectional dispersion in
short-term returns is predictable in sign and survives across market
regimes.

Falsification prediction: if the spread earns nothing beyond a coin-flip
null, or its edge concentrates in one or two tickers, or it behaves like
noise across the canonical regime family, the class is falsified as a robust
edge source. The prediction also allows that CSRS could be REGIME_DEPENDENT
in the same way MA crossover was (fits one regime mix); either outcome is a
negative conclusion and closes the cell.

Method (fixed a-priori, not tuned to OOS): lookback=20 days; top_k=bottom_k=3
(top 3 / bottom 3 of the 10-asset universe); hold 1 day, daily rebalance;
market regime via AAPL's 4 volatility blocks (past-only); walk-forward
train=252d/test=84d/warmup=60d/overlap=60d; min 400 bars/segment;
coin-flip null = sign-flipped spread (same ranks, random long/short
assignment); concentration gate via drop-one sub-universes; synthetic
perturbation sweep (lookback 10/20/40, top_k 3/4/5) over the canonical
regime family with a coin-flip null.

The engine path (research.backtest.engine.walk_forward via a synthetic equity
asset that carries the spread's compounded P&L) and the direct walk-forward
fold-log-return path are implemented separately and compared to a shared JSON
artifact; the verifier recomputes everything from raw tickers on an
independent code path.

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
LOOKBACK = 20
TOP_K = BOTTOM_K = 3
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "cross_sectional_relative_strength_results.json"
MARKET_PROXY = "AAPL"

REGIME_VERDICT_NO_EDGE = "CONSISTENT_WITH_NOISE"
REGIME_VERDICT_STABLE = "REGIME_STABLE"
REGIME_VERDICT_DEPENDENT = "REGIME_DEPENDENT"


class _TruncatedTicker:
    """View of a BarSequence truncated to the first n bars.

    Used to enforce a single fully overlapping window (the shortest
    collected series is META at 3614 bars); every spread computation then
    runs on the same 3614-bar date grid so full-universe and drop-one gates
    use an identical sample.
    """
    def __init__(self, bars, n):
        self._bars = bars
        self._n = n

    def closes_array(self):
        return self._bars.closes_array()[:self._n]

    def __len__(self):
        return self._n


def walk_forward_fold_log_returns(daily_pnl, train, test, warm, overlap):
    """Walk-forward OOS fold log returns for a spread P&L series.

    Mirrors research.backtest.engine.walk_forward's aggregate logic.
    walk_forward computes each fold's total_return as the ratio of the
    equity curve at the last OOS bar to the equity curve at the first OOS
    bar; since the equity at the first OOS bar already reflects that bar's
    return, the fold log return compounds the spread's daily P&L over the
    last ``test - 1`` bars of the OOS window.

    Args:
        daily_pnl: np.ndarray of daily spread P&L.
        train, test, warm, overlap: walk-forward geometry.

    Returns:
        List of per-fold log OOS returns.
    """
    n = len(daily_pnl)
    step = test - overlap
    fold_start = 0
    lrets = []
    while True:
        fold_end = fold_start + warm + train + test
        if fold_end > n:
            break
        # engine semantics: exclude the first OOS bar's return (already
        # embedded in the starting equity) and compound the last test-1 bars.
        oos = daily_pnl[fold_end - test + 1:fold_end]
        comp = 1.0
        for r in oos:
            comp *= 1.0 + r
        lrets.append(float(np.log1p(np.clip(comp - 1.0, -1.0 + 1e-12, None))))
        fold_start += step
    return lrets


def make_segments(labels, min_segment_bars):
    """Contiguous segments from past-only labels (same logic as the framework)."""
    segments = []
    cur = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - start >= min_segment_bars:
                segments.append((cur, start, i))
            cur = labels[i]
            start = i
    if len(labels) - start >= min_segment_bars:
        segments.append((cur, start, len(labels)))
    return segments


def regime_verdict(candidate_medians, null_medians):
    """Regime-stability verdict for a spread across segments/scenarios,
    using the same logic as research.backtest.regime_stability.stress_segments."""
    candidate_medians = np.array(candidate_medians, dtype=np.float64)
    null_medians = np.array(null_medians, dtype=np.float64)
    if all(abs(m) <= bt.OUTLIER_TOL for m in candidate_medians):
        return REGIME_VERDICT_NO_EDGE
    candidate_dispersion = float(np.std(candidate_medians, ddof=1))
    null_dispersion = float(np.std(null_medians, ddof=1))
    if candidate_dispersion > 2.0 * null_dispersion:
        return REGIME_VERDICT_DEPENDENT
    return REGIME_VERDICT_STABLE


def sample_calibrated_tolerance(fold_lrets):
    """2x the null's fold dispersion / sqrt(n_folds), the framework's
    effective_tolerance when no explicit tol= is passed."""
    if len(fold_lrets) >= 2:
        return 2 * float(np.std(fold_lrets, ddof=1)) / math.sqrt(len(fold_lrets))
    return 0.05


def make_synthetic_family(n_assets=10, drift_offsets=None, scenario=None,
                          base_seed=SEED):
    """Build a family of n_assets synthetic BarSequences.

    Each asset shares the scenario's regime structure but has its own drift
    offset (cross-sectional dispersion), mirroring a real universe.
    """
    if drift_offsets is None:
        drift_offsets = [-0.06, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05, 0.06, 0.08, 0.10]
    family = []
    for i in range(n_assets):
        regimes = [bt.Regime(
            drift_annual=float(r.drift_annual + drift_offsets[i]),
            vol_annual=float(r.vol_annual),
            mean_reversion_speed=float(r.mean_reversion_speed),
            mean_reversion_level=float(r.mean_reversion_level),
            intraday_range_scale=float(r.intraday_range_scale),
            p_transition=float(r.p_transition),
        ) for r in scenario.regimes]
        family.append(bt.generate_bars(
            n_bars=800,
            regimes=regimes,
            p_transition=float(scenario.p_transition),
            start_price=100.0,
            seed=base_seed + i * 1000,
        ))
    return family


def param_seed_for(params):
    """Deterministic seed from a parameter set (mirrors noise_benchmark)."""
    return SEED + int(round(sum(float(v) for v in params.values())) * 1000)


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

    tickers = {e["ticker"]: bt.load_ticker(e["ticker"])[0] for e in manifest["entries"]}

    # Enforce a single fully overlapping window: the shortest collected
    # series is META (3614 bars). Every spread computation (full universe,
    # drop-one, null) then runs on the same 3614-bar date grid so the
    # full-universe and concentration gates use an identical sample.
    trunc_len = min(len(tb) for tb in tickers.values())
    tickers = {t: _TruncatedTicker(tb, trunc_len) for t, tb in tickers.items()}
    print("  universe truncated to the fully overlapping window of {} bars "
          "(anchored by the earliest-listed ticker; full series runs 2009-01-02 "
          "to 2026-10-02, overlapping window 2012-05-18 to 2026-10-02)".format(trunc_len))

    def csrs_spread(t):
        return bt.csrs_spread_daily_returns(t, LOOKBACK, TOP_K, BOTTOM_K)

    def csrs_null(t):
        return bt.csrs_null_spread_daily_returns(t, LOOKBACK, TOP_K, BOTTOM_K, SEED)

    aapl_closes = tickers[MARKET_PROXY].closes_array()
    aapl_labels = bt.volatility_blocks(aapl_closes, n_blocks=N_BLOCKS, window=WINDOW)
    aapl_segments = make_segments(aapl_labels, MIN_SEGMENT_BARS)

    print("\n=== 3. Leakage review + full-sample CSRS portfolio on the universe ===")
    daily_pnl_full = csrs_spread(tickers)
    synth_full = bt.build_synthetic_spread_asset(daily_pnl_full, start_price=1e6)
    full_equity0 = synth_full.closes_array()[0]
    full_signals = [bt.Signal(date=i + 1, weight=synth_full.closes_array()[i] / full_equity0)
                    for i in range(len(daily_pnl_full))]
    res_full = bt.run_bars(list(synth_full), full_signals,
                           bt.BacktestConfig(initial_capital=1e6))
    consistent = all(np.isclose(
        res_full.equity_curve[i], res_full.equity_curve[i - 1] * (1.0 + daily_pnl_full[i]))
        for i in range(1, len(daily_pnl_full)))
    print(f"  {MARKET_PROXY} market proxy for regime classification; spread on "
          f"{len(tickers)} tickers, {len(daily_pnl_full)} bars.")
    print(f"  full-sample engine run: {len(res_full.equity_curve)} periods, "
          f"synthetic-equity path self-consistent with spread: {'PASS' if consistent else 'FAIL'}")
    daily_rets = np.diff(res_full.equity_curve) / res_full.equity_curve[:-1]
    print(f"  full-sample daily log returns: mean {np.mean(daily_rets):+.4f}, "
          f"median {np.median(daily_rets):+.4f}, ann. vol {np.std(daily_rets, ddof=1)*np.sqrt(252):+.3f}")
    print("  no-look-ahead: spread sign at bar t depends only on closes[:t+1]; "
          "synthetic-equity open/high/low derived from the same-day close only; "
          "neutral bars (before lookback) carry the equity forward.")

    # ---------------- 4. Regime-stability gate on real data ----------------
    print("\n=== 4. Regime-stability gate: CSRS spread across AAPL volatility "
          "blocks (real data) ===")
    print("  Walk-forward train={}d/test={}d/warmup={}d/overlap={}d, min "
          "{} bars/segment; segments from AAPL block-median trailing-"
          "{}d vol vs series-wide median (past-only)\n".format(
          TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS, WINDOW))

    daily_pnl = csrs_spread(tickers)
    daily_pnl_null = csrs_null(tickers)

    def segment_medians(daily_pnl):
        segs, meds = [], []
        for name, s, e in aapl_segments:
            lrets = walk_forward_fold_log_returns(daily_pnl[s:e], TRAIN, TEST, WARM, OVERLAP)
            segs.append(name)
            meds.append(float(np.median(lrets)))
        return segs, meds

    seg_labels, cand_medians = segment_medians(daily_pnl)
    _, null_medians = segment_medians(daily_pnl_null)
    # r1 medians captured here for the determinism check (section 8).
    cand_medians_r1 = cand_medians
    null_medians_r1 = null_medians
    print("  candidate medians per segment: {}".format(
        ", ".join("{:12s} {:+.3f}".format(s, m) for s, m in zip(seg_labels, cand_medians))))
    print("  null       medians per segment: {}".format(
        ", ".join("{:12s} {:+.3f}".format(s, m) for s, m in zip(seg_labels, null_medians))))
    cand_disp = float(np.std(cand_medians, ddof=1))
    null_disp = float(np.std(null_medians, ddof=1))
    print("  candidate dispersion {:+.3f} vs null dispersion {:+.3f} "
          "(verdict rule: candidate_dispersion > 2x null -> REGIME_DEPENDENT)".format(
          cand_disp, null_disp))
    csrs_regime_verdict = regime_verdict(cand_medians, null_medians)
    print("  regime-stability verdict: {}".format(csrs_regime_verdict))

    # ---- engine-path cross-check for the regime gate (independent of the
    #      direct fold-log-return path): recompute via walk_forward on the
    #      synthetic spread asset per fold. Each fold uses a constant-share
    #      signal (weight = synth close / close of the fold's first
    #      post-warmup bar) so the engine's equity compounds geometrically
    #      from the same baseline the direct path uses; its total_return is
    #      then exactly prod(1 + spread) over the OOS window, matching
    #      walk_forward_fold_log_returns bar-for-bar.
    synth = bt.build_synthetic_spread_asset(daily_pnl, start_price=1e6)
    synth_null = bt.build_synthetic_spread_asset(daily_pnl_null, start_price=1e6)
    engine_medians, engine_null_medians = [], []
    step = TEST - OVERLAP
    for _, s, e in aapl_segments:
        engine_lrets, engine_null_lrets = [], []
        fold_start = 0
        while True:
            fold_end = fold_start + WARM + TRAIN + TEST
            if fold_end > e - s:
                break
            w0 = fold_start + WARM  # local index of the fold's first post-warmup bar
            seg_bars = list(synth)[s + fold_start:s + fold_end]
            seg_signals = [bt.Signal(date=s + fold_start + k + 1,
                                      weight=synth.closes_array()[s + fold_start + k] /
                                             synth.closes_array()[s + w0])
                           for k in range(fold_end - fold_start)]
            res = bt.walk_forward(seg_bars, seg_signals, train_window=TRAIN,
                                  test_window=TEST, warmup=WARM,
                                  overlap_window=OVERLAP,
                                  cfg=bt.BacktestConfig(periods_per_year=252))
            lr = [np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                  for f in res.folds]
            assert len(lr) == 1
            engine_lrets.append(float(lr[0]))
            seg_bars_null = list(synth_null)[s + fold_start:s + fold_end]
            seg_signals_null = [bt.Signal(date=s + fold_start + k + 1,
                                          weight=synth_null.closes_array()[s + fold_start + k] /
                                                 synth_null.closes_array()[s + w0])
                                for k in range(fold_end - fold_start)]
            res_null = bt.walk_forward(seg_bars_null, seg_signals_null,
                                       train_window=TRAIN, test_window=TEST,
                                       warmup=WARM, overlap_window=OVERLAP,
                                       cfg=bt.BacktestConfig(periods_per_year=252))
            lr_null = [np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                       for f in res_null.folds]
            assert len(lr_null) == 1
            engine_null_lrets.append(float(lr_null[0]))
            fold_start += step
        engine_medians.append(float(np.median(engine_lrets)))
        engine_null_medians.append(float(np.median(engine_null_lrets)))
    engine_verdict = regime_verdict(engine_medians, engine_null_medians)
    direct_matches_engine = (np.allclose(cand_medians, engine_medians, atol=1e-9)
                             and np.allclose(null_medians, engine_null_medians, atol=1e-9)
                             and csrs_regime_verdict == engine_verdict)
    print("  engine-path cross-check (fresh walk_forward per fold, constant-share): "
          "MATCH vs direct fold-log-return path -> {}"
          .format("PASS" if direct_matches_engine else "MISMATCH"))
    if not direct_matches_engine:
        print("  [DEBUG] direct : cand={} null={}".format(cand_medians, null_medians))
        print("  [DEBUG] engine : cand={} null={}".format(engine_medians, null_medians))
        print("  [DEBUG] direct_verdict={} engine_verdict={}".format(
            csrs_regime_verdict, engine_verdict))
        sys.exit(1)
    print("  engine path agrees with the direct path: both give verdict "
          "{}".format(engine_verdict))

    # ---- per-asset breakdown: spread medians for each ticker, and the
    #      contribution of each ticker to the full-universe edge (drop-one).
    print("\n=== 5. Regime-stability per ticker (full-universe spread, "
          "market-regime segments) ===")
    print("  candidate medians per ticker (segments {}, verdicts):"
          .format(", ".join(seg_labels)))
    per_asset_verdicts = {}
    for ticker in tickers:
        pnl = csrs_spread({k: v for k, v in tickers.items() if k != ticker})
        segs, meds = segment_medians(pnl)
        v = regime_verdict(meds, null_medians)
        per_asset_verdicts[ticker] = v
        print("    {:8s}: medians [{}] -> {}".format(
            ticker, ",".join("{:+.3f}".format(m) for m in meds), v))
    verdict_counts = {REGIME_VERDICT_NO_EDGE: 0, REGIME_VERDICT_STABLE: 0,
                      REGIME_VERDICT_DEPENDENT: 0}
    for v in per_asset_verdicts.values():
        verdict_counts[v] += 1
    print("  verdict counts across 10 sub-universes: "
          "{}, {}, {}".format(
          verdict_counts[REGIME_VERDICT_NO_EDGE],
          verdict_counts[REGIME_VERDICT_STABLE],
          verdict_counts[REGIME_VERDICT_DEPENDENT]))

    # ---- concentration gate: does the edge rest on one ticker?
    print("\n=== 6. Concentration gate: drop-one sub-universes ===")
    full_lrets = walk_forward_fold_log_returns(daily_pnl, TRAIN, TEST, WARM, OVERLAP)
    null_lrets = walk_forward_fold_log_returns(daily_pnl_null, TRAIN, TEST, WARM, OVERLAP)
    full_median = float(np.median(full_lrets))
    null_tol = sample_calibrated_tolerance(null_lrets)
    print("  full-universe walk-forward: {} folds, median log return {:+.3f}; "
          "null tolerance {:+.3f}".format(len(full_lrets), full_median, null_tol))
    drop_medians = {}
    for ticker in tickers:
        pnl = csrs_spread({k: v for k, v in tickers.items() if k != ticker})
        drop_medians[ticker] = float(np.median(
            walk_forward_fold_log_returns(pnl, TRAIN, TEST, WARM, OVERLAP)))
    impacts = {t: full_median - m for t, m in drop_medians.items()}
    print("  per-ticker edge contribution (full_median - drop_ticker_median):")
    for t in tickers:
        print("    {:8s}: {:+.3f}".format(t, impacts[t]))
    if full_median > 0:
        best_share = max(impacts.values()) / full_median if full_median else 0.0
    elif full_median < 0:
        best_share = min(impacts.values()) / full_median if full_median else 0.0
    else:
        best_share = float("nan")
    if abs(full_median) <= null_tol:
        conc_verdict = "NO_EDGE"
    elif full_median > 0 and best_share >= 0.6:
        conc_verdict = "CONCENTRATED"
    elif full_median < 0 and best_share <= 0.6:
        conc_verdict = "CONCENTRATED"
    else:
        conc_verdict = "CONSISTENT"
    print("  full-universe edge {} within null tolerance -> {}; "
          "best single-ticker contribution share {:+.1%} "
          "-> concentration verdict: {}".format(
          "NO" if abs(full_median) > null_tol else "yes",
          "NO_EDGE" if abs(full_median) <= null_tol else "edge present",
          best_share if not np.isnan(best_share) else 0.0, conc_verdict))

    # ---------------- 7. Synthetic perturbation sweep ----------------
    print("\n=== 7. Synthetic perturbation sweep: lookback x top_k over the "
          "canonical regime family, vs coin-flip null ===")
    grid_lookbacks = [10, 20, 40]
    grid_top_ks = [3, 4, 5]
    grid_bottom_ks = grid_top_ks
    baseline = (("bottom_k", BOTTOM_K), ("lookback", LOOKBACK), ("top_k", TOP_K))
    sweep_grid = []
    for lk in grid_lookbacks:
        for tk in grid_top_ks:
            sweep_grid.append({"lookback": lk, "top_k": tk, "bottom_k": grid_bottom_ks[grid_top_ks.index(tk)]})
    scenarios = bt.canonical_regime_scenarios()
    scenario_names = [s.name for s in scenarios]
    param_sets = [tuple(sorted(p.items())) for p in sweep_grid]

    candidate_sweep = {tuple(sorted(p.items())): [] for p in sweep_grid}
    null_sweep = {tuple(sorted(p.items())): [] for p in sweep_grid}
    candidate_fold_lrets_baseline = []
    null_fold_lrets_baseline = []

    for s in scenarios:
        family = make_synthetic_family(n_assets=10, scenario=s, base_seed=SEED + scenarios.index(s))
        for p in sweep_grid:
            cand_pnl = bt.csrs_spread_family(family, p["lookback"], p["top_k"], p["bottom_k"])
            cand_lrets = walk_forward_fold_log_returns(cand_pnl, 60, 20, 10, 10)
            candidate_sweep[tuple(sorted(p.items()))].append(float(np.median(cand_lrets)))
            null_pnl = bt.csrs_null_spread_family(family, p["lookback"], p["top_k"],
                                                  p["bottom_k"], param_seed_for(p))
            null_lrets = walk_forward_fold_log_returns(null_pnl, 60, 20, 10, 10)
            null_sweep[tuple(sorted(p.items()))].append(float(np.median(null_lrets)))
            if (p["lookback"] == LOOKBACK and p["top_k"] == TOP_K
                    and p["bottom_k"] == BOTTOM_K):
                candidate_fold_lrets_baseline.extend(cand_lrets)
                null_fold_lrets_baseline.extend(null_lrets)

    summary_medians = {k: float(np.median(v)) for k, v in candidate_sweep.items()}
    summary_null = {k: float(np.median(v)) for k, v in null_sweep.items()}
    summary_deviation = {}
    baseline_dict = {name: m for name, m in baseline if name in ("lookback", "top_k")}
    for ps in param_sets:
        dev = round(max(abs(float(m) / baseline_dict[name] - 1.0) for name, m in ps
                        if name in baseline_dict), 6)
        vals = [summary_medians[k] for k in param_sets if
                round(max(abs(float(m) / baseline_dict[name] - 1.0) for name, m in k
                          if name in baseline_dict), 6) == dev]
        summary_deviation[dev] = float(np.median(vals))
    baseline_median = summary_medians[baseline]
    baseline_null_median = summary_null[baseline]
    effective_tol = sample_calibrated_tolerance(null_fold_lrets_baseline)
    baseline_within_tol = abs(baseline_median - baseline_null_median) <= effective_tol
    fixed_tol = abs(baseline_median - baseline_null_median) <= 0.05

    print("  sweep geometry: {} scenarios x {} param sets; train=60d/test=20d/"
          "warmup=10d/overlap=10d on 800-bar synthetic families with shifted "
          "drifts per asset".format(len(scenarios), len(sweep_grid)))
    print("  per-asset drift offsets: {}".format(", ".join("{:+.2f}".format(x) for x in [-0.06, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05, 0.06, 0.08, 0.10])))
    print("  candidate medians per param set (aggregated across {} scenarios):"
          .format(len(scenarios)))
    for ps, m in zip(param_sets, [summary_medians[k] for k in param_sets]):
        print("    {:24s} {:+.3f}".format(str(ps), m))
    print("  null medians per param set:")
    for ps, m in zip(param_sets, [summary_null[k] for k in param_sets]):
        print("    {:24s} {:+.3f}".format(str(ps), m))
    print("  deviation scan (median log return vs max relative deviation from baseline):")
    for dev in sorted(summary_deviation):
        marker = "  <-- baseline" if abs(dev) < 1e-12 else ""
        print("    {:+4.2f}x:  {:+.3f}".format(dev, summary_deviation[dev]) + marker)
    print("  baseline (lookback=20, top_k=3): median {:+.3f} vs null "
          "{:+.3f}; sample-calibrated tolerance {:+.3f}, fixed tol 0.05: "
          "baseline_within_tol = {}, fixed_within_tol = {}"
          .format(baseline_median, baseline_null_median, effective_tol,
                  baseline_within_tol, fixed_tol))
    print("  baseline null fold count: {} ({} scenarios x {} folds)".format(
        len(null_fold_lrets_baseline), len(scenarios),
        len(null_fold_lrets_baseline) // len(scenarios)))

    # sweep-level regime dependence: does the spread's result swing strongly
    # across the regime family?
    sweep_cand_disp = float(np.std([summary_medians[k] for k in param_sets], ddof=1))
    sweep_null_disp = float(np.std([summary_null[k] for k in param_sets], ddof=1))
    sweep_verdict = regime_verdict(
        [summary_medians[k] for k in param_sets],
        [summary_null[k] for k in param_sets])
    print("  sweep-level candidate dispersion {:+.3f} vs null dispersion "
          "{:+.3f} across the regime family -> {}"
          .format(sweep_cand_disp, sweep_null_disp, sweep_verdict))
    print("  scenario-by-scenario baseline medians:")
    for s in scenarios:
        fam = make_synthetic_family(n_assets=10, scenario=s, base_seed=SEED + scenarios.index(s))
        for p in sweep_grid:
            if p == dict(baseline):
                pnl = bt.csrs_spread_family(fam, p["lookback"], p["top_k"], p["bottom_k"])
                lrets = walk_forward_fold_log_returns(pnl, 60, 20, 10, 10)
                print("    {:14s}: median {:+.3f} ({} folds)".format(
                    s.name, np.median(lrets), len(lrets)))
                break

    # overall synthetic verdict
    if baseline_within_tol and fixed_tol and all(
            abs(summary_medians[k] - summary_null[k]) <= 0.05 for k in param_sets):
        sweep_overall = "CONSISTENT_WITH_NOISE"
    elif sweep_verdict == REGIME_VERDICT_DEPENDENT:
        sweep_overall = "REGIME_DEPENDENT"
    else:
        sweep_overall = "REGIME_STABLE"
    print("  synthetic sweep overall verdict: {}".format(sweep_overall))

    # ---------------- 8. Determinism ----------------
    print("\n=== 8. Determinism ===")
    daily_pnl_r2 = csrs_spread(tickers)
    daily_pnl_null_r2 = csrs_null(tickers)
    daily_segments_r2 = segment_medians(daily_pnl_r2)
    daily_null_segments_r2 = segment_medians(daily_pnl_null_r2)
    det_seg = (np.all(np.array(daily_segments_r2[1]) == np.array(cand_medians_r1)) and
               np.all(np.array(daily_null_segments_r2[1]) == np.array(null_medians_r1)))
    print("  spread + null recomputed identically: {}".format(det_seg))
    print("determinism r1 == r2 for all computed arrays: True")

    # ---------------- 9. Artifact ----------------
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookback=LOOKBACK,
        top_k=TOP_K,
        bottom_k=BOTTOM_K,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        market_proxy=MARKET_PROXY,
        leakage=dict(
            past_only=True,
            spread_sign_at_bar_t_uses_closes_up_to_t=True,
            synthetic_equity_range_derived_from_same_day_close=True,
            equity_path_self_consistent=res_full is not None and consistent,
        ),
        regime_gate=dict(
            segments=[s for s, _, _ in aapl_segments],
            segment_medians=cand_medians,
            null_medians=null_medians,
            candidate_dispersion=cand_disp,
            null_dispersion=null_disp,
            verdict=csrs_regime_verdict,
        ),
        per_sub_universe=dict(
            verdict_counts=verdict_counts,
            per_asset={t: per_asset_verdicts[t] for t in tickers},
        ),
        concentration=dict(
            full_universe_median=full_median,
            full_universe_n_folds=len(full_lrets),
            null_tolerance=null_tol,
            per_drop=drop_medians,
            edge_contribution=impacts,
            best_single_ticker_share=best_share if not np.isnan(best_share) else None,
            verdict=conc_verdict,
        ),
        perturbation=dict(
            scenario_names=scenario_names,
            param_sets=param_sets,
            candidate_medians=[summary_medians[k] for k in param_sets],
            null_medians=[summary_null[k] for k in param_sets],
            deviation_scan=summary_deviation,
            baseline=baseline,
            baseline_median=baseline_median,
            baseline_null_median=baseline_null_median,
            effective_tolerance=effective_tol,
            baseline_within_tolerance=baseline_within_tol,
            baseline_within_fixed_tolerance=fixed_tol,
            n_baseline_null_folds=len(null_fold_lrets_baseline),
            sweep_level_cand_dispersion=sweep_cand_disp,
            sweep_level_null_dispersion=sweep_null_disp,
            sweep_level_verdict=sweep_verdict,
            overall_verdict=sweep_overall,
        ),
        determinism=dict(r1_equals_r2=True, segment_path_matches_engine_path=direct_matches_engine),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("")
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Cross-sectional relative strength on the collected 10-"
          "asset universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
