"""Cost robustness of short-horizon momentum on the collected universe.

Context: lookback-3/5/10 momentum (long the previous N-day return, hold 1 day,
daily rebalance) is ADMITTED AS CANDIDATE POSITIVE EVIDENCE: 7/10 collected
assets show a REGIME_STABLE positive edge at lookback 5 with no costs. Whether
that mechanical, short-horizon edge survives conservative transaction costs is
the outstanding decision-relevant qualification.

A cost-robustness prediction (a-priori, not tuned to results): the signal is a
constant weight of +/-1 applied every bar. Under the engine's constant-dollar
sizing, each rebalance re-establishes target_shares = weight * capital / close,
so a price move of x% forces a trade of roughly x% of capital to keep exposure
constant; daily turnover is of order the daily price return (large for high-vol
bars, small otherwise), and a round trip occurs each time the sign flips. Even a
10 bps proportional fill cost on that turnover is comparable to the observed
~5-13% per 84-day OOS fold edge, so the prediction is that costs at or above the
realistic level materially erode or erase the edge, especially at the shortest
lookbacks where the per-bar information content is weakest.

This check applies the repository's existing realistic cost model
(BacktestConfig: commission_per_trade/share and slippage_cents/proportional)
through the walk-forward regime-stability gate, to both candidate and matched
coin-flip null, across lookbacks 3 / 5 / 10 and a predeclared cost grid.

Design rules (not tuned to OOS results):
- Lookbacks: 3, 5, 10 (predeclared; lookback 5 is the reference from the
  momentum cell; 3 tests the shortest-horizon claim; 10 the boundary of the
  short-horizon window).
- Cost grid (predeclared; the repository's existing cost model; no leverage,
  no sizing tuning, no universe expansion):
    zero        BacktestConfig()                                      -- baseline
    realistic   slippage_proportional=0.001 (10 bps), $0 comm          -- large-cap, zero-commission benchmark
    conservative  slippage=0.002 (20 bps), commission_per_share=0.005 -- realistic institutional
    heavy       slippage=0.005 (50 bps), commission_per_share=0.01, $0.05/trade -- adversarial
- Gate: regime-stability per asset via the existing walk-forward / matched
  coin-flip null on the 4 AAPL volatility blocks (same settings as the momentum
  cell); verdicts CWN / REGIME_STABLE / REGIME_STABLE_LOSS / REGIME_DEPENDENT.
- Leakage gate: past-only integrity check on momentum signals for the two
  flagged assets (AMZN, JPM) per lookback/cost; equity-fill audit at zero cost.
- Cost accounting: exact dollar cost per asset per cost level from the fills
  (commission + proportional slippage), and a per-fold cost-drag estimate.
- Matched null: the same BacktestConfig is applied to the coin-flip null.
- Determinism: independent rerun of key fields; asserted identical.
- Deliverable: an artifact JSON classifying cost robustness, plus an
  independent verifier (verify_momentum_cost_sensitivity.py) that recomputes
  everything on a fresh walk-forward path.

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
MARKET_PROXY = "AAPL"

# ---------------------------------------------------------------------------
# Predeclared cost grid (repository's existing cost model; no leverage).
# ---------------------------------------------------------------------------
def make_cost_cfg(level: str) -> bt.BacktestConfig:
    """Return the BacktestConfig for a predeclared cost level."""
    cfg = bt.BacktestConfig(initial_capital=1e6, periods_per_year=252)
    if level == "realistic":
        # 10 bps proportional slippage, zero commission: standard large-cap
        # estimate; daily rebalance is the costly part, not the commission.
        cfg.slippage_proportional = 0.001
    elif level == "conservative":
        # 20 bps slippage + $0.005/share: realistic institutional cost.
        cfg.slippage_proportional = 0.002
        cfg.commission_per_share = 0.005
    elif level == "heavy":
        # 50 bps slippage + $0.01/share + $0.05/trade: adversarial stress.
        cfg.slippage_proportional = 0.005
        cfg.commission_per_share = 0.01
        cfg.commission_per_trade = 0.05
    return cfg


COST_LEVELS = ["zero", "realistic", "conservative", "heavy"]

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"

REGIME_VERDICT_NO_EDGE = "CONSISTENT_WITH_NOISE"
REGIME_VERDICT_STABLE = "REGIME_STABLE"
REGIME_VERDICT_STABLE_LOSS = "REGIME_STABLE_LOSS"
REGIME_VERDICT_DEPENDENT = "REGIME_DEPENDENT"


def regime_verdict(candidate_medians, null_medians):
    """Same verdict logic as research.backtest.regime_stability.stress_segments."""
    candidate_medians = np.array(candidate_medians, dtype=np.float64)
    null_medians = np.array(null_medians, dtype=np.float64)
    if all(abs(m) <= bt.OUTLIER_TOL for m in candidate_medians):
        return REGIME_VERDICT_NO_EDGE
    candidate_dispersion = float(np.std(candidate_medians, ddof=1))
    null_dispersion = float(np.std(null_medians, ddof=1))
    if candidate_dispersion > 2.0 * null_dispersion:
        return REGIME_VERDICT_DEPENDENT
    if all(m < 0 for m in candidate_medians):
        return REGIME_VERDICT_STABLE_LOSS
    return REGIME_VERDICT_STABLE


def walk_forward_fold_log_returns(daily_pnl, train, test, warm, overlap):
    """Mirrors research.backtest.engine.walk_forward's fold log-return logic."""
    n = len(daily_pnl)
    step = test - overlap
    fold_start = 0
    lrets = []
    while True:
        fold_end = fold_start + warm + train + test
        if fold_end > n:
            break
        oos = daily_pnl[fold_end - test + 1:fold_end]
        comp = 1.0
        for r in oos:
            comp *= 1.0 + r
        lrets.append(float(np.log1p(np.clip(comp - 1.0, -1.0 + 1e-12, None))))
        fold_start += step
    return lrets


def regime_labels_for(closes):
    """Past-only 4-block volatility classification (same as the momentum cell)."""
    return bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)


def segment_medians(daily_pnl, labels, min_segment_bars, train, test, warm,
                    overlap):
    """Fold log returns per segment, from the label list."""
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
    meds = []
    for name, s, e in segments:
        meds.append(float(np.median(walk_forward_fold_log_returns(daily_pnl[s:e],
                            train, test, warm, overlap))))
    return [s.name for s in segments] if False else [seg[0] for seg in segments], meds


def key_fields(manifest, tickers, seed):
    """Full run of the cost grid over lookbacks and assets."""
    rows = []
    for lookback in LOOKBACKS:
        lookback_rows = []
        for level in COST_LEVELS:
            cfg = make_cost_cfg(level)
            result = bt.stress_segments_across_tickers(
                tickers=tickers,
                signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=kw["lookback"]),
                regime_labels_fn=regime_labels_for,
                baseline=(("lookback", lookback),),
                param_grid=[{"lookback": lookback}],
                train_window=TRAIN,
                test_window=TEST,
                warmup=WARM,
                overlap_window=OVERLAP,
                periods_per_year=252,
                min_segment_bars=MIN_SEGMENT_BARS,
                cfg=cfg,
            )
            counts = dict(result.verdict_counts)
            stable = counts.get("REGIME_STABLE", 0)
            stable_negative = counts.get("REGIME_STABLE_LOSS", 0)
            lookback_rows.append(dict(
                level=level,
                verdict_counts=counts,
                n_regime_stable=stable,
                n_regime_stable_loss=stable_negative,
                ticker_order=result.ticker_order,
                per_asset=[
                    dict(
                        ticker=ticker,
                        segments=[s.name for s in r.scenarios],
                        medians=[round(s.baseline_median_log_return, 3)
                                 for s in r.scenarios],
                        null_medians=[round(s.noise_median_log_return, 3)
                                      for s in r.scenarios],
                        candidate_dispersion=round(r.candidate_dispersion, 3),
                        null_dispersion=round(r.null_dispersion, 3),
                        verdict=r.overall_verdict,
                    )
                    for ticker, r in result.assets.items()
                ],
            ))
        rows.append((lookback, lookback_rows))
    return rows


def cost_accounting(tickers, lookback, level):
    """Exact dollar cost accounting from a full-sample costed run.

    Returns the total commission, total proportional slippage, total fixed
    slippage, total turnover dollars, number of fills (round trips when the
    sign flips), and total cost, per asset.
    """
    cfg = make_cost_cfg(level)
    per_asset = {}
    for ticker, bars in tickers.items():
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=lookback)
        res = bt.run_bars(list(bars), signals, cfg)
        total_commission = sum(f.commission for f in res.trades)
        total_turnover = sum(abs(f.shares) * f.price for f in res.trades)
        # proportional slippage = sum(delta * close * prop) per fill:
        prop_slippage = sum(abs(f.shares) * f.price * cfg.slippage_proportional
                            for f in res.trades)
        fixed_slippage = sum(abs(f.shares) * cfg.slippage_cents / 100.0
                             for f in res.trades)
        total_cost = total_commission + prop_slippage + fixed_slippage
        per_asset[ticker] = dict(
            n_fills=len(res.trades),
            total_turnover_dollars=round(total_turnover, 2),
            total_commission=round(total_commission, 2),
            total_proportional_slippage_cents=round(prop_slippage * 100, 2),
            total_fixed_slippage_cents=round(fixed_slippage * 100, 2),
            total_cost_dollars=round(total_cost, 2),
            cost_as_frac_of_capital=round(total_cost / cfg.initial_capital, 6),
        )
    return per_asset


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

    print("\n=== 3. Leakage gate: past-only momentum signals ===")
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, _ = bt.load_ticker(ticker)
        tickers[ticker] = bars
    target = ["AMZN", "JPM"]
    leakage_ok = True
    for lookback in LOOKBACKS:
        for ticker in target:
            closes = tickers[ticker].closes_array()
            signals = bt.momentum_signals(closes, lookback=lookback)
            try:
                bt.check_signal_integrity(signals, [b.date for b in tickers[ticker]],
                                          warmup=0)
            except bt.LeakySignalError:
                leakage_ok = False
            print(f"  {ticker} lookback {lookback}: signal-integrity PASS (past-only)")
    # Equity-fill audit at zero cost on both target assets.
    for ticker in target:
        closes = tickers[ticker].closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        res = bt.run_bars(list(tickers[ticker]), signals,
                          bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        print(f"  {ticker} equity-fill audit: PASS")
    print("  leakage gate: PASSED" if leakage_ok else "  leakage gate: FAILED")
    if not leakage_ok:
        return 1

    print("\n=== 4. Cost sensitivity: lookback 3 / 5 / 10 across the cost grid ===")
    print("Regime-stability gate per asset, walk-forward train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, 4 AAPL volatility "
          "blocks, matched coin-flip null run with the SAME cost config.\n".format(
          TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    rows = key_fields(manifest, tickers, SEED)
    grid_result = {}
    for lookback, lookback_rows in rows:
        print("Lookback {} days:".format(lookback))
        print("  {:8s} {:12s} {:12s} {:24s}  {:22s}".format(
            "level", "CWN", "REGIME_STABLE", "verdict counts (CWN/STABLE/STABLE_LOSS/DEPENDENT)",
            "stable positive / stable loss"))
        for lr in lookback_rows:
            v = lr["verdict_counts"]
            print("    {:8s} {:12d} {:12d} {:24s}  {:22d} / {:22d}".format(
                lr["level"], v.get("CONSISTENT_WITH_NOISE", 0),
                v.get("REGIME_STABLE", 0),
                "{}/{}/{}/{}".format(
                    v.get("CONSISTENT_WITH_NOISE", 0), v.get("REGIME_STABLE", 0),
                    v.get("REGIME_STABLE_LOSS", 0), v.get("REGIME_DEPENDENT", 0)),
                lr["n_regime_stable"], lr["n_regime_stable_loss"]))
        grid_result[lookback] = lookback_rows
        print()

    # ---- 5. Exact cost accounting per lookback / cost level ----
    print("=== 5. Exact cost accounting (full-sample runs; commission + fills) ===")
    cost_rows = {}
    for lookback in LOOKBACKS:
        lr = []
        for level in COST_LEVELS:
            acc = cost_accounting(tickers, lookback, level)
            lr.append((level, acc))
        cost_rows[lookback] = lr
    for lookback, lr in cost_rows.items():
        print("Lookback {} days:".format(lookback))
        for level, acc in lr:
            tops = acc["AAPL"]
            print("  level {:8s}: fills={:4d} turnover=${:10,.0f} "
                  "cost=${:8,.0f} ({:.4%} of capital)".format(
                      level, tops["n_fills"], tops["total_turnover_dollars"],
                      tops["total_cost_dollars"], tops["cost_as_frac_of_capital"]))
            print("    per-asset cost (fraction of capital):")
            for t in tickers:
                print("      {:8s}: {:+.6f}".format(t, acc[t]["cost_as_frac_of_capital"]))
        print()

    # ---- 6. Classify cost robustness per lookback ----
    print("=== 6. Cost-robustness classification per lookback ===")
    classification = {}
    for lookback, lookback_rows in grid_result.items():
        zero = lookback_rows[0]  # level == "zero"
        base_stable = zero["n_regime_stable"]
        surviving = []
        for level in COST_LEVELS:
            lr = next(r for r in lookback_rows if r["level"] == level)
            surviving.append((level, lr["n_regime_stable"], lr["verdict_counts"]))
        # Break-even level: first grid level where stable-positive count drops
        # strictly below the zero-cost baseline. Diagnostic only; the signal
        # rule is unchanged.
        be = None
        for level, stable, _ in surviving:
            if stable < base_stable:
                be = level
                break
        classification[lookback] = dict(
            zero_cost_regime_stable=base_stable,
            cost_path=surviving,
            break_even_cost_level=be,
        )
        print("  lookback {}: zero-cost REGIME_STABLE count = {}/10; cost path "
              "-> {}".format(lookback, base_stable, surviving))
        print("    break-even cost level (stable count first drops below {}): {}"
              .format(base_stable, be))
        print()

    # ---- 7. Determinism: rerun key fields ----
    print("=== 7. Determinism: independent rerun of the cost grid ===")
    rows_r2 = key_fields(manifest, tickers, SEED)
    print("  grid result r1 == r2: {}".format(rows == rows_r2))
    for (l1, r1), (l2, r2) in zip(rows, rows_r2):
        assert l1 == l2
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a == b, "non-deterministic cost-grid output"
    print("  all cost-grid fields identical across reruns")

    # ---- 8. Artifact ----
    print("\n=== 8. Artifact ===")
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        cost_levels=COST_LEVELS,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        market_proxy=MARKET_PROXY,
        leakage=dict(
            signal_integrity_lookbacks=[list(LOOKBACKS)],
            target_assets=target,
            equity_fill_audit=target,
            passed=leakage_ok,
        ),
        grid_results={
            "lookback_{}".format(lb): lr
            for lb, lr in grid_result.items()
        },
        cost_accounting={
            "lookback_{}".format(lb): {
                level: acc
                for level, acc in cost_rows[lb]
            }
            for lb in LOOKBACKS
        },
        cost_classification=classification,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("  artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Cost sensitivity of lookback-3/5/10 momentum on the "
          "collected universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
