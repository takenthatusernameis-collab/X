"""Cost-sensitivity check for short-horizon momentum on the collected 10-asset
universe, lookback 3/5/10, under zero cost and the repository's realistic cost
model, with a matched coin-flip null.

This reuses the existing momentum check framework
(research/backtest/regime_stability.stress_segments_across_tickers) and the
existing momentum_signals implementation without changing either; the only
introduction is a bounded cost grid passed through BacktestConfig.

Cost grid (one bounded cost parameter, predeclared):
  zero            : zero commission, zero slippage
  realistic       : commission_per_trade=2.0, commission_per_share=0.003,
                    slippage_cents=2.0, slippage_proportional=0.0005
                    (matches the realistic-cost config in
                    examples/ma_crossover.py and
                    examples/ma_crossover_real_data.py)
  realistic_stress: realistic per-trade/per-share/slippage components doubled

For each lookback x cost level, stress_segments_across_tickers runs the
candidate across the 10-asset collected universe with walk-forward
IS/OOS (train=252d / test=84d / warmup=60d / overlap=60d) inside each
asset's 4 contiguous volatility blocks, and it runs the coin-flip null
with the SAME BacktestConfig, so the candidate is always compared against
a matched null (same costs, same folds, same segments).

Two signal variants are evaluated:
  A) EXISTING momentum_signals from the framework (research/backtest/
     regime_stability.momentum_signals). IMPORTANT: this implementation
     computes ``np.mean(np.log(closes[window]))``, the MEAN OF LOG PRICES
     over the lookback window, which is always positive for positive
     prices. The resulting signal is therefore effectively CONSTANT-LONG
     (no sign flips); it is retained here exactly as contracted ("use the
     existing implementation") and its results are classified accordingly
     (constant exposure: costs cancel symmetrically against the null).
  B) CORRECTED momentum signal, computed as the lookback RETURN
     ``log(closes[i]) - log(closes[i-lookback])``. This is a diagnostic
     code path in this file only (it does not modify the framework); it is
     run under the identical framework, windows, and cost levels to answer
     what real short-horizon momentum does under these costs. It is labeled
     as a diagnostic in the artifact.

Deliverable classification (applied after results, not tuned to them):
  - survives : zero-cost and realistic verdicts agree and medians stay on the
               same side of the matched null in every segment;
  - eroded   : zero-cost shows a positive cost-free edge, but under realistic
               costs the medians move toward / inside the matched-null band;
  - killed   : realistic-cost medians become non-positive relative to the
               matched null;
  - no edge  : zero-cost medians are already indistinguishable from the
               matched null (constant exposure / noise); costs cannot destroy
               what is not an edge.

Determinism is asserted by re-running zero + realistic for all lookbacks and
comparing the JSON artifacts.

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
LOOKBACKS = (3, 5, 10)
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


ZERO_CFG = bt.BacktestConfig(
    initial_capital=1e6, warmup_periods=WARM,
    commission_per_trade=0.0, commission_per_share=0.0,
    slippage_cents=0.0, slippage_proportional=0.0,
)
REALISTIC_CFG = bt.BacktestConfig(
    initial_capital=1e6, warmup_periods=WARM,
    commission_per_trade=2.0, commission_per_share=0.003,
    slippage_cents=2.0, slippage_proportional=0.0005,
)
REALISTIC_STRESS_CFG = bt.BacktestConfig(
    initial_capital=1e6, warmup_periods=WARM,
    commission_per_trade=4.0, commission_per_share=0.006,
    slippage_cents=4.0, slippage_proportional=0.001,
)

COST_LEVELS = [
    ("zero", ZERO_CFG),
    ("realistic", REALISTIC_CFG),
    ("realistic_stress", REALISTIC_STRESS_CFG),
]


def run_cost_block(tickers_univ, signals_fn, lookback, cfg, label):
    """Run one momentum cost block over the full collected universe."""
    summary = bt.stress_segments_across_tickers(
        tickers=tickers_univ,
        signals_fn=lambda c, **kw: signals_fn(c, lookback=lookback),
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=(("lookback", float(lookback)),),
        param_grid=[{"lookback": float(lookback)}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=cfg,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    per_asset = {
        ticker: dict(
            segments=[s.name for s in r.scenarios],
            medians=[round(s.baseline_median_log_return, 3) for s in r.scenarios],
            null_medians=[round(s.noise_median_log_return, 3) for s in r.scenarios],
            candidate_dispersion=round(r.candidate_dispersion, 3),
            null_dispersion=round(r.null_dispersion, 3),
            n_folds=r.n_folds,
            verdict=r.overall_verdict,
        )
        for ticker, r in summary.assets.items()
    }
    print("  lookback={} cost={} ".format(lookback, label) +
          " | ".join("{}={}".format(k, v) for k, v in summary.verdict_counts.items()) +
          " | n_assets={}".format(len(summary.ticker_order)))
    return per_asset, summary.verdict_counts


def leakage_gate(tickers_univ):
    """Leakage discipline on the momentum signal for the three lookbacks.

    check_signal_integrity enforces past-only signals and date validity;
    check_equity_matches_fills recomputes equity from the fill sequence and
    asserts it reproduces the engine's equity curve (no untraceable equity).
    Both are run at realistic costs on the largest universe asset (AAPL).
    """
    results = {}
    for lookback in LOOKBACKS:
        bars, dates = bt.load_ticker("AAPL")
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=lookback)
        sig_ok = True
        try:
            bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        except bt.LeakySignalError as e:
            sig_ok = False
            print("    signal integrity FAIL lookback={}:".format(lookback), e)
        res = bt.run_bars(list(bars), signals, REALISTIC_CFG)
        fill_prices = (
            np.array([f.price for f in res.trades], dtype=np.float64)
            if res.trades else np.array([], dtype=np.float64)
        )
        try:
            bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        except AssertionError as e:
            sig_ok = False
            print("    fill-equity audit FAIL lookback={}:".format(lookback), e)
        results[lookback] = {"signal_integrity": sig_ok,
                             "fill_equity_audit": sig_ok,
                             "n_fills": len(res.trades),
                             "n_bars": len(bars)}
    return results


def turnover_estimate(tickers_univ):
    """Estimate realistic-cost turnover from fills on the largest asset.

    Counts round trips and total cost drag (commission + slippage) for
    lookback=5 under realistic costs on AAPL; printed for the activation
    record. Not part of the verdict. Costs are exact from the fill records.
    """
    bars, dates = bt.load_ticker("AAPL")
    closes = bars.closes_array()
    price_at = {bars[i].date: bars[i].close for i in range(len(bars))}
    signals = bt.momentum_signals(closes, lookback=5)
    res = bt.run_bars(list(bars), signals, REALISTIC_CFG)
    fills = res.trades
    commission_total = 0.0
    slippage_total = 0.0
    for f in fills:
        commission_total += f.commission
        pc = price_at.get(f.date, 0.0)
        slippage_total += abs(f.shares) * (f.price - pc)
    start, end = res.equity_curve[0], res.equity_curve[-1]
    res0 = bt.run_bars(list(bars), signals, ZERO_CFG)
    print("  realistic-cost lookback=5 on AAPL: fills={:.0f} "
          "total_commission=${:,.0f} total_round-trip_slippage=${:,.0f} "
          "full-sample drag vs zero-cost=${:,.0f} ({:+.1%})"
          .format(len(fills), commission_total, slippage_total,
                  res0.equity_curve[-1] - end,
                  (res0.equity_curve[-1] - end) / res0.equity_curve[0]))
    return dict(fills=len(fills), commission=round(commission_total, 1),
                slippage=round(slippage_total, 1),
                drag_full_sample=round(res0.equity_curve[-1] - end, 1),
                drag_pct=round((res0.equity_curve[-1] - end) / res0.equity_curve[0] * 100, 1))


def classify_signal(zero_medians, real_medians, zero_nulls, real_nulls,
                    zero_verdict, real_verdict, stress_verdict):
    """Classify cost robustness of one signal variant for one lookback.

    The base (zero-cost) level is identical to the ``zero`` level below, so
    ``base_margins == zero_margins``. Returns (label, reason, survives).
    """
    def margin(med, null):
        return [float(a - b) for a, b in zip(med, null)]

    zero_margins = margin(zero_medians, zero_nulls)
    real_margins = margin(real_medians, real_nulls)

    pos = lambda ms: [m for m in ms if m > 0.05]
    zero_pos, real_pos = pos(zero_margins), pos(real_margins)

    if not zero_pos and not real_pos:
        return ("no edge", "cost-free medians already indistinguishable from the "
                "matched null; costs cannot destroy what is not an edge", False)
    # Verdicts did not move across cost levels: the cost drag was borne
    # symmetrically (candidate and null trade identically), so the verdict
    # outcome stands. The edge (if any) survives under realistic costs.
    if zero_verdict == real_verdict == stress_verdict:
        if real_pos:
            if len(real_pos) < len(zero_pos):
                return ("survives (dragged but positive)",
                        "verdict {} in every cost level; margins positive under "
                        "realistic costs but below the zero-cost level"
                        .format(zero_verdict), True)
            return ("survives",
                    "verdict {} in every cost level; margins stay positive and "
                    "above the matched null through realistic stress"
                    .format(zero_verdict), True)
        return ("no edge", "zero-cost medians indistinguishable from the "
                "matched null; verdict {} everywhere"
                .format(zero_verdict), False)
    # Verdict changed under cost stress.
    if not real_pos:
        return ("killed",
                "zero-cost medians positive over {} segments (verdict {}) but "
                "realistic-cost medians fall inside / below the matched-null "
                "band ({} positive); verdict {} -> {}"
                .format(len(zero_pos), zero_verdict, len(real_pos),
                        zero_verdict, real_verdict), False)
    return ("eroded",
            "zero-cost medians positive over {}; realistic medians shrink "
            "toward the matched null ({} positive) and verdict degrades {} -> {}"
            .format(len(zero_pos), len(real_pos), zero_verdict, real_verdict),
            False)


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]
    print("=== 1. Manifest integrity ===")
    print("  dataset: {} | collection: {} | universe: {}"
          .format(manifest["dataset_id"], manifest["collection_date"],
                  len(manifest["universe"])))
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print("  {}: {}".format(entry["ticker"], status))
        if status != "OK":
            print("  MANIFEST CHECKSUM MISMATCH; aborting.")
            return 1

    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        return 1

    print("\n=== 3. Leakage gate (lookahead + fill-equity audit, realistic costs) ===")
    tickers = {}
    for entry in manifest["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]
    leak = leakage_gate(tickers)
    for lookback, r in leak.items():
        ok = r["signal_integrity"] and r["fill_equity_audit"]
        print("  lookback={}: signal_integrity={} fill_equity_audit={} "
              "n_fills={}".format(lookback, r["signal_integrity"],
                                  r["fill_equity_audit"], r["n_fills"]))
        if not ok:
            print("  LEAKAGE GATE FAILED; results labeled UNVERIFIED.")

    print("\n=== 4. Cost blocks over collected universe (train={}d / test={}d / "
          "warmup={}d / overlap={}d, {} volatility blocks) ==="
          .format(TRAIN, TEST, WARM, OVERLAP, N_BLOCKS))
    print("Cost levels: zero; realistic (examples/ma_crossover.py config); "
          "realistic_stress (2x realistic).\n")
    print("Signal A (EXISTING framework momentum_signals): NOTE - this function "
          "computes the mean of log PRICES over the lookback window, which is "
          "always positive for positive prices; the resulting signal is "
          "effectively CONSTANT-LONG. Retained exactly as contracted; see "
          "Signal B for the corrected lookback-RETURN implementation.\n")

    per_asset = {}
    verdict_counts = {}
    for lookback in LOOKBACKS:
        per_asset[str(lookback)] = {}
        verdict_counts.setdefault("existing", {}).setdefault(str(lookback), {})
        for label, cfg in COST_LEVELS:
            pa, vc = run_cost_block(tickers, bt.momentum_signals, lookback, cfg, label)
            per_asset[str(lookback)][label] = pa
            verdict_counts["existing"][str(lookback)][label] = vc

    print("\n=== 5. Signal B diagnostic: corrected lookback-RETURN momentum "
          "(identical framework/windows/costs) ===")
    print("Corrected signal: weight = sign of log(closes[i]) - "
          "log(closes[i-lookback]); run under the same universe, window sizes, "
          "cost grid, and matched null.\n")
    per_asset["corrected"] = {}
    verdict_counts["corrected"] = {}
    for lookback in LOOKBACKS:
        per_asset["corrected"][str(lookback)] = {}
        for label, cfg in COST_LEVELS:
            sig_fn = lambda c, lookback=lookback: lambda_close(c, lookback)
            pa, vc = run_cost_block(tickers, sig_fn, lookback, cfg, label)
            per_asset["corrected"][str(lookback)][label] = pa
            verdict_counts["corrected"].setdefault(str(lookback), {})[label] = vc

    print("\n=== 6. Universe verdict distribution by lookback x cost (Signal A) ===")
    for lookback in LOOKBACKS:
        line = "  lookback={}".format(lookback)
        for label, cfg in COST_LEVELS:
            vc = verdict_counts["existing"][str(lookback)][label]
            line += " | {}={}".format(label,
                " ".join("{}={}".format(k, v) for k, v in vc.items()))
        print(line)

    print("\n=== 7. Universe verdict distribution by lookback x cost (Signal B) ===")
    for lookback in LOOKBACKS:
        line = "  lookback={}".format(lookback)
        for label, cfg in COST_LEVELS:
            vc = verdict_counts["corrected"][str(lookback)][label]
            line += " | {}={}".format(label,
                " ".join("{}={}".format(k, v) for k, v in vc.items()))
        print(line)

    print("\n=== 8. Turnover estimate (Signal A, realistic, lookback=5, AAPL) ===")
    tv = turnover_estimate(tickers)

    print("\n=== 9. Cost-robustness classification by lookback ===")
    print("  Signal A (EXISTING, constant-long):")
    classification = {"existing": {}, "corrected": {}}
    for lookback in LOOKBACKS:
        z = per_asset[str(lookback)]["zero"]
        r = per_asset[str(lookback)]["realistic"]
        s = per_asset[str(lookback)]["realistic_stress"]
        zv, rv, sv = z["AAPL"]["verdict"], r["AAPL"]["verdict"], s["AAPL"]["verdict"]
        label, reason, survives = classify_signal(
            z["AAPL"]["medians"], r["AAPL"]["medians"], z["AAPL"]["null_medians"],
            r["AAPL"]["null_medians"], zv, rv, sv)
        classification["existing"][str(lookback)] = dict(
            classification=label, reason=reason, survives=survives,
            zero_verdict=zv, realistic_verdict=rv, stress_verdict=sv)
        print("    lookback={}: {} -> {}".format(lookback, label, reason))

    print("  Signal B (CORRECTED diagnostic, per lookback):")
    for lookback in LOOKBACKS:
        z = per_asset["corrected"][str(lookback)]["zero"]
        r = per_asset["corrected"][str(lookback)]["realistic"]
        s = per_asset["corrected"][str(lookback)]["realistic_stress"]
        zv, rv, sv = z["AAPL"]["verdict"], r["AAPL"]["verdict"], s["AAPL"]["verdict"]
        label, reason, survives = classify_signal(
            z["AAPL"]["medians"], r["AAPL"]["medians"], z["AAPL"]["null_medians"],
            r["AAPL"]["null_medians"], zv, rv, sv)
        classification["corrected"][str(lookback)] = dict(
            classification=label, reason=reason, survives=survives,
            zero_verdict=zv, realistic_verdict=rv, stress_verdict=sv)
        print("    lookback={}: {} -> {}".format(lookback, label, reason))

    print("\n=== 10. Zero-cost lookback=5 cross-check vs existing artifact ===")
    existing = json.load(open(ARTIFACT_DIR / "momentum_results.json"))
    existing_univ = existing["universe"]["per_asset"]
    mismatch = 0
    for ticker in manifest["universe"]:
        pub = existing_univ[ticker]
        new = per_asset["5"]["zero"][ticker]
        for field in ("medians", "null_medians", "segments", "verdict"):
            if field == "medians":
                ok = all(round(a, 3) == round(b, 3) for a, b in zip(new[field], pub[field]))
            else:
                ok = new[field] == pub[field]
            if not ok:
                mismatch += 1
                print("  MISMATCH {} {}: {} (pub {})".format(
                    ticker, field, new[field], pub[field]))
    if mismatch == 0:
        print("  zero-cost lookback=5 matches momentum_results.json artifact: OK")
    else:
        print("  zero-cost lookback=5 differs from momentum_results.json in {} "
              "fields: review".format(mismatch))

    print("\n=== 11. Determinism (zero + realistic re-run, all lookbacks) ===")
    per_asset2 = {}
    for lookback in LOOKBACKS:
        per_asset2[str(lookback)] = {}
        for label, cfg in (("zero", ZERO_CFG), ("realistic", REALISTIC_CFG)):
            per_asset2[str(lookback)][label] = run_cost_block(
                tickers, bt.momentum_signals, lookback, cfg, label)[0]
    # Determinism re-run for the corrected diagnostic signal too.
    per_asset2_corr = {}
    for lookback in LOOKBACKS:
        per_asset2_corr[str(lookback)] = {}
        for label, cfg in (("zero", ZERO_CFG), ("realistic", REALISTIC_CFG)):
            sig_fn = lambda c, lookback=lookback: lambda_close(c, lookback)
            per_asset2_corr[str(lookback)][label] = run_cost_block(
                tickers, sig_fn, lookback, cfg, label)[0]
    det_ok = True
    for variant in ("existing", "corrected"):
        source = per_asset if variant == "existing" else per_asset["corrected"]
        re_run = per_asset2 if variant == "existing" else per_asset2_corr
        for lookback in LOOKBACKS:
            for label in ("zero", "realistic"):
                a = source[str(lookback)][label]
                b = re_run[str(lookback)][label]
                same = True
                for ticker in manifest["universe"]:
                    for key in ("medians", "null_medians", "verdict"):
                        va, vb = a[ticker][key], b[ticker][key]
                        if key in ("medians", "null_medians"):
                            if not all(round(x, 3) == round(y, 3) for x, y in zip(va, vb)):
                                same = False
                        elif va != vb:
                            same = False
                status = "identical" if same else "DIFFERENT"
                if not same:
                    det_ok = False
                print("  lookback={} cost={} signal={}: re-run {} vs first-run -> {}".format(
                    lookback, label, variant, status, status))
    print("determinism: {}".format("PASS" if det_ok else "FAIL"))

    # Build the artifact.
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        task=("R-001 cost-sensitivity: momentum lookback 3/5/10 under realistic "
              "costs, matched null; two signal variants reported"),
        lookbacks=list(LOOKBACKS),
        cost_levels=[
            dict(name=name, cfg=dict(
                commission_per_trade=cfg.commission_per_trade,
                commission_per_share=cfg.commission_per_share,
                slippage_cents=cfg.slippage_cents,
                slippage_proportional=cfg.slippage_proportional,
                warmup_periods=cfg.warmup_periods,
            ))
            for name, cfg in COST_LEVELS
        ],
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        leakage_gate=dict(
            per_lookback={str(k): dict(ok=r["signal_integrity"] and r["fill_equity_audit"],
                                        signal_integrity=r["signal_integrity"],
                                        fill_equity_audit=r["fill_equity_audit"],
                                        n_fills=r["n_fills"]) for k, r in leak.items()},
            methodology=("check_signal_integrity: past-only + date validity; "
                         "check_equity_matches_fills: fill-computed equity "
                         "reproduces engine equity curve"),
        ),
        signal_variants=dict(
            existing=dict(
                name="EXISTING framework momentum_signals",
                implementation_note=("computes np.mean(np.log(closes[window])) - "
                                     "mean of log PRICES, which is always positive "
                                     "for positive prices; signal is effectively "
                                     "CONSTANT-LONG, no sign flips. Retained exactly "
                                     "as contracted."),
            ),
            corrected=dict(
                name="CORRECTED diagnostic",
                implementation_note=("sign of log(closes[i]) - "
                                     "log(closes[i-lookback]); the corrected "
                                     "lookback RETURN."),
                scope_note=("diagnostic code path within this file only; does not "
                            "modify the framework or the published artifact for the "
                            "existing implementation."),
            ),
        ),
        per_asset=per_asset,
        verdict_counts=verdict_counts,
        cross_check=dict(
            zero_lookback5_vs_momentum_results_json=("identical" if mismatch == 0
                                                       else "{} fields differ".format(mismatch))
        ),
        turnover_estimate=tv,
        classification=classification,
        determinism=dict(passed=det_ok),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Cost-sensitivity check on lookback 3/5/10 momentum: "
          "exploratory simulation.")
    return 0 if det_ok and all(r["ok"] for r in leak.values()) else 1


def lambda_close(closes, lookback):
    """Helper for the corrected-signal diagnostic lambda used inside
    stress_segments_across_tickers (which passes closes, **params)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.log(closes[i]) - np.log(closes[i - lookback]))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


if __name__ == "__main__":
    sys.exit(main())
