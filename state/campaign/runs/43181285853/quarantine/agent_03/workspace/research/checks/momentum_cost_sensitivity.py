"""Real-data transaction-cost sensitivity for short-horizon momentum.

Objective: decide whether the short-horizon momentum edge (lookback 3 / 5 / 10)
admitted as candidate positive evidence in `momentum.py` /
`momentum_lookback_sweep.py` survives conservative execution-cost stress on the
collected 10-asset adjusted-close universe, judged against the repository's
existing realistic cost model and the framework's matched coin-flip null.

Cost model (predeclared, not tuned to results): the repository's existing
realistic cost configuration, as used in
`examples/ma_crossover_real_data.py` and the other real-data examples:
commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0,
slippage_proportional=0.0005 (5 bps per side). The cost grid is:

  zero         : no cost (reference; also cross-checks the existing
                 `momentum_lookback_sweep_results.json` column).
  realistic    : the existing realistic cost model above.
  conservative : exactly 2x the realistic variable costs
                 (commission_per_trade=4.0, commission_per_share=0.006,
                 slippage_cents=4.0, slippage_proportional=0.0010), a bounded
                 a-priori stress level, not a tuned value.

Method (fixed a-priori, not tuned to OOS): for each cost level, lookback, and
ticker, run the existing `stress_segments` walk-forward regime gate over the 4
contiguous volatility blocks (train=252d / test=84d / warmup=60d / overlap=60d,
min 400 bars/segment) with a matched coin-flip null run on the same segments
under the same cost model. Also record a full-sample reference total return per
cost level, which is an unambiguous (no fold-base) measure of the realized
cost drag, because the per-fold log-return metric divides by an equity base that
already contains prior-segment cost.

Classification (predeclared before execution):
  An asset SURVIVES a cost level iff every segment's net-of-cost candidate
  median log return is strictly positive.
  Universe verdict at a cost level: COST_ROBUST if >= 6/10 assets survive
  (the same majority threshold used to admit the lookback-robust momentum
  evidence), COST_FRAGILE if 0/10 survive, PARTIAL otherwise.
  The matched-null "beats" count (candidate median > null median in every
  segment) is reported as a secondary diagnostic. The framework null is a
  price-independent coin-flip signal and is NOT turnover-matched to the
  candidate, so its net-of-cost medians are not a clean cost benchmark; the
  absolute net-of-cost candidate medians are the primary evidence.

A-priori prediction (recorded before the run): because the constant-dollar
sizing rebalances only the change in notional (turnover is far below one full
round trip per bar), realized costs are expected to be small relative to the
per-fold signal, so the momentum edge is expected to survive the realistic and
conservative cost levels. A COST_FRAGILE outcome would instead show that the
admitted momentum evidence is an execution-cost artifact.

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
SURVIVE_THRESHOLD = 6  # of 10 assets, matching the documented admission threshold

# Predeclared cost grid. Each entry is a factory so no mutable config is shared.
COST_MODELS = {
    "zero": {},
    "realistic": {
        "commission_per_trade": 2.0,
        "commission_per_share": 0.003,
        "slippage_cents": 2.0,
        "slippage_proportional": 0.0005,
    },
    "conservative": {
        "commission_per_trade": 4.0,
        "commission_per_share": 0.006,
        "slippage_cents": 4.0,
        "slippage_proportional": 0.0010,
    },
}

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
SWEEP_ARTIFACT_PATH = ARTIFACT_DIR / "momentum_lookback_sweep_results.json"


def make_cfg(cost_name: str) -> bt.BacktestConfig:
    return bt.BacktestConfig(**COST_MODELS[cost_name])


def segment_result(bars, lookback: int, cfg: bt.BacktestConfig) -> dict:
    """Walk-forward regime-stability result for one ticker, lookback, cost cfg."""
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
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cfg,
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


def full_sample_result(bars, lookback: int, cfg: bt.BacktestConfig) -> dict:
    """Unambiguous full-sample total return and realized cost, no fold base."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    sig = bt.momentum_signals(closes, lookback=lookback)
    res = bt.run_bars(list(bars), sig, cfg)
    total_return = (res.equity_curve[-1] - cfg.initial_capital) / cfg.initial_capital
    commissions = float(sum(f.commission for f in res.trades))
    gross_traded = float(sum(abs(f.shares) * f.price for f in res.trades))
    return dict(
        total_return=round(float(total_return), 6),
        n_trades=len(res.trades),
        commissions=round(commissions, 4),
        gross_traded=round(gross_traded, 4),
    )


def survives_cost(medians) -> bool:
    """A-priori survival rule: every segment's net-of-cost median is positive."""
    return all(m > 0 for m in medians)


def beats_null(medians, null_medians) -> bool:
    return all(m > n for m, n in zip(medians, null_medians))


def classify(survival_count: int) -> str:
    if survival_count >= SURVIVE_THRESHOLD:
        return "COST_ROBUST"
    if survival_count == 0:
        return "COST_FRAGILE"
    return "PARTIAL"


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

    print("\n=== 3. Leakage review per asset (lookback=5, realistic cost) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        cfg = make_cfg("realistic")
        res = bt.run_bars(list(bars), signals, cfg)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, cfg.initial_capital)
        total_return = (res.equity_curve[-1] - cfg.initial_capital) / cfg.initial_capital
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only.")

    print("\n=== 4. Cost-sensitivity grid (cost x lookback x asset) ===")
    print("  cost levels: {} | lookbacks: {}".format(list(COST_MODELS), list(LOOKBACKS)))
    print("  walk-forward train={}d / test={}d / warmup={}d / overlap={}d, "
          "min {} bars/segment, 4 volatility blocks, matched coin-flip null\n".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    per_asset = {}
    universe = {}
    for cost_name in COST_MODELS:
        cfg = make_cfg(cost_name)
        per_asset[cost_name] = {}
        universe[cost_name] = {}
        for lb in LOOKBACKS:
            col = {}
            for ticker in ticker_order:
                sr = segment_result(all_tickers[ticker], lb, cfg)
                fs = full_sample_result(all_tickers[ticker], lb, cfg)
                sr["full_sample"] = fs
                sr["survives"] = survives_cost(sr["medians"])
                sr["beats_null"] = beats_null(sr["medians"], sr["null_medians"])
                col[ticker] = sr
            per_asset[cost_name][str(lb)] = col

            surv = [t for t in ticker_order if col[t]["survives"]]
            beat = [t for t in ticker_order if col[t]["beats_null"]]
            vcounts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                       "REGIME_STABLE_LOSS": 0, "REGIME_DEPENDENT": 0}
            for t in ticker_order:
                vcounts[col[t]["verdict"]] += 1
            universe[cost_name][str(lb)] = dict(
                survival_count=len(surv),
                survival_tickers=surv,
                beats_null_count=len(beat),
                beats_null_tickers=beat,
                verdict_counts=vcounts,
                classification=classify(len(surv)),
            )
            print("  cost={:12s} lookback={:2d} -> survive {}/10 {} | beats_null {}/10 | "
                  "verdicts {} -> {}".format(
                      cost_name, lb, len(surv), surv, len(beat), vcounts,
                      classify(len(surv))))

    # Retained-edge summary: net-of-cost mean segment median / zero-cost mean.
    print("\n=== 5. Retained edge (net-of-cost vs zero cost, mean segment median) ===")
    retained = {}
    for lb in LOOKBACKS:
        retained[str(lb)] = {}
        for ticker in ticker_order:
            base = per_asset["zero"][str(lb)][ticker]["medians"]
            base_mean = float(np.mean(base))
            row = {}
            for cost_name in COST_MODELS:
                m = float(np.mean(per_asset[cost_name][str(lb)][ticker]["medians"]))
                row[cost_name] = round(m / base_mean, 6) if base_mean > 0 else None
            retained[str(lb)][ticker] = row
        vals = [retained[str(lb)][t]["conservative"] for t in ticker_order
                if retained[str(lb)][t]["conservative"] is not None]
        print("  lookback {}: conservative retained fraction (mean over assets) = {}".format(
            lb, round(float(np.mean(vals)), 6) if vals else None))

    # Determinism: rerun the target subset and assert identical output.
    res2 = {c: {lb: {t: segment_result(tickers[t], lb, make_cfg(c)) for t in target}
                for lb in LOOKBACKS}
            for c in COST_MODELS}

    def core(sr):
        return {k: sr[k] for k in (
            "segments", "medians", "null_medians", "candidate_dispersion",
            "null_dispersion", "n_folds", "verdict")}

    out1 = json.dumps({c: {str(lb): {t: core(per_asset[c][str(lb)][t]) for t in target}
                           for lb in LOOKBACKS} for c in COST_MODELS}, sort_keys=True)
    out2 = json.dumps({c: {str(lb): {t: core(res2[c][lb][t]) for t in target}
                           for lb in LOOKBACKS} for c in COST_MODELS}, sort_keys=True)
    print("\ndeterminism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        cost_models={k: dict(v) for k, v in COST_MODELS.items()},
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        survive_threshold=SURVIVE_THRESHOLD,
        classification_rule=(
            "asset SURVIVES a cost level iff all segment net-of-cost candidate "
            "medians > 0; COST_ROBUST if >= 6/10 survive, COST_FRAGILE if 0/10, "
            "PARTIAL otherwise"
        ),
        a_priori_prediction=(
            "momentum expected to survive realistic and conservative cost because "
            "constant-dollar sizing rebalances only the change in notional"
        ),
        per_asset=per_asset,
        universe=universe,
        retained=retained,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity grid: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())