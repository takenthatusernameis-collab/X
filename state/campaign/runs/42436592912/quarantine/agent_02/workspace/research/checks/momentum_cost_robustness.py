"""Cost-sensitivity check for lookback 3/5/10 momentum.

Objective: Test whether the short-horizon momentum edge survives conservative
transaction-cost stress on the collected real-data universe. This is the
repository's existing realistic cost model matched against a coin-flip null.

Scope:
- Use existing 10-asset collected universe and existing momentum implementation/check
  framework.
- Test lookback values: 3, 5, 10 (matching the existing momentum_lookback_sweep.py)
- Realistic cost model (same as examples)
- Walk-forward validation (train=252d/test=84d/warmup=60d/overlap=60d)
- 4 contiguous volatility blocks per asset (same as momentum.py)
- Minimum 400 bars per segment
- Deterministic: rerun the AMZN/JPM subset and assert identical output

Method (fixed a-priori, not tuned to OOS):
1. Load the collected universe from the manifest
2. For each asset and each lookback, run stress_segments with realistic costs
3. Compute medians, dispersions, and verdicts vs. coin-flip null
4. Compare against a matched null benchmark (same costs, same parameters)
5. Produce artifact for independent verification

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
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_results.json"
REALISTIC_COSTS = dict(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

\ndef realistic_cfg(warmup: int = 0) -> bt.BacktestConfig:\n    \"\"\"Backtest config with realistic costs.\"\"\"\n    return bt.BacktestConfig(\n        initial_capital=1e6,\n        warmup_periods=warmup,\n        **REALISTIC_COSTS,\n    )
\n\ndef zero_cost_cfg(warmup: int = 0) -> bt.BacktestConfig:\n    \"\"\"Backtest config with zero costs for null comparison.\"\"\"\n    return bt.BacktestConfig(\n        initial_capital=1e6,\n        warmup_periods=warmup,\n    )
\n\ndef segment_result(ticker: str, bars: bt.BarSequence, lookback: int):\n    \"\"\"Walk-forward regime-stability result for one ticker, one lookback, with realistic costs.\"\"\"\n    closes = bars.closes_array()\n    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)\n    seg_fn = bt.segment_fn_from_labels(labels)\n    sig = bt.momentum_signals(closes, lookback=lookback)\n\n    if len(sig) != len(closes):\n        raise ValueError(f\"{ticker}: signals {len(sig)} != bars {len(closes)}\")\n\n    res = bt.stress_segments(\n        signals_fn=bt.momentum_signals,\n        bars=list(bars),\n        signals=sig,\n        param_grid=[{\"lookback\": lookback}],\n        baseline=((\"lookback\", lookback),),\n        segment_fn=seg_fn,\n        train_window=TRAIN,\n        test_window=TEST,\n        warmup=WARM,\n        overlap_window=OVERLAP,\n        periods_per_year=252,\n        min_segment_bars=MIN_SEGMENT_BARS,\n        cfg=realistic_cfg(warmup=0),\n    )\n    scen = res.scenarios[phobic\n    return dict(\n        segments=[s.name for s in res.scenarios],\n        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],\n        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],\n        candidate_dispersion=round(res.candidate_dispersion, 3),\n        null_dispersion=round(res.null_dispersion, 3),\n        n_folds=res.n_folds,\n        verdict=res.overall_verdict,\n    )
\n\ndef main() -> int:\n    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)\n\n    manifest = load_manifest()\n    assert manifest[\"dataset_id\"] == DATASET_ID, manifest[\"dataset_id\"]\n\n    print(\"=== 1. Manifest integrity ===\")\n    for entry in manifest[\"entries\"]:\n        actual = sha256_file(Path(entry[\"location\"]))\n        status = \"OK\" if actual == entry[\"checksum_sha256\"] else \"MISMATCH\"\n        print(f\"  {entry['ticker']}: {status}\")\n    print(f\"  dataset: {manifest['dataset_id']} | collection: \"\n          f\"{manifest['collection_date']} | universe: {len(manifest['universe'])}\")\n\n    print(\"\\n=== 2. Data preflight (research/data/preflight.py) ===\")\n    sys.path.insert(0, str(Path.cwd() / \"research\" / \"data\"))\n    from importlib import import_module\n    pf = import_module(\"preflight\")\n    rc = pf.main()\n    if rc != 0:\n        print(\"PREFLIGHT FAILED; aborting cost-robustness run.\")\n        sys.exit(1)\n\n    print(\"\\n=== 3. Leakage review per asset (lookback=5 reference) ===\")\n    target = [\"AMZN\", \"JPM\"]\n    tickers = {}\n    for ticker in target:\n        bars, dates = bt.load_ticker(ticker)\n        tickers[ticker] = bars\n        closes = bars.closes_array()\n        signals = bt.momentum_signals(closes, lookback=5)\n        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)\n        res = bt.run_bars(list(bars), signals, realistic_cfg(warmup=0))\n        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)\n        total_return = (res.equity_curve[-1] - 1e6) / 1e6\n        print(f\"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, \"\n              f\"leakage [PASS], total_return {total_return:+.3f}\")\n    print(\"  no-look-ahead: momentum sign at bar t uses closes[:t] only; \"\n          \"neutral bars before the lookback window carry the equity forward.\")\n\n    print(\"\\n=== 4. Momentum cost-robustness check (lookback 3 / 5 / 10) ===\")\n    print(\"  Params: walk-forward per segment train={}d / test={}d / \"\n          \"warmup={}d / overlap={}d, min {} bars/segment, \"\n          \"4 contiguous volatility blocks, each lookback vs a matched null \"\n          \"(zero cost)\".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))\n\n    # Load the full collected universe.\n    all_tickers = {}\n    for entry in manifest[\"entries\"]:\n        ticker = entry[\"ticker\"]\n        all_tickers[ticker] = bt.load_ticker(ticker)[0]\n    ticker_order = list(all_tickers.keys())\n\n    # Results indexed [lookback][ticker][segment].\n    cost_robustness_results = {lb: {t: segment_result(t, all_tickers[t], lb) for t in ticker_order}\n                               for lb in LOOKBACKS}\n\n    print(\"  cost-robustness verdict counts across universe:\")\n    print(f\"    COST_ROBUST_POSITIVE (>=2 lookbacks with positive edge): {robustness_counts['COST_ROBUST_POSITIVE']}\")\n    print(f\"    COST_ROBUST_NEGATIVE (>=2 lookbacks with negative edge): {robustness_counts['COST_ROBUST_NEGATIVE']}\")\n    عليك\n    print(f\"    COST_ROBUST_INCONCLUSIVE (<2 clear verdicts): {robustness_counts['COST_ROBUST_INCONCLUSIVE']}\")\n\n    print(\"  Per-asset cost-robustness profiles:\")\n    for ticker in ticker_order:\n        p = per_asset[ticker]\n        verdicts = \", \".join(\n            \"{}:{:6.3f} {}\".format(lb, p[str(lb)][\"medians\"][0], robustness_verdict(p[str(lb)]))\n            for lb in LOOKBACKS\n        )\n        print(\"    {:6s} segments={:<8} medians3510={:<40s} rob={:<8s} \"\n              \"edge_counts={}\".format(\n                ticker, str(p[\"5\"][\"segments\"]), verdicts, p[\"cost_robustness_verdict\"],\n                p[\"edge_counts\"]))\n\n    # Determinism: rerun the target subset and assert identical output.\n    res2 = {lb: {t: segment_result(t, tickers[t], lb) for t in target}\n            for lb in LOOKBACKS}\n    out1 = json.dumps({lb: {t: cost_robustness_results[lb][t] for t in target}\n                       for lb in LOOKBACKS}, sort_keys=True)\n    out2 = json.dumps(res2, sort_keys=True)\n    print(\"determinism: r1 == r2: {}\".format(out1 == out2))\n    assert out1 == out2, \"non-deterministic output\"\n\n    artifact = dict(\n        dataset_id=DATASET_ID,\n        seed=SEED,\n        lookbacks=list(LOOKBACKS),\n        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,\n        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,\n        realistic_costs=REALISTIC_COSTS,\n        per_asset=per_asset,\n        universe=dict(\n            cost_robustness_counts=robustness_counts,\n            ticker_order=ticker_order,\n            per_asset=dict(\n                (t, dict(\n                    lookback_medians={str(lb): per_asset[t][str(lb)][\"medians\"] for lb in LOOKBACKS},\n                    lookback_null_medians={str(lb): per_asset[t][str(lb)][\"null_medians\"] for lb in LOOKBACKS},\n                    lookback_5_verdict=per_asset[t][\"5\"][\"verdict\"],\n                    cost_robustness_verdict=per_asset[t][\"cost_robustness_verdict\"],\n                    edge_counts=per_asset[t][\"edge_counts\"],\n                    n_folds_5=per_asset[t][\"5\"][\"n_folds\"],\n                ))\n                for t in ticker_order\n            ),\n        ),\n    )\n    with open(ARTIFACT_PATH, \"w\") as f:\n        json.dump(artifact, f, indent=2, sort_keys=True)\n    print(\"artifact written to: {}\".format(ARTIFACT_PATH))\n    print(\"NOTE: research/simulation only. No live trading or production \"\n          \"execution. Momentum cost-robustness check on collected universe: \"\n          \"exploratory simulation.\")\n    return 0\n\n\nif __name__ == \"__main__\":\n    sys.exit(main())\nEOF
