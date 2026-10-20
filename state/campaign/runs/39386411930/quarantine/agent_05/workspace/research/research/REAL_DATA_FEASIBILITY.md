# Real-Data Feasibility Note

Research/simulation only. This repository is not a live-trading or
production-execution system. No exchange credentials, production
secrets, or personal files belong here.

## Purpose

This note defines what data may enter research-only simulation work,
how provenance must be recorded, and the pre-run checks that must pass
before any real-data backtest is executed. It is a precondition document:
it does not authorize any real-data run, and no strategy evaluated on
real data here may be treated as evidence suitable for live capital.

## Scope boundary

The repository's validation pipeline (`research/backtest/`: engine,
metrics, leakage checks, perturbation tests) is calibrated and unit-tested
on synthetic data only. Synthetic data validates the tooling; it does not
validate a strategy. Any real-data work inherits the same tooling but
must additionally prove data quality and pass every leakage check before
its results are admitted into the evidence base.

## Data that is in-scope for research-only simulation

In-scope sources share these properties: public, non-proprietary,
credential-free, and usable without any exchange, broker, or paid-feed
account.

- Government and intergovernmental statistical releases: e.g. SEC EDGAR
  filings, FRED (Federal Reserve Economic Data), BEA, Census Bureau,
  BLS, CFTC Commitments of Traders, IMF, World Bank, OECD.
- Public exchange- or trade-reporting snapshots that require no key and
  no account (e.g. posted OHLCV archives, public trade aggregation
  endpoints that do not gate behind terms requiring credentials).
- Academic and research-grade public datasets (e.g. Fama-French factor
  tables, CRSP/Compustat mirrors distributed for teaching).
- Self-collected public data obtained with due respect to robots.txt and
  each source's terms of service, provided collection requires no
  authentication.

## Data that is out of scope

- Anything requiring exchange, broker, or paid-feed credentials (API
  keys, account sign-up, subscription).
- Proprietary or commercial feeds, private repositories, and personal
  files.
- Data derived from production infrastructure, order books requiring
  latency-aware feeds, or any source that would blur the line between
  research and live execution.

If an in-scope source is accessed from a different mirror or with a
different collection date, treat it as a different dataset: give it a new
manifest entry.

## Provenance and manifest

Every real dataset used in a run must be accompanied by a manifest file
placed alongside the data under `research/data/`. The manifest records:

- `source`: canonical name and URL of the source;
- `dataset_id`: stable identifier (e.g. `sec-edgar-macd-2026-01-15`);
- `collection_date` and `accessed_date`: what period the data covers, and
  when it was downloaded;
- `location`: filesystem path and a checksum (sha256) of the raw file;
- `license`: license and any reuse restrictions;
- `collection_method`: script or manual process used, with its version;
- `transform_log`: every transformation applied (renames, filters,
  merges, forward-fill decisions), with rationale;
- `known_issues`: gaps, duplicates, or inconsistencies found during
  intake;
- `owner` and `created` timestamp.

The manifest is the audit trail. A real-data run summary (per
`logs/ACTIVATION-YYYY-MM-DD.md`) must cite the manifest id used for that run.

## Data-quality preflight checklist

Before any real-data signal is generated, a preflight must pass and its
results recorded.

1. **Existence and integrity**: raw file matches its recorded checksum;
   manifest present and complete.
2. **Completeness**: no missing dates in the intended calendar (document
   legitimate holidays/exchanges shutdowns); no unexpected duplicates.
3. **Ordering**: timestamps are monotonically increasing and correctly
   timezone-normalized to a single reference (prefer UTC).
4. **No future dates**: no bar date exceeds the collection/end date of
   the dataset; no look-ahead into the signal's decision time.
5. **OHLC cross-consistency**: `low <= open`, `low <= close`,
   `high >= open`, `high >= close`, `high >= low`, per bar.
6. **Volume and count fields**: non-negative, finite; consistent units.
7. **Price plausibility**: positive prices; no division-by-zero-prone
   zero-price bars (document handling).
8. **Survivorship**: the investable universe membership is fixed as of
   the decision date; no delisted names or reconstituted indices appear
   mid-sample.
9. **Feature consistency**: all constructed features use only data
   available at the signal bar's close; no point-in-time corrections
   retroactively injected.

## Leakage review checklist (pre-run)

A real-data run must not start until a reviewer signs off on each item.
This checklist mirrors the engine's no-look-ahead rules and the
`check_signal_integrity` / `check_equity_matches_fills` checks.

1. **Close-only execution**: every fill prices at close plus fixed and
   proportional slippage; open/high/low are never used for execution.
2. **Warmup padding**: the first `warmup_periods` bars hold initial
   capital and emit neutral signals; signal code pads indicators so no
   real signal exists before the warmup boundary.
3. **Signal dates**: every signal references a bar date that exists in
   the series and never a future bar; the series is fully covered (a
   signal exists at the last bar date).
4. **Feature construction**: rolling, expanding, and lagged features
   reference only closes up to and including the signal bar; label
   construction (e.g. next-bar return) never uses the decision-bar's own
   return.
5. **Static thresholds**: regime thresholds, correlations, and universe
   definitions are fixed before the walk-forward; nothing is re-fitted
   inside a test segment.
6. **Fill-equity audit**: `check_equity_matches_fills` recomputes the
   equity curve from the fill sequence and asserts it matches; every
   equity change is traceable to a recorded fill.
7. **Cost model**: commission, slippage, and funding/carry assumptions
   are recorded in the manifest and held identical across folds.
8. **IS/OOS discipline**: only test segments are backtested and reported;
   train segments are information available at fold start.
9. **Walk-forward sanity**: fold count, OOS periods, and per-fold metrics
   are internally consistent (unit-tested in `tests/test_engine.py`).
10. **Perturbation pass**: before any result is admitted, the strategy is
    swept across a +-x2 window grid around the canonical parameters and
    shows no single-point peak and no systematic collapse as parameters
    deviate. The noise benchmark (`bt.noise_benchmark`) must remain near
    zero on the same grid.

## Validation hierarchy

- Synthetic-data run: tooling is validated; nothing about a strategy.
- Real-data run without the above preflight/checklist: not admitted;
  treated as unvalidated exploration at best.
- Real-data run passing preflight + all leakage checks + walk-forward +
  perturbation: candidate evidence, still to be judged for regime
  stability, parameter sensitivity, and robustness.

## Decision rule

No live-trading or production-execution action is permitted from this
repository. Real-data results inform only the decision about whether to
continue research. The repository remains research/simulation only.
