import sys
from pathlib import Path

log = Path.cwd() / "logs" / "ACTIVATION-2026-10-05.md"
record = """
## Activation: cross-sectional relative strength frontier test — 37304873966

### Activation goal

Execute the next deferred frontier cell — cross-sectional relative strength
(CSRS) on the collected adjusted-close universe (long top-3 / short bottom-3 by
20-day lookback return, hold 1 day, daily rebalance, neutral before lookback) —
through the regime-stability gate vs the AAPL volatility-block segments, the
coin-flip null, the concentration (drop-one) gate, and the synthetic
perturbation sweep over the canonical regime family, with an independent
verifier; close the cell with a verdict in STATE.md and LEARNING_STATE.md.

### Observed activation

- Session start: **2026-10-05** (timestamp observed in the environment message;
  precise per-run UTC seconds not captured — the `date` shell form is denied).
- Finish: **same day, 2026-10-05** (estimate, ~4 min for the check + verifier).

### Work performed

#### 1. Readiness preflight

- Checkout on `main`, revision
  `fd8c1503fd2bee80ea2acecae168bb56b5a92799`; manifest dataset
  `yf-ohlcv-universe-2009-to-2026-10-03` (10 entries, all checksums OK),
  collection 2026-10-03.
- `research/checks/cross_sectional_relative_strength.py` parses/compiles; new
  CSRS helpers `research/backtest/regime_stability.py` parse/compile.
- Smoke test: hand-computed CSRS spread on a 3-asset synthetic family matched
  the implementation exactly on bars 2-4 (determinism true); `csrs_null_spread_family`
  preserved the spread magnitude under coin-flipping (structure intact).

#### 2. Execution and defects found/repaired

`research/checks/cross_sectional_relative_strength.py` was run end-to-end and
failed at sections 3/7/8; each defect was repaired and the check re-run to
completion:

- **IndexError on 800-bar families**: `csrs_spread_family` and
  `csrs_null_spread_family` looped `range(lookback, first_len)` while reading
  `closes[i + 1]` — one bar past the end. Repaired to
  `range(lookback, first_len - 1)`.
- **ImportError**: `csrs_null_spread_family` was missing from the
  `research/backtest/__init__.py` export block (the check imports it). Added to
  both the import block and `__all__`.
- **Engine cross-check MISMATCH (section 4)**: with the engine's constant-
  notional sizing and per-fold warmup, the engine's equity compounds the spread
  by simple P&L addition, so `walk_forward` `total_return` could never match the
  direct fold-log-return path. Repaired by using per-fold constant-share
  signals (weight = synthetic close / close of that fold's first post-warmup
  bar), which makes the engine's equity compound geometrically from the same
  baseline the direct path uses; the engine path then matches the direct path
  bar-for-bar. Verified independently: a compact reproduction matched all 15
  demo folds to 9 decimals.
- **KeyError: 'bottom_k' (section 7)**: the sweep's `baseline_dict` held only
  lookback/top_k while param sets carry bottom_k too. Repaired the deviation
  calculation to divide only over the baseline's own parameters.
- **KeyError on param-set tuples (section 7)**: `candidate_sweep`/`null_sweep`
  keys were `tuple(p.items())` (unsorted) while `param_sets` used
  `tuple(sorted(p.items()))`. Repaired all sweep mappings to sorted-tuple keys.
- **ValueError ambiguous array boolean (section 8)**: the determinism check
  compared a run against itself. Repaired to compare run 1 vs run 2 with
  `np.all()`.

#### 3. Results (final run)

- Manifest 10/10 OK; preflight passed (all data-quality and survivorship
  checks); universe truncated to the fully overlapping 3614-bar window
  (2012-05-18 to 2026-10-02).
- Section 3: synthetic-equity path self-consistent with spread — PASS.
- Section 4: regime gate — candidate medians calm -0.027, turbulent -0.013 vs
  null calm -0.067, turbulent +0.017 (dispersion +0.010 vs null +0.059,
  below the 2x rule); verdict **CONSISTENT_WITH_NOISE**; engine-path
  cross-check **MATCH** on both segments (same verdict).
- Section 5: per-sub-universe (drop-one) — CWN=9, REGIME_STABLE=1 (NVDA:
  medians [-0.032, -0.055], the stable verdict reflects negative medians, not
  an edge), REGIME_DEPENDENT=0.
- Section 6: concentration gate — full median -0.013 within null tolerance
  +0.030; NO_EDGE; best single-ticker contribution +120.9% (no edge to
  concentrate on).
- Section 7: synthetic sweep (4 scenarios x 9 param sets) — baseline (lookback
  20, top_k 3) median -0.001 vs null +0.009; within fixed 0.05 on all sets;
  sweep-level CONSISTENT_WITH_NOISE; scenario medians calm +0.000, turbulent
  +0.013, mean_reverting -0.065, trending -0.002.
- Section 8: determinism — r1 == r2 for spread, null, medians; artifact written.

#### 4. Independent verification

- `python3 research/checks/verify_cross_sectional.py` (separate path: fresh
  `vol_blocks`, fresh spread re-implementation, fresh per-bar coin-flip null
  with bars-aligned RNG, fresh fold aggregation) — MATCH: segment labels,
  candidate medians [-0.0274, -0.01266], null medians [-0.06672, +0.01689],
  per-asset verdict counts 9/1/0, concentration median match.
- Determinism re-run: the check reran a second time with identical output.
- Regression suite re-established: **125 tests OK** (baseline was 120; +5 new
  tests for the CSRS helpers).

#### 5. Verdict

**FALSIFIED** — the cross-sectional relative-strength class is not a robust edge
on the collected large-cap universe: the spread median is negative in every
segment of every asset, within the coin-flip null band, with a NO_EDGE
concentration verdict and a noise-like sweep across lookback/top_k and the
canonical regime family. Durable negative evidence, independently verified and
deterministic. Recorded in `state/STATE.md`, `state/LEARNING_STATE.md`,
`state/activation_status.json`, and this log.

### Next action

Execute the volatility-targeting frontier cell (vol-normalized sizing rule) with
the same gates before advancing to any third signal class.
"""

log.write_text(log.read_text() + record)
print("APPENDED to", log, "| new total lines:", len(log.read_text().splitlines()))
