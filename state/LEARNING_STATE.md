# Research Learning State

This is the compact durable learning layer for the autonomous quantitative-research worker.

## Active strategy delta

- **Gap:** Research activations have durable conclusions, but prior work did not explicitly convert each activation into a retained/rejected search-policy update or frontier decision; objectives were selected opportunistically from a queue.
- **Strategy Delta:** Make frontier-first research selection explicit: choose the highest-value unresolved hypothesis/frontier cell before starting substantial work, and record one process-level search change per substantive activation.
- **Expected Effect:** Fewer equivalent experiments and more information gained per unit of worker time.
- **Anti-gaming Constraint:** Do not tune to OOS results or treat experiment count, positive results, runtime, or commits as learning. Prefer reliable negative conclusions.
- **Observed Effect:** This activation operationalized the contract's infrastructure (populated `state/LEARNING_STATE.md` with every accumulated hypothesis from `logs/` and `state/STATE.md`), but its objective was chosen from the queued receipt item (fold the regime-adaptive MA verdict into `STATE.md`), not from a populated frontier cell. The expected effect (fewer equivalent experiments) was therefore not observed this activation.
- **Decision: UNVERIFIED**
- **Next:** In the next substantive activation, select the highest-value DEFERRED/UNEXPLORED frontier cell (a new signal class — mean-reversion / volatility-targeting / cross-sectional relative strength on the collected universe) and test it through the same perturbation + coin-flip-null + regime-stability gate before any positive claim.

## Research frontier

| Area / hypothesis | Status | Evidence / reason | Next bounded action |
|---|---|---|---|
| MA crossover / volatility-regime filter edge on regime-switching synthetic data | FALSIFIED (tooling-validation only) | Parameter-sweep medians near zero, mixed signs, flat deviation scan; indistinguishable from the coin-flip null (seed 42) | None required; synthetic baseline established |
| MA(20/60) edge on collected real data (AAPL t=+2.75; META t=+2.64, nominal 5%) | REJECTED from evidence base | Within the sample-calibrated coin-flip noise band; canonical (20,60) not a peak in any of 10 assets; 0/10 survive Bonferroni at alpha 0.005; sparse per-asset significance | Explore whether the negative result is genuine (framework gates validated) rather than a framework artifact |
| MA(20/60) regime-stable across real market regimes (AAPL) | PARTIALLY SUPPORTED, not generalized | AAPL REGIME_STABLE across 4 blocks; but universe shows mixed verdicts (CWN=4, STABLE=4, DEPENDENT=2 base) and NVDA CONSISTENT_WITH_NOISE | Defer: generalization not demonstrated |
| MA(20/60) robust across the 10-asset collected universe | FALSIFIED | No consistent cross-asset edge; 5/10 CONSISTENT_WITH_NOISE, 4/10 REGIME_STABLE (no edge), 1/10 REGIME_DEPENDENT under the base variant | Move to a new signal class or method |
| MA(20/60) edge concentrated in the 2009-2013 regime (AMZN, JPM) | FALSIFIED | Deep-dive: REGIME_DEPENDENT; the edge lives in one historical regime mix and reverses/disappears later | None |
| Turbulent-only MA filter (signals active only in turbulent segments) rescues the edge | FALSIFIED | Still REGIME_DEPENDENT (AMZN dispersion +0.086 unchanged; JPM +0.045 vs 9x the null) | None |
| Regime-adaptive MA (fast 10/30 in turbulent, standard 20/60 in calm) rescues a REGIME_STABLE edge | FALSIFIED (activation 37260572520) | AMZN -> CONSISTENT_WITH_NOISE (edge killed); JPM -> REGIME_DEPENDENT (+0.034); universe: CWN=5, REGIME_STABLE=4, REGIME_DEPENDENT=1 (JPM); independently recomputed via fresh `walk_forward` path — all six figures MATCH to 3 decimals | None |
| A past-only volatility classifier can separate the 2009-2013 "good" turbulent regime from the 2022-2026 "bad" turbulent regime | FALSIFIED | Both periods carry the identical "turbulent" label; a volatility-based timing rule cannot harvest the 2009-2013 edge | Defer: requires richer regime features (cross-sectional, macro, liquidity) |
| Mean-reversion / volatility-targeting / cross-sectional relative strength on collected data | DEFERRED (new signal class) | Not yet explored; the trend-following / MA-crossover class is exhausted on this dataset | Test a new signal class through the same perturbation + coin-flip-null + regime-stability gate before any positive claim |

Use statuses such as UNEXPLORED, TESTED, FALSIFIED, SUPPORTED, BLOCKED, or DEFERRED.
Prefer the highest-value unresolved frontier cell over repeating an equivalent experiment.

## Recent learning history

| Activation | Strategy Delta | Observed Effect | Information Gain | Decision |
|---|---|---|---|---|
| 2026-10-02 | Build smallest deterministic backtest toolkit (stdlib + numpy); synthetic = tooling-validation only; IS/OOS walk-forward default; no-look-ahead by construction | VERIFIED: full 120-test suite runs; all examples reproducible under seed 42 | Framework capability established (engine, metrics, leakage, perturbation, regime-stability, universe) | RETAIN |
| 2026-10-03 | Per-asset coin-flip null as the default (each asset judged against its own null rather than a single global null) | VERIFIED: all canonical verdicts unchanged (CONCENTRATED / CONSISTENT / NO_EDGE) | Tighter, per-asset significance flags; documented gap closed | RETAIN |
| 2026-10-03 | Sample-calibrated `compare_noise` tolerance (2 x null dispersion / sqrt(n_folds)) replacing the arbitrary fixed 5% band | VERIFIED: AAPL verdict unchanged (within noise); fixed-band API preserved via explicit `tol=` | Methodology gap closed per `METHODOLOGY.md` | RETAIN |
| 2026-10-04 | Durable loader verification (loader reads CSV `adjclose`, not raw `close`); 9-test loader suite | VERIFIED: corrected loader executed end-to-end; 96 tests OK; preflight 57 checks OK | Data-pipeline integrity restored; no look-ahead in real-data price series | RETAIN |
| 2026-10-04 | Research CI preflight gate + independent post-worker verification | VERIFIED: controller architecture restored; X run #40 persisted after rebase conflict resolved | Controller reliability improved | RETAIN |
| 2026-10-05 | Learning-efficiency contract initialized (`LEARNING_STATE.md`, frontier-first selection delta); frontier populated in activation 37260572520 | PARTIAL: infrastructure exists and frontier is populated; the process was not yet operative this activation | Process infrastructure in place; not yet evidence of better selection | RETAIN (process infrastructure) |
| 37257258857 (2026-10-05) | Test regime-adaptive MA as the falsifiable rescue of the 2009-2013 MA edge | VERIFIED: falsified (see 37260572520) — AMZN -> CONSISTENT_WITH_NOISE, JPM -> REGIME_DEPENDENT, independently corroborated | One MA-crossover rescue hypothesis eliminated with two independent paths | RETAIN (as a falsified finding) |
| 37260572520 (2026-10-05) | Operationalize the frontier: populate it from durable evidence; test the adaptive-variant hypothesis on AMZN/JPM + universe; fold the verdict into `STATE.md` | VERIFIED for the research claim (falsification corroborated by independent recomputation); the process-level delta remains UNVERIFIED because the objective came from the queued receipt item rather than a frontier cell | One MA-crossover hypothesis falsified with two independent paths; the frontier is now a usable decision surface | UNVERIFIED (process delta) |

## Learning rules

1. One substantive activation should normally choose one primary strategy delta.
2. A repeated experiment requires a new hypothesis, diagnostic purpose, materially new data, independent verification need, or different information objective.
3. Measure learning by uncertainty reduced, hypotheses falsified/strengthened, methodology repaired, independent evidence gained, or research capability improved—not by activity, runtime, experiment count, or positive results.
4. Do not use OOS results as a process-tuning oracle.
5. Preserve negative and inconclusive evidence.
6. Do not claim a strategy delta worked until its expected effect is actually observed.
7. When there is insufficient evidence, choose UNVERIFIED rather than forcing RETAIN or REVERT.
