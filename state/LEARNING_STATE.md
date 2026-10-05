# Research Learning State

This is the compact durable learning layer for the autonomous quantitative-research worker.

## Active strategy delta

- Gap: Research activations have durable conclusions, but prior work did not explicitly convert each activation into a retained/rejected search-policy update or frontier decision.
- Strategy Delta: Make frontier-first research selection explicit: choose the highest-value unresolved hypothesis/frontier cell before starting substantial work, and record one process-level search change per substantive activation.
- Expected Effect: Fewer equivalent experiments and more information gained per unit of worker time.
- Anti-gaming Constraint: Do not tune to OOS results or treat experiment count, positive results, runtime, or commits as learning.
- Observed Effect: UNVERIFIED — this is the first operational activation of the learning contract.
- Decision: UNVERIFIED
- Next: On the next substantive activation, select one unresolved frontier item and complete the strategy-delta receipt before handoff.

## Research frontier

| Area / hypothesis | Status | Evidence / reason | Next bounded action |
|---|---|---|---|
| | UNEXPLORED | | |

Use statuses such as UNEXPLORED, TESTED, FALSIFIED, SUPPORTED, BLOCKED, or DEFERRED.

Prefer the highest-value unresolved frontier cell over repeating an equivalent experiment.

## Recent learning history

| Activation | Strategy Delta | Observed Effect | Information Gain | Decision |
|---|---|---|---|---|
| | | | | |

## Learning rules

1. One substantive activation should normally choose one primary strategy delta.
2. A repeated experiment requires a new hypothesis, diagnostic purpose, materially new data, independent verification need, or different information objective.
3. Measure learning by uncertainty reduced, hypotheses falsified/strengthened, methodology repaired, independent evidence gained, or research capability improved—not by activity, runtime, experiment count, or positive results.
4. Do not use OOS results as a process-tuning oracle.
5. Preserve negative and inconclusive evidence.
6. Do not claim a strategy delta worked until its expected effect is actually observed.
7. When there is insufficient evidence, choose UNVERIFIED rather than forcing RETAIN or REVERT.
