# Kilo Worker Learning Efficiency

## Current diagnosis

The worker has a strong research-methodology prompt and durable state. The controller now also provides an independent per-activation process evaluator and per-agent worktree validation, so the learning loop has an execution-quality feedback path distinct from the worker's own claims.

The main learning bottleneck is therefore not adding more instructions. It is converting each activation into a compact, falsifiable **strategy update** that improves the next activation without encouraging score/activity gaming.

## Recommended learning loop

Use:

**Observe prior evidence -> choose one strategy delta -> execute -> independently inspect -> measure information gained -> retain/reject the delta**

For each substantive activation, record:

- **Strategy Delta:** one concrete change in how the next research search will be conducted.
- **Reason:** the observed bottleneck or repeated failure that motivated it.
- **Expected Effect:** what should improve if the delta is useful.
- **Observed Effect:** what actually changed in evidence quality, falsification power, discovery efficiency, or compute efficiency.
- **Decision:** RETAIN / REVERT / UNVERIFIED.
- **Next:** exactly one bounded continuation.

Do not treat a positive backtest or a larger experiment count as evidence that a strategy delta worked.

## Highest-value improvements

### 1. Maintain a research frontier

Persist a compact frontier of unresolved hypotheses, failed hypotheses, unexplored assets/data regimes, and known dead ends.

Prefer the next action that most reduces uncertainty across this frontier rather than repeatedly refining the most interesting recent result.

### 2. Add a novelty guard

Before launching an experiment, identify whether an equivalent experiment already exists.

Repeat only when there is:
- a new hypothesis,
- a new diagnostic purpose,
- materially new data,
- an independent verification need, or
- a clearly different information objective.

This converts accumulated history into actual avoidance of wasted search.

### 3. Measure information gain, not activity

At handoff, classify the activation's principal contribution as one of:
- hypothesis falsified;
- hypothesis materially strengthened;
- implementation defect repaired;
- new independent evidence;
- new research capability;
- uncertainty reduced;
- no substantive information gain.

A long successful execution with no new information should be treated as low learning efficiency.

### 4. Protect against self-training on OOS

A strategy delta may improve the research process, but it must not use OOS performance as a tuning oracle. Keep exploration, validation, and acceptance separate.

## Implementation priority

The first lightweight implementation should be durable strategy-delta logging and frontier state, not a second autonomous evaluator.

Use the evaluator categorically, not as a reward score: `VERIFIED_PROGRESS`, `VERIFIED_REPAIR`, `NO_SUBSTANTIVE_ACTION`, or `UNVERIFIED`. Add finer quantitative scoring only after repeated activations demonstrate that these categorical gates are insufficient.


## Controller enforcement

The current controller implements the lightweight loop above without turning it into a gamified score:

1. Each fresh campaign agent receives a post-session controller validation gate covering protected paths, syntax, regression health when research code changed, receipt structure, and semantic checkpoint transition.
2. After the campaign, an independent process evaluator compares the live worktree with the committed baseline and distinguishes durable research progress, verified research/software repair, no substantive action, and unverified execution.
3. The terminal outcome classifier requires this process result in addition to independent post-worker verification before returning `COMPLETE`.
4. Failed or unverified worker output is eligible for recovery-branch quarantine rather than automatic main-branch integration.
