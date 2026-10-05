# Kilo Worker Learning Efficiency

## Current diagnosis

The worker has a strong research-methodology prompt and durable state, but there is no independent per-activation process evaluator comparable to desktop-tutorial's hidden benchmark and aggregate solver feedback.

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

Only add a quantitative process score after repeated activations demonstrate that the qualitative learning loop is insufficient.
