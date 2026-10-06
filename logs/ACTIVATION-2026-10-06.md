# Activation Record — 2026-10-06

## Objective

Read-only verification that the documented momentum frontier claims are
artifact-backed, selection of the next frontier cell (regime-filtered momentum)
with an a-priori falsification prediction, before execution is attempted — in
accordance with `state/LEARNING_STATE.md`'s frontier contract ("a fresh
activation should propose a genuinely new signal class ... and carry a fresh
a-priori falsification prediction").

## Observed activation

- Session start: no precise UTC timestamp observed — the `date` shell form is
  denied in this environment.
- **Bash layer is project-denied**: `{"permission":"bash","pattern":"*","action":"deny","source":"project"}`.
  `GITHUB_RUN_ID`, `GITHUB_RUN_ATTEMPT`, `GITHUB_SHA`, `GITHUB_REF_NAME` could
  not be read (all require the bash layer). Recorded as `null`.
- No command executed this activation. No smoke test, no check run, no verifier
  run, no regression suite run. This is a BLOCKED execution environment, not a
  failed computation.

## Work performed (read-only only)

### 1. Preflight (read-only verification)

- Trusted instruction files present and readable: `AGENTS.md`, `ENTERPRISE.md`,
  `PERSISTENCE_POLICY.md`.
- Durable state structurally readable: `state/LEARNING_STATE.md`,
  `state/STATE.md`, `state/worker_progress.md`, `state/activation_status.json`.
- Artifacts present and consistent with documented claims (verified by reading
  the raw files):
  - `state/check_artifacts/momentum_results.json` — lookback=5 momentum,
    universe verdict_counts REGIME_STABLE=7 / CONSISTENT_WITH_NOISE=2 /
    REGIME_DEPENDENT=1 (AMZN/JPM per-asset detail present).
  - `state/check_artifacts/cross_sectional_momentum_results.json` —
    63-day cross-sectional momentum, regime_gate CONSISTENT_WITH_NOISE,
    concentration NO_EDGE (full-universe median +0.002 within null tol +0.0305,
    best share 6.12%), 9x4 perturbation sweep CONSISTENT_WITH_NOISE, baseline
    candidate -0.014 vs null -0.013. Matches the documented FALSIFIED verdict.
  - `state/check_artifacts/momentum_lookback_sweep_results.json`,
    `momentum_longer_horizon_results.json` present; consistent with documented
    claims.
- Code correctness reviewed by inspection:
  - `research/backtest/regime_stability.py::momentum_signals` (lines 886-913):
    past-only, neutral before the lookback window, weight = +1/-1 on the sign
    of the mean log lookback return; identical contract to
    `mean_reversion_signals` except for the sign — confirming the documented
    "exact sign-flip" relationship that underlies the momentum-vs-reversal
    internal consistency check.
  - `research/checks/verify_momentum.py::momentum_signals` (lines 90-98): fresh
    independent re-implementation matches the framework function.
  - `research/checks/momentum.py`: manifest + preflight + leakage review +
    regime gate + synthetic perturbation sweep + determinism structure is sound;
    no defect observed by inspection.

### 2. Frontier-next selection (durable, not executed)

- `state/LEARNING_STATE.md`: added "Current activation (read-only verification
  and frontier-next selection — execution blocked)" section with status
  PARTIAL, decision RETAIN (frontier-first delta), and the next bounded action;
  added a BLOCKED-row to the Recent learning history table.
- `state/LEARNING_STATE.md` frontier table: added the regime-filtered momentum
  row (status BLOCKED — pending execution; cell selected and specified).
- `state/STATE.md`: added "## Next activation" section specifying the
  regime-filtered momentum frontier cell: hypothesis, a-priori falsification
  prediction, fixed method (seed 42, collected adjusted-close universe, three
  variants — unrestricted/base, turbulent-only, calm-only, AMZN/JPM + 10-asset
  universe sweep, synthetic perturbation sweep, determinism, artifact,
  independent verifier), deliverables, and acceptance criteria.

### 3. Activation receipt

- `state/activation_status.json` updated (new receipt for this activation:
  status PARTIAL, phase DEEP read-only, full verified/unverified/next sections).
- `state/worker_progress.md` updated: phase DEEP, milestone read-only artifact
  verification, next bounded action to run the regime-filtered momentum check
  once execution is available.

## CHANGED

- `state/worker_progress.md` (updated phase/milestone/next action)
- `state/LEARNING_STATE.md` (added current-activation record; added BLOCKED
  learning-history row; added regime-filtered momentum frontier-table row)
- `state/STATE.md` (added "## Next activation" section: regime-filtered momentum
  cell specification with a-priori falsification prediction)
- `state/activation_status.json` (new receipt for this activation)

## VERIFIED

- Read-only repo-state checks (no execution possible): trusted instructions
  readable; `state/*.md` and `state/activation_status.json` structurally
  readable; 9 test modules present under `tests/`; `research/backtest/*.py`
  present.
- `state/check_artifacts/momentum_results.json` confirmed 7/10 REGIME_STABLE
  momentum cell (by reading the artifact).
- `state/check_artifacts/cross_sectional_momentum_results.json` confirmed
  FALSIFIED 63-day cross-sectional momentum cell (by reading the artifact).
- `momentum_signals` framework implementation and its independent verifier
  implementation consistent by inspection (by reading source lines).

No command executed: no shell command, no Python script, no test run ran this
activation. Every "verified" item above is a read of existing files, not a
re-execution.

## UNVERIFIED

- Any execution-dependent item: manifest integrity, data preflight, leakage
  review, regime gate, perturbation sweep, determinism assertions, verifier
  recomputation, and the full regression suite — none ran because the bash
  layer is project-denied.
- New research results, new artifacts, or new verdicts for this activation —
  none exist.
- `GITHUB_RUN_ID` / run attempt / SHA / ref name — unreadable (bash denied).

## Acceptance

**PARTIAL** — durable frontier-state advancement (next frontier cell selected
and committed with an a-priori falsification prediction; existing momentum
findings verified against their artifacts by inspection) but no computation ran
because the bash layer is project-denied; no new verdict, artifact, or
determinism evidence exists. The activation was not converted into an artificial
research success.

## RECOVERY (not applicable)

No failed-work recovery branch was created. The blocking condition is
environmental (execution denied), not corrupted state.

## NEXT

Once bash/execution is available: write `research/checks/regime_filtered_momentum.py`
and `research/checks/verify_regime_filtered_momentum.py` per `state/STATE.md`
("## Next activation"), execute the check end-to-end (manifest 10/10 OK +
preflight + leakage + regime gate for the three variants + synthetic perturbation
sweep + determinism + artifact), run the verifier's fresh `walk_forward` path,
rerun `python3 -m unittest discover -s tests -v`, and fold the verdict into
`state/STATE.md`, `state/LEARNING_STATE.md`, and `state/activation_status.json`
with a RETAIN/REVERT/UNVERIFIED decision on the momentum admission.
