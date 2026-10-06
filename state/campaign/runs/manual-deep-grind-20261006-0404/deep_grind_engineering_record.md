# X Deep-Grind Engineering Record — 2026-10-06

## Session objective

Deep-grind the sequential Kilo research campaign architecture for latent and observed engineering errors, then make the smallest evidence-backed fixes and preserve reusable lessons.

## Prompt improvements made before execution

The supplied prompt was strengthened in five important ways:

1. **Concurrency/race audit:** explicitly inspect scheduled-run overlap because the workflow executes every five minutes while campaigns can run for hours.
2. **Evidence integrity audit:** distinguish staged, unstaged, and untracked changes when validating workers.
3. **Control-plane integrity:** protect the campaign architecture regression test from worker modification.
4. **Security-scanner correctness:** require behavioral testing of credential detection rather than text-only assertions, and verify secret signatures against authoritative provider documentation.
5. **Durable record contract:** use this exact session record destination and distinguish verified live behavior from static/local verification.

## Observed failures

### 1. Staged credential scanner false positive

The latest live run on commit `c5f8ddc0a361e984b026959b540be37df9ab58ab` reached successful readiness, Kilo execution, controller validation, final process evaluation, research verification, synthesis, and persistence preparation, but the persistence gate failed because the credential scanner matched the benign text:

`state/smoke/kilo_campaign_dummy_trigger.txt`

against a guessed `kilo_...` token pattern.

The first scanner repair changed the scanner from scanning diff headers to staged blob contents, but the next live run demonstrated that the underlying regex was still too broad.

### 2. Missing campaign concurrency guard

The five-minute schedule had no workflow-level concurrency group. A campaign can run much longer than five minutes, so overlapping scheduled/manual executions could race durable task state and the persistent global-agent counter.

### 3. Validator missed staged-only worker changes

`.github/scripts/validate_campaign_agent.py` originally collected only unstaged and untracked paths. A worker could stage a modified path and potentially evade the change-path inventory used by validation.

### 4. Campaign architecture regression test was not protected

Workers were prohibited from modifying controller/workflow infrastructure, but `tests/test_campaign_architecture.py` was not controller-protected. A worker could potentially weaken an architectural regression test while still leaving the control plane unchanged.

### 5. Global identity reservation happened before queue validation

`campaign_controller.py` reserved persistent global agent numbers before fully validating the durable task queue and role availability. A failed initialization could therefore consume an identity range without creating a valid campaign.

## Fixes applied

### Scanner hardening

Added:

` .github/scripts/scan_staged_secrets.py `

The scanner now:

- reads only the staged blob content;
- does not scan diff headers or filenames as credential evidence;
- uses behaviorally testable patterns;
- detects private-key markers;
- detects GitHub token formats;
- detects Kilo Gateway API keys as JWTs only when they occur in an authentication context;
- reports only the affected path/pattern class, not secret contents.

The Kilo JWT rule was based on Kilo's current Gateway authentication documentation, which states that Kilo Gateway API keys are JWT tokens.

Added:

`tests/test_staged_secret_scan.py`

Coverage includes:
- benign `kilo_campaign_dummy_trigger.txt` reference;
- `KILO_API_KEY=<JWT>`;
- `Authorization: Bearer <JWT>`;
- GitHub classic token;
- private-key marker.

### Workflow serialization

Added a workflow-level concurrency group:

`x-kilo-research-campaign`

with `cancel-in-progress: false`.

This prevents multiple campaign runs from executing concurrently and racing durable campaign state.

### Validator hardening

`current_changed()` now includes:

- unstaged tracked changes;
- staged tracked changes;
- untracked files.

`tests/test_campaign_architecture.py` is now controller-protected.

### Controller hardening

Persistent global-agent identity allocation now occurs only after the task queue and required roles have successfully validated.

This reduces accidental identity gaps caused by failed campaign initialization.

### Regression coverage

Expanded `tests/test_campaign_architecture.py` to cover:

- workflow concurrency;
- delegated behavioral credential scanning;
- absence of the old broad Kilo prefix regex;
- staged-change visibility;
- protection of the campaign architecture test.

## Verification evidence

### Verified locally

- Secret-scanner regex behavior tested directly.
- Benign `kilo_campaign_dummy_trigger.txt` content does not match.
- Context-bound Kilo JWT detection matches both `KILO_API_KEY=` and `Authorization: Bearer`.
- Arbitrary `kilo_...` text no longer matches.
- New scanner Python source passed terminal `py_compile`.
- Workflow, controller, validator, and test changes were inspected on the current `main` tree.

### Live evidence

The latest pre-fix live run demonstrated the original false-positive failure and is useful evidence for the root cause.

The new deep-grind head has **not yet received a fresh live GitHub Actions run**, so the post-fix workflow behavior remains UNVERIFIED.

Main head at the original record creation:
`64635ac522a89f5f1b08fdd5ec11cb70ae4f5165`

Subsequent adaptive-fix commits now advance the X main head beyond that original record snapshot.

## Adaptive follow-up findings

The first repair set exposed two additional boundary failures before a new live campaign could validate it:

1. **Regression-test drift:** the architecture test still asserted the removed inline credential-scanning shell code. The hardened workflow therefore had a test that would fail even though the implementation was more correct.
2. **Duplicate-path security drift:** the legacy manual one-shot workflow still contained the old broad `kilo_...` credential regex and diff-based scanner, so the repository had two inconsistent persistence security boundaries.

## Adaptive fixes

- Updated `tests/test_campaign_architecture.py` to treat the dedicated scanner as the canonical persistence boundary and require both Kilo workflows to delegate to it.
- Updated `.github/workflows/kilo-prompt-execution-one-shot.yml` to use `.github/scripts/scan_staged_secrets.py` rather than its duplicated inline scanner.
- Extended one-shot control-plane protection to `.github/scripts/`.
- Protected `tests/test_staged_secret_scan.py` from worker modification.
- Rechecked the current main tree for residual legacy scanner execution paths.

These fixes reduce implementation drift between execution modes and make the scanner/test contract executable at the architecture-test layer.

## Remaining uncertainties

1. **Global identity reuse after a failed/quarantined campaign:** persistent numbering is still stored in normal repository state, so a campaign that never persists its reservation to the main branch could theoretically reuse a range on a later campaign. The new workflow concurrency guard removes overlap but does not fully solve failure-path reservation durability.
2. **Full live post-fix campaign behavior:** requires a new GitHub Actions execution on the current head.
3. **Secret signature completeness:** the scanner now matches the documented Kilo JWT format, but no generic secret scanner can prove absence of every possible credential format.

These remain explicitly UNVERIFIED rather than being treated as failures.

## Generalized lessons

- A security scanner must inspect the data boundary that will actually be persisted, not surrounding filenames or diff metadata.
- Credential signatures should be based on authoritative provider formats, not guessed prefixes.
- Any scheduled workflow that mutates durable counters/queues needs an explicit non-overlap policy.
- Change detection must account for staged, unstaged, and untracked state.
- Regression tests governing the control plane should themselves be protected from worker modification.
- Reserve persistent identities only after all campaign initialization prerequisites pass.
- A green workflow before the current commit is not evidence that the current commit is healthy.

## Highest-value next action

Run one fresh campaign on the current `main` head and inspect the complete path from preflight through persistence and terminal classification. Treat only that run as live verification of this deep-grind repair set.
