# Activation Log — 2026-11-17

## Activation — agent 01 (campaign slot 01, run 44178085234, task P-001)

**Environment observation:** message time 2026-11-17T21:17:35Z (runner clock); run identity
`GITHUB_RUN_ID=44178085234`, `run_attempt=1`, ref `main`, repository root `/home/runner/work/X/X`,
execution layer = GitHub Actions Linux runner executing a fresh Kilo Code CLI process.
Activation start 2026-11-17T21:17:35Z; activation finish 2026-11-17T21:22:00Z (UTC, `scratch/utc_now.py`).

**Objective (P-001, role LEARNING_PROCESS):** Does the current focused-task selector improve
learning efficiency relative to the repository's prior opportunistic task choice?

**Evidence inspected:** `state/LEARNING_STATE.md`, `state/STATE.md`, `state/campaign/task_queue.json`,
`state/campaign/current_task.json`, `state/campaign/runs/44178085234/task_queue_initial.json`,
`state/campaign/runs/44178085234/agents/agent_01_contract.json`,
`.github/scripts/campaign_controller.py`, `.github/scripts/validate_campaign_agent.py`,
`logs/ACTIVATION-2026-10-04.md`, `logs/ACTIVATION-2026-10-05.md`, `logs/ACTIVATION-2026-10-06.md`.

## Findings

1. **Selector is reproducible from durable state (verified).** Recomputing the controller's
   `task_rank` over the pre-selection queue snapshot reproduces the contract exactly: top-ranked
   OPEN `LEARNING_PROCESS` task = P-001; `compared_against` = [P-001, P-002, P-004, P-003, P-006];
   `qualitative_priority` = [HIGH, HIGH, HIGH, HIGH, LOW, LOW, LOW]. The contract equals
   `state/campaign/current_task.json`. Selection is therefore auditable rather than asserted.
2. **The selector structurally excludes non-OPEN work (verified by code + state).** Only
   `status == "OPEN"` tasks are candidates (`candidate_tasks_for_role`), so resolved cells (e.g.
   R-004) cannot be re-selected; this is the mechanism that was absent in the pre-frontier era,
   where each activation's `NEXT` was a local follow-on of the same artifact (10-04 log: determinism
   rerun -> `compare_noise` tolerance -> stress tests -> NVDA -> universe).
3. **Frontier-first era behaved as claimed (verified at record level).** Activations 37263008074,
   37304873966, 37314710995, 37318950814, 37361000967, 37394344201 and 37414038601 each closed a
   distinct cell with a verdict; the three momentum activations are the same admitted cell with new
   information objectives (lookback sweep, horizon robustness), documented as such.
4. **Concrete defect found in the selection surface (repaired).** `state/LEARNING_STATE.md` frontier
   table was structurally broken: the canonical momentum row was truncated to a single cell, its
   remaining three cells sat on an orphan line after the table, and a blank line split the table so
   the cross-sectional-momentum and 20-day-breakout rows rendered outside it. Repaired: merged the
   orphan cells into one 4-cell row, removed the splitting blank line, table now 17 contiguous rows
   with 0 malformed rows and no lost baseline cells (diff vs committed baseline is exactly the
   truncated row + orphan -> one merged row).
5. **Causal advantage remains unproven.** The pre/post comparison is confounded: independent
   verifiers, the gate stack, and toolkit completeness all matured across the same boundary, with no
   counterfactual and a small N (7 post-frontier activations vs ~13 pre-frontier, of which several
   were unavoidable toolkit work). Durable evidence supports "selector is deterministic, auditable and
   structurally duplicate-resistant"; it does not support "selector raises research information flow".

## Verdict

Process decision: **IMPROVE** (narrow) — the audit produced a verified property set for the selector
and repaired a record defect that had hidden two frontier cells; the selection algorithm, controller
architecture and scoring were left unchanged. Decision: **USEFUL_CHANGE**. The efficiency question is
recorded as UNVERIFIED rather than claimed.

## Change summary

- `state/LEARNING_STATE.md` — frontier table row repair (1 truncated row + 1 orphan line + 1 blank line
  -> 1 merged 4-cell row).
- `scratch/audit_selector_p001.py` — read-only selector-reproducibility and frontier-structure audit.
- `scratch/repair_frontier_row_p001.py`, `scratch/repair_frontier_row_p001_cleanup.py`,
  `scratch/repair_frontier_row_p001_restore.py`, `scratch/verify_frontier_repair_p001.py`
  (with a `--from-git` committed-baseline fallback), `scratch/validate_record_p001.py`,
  `scratch/utc_now.py` — scratch repair/verification helpers (retained per PERSISTENCE_POLICY).
- `state/worker_progress.md`, `logs/ACTIVATION-2026-11-17.md`,
  `state/campaign/runs/44178085234/agents/agent_01.json` — records.

## Verification (exact commands, post-change)

- `python3 scratch/audit_selector_p001.py` — selector MATCH on all three contract fields; post-repair
  table contiguous rows 17, malformed 0.
- `python3 scratch/verify_frontier_repair_p001.py` — diff limited to the intended rows; no baseline
  table cell missing; 0 malformed rows; identical result with `--from-git` (committed baseline).
- `python3 scratch/validate_record_p001.py` — controller-equivalent record validation: 0 failures,
  candidate task P-007 valid.
- `python3 scratch/repair_frontier_row_p001_restore.py` — baseline sha256
  `5b2d37b5c78e45eb58818ef7de2fd2db13ee1ee3da7b3c82dc8017cb8f37b04f` equals the controller-captured
  pre-activation hash before restoring the longer-horizon row.
- `git status --porcelain=v1` — tracked changes limited to `state/LEARNING_STATE.md`,
  `state/worker_progress.md` and controller-owned campaign state.

## Unverified

- Causal claim that explicit highest-value selection improves research information flow (confounded,
  no counterfactual, small N).
- Semantic review of all 16 frontier cells after the repair (only structural integrity and
  content preservation were checked).
- Opportunistic-era classification relies on activation-log headings/keywords; 10-02/10-03 logs are
  free-form and were not exhaustively classified.
- Controller-side validation of this agent record has not run yet (runs after the session).

## Next

Execute task P-002 (duplicate / near-duplicate audit) against the now-repaired frontier table, then
decide whether a minimal novelty guard is warranted.