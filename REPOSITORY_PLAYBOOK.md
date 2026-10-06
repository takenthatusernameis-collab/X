# Repository Interaction Playbook

## Purpose

This file is the repository's **procedural generational memory**.

It records verified, reusable knowledge about how to operate this repository efficiently and reliably. It is deliberately separate from research conclusions in `state/LEARNING_STATE.md` and `state/STATE.md`.

The objective is to help a fresh generation avoid rediscovering operational knowledge that previous generations already paid to learn.

## How to use this playbook

Read this file early in every activation.

Treat entries as **operational guidance**, not research evidence. A playbook entry may tell you how to inspect, execute, validate, persist, or recover work efficiently; it does not establish that a research hypothesis is true.

Prefer:
- existing tested repository paths over new tooling;
- the narrowest command that can answer the question;
- targeted validation before broad validation;
- durable artifacts over transient output;
- one generalized rule over many incident notes.

When a verified lesson conflicts with an older entry, update the older entry rather than creating a duplicate.

## Fast path for a fresh activation

1. Read `state/LEARNING_STATE.md` and `state/STATE.md`.
2. Read the latest relevant activation record.
3. Read this playbook for repository-specific operating shortcuts and known failure modes.
4. Inspect the existing check/verifier before creating new research code.
5. Run the smallest representative test before scaling to the full universe or full suite.
6. Independently verify consequential results before treating them as durable evidence.
7. Persist the research conclusion in the research state and the reusable operating lesson here.

## Efficient interaction patterns

### Repository discovery

**Rule:** Start from durable state and the existing research/check infrastructure before broad file exploration.

### Research implementation

**Rule:** Reuse tested framework primitives and existing checks/verifiers before introducing new machinery.

### Verification

**Rule:** For consequential research changes, run the smallest targeted verifier first, then the full regression suite when the targeted path passes.

### Failure handling

**Rule:** Classify whether a failure is infrastructure, implementation, validation, persistence, or research evidence before changing the research hypothesis.

### Persistence

**Rule:** Preserve the minimal artifact needed to reconstruct the result, and keep negative or inconclusive evidence durable.

## Verified interaction lessons

Add only lessons that are generalizable beyond one transient incident.

| Lesson | Status | Evidence / context | Reusable rule |
|---|---|---|---|
| Read durable frontier state before selecting new research work | VERIFIED | Frontier-first campaign history | Start from the unresolved frontier; do not rediscover already-closed cells. |
| Distinguish worker execution from verification and acceptance | VERIFIED | Controller lifecycle and independent-verification design | A successful command/session is execution evidence only; acceptance requires the relevant verifier. |
| Use the canonical staged-secret scanner rather than inline workflow regexes | VERIFIED | Persistence security-boundary repairs | Reuse `.github/scripts/scan_staged_secrets.py`; do not duplicate scanner logic. |
| Use the Kilo `openai-compatible/free-kilo` alias for worker execution | VERIFIED | Model-interface repair | Let model discovery populate the alias and invoke the alias consistently. |
| Separate workflow-environment observations from research evidence | VERIFIED | Environment-awareness repair on 2026-10-06 | Classify runner, gateway, permission, and workflow behavior as infrastructure evidence unless independently connected to the research question. |

## Environment constraints worth remembering

Record stable constraints such as tool permissions, command restrictions, network limitations, or runner behavior only when they materially change the efficient execution path.

Do not record secrets, private reasoning, raw credential material, or transient logs.

## Failed approaches worth avoiding

| Approach | Status | Why it is inefficient or unsafe | Better path |
|---|---|---|---|
| Reimplementing a tested research primitive from scratch | VERIFIED_NEGATIVE | Adds unnecessary defect surface and weakens comparability | Reuse the existing primitive and add a narrow wrapper/check. |
| Treating a Kilo gateway/model error as a research result | VERIFIED_NEGATIVE | The failure is environmental/tooling evidence | Classify the failure, repair the execution path, then rerun the actual research gate. |
| Treating a workflow/runner artifact as proof of a market or research claim | VERIFIED_NEGATIVE | Infrastructure signals can be mistaken for external-world evidence | Establish the research claim independently through the repository's research gates. |

## Update contract for future generations

A fresh agent should add or revise a playbook entry only when all are true:

1. The interaction lesson is generalizable.
2. The lesson was directly observed or independently verified.
3. The lesson can save future research time, reduce error, or improve evidence quality.
4. The lesson does not duplicate an existing entry.

Use this compact format:

```
Lesson: <generalizable operational lesson>
Status: VERIFIED | UNVERIFIED | VERIFIED_NEGATIVE
Evidence: <where it was established>
Reusable rule: <one concrete instruction>
```

Do not turn this file into an activation diary. Detailed chronology belongs in activation logs; research conclusions belong in research state; this file stores reusable **how-to-operate knowledge**.
