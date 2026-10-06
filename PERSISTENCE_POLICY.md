# Persistence Policy

## Purpose

The runner is disposable, but useful work should be allowed to become durable repository state.

This policy deliberately does **not** require Kilo to keep the repository artificially small or to delete every diagnostic or scratch file. New research code, tests, examples, documentation, verification helpers, diagnostics, and supporting infrastructure may be persisted when they provide future value.

The persistence boundary is therefore **safety-based, not cleanliness-based**.

## Allowed

Kilo may persist ordinary repository content when it is relevant to the research mission, including:

- research and backtest code;
- tests and verification tooling;
- examples and diagnostics;
- scratch/debug helpers that have useful future value;
- state, logs, experiment records, methodology and documentation;
- new directories or supporting files needed by the research system.

A temporary-looking filename is not unsafe merely because it is temporary-looking.

## Always protected

The autonomous agent must not modify or persist changes to trusted control-plane files:

- `.github/workflows/**`
- `.github/scripts/**`
- `.kilo/**`
- `AGENTS.md`
- `ENTERPRISE.md`
- `MANUAL_SETUP.md`
- `PERSISTENCE_POLICY.md`

The trusted workflow restores these paths before persistence and rejects any staged violation. Controller-evaluation helpers under `.github/scripts/**` are part of the same trust boundary and are never worker-owned.

## Never persist

The persistence gate must reject high-confidence unsafe material, including:

- private keys and private-key blocks;
- obvious GitHub, cloud, Kilo, or other API credentials/tokens when recognizable by standard credential prefixes or formats;
- credential/secret files such as `.env`, `.env.*`, `credentials.*`, `secrets.*`, or `*.pem` / `*.key` when they contain credential material;
- symbolic links;
- generated Python bytecode/build artifacts such as `__pycache__/**`, `*.pyc`, and `*.py.c`.

Ambiguous patterns should not by themselves block legitimate research code; the gate targets high-confidence unsafe material. High-confidence unsafe material must fail the persistence step rather than being silently committed.

## Behavior on violation

A persistence-policy violation is a failed activation persistence step, not a reason to weaken the policy.

The agent should report:

- what category was blocked;
- which path(s) triggered the block;
- whether the underlying research work remains available only in the disposable runner;
- the smallest safe next action.

## Promotion rule

Useful verification or diagnostic tooling may be promoted into durable infrastructure when future activations benefit from it. The agent should document why the helper is durable when that distinction matters.

The objective is:

**maximum useful research continuity subject to a small, explicit safety boundary.**
