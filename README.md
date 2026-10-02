# X

## Autonomous Quantitative Research Enterprise

This public repository is a research/simulation environment for a periodically waking Kilo Code agent.

### Purpose

Continuously improve the enterprise's ability to make increasingly effective decisions about how to discover, evaluate, falsify, validate, and learn about potentially robust quantitative trading opportunities under uncertainty.

### Architecture

- GitHub Actions: deterministic 5-minute heartbeat, concurrency control, disposable Linux runner.
- Kilo Code CLI: autonomous agent operating inside the current runner.
- Repository: persistent institutional memory and research state.
- Git history: durable audit trail.
- Deterministic research/backtest infrastructure: `research/backtest/`
  (event-driven backtest, walk-forward IS/OOS, synthetic-data
  validation, leakage checks, parameter-sensitivity / robustness testing)
  plus `tests/`. Preferred execution layer.

### Safety boundary

This repository is deliberately research-only. It has no live-trading credentials, no production execution path, and no access to unrelated repositories or personal files.

Kilo is not given the workflow's GitHub write token and cannot dispatch another workflow. The trusted workflow persists safe working-tree changes after the Kilo step.

See `ENTERPRISE.md`, `AGENTS.md`, `MANUAL_SETUP.md`, and `.github/workflows/kilo-wakeup.yml`.
