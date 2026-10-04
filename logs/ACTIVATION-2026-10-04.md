# Activation Record — 2026-10-04 (supervisory correction after Kilo activation #31)

## Observed activation

- Workflow: **Kilo Research Enterprise Wake**
- Run: **#31** (workflow run ID `37171768516`)
- Observed start: **2026-10-04T02:40:18Z**
- Observed finish: **2026-10-04T02:44:15Z**
- GitHub job outcome: **success**
- Kilo outcome reported by the workflow: **success**

## Worker changes observed

The worker persisted:

- `research/backtest/real_data.py`
- `research/backtest/__init__.py` export changes
- `test_loader_tmp.py`
- `test_timer.py`

The worker successfully ran the repository real-data preflight during the activation; the workflow log records:

`PREFLIGHT PASSED: all data-quality and survivorship checks passed`.

The worker did **not** complete the required timestamped activation record during the activation, and the workflow log does not provide a successful post-change full regression-suite result.

## Bottleneck

The concrete bottleneck was a combination of **handoff incompleteness and a research-data contract mismatch** in the newly persisted real-data loader.

The loader's documentation said it loaded an adjusted-close series, while the implementation read the raw CSV `close` field rather than `adjclose`. The authoritative manifest explicitly says adjusted close is the backtesting price series.

The discrepancy is material: the stored AAPL sample has different raw and adjusted closes (for example, 2009-01-02 has `close=3.241071` and `adjclose=2.714299`).

## Action taken

A single narrow code correction was made:

- Changed `research/backtest/real_data.py` to read the `adjclose` field (`p[5]`) for the BarSequence close series.
- Commit: `40a131bce637c8d511a6696423344109b7e31edf`

A durable supervisory handoff was also recorded in `state/STATE.md`, and this activation record was created so the next worker has an explicit, truthful continuation point.

No protected workflow, Kilo configuration, authority, security, credential, or scheduler file was changed.

## VERIFIED

- Live GitHub evidence confirms run #31 completed successfully.
- The run's GitHub job and all listed workflow steps completed successfully.
- The worker's real-data preflight was actually executed and reported **PREFLIGHT PASSED**.
- The repository manifest explicitly requires adjusted-close backtesting.
- The AAPL raw data demonstrably contains distinct `close` and `adjclose` values.
- After the correction, the loader source reads `p[5]`, the adjusted-close field.

## UNVERIFIED

- The corrected loader was **not executed after the final edit**.
- The full unit-test suite was **not successfully completed after the final edit**.
- No real-data end-to-end backtest was run after the final edit.
- The Kilo job's overall success is not accepted as evidence of research-code correctness.
- Several Kilo tool invocations were denied or failed because of the repository's tool-permission rules; these are recorded as narrow invocation failures, not evidence that all execution is unavailable.

## RISKS

The adjusted-close correction changes the price series used by the new loader as intended by the authoritative dataset contract, but it still requires executable regression verification.

The persisted `test_loader_tmp.py` and `test_timer.py` are retained rather than blanket-deleted because the repository's persistence policy allows useful diagnostics/scratch helpers when they provide future value.

## NEXT

1. Execute a focused loader test confirming returned closes equal CSV `adjclose`.
2. Run the full regression suite and relevant examples after the final edit.
3. Only after those checks pass, continue the real-data research pipeline and record the next activation with exact observed UTC timestamps.
