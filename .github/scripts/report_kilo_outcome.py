#!/usr/bin/env python3
"""Deterministic Kilo outcome classifier shared by preflight and hand-off."""

from __future__ import annotations

import argparse


def classify(
    *,
    preflight: str,
    smoke: str,
    worker: str,
    verify: str,
    liveness: str,
    persistence: str,
) -> tuple[int, str]:
    if preflight != "success":
        return 1, "FAILED"
    if smoke != "success":
        return 1, "FAILED"
    if liveness == "STALLED":
        return 1, "FAILED"
    if liveness == "COMPLETED_WITHOUT_SEMANTIC_CHECKPOINT":
        return 1, "FAILED"
    if worker == "skipped":
        return 1, "FAILED"
    if verify != "success":
        return 1, "PARTIAL"
    if persistence == "MAIN":
        if worker == "success":
            return 0, "COMPLETE"
        return 1, "PARTIAL"
    if persistence == "RECOVERY_BRANCH":
        return 1, "PARTIAL"
    if persistence == "NO_CHANGES":
        return 0, "NO_SUBSTANTIVE_ACTION"
    return 1, "FAILED"


def self_test() -> int:
    common = {
        "preflight": "success",
        "smoke": "success",
    }

    assert classify(**common, verify="success", worker="success", liveness="COMPLETED_WITH_SEMANTIC_CHECKPOINTS", persistence="MAIN") == (0, "COMPLETE")
    assert classify(**common, verify="success", worker="failure", liveness="COMPLETED_WITH_SEMANTIC_CHECKPOINTS", persistence="MAIN") == (1, "PARTIAL")
    assert classify(**common, verify="success", worker="failure", liveness="COMPLETED_WITH_SEMANTIC_CHECKPOINTS", persistence="RECOVERY_BRANCH") == (1, "PARTIAL")
    assert classify(**common, verify="success", worker="success", liveness="COMPLETED_WITH_SEMANTIC_CHECKPOINTS", persistence="NO_CHANGES") == (0, "NO_SUBSTANTIVE_ACTION")
    assert classify(**common, verify="success", worker="skipped", liveness="COMPLETED_WITH_SEMANTIC_CHECKPOINTS", persistence="MAIN") == (1, "FAILED")
    assert classify(**common, worker="success", verify="failure", liveness="COMPLETED_WITH_SEMANTIC_CHECKPOINTS", persistence="MAIN") == (1, "PARTIAL")
    assert classify(**common, verify="success", worker="success", liveness="STALLED", persistence="MAIN") == (1, "FAILED")
    assert classify(**common, verify="success", worker="success", liveness="COMPLETED_WITHOUT_SEMANTIC_CHECKPOINT", persistence="MAIN") == (1, "FAILED")

    print("Kilo outcome reporter self-test: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preflight")
    parser.add_argument("--smoke")
    parser.add_argument("--worker")
    parser.add_argument("--verify")
    parser.add_argument("--liveness")
    parser.add_argument("--persistence")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    values = {
        "preflight": args.preflight,
        "smoke": args.smoke,
        "worker": args.worker,
        "verify": args.verify,
        "liveness": args.liveness,
        "persistence": args.persistence,
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        parser.error("missing outcome arguments: " + ", ".join(missing))

    rc, outcome = classify(**values)
    print(f"KILO_WORKER_OUTCOME={outcome}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
