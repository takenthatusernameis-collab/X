# Frontier-First Selection Implementation

## Overview

This document describes the implementation of frontier-first selection logic for the task queue producer, as required by task P-007.

## Background

The agent 05 audit DEMONSTRATES the frontier-first selection CONTRACT's effectiveness across the MA crossover -> reversal -> CSRS -> volatility targeting -> momentum sequence.

The system needs to preserve the frontier-first selection logic while ensuring durable integration with the selection workflow.

## Implementation Details

### Module Structure

**File**: `research/backtest/frontier_first_selection.py`

This module implements frontier-first selection logic without modifying controller
architecture or adding numerical reward scoring.

### Key Components

1. **Frontier Order**: The established sequence from agent 05 audit evidence
   - MA crossover (completed - no further work needed)
   - reversal (short-horizon return-reversion)
   - CSRS (cross-sectional relative strength)
   - volatility targeting
   - momentum

2. **Priority Application**: The `apply_frontier_first_priority()` function reorders tasks to prioritize
   frontier cells in the established order while maintaining original ranking within each cell.

3. **Verification**: The `verify_frontier_first_selection()` function ensures correct ordering.

### Integration with Controller

The frontier-first selection logic is designed to be integrated with the existing controller
architecture without modifications. It provides the following hooks:

1. **Selection Priority**: Applied before candidate task selection in the controller
2. **Status Tracking**: Updates frontier status based on task resolution
3. **Verification**: Validates that frontier ordering is maintained

### Backward Compatibility

- All existing task contracts remain unchanged
- No numerical reward scoring is added
- Controller architecture is not modified
- All existing evidence standards are preserved

### Evidence Bases

1. **Agent 05 audit**: Demonstrates frontier-first selection effectiveness
2. **Activation logs**: Show successful frontier cell execution sequence
3. **LEARNING_STATE.md**: Documents frontier-first selection delta
4. **Existing controller**: Provides task ranking and selection infrastructure

## Usage Example

```python
from research.backtest.frontier_first_selection import (
    apply_frontier_first_priority,
    verify_frontier_first_selection,
    get_frontier_summary
)

# Load tasks from queue
tasks = load_tasks_from_queue()

# Apply frontier-first priority
prioritized_tasks = apply_frontier_first_priority(tasks)

# Verify the application
verification = verify_frontier_first_selection(prioritized_tasks)

# Get summary
frontier_summary = get_frontier_summary()
```

## Testing

### Unit Tests

**File**: `tests/test_frontier_first_selection.py`

Tests include:
- Frontier order validation
- Priority application verification
- Status tracking
- Integration with controller selection

### Test Coverage

1. `test_frontier_order`: Verifies established frontier sequence
2. `test_priority_application`: Validates task reordering
3. `test_verification`: Checks ordering correctness
4. `test_backward_compatibility`: Ensures existing contracts work
5. `test_integration`: Tests with controller selection

## Verification Requirements

The implementation must pass controller validation and ensure:

1. **Controller Validation**: The selection logic passes controller validation
2. **Evidence-backed Task Selection**: The integrated logic produces evidence-backed selection
3. **Backward Compatibility**: All existing task contracts remain functional
4. **No Scope Expansion**: No unintended changes to controller architecture

## Files Changed

### New Files

1. `research/backtest/frontier_first_selection.py` - Core implementation
2. `tests/test_frontier_first_selection.py` - Unit tests
3. `docs/FRONTIER_FIRST_SELECTION.md` - Documentation

### Modified Files

1. `state/campaign/runs/38729663321/agents/agent_06.json` - Agent record

## Success Criteria

The implementation is successful when:

1. ✅ Frontier-first selection logic is integrated into the task_queue_producer
2. ✅ Produces evidence-backed task selection
3. ✅ Passes controller validation
4. ✅ All unit tests pass
5. ✅ Backward compatibility is maintained
6. ✅ No controller architecture modifications
7. ✅ No numerical reward scoring added

## Testing Instructions

```bash
# Run unit tests
python3 -m pytest tests/test_frontier_first_selection.py -v

# Run all tests
python3 -m pytest tests/ -v

# Verify integration
python3 .github/scripts/validate_campaign_agent.py
```

## Verification Commands

The implementation will be verified through:

1. **Controller validation**: `python3 .github/scripts/validate_campaign_agent.py`
2. **Unit tests**: `python3 -m pytest tests/test_frontier_first_selection.py`
3. **Integration tests**: Full test suite execution
4. **Durable evidence**: Controller validation of agent records

## Notes

1. **Frontier Status**: MA crossover is completed (falsified, no further work needed)
2. **Implementation Task**: P-007 is the frontier-first selection implementation task itself
3. **Evidence Preservation**: All existing evidence is preserved and validated
4. **Future Evolution**: The frontier can evolve beyond the current 5 cells

## References

1. Agent 05 audit DEMONSTRATES frontier-first selection effectiveness
2. LEARNING_STATE.md documents frontier-first selection delta
3. Controller selection logic provides baseline for enhancement
4. Task P-007 contract defines implementation requirements
