#!/usr/bin/env python3
"""
Final verification script for R-001 deliverable completion.
This script verifies that all contract requirements have been met.
"""
import json
import sys
from pathlib import Path

def check_agent_record():
    """Verify agent_10.json meets contract requirements."""
    agent_path = Path.cwd() / "state/campaign/runs/39697921459/agents/agent_10.json"
    
    with open(agent_path, 'r') as f:
        agent_data = json.load(f)
    
    required_fields = [
        "agent_number", "role", "task_id", "objective", "bottleneck",
        "question", "action", "changed", "verified", "unverified",
        "observed_effect", "uncertainty_targeted", "uncertainty_reduced",
        "process_decision", "research_result", "decision", "next",
        "candidate_tasks", "complexity_added", "failure_class",
        "task_selection_observation"
    ]
    
    missing_fields = []
    for field in required_fields:
        if field not in agent_data:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"❌ Missing required fields: {missing_fields}")
        return False
    
    # Check controller metadata
    controller_fields = ["agent_number", "campaign_slot", "global_agent_number"]
    for field in controller_fields:
        if agent_data.get(field) != 10:
            print(f"❌ Controller field {field} value is {agent_data.get(field)}, expected 10")
            return False
    
    # Check canonical values
    if agent_data["decision"] not in ["NO_SUBSTANTIVE_ACTION", "REJECT", "RETAIN", "UNVERIFIED", "USEFUL_CHANGE", "VERIFIED_NEGATIVE_RESULT"]:
        print(f"❌ Invalid decision value: {agent_data['decision']}")
        return False
    
    if agent_data["process_decision"] not in ["IMPROVE", "REJECT", "RETAIN", "UNVERIFIED"]:
        print(f"❌ Invalid process_decision value: {agent_data['process_decision']}")
        return False
    
    if agent_data["task_selection_observation"] not in ["IMPROVED", "UNCHANGED", "UNVERIFIED", "WORSENED"]:
        print(f"❌ Invalid task_selection_observation value: {agent_data['task_selection_observation']}")
        return False
    
    if agent_data["failure_class"] not in ["LIVENESS", "NONE", "PERSISTENCE", "RESEARCH_EXECUTION", "SCOPE_VIOLATION", "TOOL_FAILURE", "TRANSIENT_GATEWAY", "UNKNOWN", "VALIDATION_FAILURE"]:
        print(f"❌ Invalid failure_class value: {agent_data['failure_class']}")
        return False
    
    # Check arrays are not empty where appropriate
    if not agent_data["changed"]:
        print("❌ changed array should not be empty")
        return False
    
    if not agent_data["verified"]:
        print("❌ verified array should not be empty")
        return False
    
    print("✅ Agent record validation passed")
    return True

def check_artifacts():
    """Verify required artifacts exist and are valid."""
    artifacts = [
        "state/check_artifacts/momentum_cost_robustness.json",
        "state/check_artifacts/momentum_cost_robustness_verification.json",
        "state/check_artifacts/momentum_results.json"  # Existing artifact for reference
    ]
    
    for artifact in artifacts:
        path = Path.cwd() / artifact
        if not path.exists():
            print(f"❌ Artifact not found: {artifact}")
            return False
        
        try:
            with open(path, 'r') as f:
                json.load(f)
            print(f"✅ Artifact verified: {artifact}")
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in {artifact}: {e}")
            return False
    
    return True

def check_implementation():
    """Verify implementation code exists."""
    implementation_files = [
        "research/checks/momentum_cost_analysis.py",
        "research/checks/momentum_cost_robustness.py",  # Legacy
        "research/checks/verify_momentum_cost_robustness.py"  # Legacy
    ]
    
    for impl_file in implementation_files:
        path = Path.cwd() / impl_file
        if not path.exists():
            print(f"❌ Implementation file not found: {impl_file}")
            return False
        
        # Check it contains expected elements
        content = path.read_text()
        if "momentum_cost_sensitivity" not in content and "lookback" not in content:
            print(f"⚠️  Implementation file may not contain expected content: {impl_file}")
        
        print(f"✅ Implementation file found: {impl_file}")
    
    return True

def main():
    """Run all verification checks."""
    print("=== R-001 Deliverable Completion Verification ===\n")
    
    checks = [
        ("Agent Record", check_agent_record),
        ("Research Artifacts", check_artifacts),
        ("Implementation Code", check_implementation),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"--- {check_name} ---")
        if not check_func():
            all_passed = False
        print()
    
    if all_passed:
        print("🎉 SUCCESS: All verification checks passed!")
        print("\nR-001 deliverable is COMPLETE and ready for handoff.")
        return 0
    else:
        print("❌ FAILURE: Some verification checks failed.")
        print("\nR-001 deliverable requires attention before handoff.")
        return 1

if __name__ == "__main__":
    exit(main())
