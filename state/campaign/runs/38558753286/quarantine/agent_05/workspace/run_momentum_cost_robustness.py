#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from research.checks.momentum_cost_robustness import MomentumCostRobustnessChecker

if __name__ == "__main__":
    checker = MomentumCostRobustnessChecker()
    artifact = checker.run()
    print("\n" + "="*60)
    print("MOMENTUM COST-ROBUSTNESS CHECK COMPLETE")
    print(f"Overall verdict: {artifact['overall_verdict']}")
    print(f"Conclusion: {artifact['conclusion']}")
    print("="*60)