"""Agent 2 - Campaign Slot 2: Cost-robustness verification for short-horizon momentum

NOTES:
1. What is the system ultimately trying to become better at learning?
   The system is trying to become better at choosing what is worth learning, learning it efficiently, falsifying it, validating it independently, preserving evidence, and choosing what to learn next.

2. What currently most constrains useful learning?
   The candidate positive momentum evidence is short-horizon and mechanical; cost robustness remains decision-relevant.

3. What uncertainty prevents a better decision?
   Whether the short-horizon momentum edge survives conservative transaction-cost stress on the collected real-data universe.

4. What single bounded action is most likely to reduce it?
   Run a bounded cost-sensitivity check for lookback 3/5/10 momentum under the repository's existing realistic cost model and matched null.

5. Why is that action worth its effort, complexity, and execution risk?
   This directly addresses the bottleneck by testing whether the momentum signal survives realistic transaction costs. It uses existing infrastructure, provides a clear go/no-go decision point, and preserves the evidence regardless of outcome.

EXECUTION:
- Set up cost-sensitive momentum analysis using the existing 10-asset collected universe
- Use realistic cost parameters from examples (commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005)
- Run bounded lookback sweep (3/5/10) with cost sensitivity analysis
- Compare momentum performance vs matched null on the same segments
- Generate reproducible artifact and independent verification

DELIVERABLE:
- Reproducible cost-robustness artifact classifying momentum edge survival under conservative transaction costs
- Independent verification result that matches the check artifact

SUCCESS CRITERIA:
- Result is reproducible and independently verified, regardless of positive or negative outcome
- Uses existing momentum implementation/check framework
- Real-data preflight, leakage gate, independent recomputation, determinism, and existing regression suite

SCOPE BOUNDARIES:
- Use existing 10-asset collected universe
- Use existing momentum implementation/check framework
- Do not tune signal rules after seeing cost results
- Do not expand the universe
- Do not introduce leverage
- Exactly one bounded objective
- Exactly one meaningful deliverable
"""