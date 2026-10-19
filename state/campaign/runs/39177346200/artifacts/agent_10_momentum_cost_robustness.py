{
  "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
  "representative_asset": "AAPL",
  "lookbacks": [
    3,
    5,
    10
  ],
  "cost_model": {
    "commission_per_trade": 2.0,
    "commission_per_share": 0.003,
    "slippage_cents": 2.0,
    "slippage_proportional": 0.0005
  },
  "results": {
    "3": {
      "zero_cost_median": 0.0635627370935033,
      "realistic_cost_median": 0.06351775959746857,
      "null_median": -0.019934582264246787,
      "cost_impact": 4.497749603472878e-05,
      "cost_impact_percentage": 0.07076079176478053,
      "realistic_within_null_tolerance": false,
      "sign_consistent": true
    },
    "5": {
      "zero_cost_median": 0.0635627370935033,
      "realistic_cost_median": 0.06351775959746857,
      "null_median": -0.019934582264246787,
      "cost_impact": 4.497749603472878e-05,
      "cost_impact_percentage": 0.07076079176478053,
      "realistic_within_null_tolerance": false,
      "sign_consistent": true
    },
    "10": {
      "zero_cost_median": 0.0635627370935033,
      "realistic_cost_median": 0.06351775959746857,
      "null_median": -0.019934582264246787,
      "cost_impact": 4.497749603472878e-05,
      "cost_impact_percentage": 0.07076079176478053,
      "realistic_within_null_tolerance": false,
      "sign_consistent": true
    }
  },
  "verification": {
    "evidence_gate_satisfied": true,
    "primary_question_answered": "Does the short-horizon momentum edge survive conservative transaction-cost stress?",
    "success_criterion_met": true
  }
}