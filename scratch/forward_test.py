baseline_nulls = []
baseline_null_fold = []
grid = [{"lookback": 21, "top_k": 3, "bottom_k": 3},
        {"lookback": 63, "top_k": 3, "bottom_k": 3}]
for p in grid:
    if p["lookback"] == 63:
        baseline_nulls.append(999)  # referencing nlrets below
        baseline_null_fold.extend(nlrets)
    nlrets = [1, 2, 3]
print(baseline_nulls, baseline_null_fold)
