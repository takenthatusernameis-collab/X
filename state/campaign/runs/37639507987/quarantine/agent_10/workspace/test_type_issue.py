# Test the logic
test_dict = {1: "one", 2: "two", 3: "three", 5: "five"}

print("Testing type conversion:")
for h2 in [1, 2, 3, 5]:
    str_key = str(h2)
    print(f"h2={h2}, str(h2)={str_key}, type=str_key={type(str_key)}")
    if str_key in test_dict:
        print(f"  Value: {test_dict[str_key]}")
    else:
        print(f"  Key not found")

print("\nTesting the actual problem:")
# Simulate the issue
profiles = {
    "AAPL": {
        "1": {"verdict": "REGIME_STABLE", "medians": [0.1, 0.2, 0.3]},
        "2": {"verdict": "REGIME_STABLE", "medians": [0.1, 0.2, 0.3]},
        "3": {"verdict": "REGIME_STABLE", "medians": [0.1, 0.2, 0.3]},
        "5": {"verdict": "REGIME_STABLE", "medians": [0.1, 0.2, 0.3]},
    }
}

ticker = "AAPL"
results = []
for h2 in [1, 2, 3, 5]:
    str_h2 = str(h2)
    if str_h2 in profiles[ticker]:
        condition = (
            profiles[ticker][str_h2]["verdict"] == "REGIME_STABLE"
            and all(m > 0 for m in profiles[ticker][str_h2]["medians"])
        )
        print(f"h2={h2}, str_h2={str_h2}, condition={condition}, type={type(condition)}")
        results.append(condition)

print(f"\nResults: {results}")
print(f"Sum of results: {sum(results)}")
print(f"Number of True results: {len([r for r in results if r])}")