import subprocess, hashlib, sys
sys.path.insert(0, "/home/runner/work/X/X")

def run_and_hash(args):
    result = subprocess.run(args, capture_output=True, text=True, cwd="/home/runner/work/X/X")
    out = result.stdout
    return hashlib.sha256(out.encode()).hexdigest(), out, result.returncode

files = [
    ["python3", "-m", "examples.ma_crossover_real_data"],
    ["python3", "-m", "examples.regime_stability_real_data"],
    ["python3", "-m", "examples.ma_crossover"],
    ["python3", "-m", "examples.volatility_regime_filter"],
    ["python3", "-m", "examples.regime_stability_demo"],
    ["python3", "-m", "examples.universe_sweep"],
]

for i, args in enumerate(files):
    h, out, rc = run_and_hash(args)
    print(f"run {i+1}: rc={rc} sha256={h[:16]}...")
    if rc != 0:
        print("  STDERR:", out[:500])

# Second pass: compare byte-identical across independent runs
second = [run_and_hash(f)[0] for f in files]
all_ids = True
for idx, f in enumerate(files):
    first = run_and_hash(f)[0]
    same = first == second[idx]
    all_ids = all_ids and same
    print(f"{f[2]}: {'IDENTICAL' if same else 'DIFFERS'}")
print("ALL_DETERMINISTIC:", all_ids)
