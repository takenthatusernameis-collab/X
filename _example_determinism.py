import subprocess
out1 = subprocess.run(["python3", "/home/runner/work/X/X/examples/ma_crossover.py"],
                      capture_output=True, text=True).stdout
out2 = subprocess.run(["python3", "/home/runner/work/X/X/examples/ma_crossover.py"],
                      capture_output=True, text=True).stdout
print("run1 == run2:", out1 == out2)
if out1 != out2:
    for l1, l2 in zip(out1.splitlines(), out2.splitlines()):
        if l1 != l2:
            print("DIFF:", l1, "|", l2)
