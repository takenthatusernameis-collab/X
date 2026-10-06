path = "/home/runner/work/X/X/research/checks/verify_cross_sectional_momentum.py"
lines = open(path).read().split("\n")
for n in range(336, 412):
    print("{:4d}: {}".format(n, lines[n-1]))
