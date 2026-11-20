import hashlib, sys
p = sys.argv[1]
print(hashlib.sha256(open(p, "rb").read()).hexdigest())