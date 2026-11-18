import hashlib
import sys

path = sys.argv[1]
with open(path, "rb") as f:
    data = f.read()
print(hashlib.sha256(data).hexdigest())