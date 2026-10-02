"""Utility: compare two captured runs for byte-identical determinism.

Usage: `python3 research/backtest/_determinism_check.py <file1> <file2>`
Prints sha256 of each and IDENTICAL / DIFFERENT, exiting 0/1.
"""
import hashlib
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if len(sys.argv) != 3:
        print("usage: _determinism_check.py <file1> <file2>", file=sys.stderr)
        sys.exit(2)
    a, b = sys.argv[1], sys.argv[2]
    ha, hb = sha256(a), sha256(b)
    print(f"{a}: {ha}")
    print(f"{b}: {hb}")
    same = ha == hb
    print("IDENTICAL" if same else "DIFFERENT")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
