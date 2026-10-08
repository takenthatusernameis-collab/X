import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
print("sys.path:", sys.path)
try:
    import research
    print("research module found")
except ImportError as e:
    print("ImportError:", e)
