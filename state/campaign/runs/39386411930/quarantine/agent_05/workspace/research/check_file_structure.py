from pathlib import Path

# Check the actual file structure
base_path = Path("/home/runner/work/X/X/state/campaign/runs/39386411930/quarantine/agent_05/workspace/research")
print(f"Base path: {base_path}")

# Check what's in the base path
print("\nContents of base path:")
for item in base_path.iterdir():
    print(f"  {item.name} (exists: {item.exists()})")
    
# Check the research subdirectory
research_dir = base_path / "research"
print(f"\nResearch subdir: {research_dir}")
print(f"Exists: {research_dir.exists()}")
if research_dir.exists():
    print(f"Has __init__.py: {(research_dir / '__init__.py').exists()}")
    
# Check if we can import directly
import sys
sys.path.insert(0, str(base_path))
try:
    import research
    print(f"\nCan import research: {research}")
    import research.backtest as bt
    print(f"Can import research.backtest: {bt}")
except Exception as e:
    print(f"Import failed: {e}")