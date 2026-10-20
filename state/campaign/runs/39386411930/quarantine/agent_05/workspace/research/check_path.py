from pathlib import Path

# Check the path manipulation in the script
script_path = Path("/home/runner/work/X/X/state/campaign/runs/39386411930/quarantine/agent_05/workspace/research/checks/cost_sensitive_momentum.py")
added_path = script_path.parent.parent.parent

print(f"Script path: {script_path}")
print(f"Added to sys.path: {added_path}")
print(f"Exists: {added_path.exists()}")

# Check what's in the added path
import os
for item in os.listdir(added_path):
    item_path = os.path.join(added_path, item)
    print(f"  {item} (exists: {os.path.exists(item_path)})")
    
# Check the research subdirectory
research_subdir = added_path / "research"
print(f"\nResearch subdir path: {research_subdir}")
print(f"Exists: {research_subdir.exists()}")
print(f"Has __init__.py: {(research_subdir / '__init__.py').exists()}")

# Check the nested research directory
workspace_research = added_path / "research" / "research"
print(f"\nWorkspace research path: {workspace_research}")
print(f"Exists: {workspace_research.exists()}")
print(f"Has __init__.py: {(workspace_research / '__init__.py').exists()}")