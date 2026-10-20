import sys
from pathlib import Path

# Simulate what the quarantine script does
script_path = Path('/home/runner/work/X/X/state/campaign/runs/39386411930/quarantine/agent_05/workspace/research/test_import_from_workspace.py')
print('Script path:', script_path)
print('Script dir:', script_path.parent)
print('Script dir parent:', script_path.parent.parent)

sys.path.insert(0, str(script_path.parent.parent))
print('Added to sys.path:', sys.path[0])

# Check what's in the path
print('Contents of path:')
for item in Path(sys.path[0]).iterdir():
    if item.is_dir():
        print(f'  DIR: {item.name}')
        # Check if it's research
        if item.name == 'research':
            research_path = item
            print(f'  Found research directory at: {research_path}')
            if (research_path / '__init__.py').exists():
                print(f'  research has __init__.py')
            else:
                print(f'  research does NOT have __init__.py')
    else:
        print(f'  FILE: {item.name}')

# Try to import
try:
    import research
    print('\nImport successful!')
    print(f'research.__file__ = {research.__file__}')
    
    import research.backtest as bt
    print(f'research.backtest.__file__ = {bt.__file__}')
    
except Exception as e:
    print(f'\nImport failed: {e}')
    import traceback
    traceback.print_exc()