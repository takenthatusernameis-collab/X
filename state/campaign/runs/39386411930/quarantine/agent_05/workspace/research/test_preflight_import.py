import sys
sys.path.insert(0, 'research/data')
print(f"Path: {sys.path[0]}")

try:
    import preflight
    print('Import successful')
except Exception as e:
    print(f'Import failed: {e}')
    import traceback
    traceback.print_exc()