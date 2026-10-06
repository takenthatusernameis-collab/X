from pathlib import Path
p = Path("/home/runner/work/X/X/state/check_artifacts/momentum_horizon_gap_analysis.json")
if p.exists():
    if p.is_dir():
        import shutil
        shutil.rmtree(p)
        print("removed directory:", p)
    else:
        p.unlink()
        print("removed file:", p)
else:
    print("does not exist")
