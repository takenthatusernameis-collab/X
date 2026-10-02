import warnings, sys
from pathlib import Path
warnings.filterwarnings('error', category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import examples.ma_crossover
examples.ma_crossover.main()
