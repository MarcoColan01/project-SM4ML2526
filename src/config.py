from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "report" / "Figures"

for _d in (DATA_RAW, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 20252026

def seeds(n, base=SEED):
    return np.random.SeedSequence(base).generate_state(n)

MOONS = dict(sigma=0.05, m_train=2000, m_test=10_000)
DIAGONAL = dict(gamma=0.12, m_train=2000, m_test=10_000)
ETAS = (0.0, 0.05, 0.10, 0.20)
N_REPS = 10

SPAMBASE_SHAPE = (4601, 57)
N_FOLDS = 5 

T_MAX = 1000

DT_DEPTHS = tuple(range(1,16))
LR_CS = tuple(10.0 ** np.linspace(-3.0, 3.0, 13))