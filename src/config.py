from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "report" / "Figures"

for dir in (DATA_DIR, RESULTS_DIR, FIG_DIR):
    dir.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(10))
SEED = SEEDS[0]

DATASETS = ("spambase", "oblique", "circles")
TEST_SIZE = 0.2
K_FOLDS = 5

N_SYNTH = 1250
ETA = 0.10
THETA = np.deg2rad(30)
RADIUS = np.sqrt(2 / np.pi)

T_MAX = 500
TOL = 1e-10

TREE_GRID = {"max_depth": [2,3,4,5,6,8,10,15,None],
             "min_samples_leaf": [1,5,10,20]}
LR_GRID = {"C": [10.0**p for p in range(-3,4)]}