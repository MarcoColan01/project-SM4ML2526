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

DATASETS = ("spambase", "oblique", "circles", "xor")
TEST_SIZE = 0.2
K_FOLDS = 5

N_SYNTH = 1250
ETA = 0.10
THETA = np.deg2rad(30)
R_IN, R_OUT = 0.5, 1.0
SIGMA = 0.15

T_MAX = 500
TOL = 1e-10

TREE_GRID = {"max_depth": [2,3,4,5,6,8,10,15,None],
             "min_samples_leaf": [1,5,10,20]}
LR_GRID = {"C": [10.0**p for p in range(-3,4)]}

NAMES = {"spambase": "Spambase", "oblique": "Oblique", "circles": "Circles", "xor": "XOR"}
MODEL_NAMES = {"adaboost": "AdaBoost", "tree": "Decision tree", "lr": "Logistic regression"}
LIM = {"oblique": 3.0, "circles": 1.7, "xor": 3.0}
CLASS_STYLE = {1: dict(c="tab:blue", marker="o", label="$y = +1$"),
               -1: dict(c="tab:orange", marker="x", label="$y = -1$")}