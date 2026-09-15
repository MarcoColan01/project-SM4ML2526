from urllib.parse import unquote
from sklearn.datasets import fetch_openml
from functools import lru_cache

import numpy as np 
from sklearn.model_selection import train_test_split
from .config import DATA_DIR, SEED, TEST_SIZE, K_FOLDS, N_SYNTH, ETA, THETA, RADIUS



@lru_cache
def load_spambase(log=True):
    ds = fetch_openml(data_id=44, data_home=DATA_DIR, as_frame=False)   
    raw = np.column_stack([ds.data, ds.target.astype(float)])
    _, first = np.unique(raw, axis=0, return_index=True)
    data = raw[np.sort(first)]                      
    print(f"Spambase: {len(raw)} rows, {len(raw) - len(data)} duplicates removed")
    X, y = data[:, :-1], np.where(data[:, -1] == 1, 1, -1)
    if log:
        X = np.log1p(X)
    names = [unquote(n) for n in ds.feature_names]  
    return X, y, names

def boundary_score(X, kind):
    if kind == "oblique":
        return X[:,1] - np.tan(THETA) * X[:,0]
    if kind == "circles":
        return RADIUS**2 - np.sum(X**2, axis=1)
    raise ValueError(f"unknown dataset: {kind}")

def make_synthetic(kind, n = N_SYNTH, eta=ETA, seed=SEED):
    rng = np.random.default_rng([seed, 0])
    X = rng.uniform(-1,1,size=(n,2))
    y_clean= np.where(boundary_score(X, kind) > 0, 1, -1)
    y = np.where(rng.random(n) < eta, -y_clean, y_clean)
    return X, y, y_clean

def load_synthetic(kind, n = N_SYNTH, eta=ETA, seed=SEED):
    path = DATA_DIR / f"{kind}_n{n}_eta{eta:.2f}_seed{seed}.npz"
    if not path.exists():
        X, y, y_clean = make_synthetic(kind, n, eta, seed)
        np.savez(path, X=X, y=y, y_clean=y_clean)
    d = np.load(path)
    return d["X"], d["y"], d["y_clean"]

def get_split(name, seed=SEED):
    if name == "spambase":
        X, y, names = load_spambase()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=seed)
        return dict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, names=names)

    X, y, y_clean = load_synthetic(name, seed=seed)
    X_train, X_test, y_train, y_test, yclean_train, yclean_test = train_test_split(
        X, y, y_clean, test_size=TEST_SIZE, stratify=y, random_state=seed)
    return dict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                yclean_train=yclean_train, yclean_test=yclean_test, names=["x1", "x2"])

def stratified_kfold(y, k=K_FOLDS, seed=SEED):
    rng = np.random.default_rng([seed, 1])
    parts = [np.array_split(rng.permutation(np.flatnonzero(y == c)), k) for c in np.unique(y)]
    val = [np.concatenate([p[f] for p in parts]) for f in range(k)]
    return [(np.concatenate(val[:f] + val[f + 1:]), val[f]) for f in range(k)]