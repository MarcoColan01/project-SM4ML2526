from dataclasses import dataclass
import numpy as np
import config

W_DIAG = np.array([1.0,1.0]) / np.sqrt(2.0)

@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    y_clean: np.ndarray
    flipped: np.ndarray
    name: str

    def __len__(self):
        return len(self.y)

    @property
    def d(self):
        return self.X.shape[1]


def _moons(m, sigma, rng):
    y = rng.choice([-1.0, 1.0], size=m)
    theta = rng.uniform(0.0, np.pi, size=m)
    X = np.empty((m,2))
    up = y == 1.0
    X[up, 0] = np.cos(theta[up])
    X[up, 1] = np.sin(theta[up])
    X[~up, 0] = 1.0 - np.cos(theta[~up])
    X[~up, 1] = 0.5 - np.sin(theta[~up])

    return X + sigma * rng.normal(size=X.shape),y

def _diagonal(m, gamma, rng):
    kept = []
    total = 0
    while total < m:
        X = rng.uniform(-1.0, 1.0, size=(2 * m, 2))
        block = X[np.abs(X @ W_DIAG) >= gamma]
        kept.append(block)
        total += len(block)

    X = np.vstack(kept)[:m]
    return X, np.sign(X @ W_DIAG)

def _flip(y, eta, rng):
    mask = rng.random(len(y)) < eta
    return np.where(mask, -y, y), mask

def make_synthetic(kind, m, seed, eta=0.0, **kwargs):
    rng = np.random.default_rng(seed)
    if kind == "moons":
        X, y = _moons(m, kwargs.get("sigma", config.MOONS["sigma"]), rng)
    elif kind == "diagonal":
        X, y = _diagonal(m, kwargs.get("gamma", config.DIAGONAL["gamma"]), rng)
    else:
        raise ValueError(f"unknown synthetic dataset: {kind!r}")

    y_noisy, mask = _flip(y, eta, rng)

    return Dataset(X, y_noisy, y, mask, f"{kind}(eta={eta:g})")

def make_synthetic_split(kind, seed, eta=0.0,**kwargs):
    cfg = config.MOONS if kind == "moons" else config.DIAGONAL
    child = np.random.SeedSequence(int(seed)).spawn(2)
    train = make_synthetic(kind, cfg["m_train"], child[0], eta, **kwargs)
    test = make_synthetic(kind, cfg["m_test"], child[1], eta, **kwargs)

    return train, test

def _moon_densities(X, sigma, n_theta=400):
    th = (np.arange(n_theta) + 0.5) * np.pi / n_theta
    mus = (
        np.stack([np.cos(th), np.sin(th)], axis=1),
        np.stack([1.0 - np.cos(th), 0.5 - np.sin(th)], axis=1),
    )
    out = []
    for mu in mus:
        d2 = (
            (X**2).sum(1)[:, None] - 2.0 * X @ mu.T + (mu**2).sum(1)[None, :]
        )
        out.append(np.exp(-d2 / (2.0 * sigma**2)).mean(1) / (2.0 * np.pi * sigma**2))

    return out

def bayes_risk_moons(sigma, eta=0.0, n_mc=20_000, seed=0):
    X, _ = _moons(n_mc, sigma, np.random.default_rng(seed))
    p_pos, p_neg = _moon_densities(X, sigma)
    r0 = float(np.mean(np.minimum(p_pos, p_neg) / (p_pos + p_neg)))

    return eta + (1.0 - 2.0 * eta) * r0, r0

def load_spambase():
    cache = config.DATA_RAW / "spambase.npz"
    if cache.exists():
        blob = np.load(cache)
        X, y = blob["X"], blob["y"]
    else:
        from sklearn.datasets import fetch_openml

        raw = fetch_openml("spambase", version=1, as_frame=False)
        X = np.asarray(raw.data, dtype=float)
        y = np.where(np.asarray(raw.target).astype(int) == 1, 1.0, -1.0)
        np.savez_compressed(cache, X=X, y=y)

    if X.shape != config.SPAMBASE_SHAPE:
        raise RuntimeError(
            f"expected Spambase with shape {config.SPAMBASE_SHAPE}, got {X.shape}; "
            "the OpenML version may have changed"
        )

    return Dataset(X, y, y, np.zeros(len(y), dtype=bool), "spambase")

