import numpy as np
from .config import T_MAX, TOL

def sgn(z):
    return np.where(z>0, 1,-1)

def stump_predict(X,j,tau,s):
    return s * sgn(X[:,j] - tau)

def presort(X):
    order = np.argsort(X, axis=0, kind="stable")
    Xs = np.take_along_axis(X, order, axis=0)
    valid = np.vstack([np.ones(X.shape[1], bool), Xs[1:] > Xs[:-1]])
    return Xs, order, valid

def best_stump(Xs, order, valid, p, y):
    C = np.cumsum((p*y)[order], axis=0)
    err_pos = p[y==-1].sum() + np.vstack([np.zeros(Xs.shape[1]), C[:-1]])
    gain = np.where(valid, np.abs(err_pos -0.5), -1.0)
    k,j = np.unravel_index(np.argmax(gain), gain.shape)
    tau = -np.inf if k == 0 else (Xs[k-1,j] + Xs[k,j]) /2
    s = 1 if err_pos[k,j] <= 0.5 else -1
    return j,tau,s, 0.5-gain[k,j]

def fit(X,y,T=T_MAX, store_weights=False):
    Xs, order, valid = presort(X)
    p = np.full(len(y), 1/len(y))
    hist = dict(feature=[], threshold=[], polarity=[], alpha=[], eps=[], weights=[p.copy()])
    for _ in range(T):
        j,tau,s,eps = best_stump(Xs, order, valid, p,y)
        if eps >= 0.5 - TOL:
            break
        if eps <= TOL:
            hist.update(feature=[j], threshold=[tau], polarity=[s], alpha=[1.0], eps=[eps])
            break
        beta = eps / (1-eps)
        beta = eps / (1 - eps)
        hist["feature"].append(j)
        hist["threshold"].append(tau)
        hist["polarity"].append(s)
        hist["alpha"].append(np.log(1 / beta))
        hist["eps"].append(eps)
        p *= np.where(stump_predict(X,j,tau,s) == y, beta, 1.0)
        p /= p.sum()
        if store_weights:
            hist["weights"].append(p.copy())
    if not store_weights:
        hist.pop("weights")
    return {key: np.array(val) for key, val in hist.items()}

def staged_scores(model, X):
    H = [a * stump_predict(X,j,tau,s) for j, tau, s, a in zip(model["feature"], model["threshold"], model["polarity"], model["alpha"])]
    return np.cumsum(H, axis=0)

def staged_errors(model, X, y, T=None):
    err = np.mean(sgn(staged_scores(model, X)) != y, axis=1)
    return err if T is None else np.pad(err, (0, T - len(err)), mode="edge")

def predict(model, X, T=None):
    return sgn(staged_scores(model, X)[(T or len(model["alpha"])) -1])
