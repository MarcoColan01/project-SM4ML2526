from itertools import product
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from . import adaboost
from .config import SEED, T_MAX, TREE_GRID, LR_GRID
from .data import stratified_kfold
from .evaluation import error_rate

def standardize(X_fit, X_other):
    mu, sd = X_fit.mean(axis=0), X_fit.std(axis=0)
    sd[sd==0] = 1.0
    return (X_fit-mu)/sd, (X_other - mu)/sd

def make_tree(**kwargs):
    return DecisionTreeClassifier(criterion="gini", random_state=SEED, **kwargs)

def make_logregr(**kwargs):
    return LogisticRegression(max_iter=10_000, **kwargs)

def adaboost_kCV(X,y, folds, T=T_MAX):
    curves = np.array([adaboost.staged_errors(adaboost.fit(X[tr], y[tr], T), X[va], y[va], T)
                       for tr, va in folds])
    return int(np.argmin(curves.mean(axis=0))) +1, curves

def cv_grid(make_model, grid, X, y, folds, scale=False):
    results = []
    for values in product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        errors = []
        for tr, va in folds:
            X_tr, X_va = standardize(X[tr], X[va]) if scale else (X[tr], X[va])
            errors.append(error_rate(y[va], make_model(**params).fit(X_tr, y[tr]).predict(X_va)))
        results.append(dict(params=params, mean=np.mean(errors), std=np.std(errors)))
    return min(results, key=lambda r: r["mean"])["params"], results

def tune(split, seed=SEED):
    X,y = split["X_train"], split["y_train"]
    folds = stratified_kfold(y, seed=seed)
    T_star, ada_curves = adaboost_kCV(X, y, folds)
    tree, tree_res = cv_grid(make_tree, TREE_GRID, X, y, folds)
    logregr, logregr_res = cv_grid(make_logregr, LR_GRID, X, y, folds, scale=True)
    return dict(T=T_star, tree=tree, lr=logregr), dict(adaboost=ada_curves, tree=tree_res, lr=logregr_res)

def fit_final(split, params):
    X,y = split["X_train"], split["y_train"]
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd==0] = 1.0
    ada = adaboost.fit(X,y, params["T"])
    tree = make_tree(**params["tree"]).fit(X,y)
    logregr = make_logregr(**params["lr"]).fit((X-mu)/sd, y)

    return {"stump": lambda Z: adaboost.predict(ada,Z,1),
            "adaboost": lambda Z: adaboost.predict(ada,Z),
            "tree": tree.predict,
             "lr": lambda Z: logregr.predict((Z-mu) / sd)}, ada