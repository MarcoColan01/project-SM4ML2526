import numpy as np

def error_rate(y_true, y_pred):
    return np.mean(np.asarray(y_true) != np.asarray(y_pred))

def effective_size(W):
    return 1.0 / np.sum(W**2, axis=1) 

def weight_share(W, mask):
    return W[:, mask].sum(axis=1)

def feature_importance(model, n_features):
    return np.bincount(model["feature"], weights=model["alpha"], minlength=n_features)

def additive_component(model, j, grid):
    m = model["feature"] == j
    return np.sum(model["alpha"][m, None] * model["polarity"][m, None]
                  * np.where(grid - model["threshold"][m, None] > 0, 1, -1), axis=0)