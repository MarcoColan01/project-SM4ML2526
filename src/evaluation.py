import numpy as np

def error_rate(y_true, y_pred):
    return np.mean(np.asarray(y_true) != np.asarray(y_pred))