import numpy as np 
from scipy.stats import spearmanr 

def spearmans_rho(y_true, y_pred):
    return np.array([
        spearmanr(y_true[:, i], y_pred[:, i])[0] 
        for i in range(y_true.shape[0])
    ]).mean()
