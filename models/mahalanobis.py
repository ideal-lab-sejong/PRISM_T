import numpy as np


def fit_mahal(F, lam_rel=1e-3, scale_aware=False):
    mu = F.mean(0)
    Fc = F - mu
    Sigma = (Fc.T @ Fc) / max(len(F) - 1, 1)
    d = F.shape[1]
    lam = lam_rel * (np.trace(Sigma) / d + 1e-12) if scale_aware else lam_rel
    return mu, np.linalg.inv(Sigma + lam * np.eye(d))


def score_mahal(F, mu, prec):
    diff = F - mu
    M = diff @ prec
    return np.sqrt(np.maximum((M * diff).sum(1), 0.0))
