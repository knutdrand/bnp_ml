import numpy as np


def estimate_p(dist, X):
    return dist.__class__(X.sum()/X.size)


def estimate_normal(dist, X):
    mu = np.sum(X, axis=0)/len(X.shape)
    sigma = np.sum((X-mu)**2, axis=0)/len(X.shape)
    return dist.__class__(mu, sigma)


def estimate_mixed_normal(dist, X):
    return dist.__class__(*dist.parameters)
