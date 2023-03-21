import pytest
from distrax import Bernoulli, Normal
from bnp_ml.jax_wrapper import class_wrapper
from bnp_ml.fisher_plot import fisher_table
import jax
import numpy as np


def estimate_p(dist, X):
    return dist.__class__(X.sum()/X.size)


def estimate_normal(dist, X):
    mu = np.sum(X, axis=0)/np.array(X.shape)
    sigma = np.sum((X-mu)**2, axis=0)/np.array(X.shape)
    return dist.__class__(mu, sigma)


@pytest.fixture
def bernoulli():
    return class_wrapper(Bernoulli, ['probs'], jax.random.PRNGKey(123))(0.4)


@pytest.fixture
def normal():
    return class_wrapper(Normal, ['loc', 'scale'], jax.random.PRNGKey(123))(0.0, 1.0)


def test_fisher_xy(bernoulli):
    table = fisher_table(bernoulli, estimate_p, sample_sizes=np.arange(1, 5))
    assert len(table['sample_size']) == 4
    assert len(table['z_score']) == 4


# @pytest.mark.xfail
def test_fisher_normal(normal):
    table = fisher_table(normal, estimate_normal, sample_sizes=np.arange(1, 5))
    assert len(table['sample_size']) == 8
    assert len(table['z_score']) == 8

