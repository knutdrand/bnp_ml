import pytest
from distrax import Bernoulli
from bnp_ml.jax_wrapper import class_wrapper
from bnp_ml.fisher_plot import fisher_table
import jax
import numpy as np


def estimate_p(dist, X):
    return dist.__class__(X.sum()/X.size)


@pytest.fixture
def bernoulli():
    return class_wrapper(Bernoulli, ['probs'], jax.random.PRNGKey(123))(0.4)


def test_fisher_xy(bernoulli):
    table = fisher_table(bernoulli, estimate_p, sample_sizes=np.arange(1, 5))
    assert table['sample_size'].shape == (4, )
    assert table['z_score'].shape == (4, )
