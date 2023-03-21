import pytest
from distrax import Bernoulli
from bnp_ml.jax_wrapper import class_wrapper
from bnp_ml.fisher_plot import fisher_table
import numpy as np


def estimate_p(dist, X):
    return dist.__class__(X.sum()/X.size)


@pytest.fixture
def bernoulli():
    class_wrapper(Bernoulli)(0.4)


def test_fisher_xy(bernoulli):
    table = fisher_table(bernoulli, estimate_p, sample_sizes=np.arange(1, 4))
    assert table['sample_size'].shape == (4, )
    assert table['z_scores'].shape == (4, )
