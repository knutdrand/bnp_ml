import pytest
# from distrax import Bernoulli, Normal
from bnp_ml.jax_wrapper import class_wrapper, mixture_class
from bnp_ml.fisher_plot import fisher_table
from bnp_ml.distributions import Normal, Bernoulli, MultiVariateNormalDiag, MixtureOfTwo
from bnp_ml.estimators import estimate_p, estimate_normal, estimate_mixed_normal
import jax
import numpy as np


@pytest.fixture
def bernoulli():
    return Bernoulli(0.4)
# return class_wrapper(Bernoulli, ['probs'], jax.random.PRNGKey(123))(0.4)


@pytest.fixture
def normal():
    return Normal(0.0, 1.0)


@pytest.fixture
def normal2():
    return Normal(2.0, 3.0)


@pytest.fixture
def multivariate_normal():
    return MultiVariateNormalDiag(np.array([1.0, 2.0]),
                                  np.array([2.0, 3.0]))


# @pytest.fixture
def mixed_normal():
    cls = mixture_class(Normal, Normal)
    return cls(0.5, 0.0, 1.0, 2.0, 3.0)


def test_fisher_xy(bernoulli):
    table = fisher_table(bernoulli, estimate_p, sample_sizes=np.arange(1, 5))
    assert len(table['sample_size']) == 4
    assert len(table['z_score']) == 4


# @pytest.mark.xfail
def test_fisher_normal(normal):
    table = fisher_table(normal, estimate_normal, sample_sizes=np.arange(1, 5))
    assert len(table['sample_size']) == 8
    assert len(table['z_score']) == 8


def test_fisher_multivariatenormal(multivariate_normal):
    table = fisher_table(multivariate_normal, estimate_normal, sample_sizes=np.arange(1, 5))
    assert len(table['sample_size']) == 16
    assert len(table['z_score']) == 16

@pytest.mark.xfail
def test_fisher_mixed(mixed_normal):
    table = fisher_table(multivariate_normal, estimate_mixed_normal, sample_sizes=np.arange(1, 5))    
