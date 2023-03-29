# from .probability_fixtures import normal
from bnp_ml.probability.model_wrap import wrap_model_func
from bnp_ml.probability.events import Normal
from bnp_ml.jax_wrapper import estimate_fisher_information, estimate_sgd
from bnp_ml.fisher_plot import fisher_table
import numpy as np
import pytest
from functools import partial
fast_sgd = partial(estimate_sgd, n_iterations=5)


@pytest.fixture
def normal_func():
    return lambda mu, sigma: Normal(mu, sigma)


@pytest.fixture
def normal_model(normal_func):
    return wrap_model_func(normal_func)


@pytest.fixture
def standard_normal(normal_model):
    return normal_model(0., 1.)


def test_wrap_model(normal_func):
    model = wrap_model_func(normal_func)
    assert hasattr(model, 'parameters')
    assert hasattr(model, 'parameter_names')


def test_fisher_information(standard_normal):
    estimate_fisher_information(standard_normal, n=10, rng=10)


def test_optimizer(standard_normal, normal_model):
    X = standard_normal.sample(10, (200, ))
    dist = estimate_sgd(normal_model(1., 0.2), X, n_iterations=1000)
    assert np.abs(dist.mu-np.mean(X)) < 0.01


def test_fisher_table(standard_normal):
    fisher_table(standard_normal, fast_sgd, sample_sizes=np.arange(1, 5), n_fisher=10, rng=10)

