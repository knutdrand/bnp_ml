# from .probability_fixtures import normal
from bnp_ml.probability.model_wrap import wrap_model_func
from bnp_ml.probability.events import Normal, Categorical
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
def mixture_func():
    return lambda ps, mus: Normal(mus, [1.0])[Categorical(probs=ps)]


@pytest.fixture
def mixture_model(mixture_func):
    model_class = wrap_model_func(mixture_func)
    return model_class(np.arange(3).astype(float), np.array([0.1, 0.5, 0.4]))


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


def test_fisher_information_mix(mixture_model):
    estimate_fisher_information(mixture_model, n=3, rng=10)


def test_backfroth_mix(mixture_model):
    X = mixture_model.sample(10, (3, 2))
    probs = mixture_model.log_prob(X)
    assert probs.shape == (3, 2)


def test_optimizer(standard_normal, normal_model):
    X = standard_normal.sample(10, (200, ))
    dist = estimate_sgd(normal_model(1., 0.2), X, n_iterations=1000)
    assert np.abs(dist.mu-np.mean(X)) < 0.01


def test_fisher_table(standard_normal):
    fisher_table(standard_normal, fast_sgd, sample_sizes=np.arange(1, 5), n_fisher=10, rng=10)

