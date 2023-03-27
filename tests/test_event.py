import numpy as np
from bnp_ml.events import (DictRandomVariable, Event, Probability,
                           P, Bernoulli, Beta, scipy_stats_wrapper,
                           Normal, Categorical)

from bnp_ml.pyprob.regression import linear_regression_model
import scipy.stats
import pytest
from numpy.testing import assert_allclose


@pytest.fixture
def dice():
    return DictRandomVariable({i: Probability(1/6) for i in range(1, 7)})


@pytest.fixture
def rng():
    return np.random.default_rng()


@pytest.fixture
def bernoulli():
    return Bernoulli(0.3)


@pytest.fixture
def normals():
    return Normal([0, 1, 2.], [1.0])

@pytest.fixture
def normal():
    return Normal([1], [1.0])


@pytest.fixture
def beta():
    return Beta(1.0, 1.0)


@pytest.fixture
def ps():
    return np.array([0.1, 0.5, 0.4])


@pytest.fixture
def categorical(ps):
    return Categorical(probs=ps)


@pytest.fixture
def coin():
    return DictRandomVariable({i: Probability(1/2) for i in ('H', 'T')})


@pytest.fixture
def dice_2(dice):
    return Event(dice, 2)


@pytest.fixture
def dice_3(dice):
    return Event(dice, 3)


@pytest.fixture
def coin_heads(coin):
    return Event(coin, 'H')


def test_random_variable(dice):
    assert P(dice == 2).equals(1/6)


def test_event_or(dice_2, dice_3):
    assert P(dice_2 | dice_3).equals(2/6)


def test_event_and(dice_2, coin_heads):
    assert P(dice_2 & coin_heads).equals(1/12)


def test_event_not(dice_2):
    assert P(~dice_2).equals(5/6)


def test_bernoulli(bernoulli):
    assert P(bernoulli == True).equals(0.3)


def test_beta(beta):
    assert P(beta == 0.5).equals(1.0)


def test_beta_sample(beta, rng):
    assert beta.sample(rng, (3, 2)).shape == (3, 2)


def test_beta_wrapper():
    beta = scipy_stats_wrapper(scipy.stats.beta)(1.0, 1.0)
    assert P(beta == 0.5).equals(1.0)


def test_beta_wrapper_sample(rng):
    beta = scipy_stats_wrapper(scipy.stats.beta)(1.0, 1.0)
    assert beta.sample(rng, (3, 2)).shape == (3, 2)


def test_index_model(normals, normal):
    assert P(normals[1] == 2.).prob() == P(normal == 2.).prob()


def test_index_model_sample(normals, rng):
    assert normals[1].sample(rng, (4, 5)).shape == (4, 5)
    # assert P(normals[1] == 2.).prob() == P(normal == 2.).prob()


def test_mixture_model(normals, categorical, ps):
    Z = categorical
    X = normals[Z]
    prob = P(X == 2.).prob()
    true = sum(p1*p2 for p1, p2 in zip(ps, P(normals == 2.).prob()))
    assert_allclose(prob, true)


def test_mixture_model_sample(normals, categorical):
    Z = categorical
    X = normals[Z]
    assert X.sample(10, (4, 5)).shape == (4, 5)
