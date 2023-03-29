import numpy as np
from bnp_ml.probability.events import (DictRandomVariable, Event, Probability,
                                       P, Bernoulli, Beta, scipy_stats_wrapper,
                                       Normal, Categorical, Geometric)
# from bnp_ml.pyprob.regression import linear_regression_model
import scipy.stats
import pytest
from numpy.testing import assert_allclose
from .probability_fixtures import *


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


def test_index_model_sample(normals):
    assert normals[1].sample(10, (4, 5)).shape == (4, 5)
    # assert P(normals[1] == 2.).prob() == P(normal == 2.).prob()


def test_mixture_model(normals, categorical, ps):
    Z = categorical
    X = normals[Z]
    prob = P(X == 2.).prob()
    true = sum(p1*p2 for p1, p2 in zip(ps, P(normals == 2.).prob()))
    assert_allclose(prob, true)


def test_mixture_model_2(normals, categorical, ps):
    Z = categorical
    X = normals[Z]
    x = np.array([2.0, 1.0])
    prob = P(X == x).prob()
    true = np.sum(ps*P(normals==x[:, np.newaxis]).prob(), axis=-1)
    # true = sum(p1*p2 for p1, p2 in zip(ps, P(normals == x[:, np.newaxis]).prob()))
    assert_allclose(prob, true)


def test_mixture_model_sample(normals, categorical):
    Z = categorical
    X = normals[Z]
    assert X.sample(10, (4, 5)).shape == (4, 5)


def test_add_scalar(normal):
    X = normal
    Z = X+2
    assert P(X == 2.0).prob() == P(Z == 4.0).prob()


def test_add_categorical(normal, categorical, ps):
    Z = normal+categorical
    true = sum(P(normal==2.0-i).prob()*p for i, p in enumerate(ps))
    assert P(Z == 2.0).prob() == true


def test_given(dice):
    assert P(dice == 2, given= (dice == 2) | (dice == 3))


def test_le(categorical):
    assert P(categorical < 2).prob() == 0.6


@pytest.mark.xfail
def test_signal_model(categorical, geometric):
    Z = categorical + geometric
    P(Z == z, given=Z < 4)

    
