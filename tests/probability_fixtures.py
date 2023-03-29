from bnp_ml.probability.events import (DictRandomVariable, Event, Probability,
                                       P, Bernoulli, Beta, scipy_stats_wrapper,
                                       Normal, Categorical, Geometric)
import numpy as np
import pytest


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
def geometric():
    return Geometric(0.4)


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
