import pytest
from numpy.testing import assert_approx_equal
from bnp_ml.signal_distribution import SignalModel
from collections import Counter
import numpy as np

max_fragment_length = 4


@pytest.fixture
def signal_model():
    return SignalModel(np.full(3, 1/3), np.arange(max_fragment_length+1)/10, 0.5)


def test_backgroun_prob(signal_model):
    assert_approx_equal(sum(signal_model.background_prob for i in range(signal_model.area_size)),
                        0.5*0.5)


def test_probability(signal_model):
    pos_prob = sum(signal_model.probability(i, '+') for i in range(signal_model.area_size))
    assert_approx_equal(pos_prob,
                        0.5)
    neg_prob = sum(signal_model.probability(i+max_fragment_length-1, '-') for i in range(signal_model.area_size))
    assert_approx_equal(pos_prob,
                        0.5)


def test_simulate(signal_model):
    rng = np.random.default_rng()
    counter = Counter(signal_model.simulate(rng) for _ in range(100))
    s = sum(counter.values())
    for key in sorted(counter.keys()):
        print(key, counter[key]/s, signal_model.probability(*key))
