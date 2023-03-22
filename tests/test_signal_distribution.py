import pytest
import torch
from numpy.testing import assert_approx_equal
from bnp_ml.signal_distribution import SignalModel
from bnp_ml.ml_check import estimate_fisher_information
from collections import Counter
from logarray.logarray import log_array

import numpy as np

max_fragment_length = 4


@pytest.fixture
def signal_model():
    return SignalModel(log_array(np.full(3, 1/3)), log_array(np.arange(max_fragment_length+1)/10), log_array(0.5))


@pytest.fixture
def torch_model():
    return SignalModel(*(torch.tensor(a) for a in (
        np.full(3, 1/3),
        np.arange(max_fragment_length+1)/10,
        0.5)))


def _test_backgroun_prob(signal_model):
    assert_approx_equal(sum(signal_model.background_prob for i in range(signal_model._area_size)),
                        0.5*0.5)


@pytest.mark.xfail
def test_probability(signal_model):
    pos_prob = sum(signal_model.probability(i, '+') for i in range(signal_model._area_size))
    print(pos_prob)
    assert_approx_equal(pos_prob.to_array(),
                        0.5)
    neg_prob = sum(signal_model.probability(i+max_fragment_length-1, '-') for i in range(signal_model._area_size))
    assert_approx_equal(pos_prob.to_array(),
                        0.5)


@pytest.mark.xfail
def test_simulate(signal_model):
    rng = np.random.default_rng()
    counter = Counter(signal_model.simulate(rng) for _ in range(100))
    s = sum(counter.values())
    for key in sorted(counter.keys()):
        print(key, counter[key]/s, signal_model.probability(*key))


@pytest.mark.xfail
def test_back_forth(signal_model):
    signal_model.log_prob(signal_model.sample((10, )))


@pytest.mark.xfail
def test_fisher_information(torch_model):
    estimate_fisher_information(torch_model, 100)
