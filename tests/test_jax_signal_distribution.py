import pytest
import torch
from numpy.testing import assert_approx_equal
from bnp_ml.jax_signal_model import JaxSignalModel
from bnp_ml.jax_wrapper import estimate_fisher_information
from collections import Counter
from logarray.logarray import log_array
from .goodness_test import assert_sample_logprob_fit
import numpy as np

max_fragment_length = 4


@pytest.fixture
def rng():
    return np.random.default_rng()


@pytest.fixture
def signal_model():
    return JaxSignalModel(np.full(3, 1/3), np.arange(max_fragment_length+1)/10)


# @pytest.fixture
# def torch_model():
#     return SignalModel(*(torch.tensor(a) for a in (
#         np.full(3, 1/3),
#         np.arange(max_fragment_length+1)/10,
#         0.5)))


def _test_backgroun_prob(signal_model):
    assert_approx_equal(sum(signal_model.background_prob for i in range(signal_model._area_size)),
                        0.5*0.5)


@pytest.mark.xfail
def test_probability(signal_model):
    area_size = len(signal_model.binding_affinity)
    pos_prob = sum(signal_model.probability(i, '+') for i in range(area_size))
    assert_approx_equal(pos_prob.to_array(),
                        0.5)
    neg_prob = sum(signal_model.probability(i+max_fragment_length-1, '-') for i in range(signal_model._area_size))
    assert_approx_equal(pos_prob.to_array(),
                        0.5)


def test_simulate(signal_model):
    assert_sample_logprob_fit(signal_model)
    # rng = np.random.default_rng()
    # counter = Counter(signal_model.sample(rng, (10000,)))
    # s = sum(counter.values())
    # for key in sorted(counter.keys()):
    #     print(key, counter[key]/s, signal_model.probability(key))
    # assert False


def test_back_forth(signal_model, rng):
    signal_model.log_prob(signal_model.sample(rng, (10, )))


def test_fisher_information(signal_model, rng):
    estimate_fisher_information(signal_model, 100, rng)
