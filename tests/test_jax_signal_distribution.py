import pytest
import torch
from numpy.testing import assert_approx_equal, assert_allclose
from bnp_ml.jax_signal_model import JaxSignalModel, MultiNomialReparametrization, NaturalSignalModel, NaturalSignalModelGeometricLength
from bnp_ml.jax_wrapper import estimate_fisher_information, estimate_sgd
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


@pytest.fixture
def natural_signal_model():
    return NaturalSignalModel(MultiNomialReparametrization.to_natural(np.full(3, 1/3)),
                              np.arange(max_fragment_length+1)/10)


@pytest.fixture
def natural_signal_model_geometric_length():
    return NaturalSignalModelGeometricLength(MultiNomialReparametrization.to_natural(np.full(3, 1/3)),
                                             np.log(0.4))


@pytest.fixture
def big_probs():
    return np.arange(1, 11)/55


@pytest.fixture
def signal_model_big(big_probs):
    return JaxSignalModel(big_probs, np.arange(max_fragment_length+1)/10)


@pytest.fixture
def simulated_data(signal_model_big, rng):
    return signal_model_big.sample(rng, (100, ))


def assert_prob_is_one(model):
    domain = list(sorted(model.domain()))
    print(domain)
    sum_prob = sum(model.probability(x) for x in domain)
    assert_approx_equal(sum_prob, 1)


def test_probability(signal_model: JaxSignalModel):
    assert_prob_is_one(signal_model)


def test_probability_natural(natural_signal_model: JaxSignalModel):
    assert_prob_is_one(natural_signal_model)


def test_probability_natural_geo(natural_signal_model_geometric_length: JaxSignalModel):
    assert_prob_is_one(natural_signal_model_geometric_length)


def test_probability_big(signal_model_big: JaxSignalModel):
    assert_prob_is_one(signal_model_big)

    # # domain_size = len(signal_model.binding_affinity) + max_fragment_length-1
    # domain = list(sorted(signal_model.domain()))
    # print(domain)
    # sum_prob = sum(signal_model.probability(x) for x in domain)
    # assert_approx_equal(sum_prob, 1)
    # pos_probs = [signal_model.probability((i, '+')) for i in range(domain_size)]
    # pos_prob = sum(pos_probs)
    # print(np.array(pos_probs))
    # assert_approx_equal(pos_prob, 0.5)
    # neg_prob = sum(signal_model.probability((i+max_fragment_length-1, '-')) for i in range(domain_size))
    # assert_approx_equal(neg_prob, 0.5)


def test_simulate(signal_model):
    assert_sample_logprob_fit(signal_model)


def test_simulate_big(signal_model_big):
    assert_sample_logprob_fit(signal_model_big)


def test_simulate_natural(natural_signal_model: JaxSignalModel):
    assert_sample_logprob_fit(natural_signal_model)


def test_simulate_natural_geom(natural_signal_model_geometric_length: JaxSignalModel):
    assert_sample_logprob_fit(natural_signal_model_geometric_length)


def test_estimation(signal_model_big, simulated_data):
    model = signal_model_big.__class__(
        np.full_like(
            signal_model_big.binding_affinity,
            1/len(signal_model_big.binding_affinity)),
        np.full_like(
            signal_model_big.fragment_length_distribution,
            1/len(signal_model_big.fragment_length_distribution)))

    estimate_sgd(model, simulated_data, n_iterations=5)


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


def test_multinomial_reparametrization(big_probs):
    reparam = MultiNomialReparametrization
    roundtrip = np.array(reparam.from_natural(reparam.to_natural(big_probs)))
    print(roundtrip)
    print(big_probs)
    assert_allclose(roundtrip, big_probs, rtol=6)
    
