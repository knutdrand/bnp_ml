import pytest
from numpy.testing import assert_approx_equal, assert_allclose
from bnp_ml.jax_signal_model import JaxSignalModel, MultiNomialReparametrization, NaturalSignalModel, NaturalSignalModelGeometricLength
from bnp_ml.jax_wrapper import estimate_fisher_information, estimate_sgd, init_like
from .goodness_test import assert_sample_logprob_fit
import numpy as np
from .jax_fixtures import *


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


@pytest.mark.parametrize('model', models)
def test_probs(model):
    assert_prob_is_one(model)


@pytest.mark.parametrize('model', models)
def test_sim(model):
    assert_sample_logprob_fit(model)


@pytest.mark.parametrize('model', models)
def test_fisher_information(model, rng):
    estimate_fisher_information(model, 10, rng)


@pytest.mark.parametrize('model', models)
def test_back_forth(model, rng):
    signal_model.log_prob(model.sample(rng, (10, )))


def test_simulate(signal_model):
    assert_sample_logprob_fit(signal_model)


def test_simulate_big(signal_model_big):
    assert_sample_logprob_fit(signal_model_big)


def test_simulate_natural(natural_signal_model: JaxSignalModel):
    assert_sample_logprob_fit(natural_signal_model)


def test_simulate_natural_geom(natural_signal_model_geometric_length: JaxSignalModel):
    assert_sample_logprob_fit(natural_signal_model_geometric_length, n_samples=100000)


@pytest.mark.parametrize('model', models)
def test_estimations(model, rng):
    simulated_data = model.sample(rng, (10,))
    init_model = init_like(model, rng)
    estimate_sgd(init_model, simulated_data, n_iterations=5)


def test_estimation(signal_model_big, simulated_data):
    model = signal_model_big.__class__(
        np.full_like(
            signal_model_big.binding_affinity,
            1/len(signal_model_big.binding_affinity)),
        np.full_like(
            signal_model_big.fragment_length_distribution,
            1/len(signal_model_big.fragment_length_distribution)))

    estimate_sgd(model, simulated_data, n_iterations=5)


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
    
