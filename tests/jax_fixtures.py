import pytest
from numpy.testing import assert_approx_equal, assert_allclose
from bnp_ml.jax_signal_model import JaxSignalModel, MultiNomialReparametrization, NaturalSignalModel, NaturalSignalModelGeometricLength
from bnp_ml.jax_wrapper import estimate_fisher_information, estimate_sgd, init_like
from .goodness_test import assert_sample_logprob_fit
import numpy as np

max_fragment_length = 4
length_dist = np.arange(max_fragment_length+1)/10

models = [
    JaxSignalModel(np.full(3, 1/3), length_dist),
    JaxSignalModel(np.full(3, 1/3), np.array([0., 0.64102566, 0.25641027, 0.10256409])),
    JaxSignalModel(np.array([0.33333334, 0.33333333, 0.33333333]), np.array([0., 0.64102566, 0.25641027, 0.10256409])),
    NaturalSignalModel(MultiNomialReparametrization.to_natural(np.full(3, 1/3)),
                       np.array([0., 0.64102566, 0.25641027, 0.10256409])),
    NaturalSignalModel(MultiNomialReparametrization.to_natural(np.full(3, 1/3)),
                       length_dist),
    NaturalSignalModelGeometricLength(MultiNomialReparametrization.to_natural(np.full(3, 1/3)),
                                      np.log(0.4))
]


@pytest.fixture
def rng():
    return np.random.default_rng()


@pytest.fixture
def signal_model():
    return JaxSignalModel(np.full(3, 1/3), length_dist)


@pytest.fixture
def natural_signal_model():
    return NaturalSignalModel(MultiNomialReparametrization.to_natural(np.full(3, 1/3)),
                              length_dist)


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
