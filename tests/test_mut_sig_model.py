from bnp_ml.mut_sig_model import MutSigModel
from bnp_ml.jax_wrapper import estimate_fisher_information
import pytest
import numpy as np


@pytest.fixture
def rng():
    return np.random.default_rng()


@pytest.fixture
def mut_sig_model():
    return MutSigModel(np.array([[1., 2., 3.],
                                 [4., 5., 6.]]),
                       np.array([[0.2, 0.5],
                                 [0.1, 0.9],
                                 [0.13, 0.2],
                                 [0.9, 0.10]]))


def test_fisher_information(mut_sig_model, rng):
    estimate_fisher_information(mut_sig_model, 100, rng)
# 
# def test_probability(mut_sig_model: JaxSignalModel):
#      assert_prob_is_one(signal_model_big)
# 
# 
# def test_simulate(signal_model):
#     assert_sample_logprob_fit(signal_model)

