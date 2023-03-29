from .probability_fixtures import normal
from bnp_ml.probability.model_wrap import wrap_model_func
from bnp_ml.probability.events import Normal
import pytest


@pytest.fixture
def normal_model():
    return lambda mu, sigma: Normal(mu, sigma)


def test_wrap_model():
    model = wrap_model_func(normal)
    assert hasattr(model, 'parameters')
    assert hasattr(model, 'parameter_names')
