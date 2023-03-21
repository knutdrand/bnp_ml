from logarray import log_array, LogArray, set_backend, as_log_array
import numpy as np
import pytest

from bnp_ml.jax_wrapper import estimate_fisher_information, Distribution, estimate_gd, estimate_sgd
import jax.numpy as jnp
from jax import random
set_backend(jnp)


class NormalDistribution:
    def __init__(self,  loc, scale):
        self.loc = jnp.asarray(loc)
        self.scale = jnp.asarray(scale)
        # self._loc = log_array(loc)
        # self._scale = log_array(scale)

    @property
    def parameters(self):
        return (self.loc, self.scale)

    @property
    def event_shape(self):
        return (1,)

    def log_prob(self, data):
        return jnp.log(1/(self.scale*jnp.sqrt(2*jnp.pi))) - 0.5*(data-self.loc)**2/self.scale**2

        r = np.log(self.probability(data))
        return r

    def probability(self, data):
        return 1/(self._scale*np.sqrt(2*np.pi))*np.exp(-0.5*(data-self._loc)**2/self._scale**2)
        return 1/(self._scale*np.sqrt(2*np.pi))*LogArray(-0.5*(data-self._loc)**2/self._scale**2)

    def sample(self, shape):
        key = random.PRNGKey(0)
        return random.normal(key, shape)*self.scale+self.loc


class Bernoulli:
    def __init__(self, p):
        self.p = p

    @property
    def event_shape(self):
        return (1, )

    @property
    def parameters(self):
        return [self.p]

    def log_prob(self, X):
        return jnp.log(self.p)*X + jnp.log(1-self.p) * (1-X)

    def sample(self, shape):
        key = random.PRNGKey(1000)
        return random.bernoulli(key, p=self.p, shape=shape)




@pytest.fixture
def standard_normal() -> Distribution:
    return NormalDistribution(0., 1.)

@pytest.fixture
def bernoulli() -> Distribution:
    return Bernoulli(0.4)


@pytest.fixture
def bernoulli_raw() -> Distribution:
    return Bernoulli(np.array(0.9))


# return dists.Normal(torch.tensor([1.0]), torch.tensor([1.0]))


def test_fisher_information(standard_normal):
    I = estimate_fisher_information(standard_normal, n=10)


def test_fisher_information(bernoulli):
    I = estimate_fisher_information(bernoulli, n=100000)
    true = 1/bernoulli.p/(1-bernoulli.p)
    assert np.abs(I[0][0]-true) < 0.1


@pytest.mark.parametrize('optimizer', [estimate_sgd])
def test_estimate(bernoulli, bernoulli_raw, optimizer):
    X = bernoulli.sample((200, ))
    dist = optimizer(bernoulli_raw, X, n_iterations=1000)
    assert np.abs(dist.p-bernoulli.p) < 0.01
