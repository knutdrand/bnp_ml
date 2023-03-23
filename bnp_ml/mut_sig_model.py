from scipy.special import loggamma
import math
import numpy as np
import jax.numpy as jnp
xp = jnp


class MutSigModel:
    def __init__(self, S, E):
        self.S = S
        self.E = E
        self.rates = self.E @ self.S

    @property
    def event_shape(self):
        return self.rates.shape

    @property
    def parameters(self):
        return (self.S, self.E)

    @classmethod
    def parameter_names(cls):
        return ['S', 'E']

    def log_prob(self, M):
        return xp.sum(M*xp.log(self.rates)-self.rates-loggamma(M+1))

    def sample(self, rng, shape):
        print(shape)
        n = math.prod(shape)
        return np.array([rng.poisson(self.rates) for _ in range(n)]).reshape(shape+self.rates.shape)
