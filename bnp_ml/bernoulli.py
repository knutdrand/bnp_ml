from jax import random
import numpy as np
import jax.numpy as jnp


class Bernoulli:
    def __init__(self, p):
        self.p = p
        self._key = random.PRNGKey(2000)

    @property
    def event_shape(self):
        return (1, )

    @property
    def parameters(self):
        return [self.p]

    def log_prob(self, X):
        return jnp.log(self.p)*X + jnp.log(1-self.p) * (1-X)

    def sample(self, shape):
        out = np.random.choice([True, False], size=shape, p=[self.p, 1-self.p])
        # out = random.bernoulli(self._key, p=self.p, shape=shape)
        print(f'sampling {shape} from', self.p, out.sum()/shape[-1], out.shape)
        return out
