import numpy as np
from typing import Protocol
import jax
import math


class Distribution(Protocol):
    def log_pmf(self, *data) -> np.ndarray:
        return NotImplemented

    def sample(self, sample_shape) -> np.ndarray:
        return NotImplemented

    @property
    def parameters(self):
        return NotImplemented

    @property
    def event_shape(self):
        return NotImplemented


def get_log_likelihood_function(distribution_class: type, data: np.ndarray):
    def log_likelihood_function(*args, **kwargs):
        return -np.mean(distribution_class(*args, **kwargs).log_prob(data))
    return log_likelihood_function


def estimate_fisher_information(model: Distribution, n: int = 10000000):
    n //= math.prod(model.event_shape)
    x = model.sample((n,))
    # print(x)
    f = get_log_likelihood_function(model.__class__, x)
    hessian = jax.hessian(f, argnums=list(range(len(model.parameters))))(*model.parameters)
    # print(hessian)
    for H in hessian:
        print([h.shape for h in H])
    return H
    return tuple(tuple(-np.mean(h, axis=-1) for h in row)
                 for row in hessian)

