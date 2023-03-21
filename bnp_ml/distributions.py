from .jax_wrapper import class_wrapper
import distrax
from jax import random

seed = random.PRNGKey(12345)

old_seed, seed = random.split(seed)
Normal = class_wrapper(distrax.Normal, ['loc', 'scale'], old_seed)
old_seed, seed = random.split(seed)
Bernoulli = class_wrapper(distrax.Bernoulli, ['probs'], old_seed)
old_seed, seed = random.split(seed)
MultiVariateNormalDiag = class_wrapper(distrax.MultivariateNormalDiag, ['loc', 'scale_diag'], old_seed)


class MixtureOfTwo:
    def __init__(self, prob_a, component_a, component_b):
        self._dist = distrax.MixtureOfTwo(prob_a, component_a._dist, component_b._dist)
        self.prob_a = prob_a
        self.component_a = component_a
        self.component_b = component_b

    @property
    def parameters(self):
        return (self.prob_a, ) + self.component_a.parameters() + self.component_b.parameters()

    @property
    def parameter_names(self):
        return ['prob_a'] + ['A_'+ name for name in self.component_a.parameter_names] + ['B_' + name for name in self.component_b.parameter_names]

    def log_prob(self, *args, **kwargs):
        return self._dist.log_prob(*args, **kwargs)

    def sample(self, shape, **kwargs):
        kwargs['seed'], self._seed = random.split(self._seed)
        kwargs['sample_shape'] = shape
        return self._dist.sample(**kwargs)
