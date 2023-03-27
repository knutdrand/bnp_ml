import logging
from functools import partial
import numpy as np
from numpy.random import default_rng
from bnp_ml.events import Normal, Categorical, P
from bnp_ml.jax_signal_model import (NaturalSignalModelGeometricLength,
                                     MultiNomialReparametrization)
from bnp_ml.fisher_plot import fisher_table, plot_table
from bnp_ml.jax_wrapper import estimate_sgd
from bnp_ml.model import curry

logging.basicConfig(level=logging.INFO)


def mixture_model(means, ps):
    Z = Categorical(probs=ps)
    return Normal(means, 1.0)[Z]


class Mixture:
    def __init__(self, means, etas):
        self.means = means
        self.etas = etas
        self.ps = MultiNomialReparametrization.from_natural(etas)
        self.X = mixture_model(self.means, self.ps)

    def log_prob(self, x):
        return P(self.X == x).log_prob()

    def sample(self, seed, shape):
        return self.X.sample(seed, shape)

    @property
    def parameters(self):
        return [self.means, self.etas]

    @classmethod
    def parameter_names(self):
        return ['means', 'etas']

    @property
    def event_shape(self):
        return (1,)


my_sgd = partial(estimate_sgd, n_iterations=10000)
means = np.arange(3).astype(float)
ps = np.array([0.3, 0.35, 0.45])
eta = MultiNomialReparametrization.to_natural(ps)

true_model = Mixture(means, eta)
t = fisher_table(true_model, my_sgd, sample_sizes=np.arange(1, 200, 2),
                 n_fisher=200, rng=12345)#default_rng())
plot_table(t).show()
