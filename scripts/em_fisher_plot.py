import logging
from functools import partial
import numpy as np
from numpy.random import default_rng
from bnp_ml.events import Normal, Categorical, P
from bnp_ml.jax_signal_model import (NaturalSignalModelGeometricLength,
                                     MultiNomialReparametrization)
from bnp_ml.fisher_plot import fisher_table, plot_table
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

        
area_size = 21
my_sgd = partial(estimate_sgd, n_iterations=10000)
weights = np.ones(area_size)
weights[area_size//2] = 10
ps = weights/weights.sum()
eta = MultiNomialReparametrization.to_natural(ps)
log_p = np.log(0.5)

Curried = curry(NaturalSignalModelGeometricLength, eta)
true_model = Curried(log_p)
t = fisher_table(true_model, my_sgd, sample_sizes=np.arange(1, 200, 2),
                 n_fisher=200, rng=default_rng())
plot_table(t).show()

true_model = NaturalSignalModelGeometricLength(eta, log_p)
t = fisher_table(true_model, my_sgd, sample_sizes=3*np.arange(1, 30),
                 n_fisher=100, rng=default_rng())
plot_table(t).show()
