import logging
from functools import partial
import numpy as np
from numpy.random import default_rng
from bnp_ml.jax_signal_model import (NaturalSignalModelGeometricLength,
                                     MultiNomialReparametrization)
from bnp_ml.jax_wrapper import estimate_sgd
from bnp_ml.fisher_plot import fisher_table, plot_table
logging.basicConfig(level=logging.INFO)
area_size = 21
my_sgd = partial(estimate_sgd, n_iterations=100000)
weights = np.ones(area_size)
weights[area_size//2] = 10
ps = weights/weights.sum()
eta = MultiNomialReparametrization.to_natural(ps)
log_p = np.log(0.5)
true_model = NaturalSignalModelGeometricLength(eta, log_p)
t = fisher_table(true_model, my_sgd, sample_sizes=3*np.arange(1, 30),
                 n_fisher=100, rng=default_rng())
plot_table(t).show()
