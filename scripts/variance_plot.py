# from bnp_ml.bernoulli import Bernoulli
from bnp_ml.distributions import Bernoulli, Normal, MultiVariateNormalDiag
from bnp_ml.jax_wrapper import estimate_fisher_information, estimate_sgd, linear_fisher_information, estimate_gd, Wrapper, mixture_class
from bnp_ml.fisher_plot import fisher_plot2, fisher_table, plot_table
from bnp_ml.estimators import estimate_mixed_normal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from functools import partial
import jax
# from distrax import Normal


def fisher_plot():
    dist = Bernoulli(0.4)
    print('getting fisher information')
    fisher_info = linear_fisher_information(dist, n=100000)
    sample_sizes = np.arange(1, 10)*10
    variances = []
    for sample_size in sample_sizes:
        print(f'running for sample_size {sample_size}')
        X = dist.sample((sample_size, ))
        estimated_ps = np.array([estimate_sgd(Bernoulli(0.6), X, n_iterations=100).p
                                 for _ in range(100)])
        variances.append(np.sum((estimated_ps-dist.p)**2)/100)
    plt.plot(sample_sizes, 1/np.array(variances))
    plt.show()


def estimate_p(dist, X):
    # dist.p = X.sum()/X.size
    return dist.__class__(X.sum()/X.size)


def estimate_normal(dist, X):
    mu = np.sum(X, axis=0)/len(X)
    sigma = np.sqrt(np.sum((X-mu)**2, axis=0)/len(X))
    return dist.__class__(mu, sigma)


cls = mixture_class(Normal, Normal, jax.random.PRNGKey(12345))
mixed_normal = cls(0.5, 0.0, 1.0, 2.0, 3.0)

# t = fisher_table(Bernoulli(0.4), estimate_p, sample_sizes=np.arange(1, 200))
#t = fisher_table(Normal(0.0, 1.0), estimate_normal, sample_sizes=np.arange(1, 100))
# t = fisher_table(MultiVariateNormalDiag(np.array([1.0, 2.0]),
#                                         np.array([2.0, 3.0])), estimate_normal, sample_sizes=np.arange(1, 200))
# t = fisher_table(MultiVariateNormalDiag(np.arange(1, 10)*0.2,
#                                         np.arange(1, 10)*1.2),
#                  estimate_normal, sample_sizes=np.arange(1, 200))
t = fisher_table(mixed_normal, estimate_mixed_normal, sample_sizes=np.arange(1, 100))

# t = fisher_table(Bernoulli(0.4), estimate_p, sample_sizes=np.arange(1, 200))
plot_table(t).show()
# fisher_plot2(Bernoulli(0.4), estimate_sgd)

# fisher_plot2(Normal(0.0, 1.0), estimate_parameters)
