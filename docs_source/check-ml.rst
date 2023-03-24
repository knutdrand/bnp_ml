========
Check-ml
========

The goal of check-ml is to check whether a optimization alogrithm finds the maximum-likelihood estimates, and in what way they fail. It does this by plotting the residuals (difference between estimates and true parameter vvalues) divided by the standard deviation predicted by the Cramer-Rao lower bound. These values should then be standard normal distributed if the estimates are as good as the maximum likelihood estimates.

Lets check this for a normal distribution first, with actual ML estimates, bad gradient descent and lastly good gradient descent:

.. plotly::
   :fig-vars: ml_fig, bad_gd_fig, good_gd_fig

    from bnp_ml import fisher_table, plot_table
    from bnp_ml.distributions import Normal
    from bnp_ml.jax_wrapper import estimate_sgd
    from functools import partial
    import numpy as np

    def estimate_normal(dist, X):
        assert not np.all(X == X.ravel()[0])
        mu = np.sum(X, axis=0)/len(X)
    sigma = np.sqrt(np.sum((X-mu)**2, axis=0)/len(X))
        return dist.__class__(mu, sigma)
    t = fisher_table(Normal(0.0, 1.0), estimate_normal, sample_sizes=np.arange(2, 100))
    ml_fig = plot_table(t)
    sgd_estimator = partial(estimate_sgd, n_iterations=10)
    t = fisher_table(Normal(0.0, 1.0), sgd_estimator, sample_sizes=np.arange(2, 100))
    bad_gd_fig = plot_table(t)
    better_sgd_estimator = partial(estimate_sgd, n_iterations=1000)
    t = fisher_table(Normal(0.0, 1.0), better_sgd_estimator, sample_sizes=np.arange(2, 100))
    good_gd_fig = plot_table(t)

This also works for vector-valued parameters, which will plot all the elements in one plot

.. plotly::

    from bnp_ml import fisher_table, plot_table
    from bnp_ml.distributions import MultiVariateNormalDiag
    from bnp_ml.jax_wrapper import estimate_sgd
    from functools import partial
    import numpy as np

    def estimate_normal(dist, X):
        mu = np.sum(X, axis=0)/len(X)
        sigma = np.sqrt(np.sum((X-mu)**2, axis=0)/len(X))
        return dist.__class__(mu, sigma)

    t = fisher_table(MultiVariateNormalDiag(np.arange(1, 10)*0.2,
                     np.arange(1, 10)*1.2),
                     estimate_normal, sample_sizes=np.arange(2, 100, 2))
    plot_table(t)


The only requirment for making this work is to have a distribution class that fulfills the `Distribution` interface, i.e. has the `log_prob`,  `sample` and `event_shape` methods (similar to `tf.distributons`) and the `parameters` property and `parameter_names` classmethod. The easiest thing is to wrap whatever distribution class you have in a class wrapper, like this one for jax distributions:
