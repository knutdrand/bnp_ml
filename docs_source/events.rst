========
 Events
========

When having specified a model with for instance:

.. math::

   Z \sim Multinomial(\theta)

   \delta \sim Geometric(\eta)

   B \sim Bernoullli(p_s)

   P(D, X | X \in W) = P(\tilde{B})/w + P(B)P(Z+\delta=X | 0<=Z+delta<w)/2

It should be easy to specify it in python with for instance:

.. code-block::

   def model(x, d, theta, eta, p_s, w):
       Z = dist.Multinomial(theta)
       delta = dist.Geometric(eta)
       B = dist.Bernoulli(p_s)
       X_random = dist.Uniform(w)
       X_signal = Z+\delta
       return ~B * (X_random) + B * Z+\delta

and translate the model to the framework of choice, i.e. tf, torch, pyro, pyMC, scipy, jax
