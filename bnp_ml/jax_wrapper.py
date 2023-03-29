import numpy as np
from typing import Protocol
import jax
import distrax
from jax import random
import jax.numpy as xp
import math
import optax
import logging
logger = logging.getLogger(__name__)


def mixture_class(class_1, class_2, seed):
    class MixtureOfTwo:
        def __init__(self, prob_a, *args):
            component_a = class_1(*args[:len(class_1.parameter_names())])
            component_b = class_2(*args[len(class_1.parameter_names()):])
            self._dist = distrax.MixtureOfTwo(prob_a, component_a._dist, component_b._dist)
            self.prob_a = prob_a
            self.component_a = component_a
            self.component_b = component_b
            self._seed = seed

        @property
        def parameters(self):
            return (self.prob_a, ) + self.component_a.parameters + self.component_b.parameters
    
        @classmethod
        def parameter_names(self):
            return ['prob_a'] + ['A_'+ name for name in class_1.parameter_names()] + ['B_' + name for name in class_2.parameter_names()]
    
        def log_prob(self, *args, **kwargs):
            return self._dist.log_prob(*args, **kwargs)
    
        def sample(self, shape, **kwargs):
            kwargs['seed'], self._seed = random.split(self._seed)
            kwargs['sample_shape'] = shape
            return self._dist.sample(**kwargs)

        @property
        def event_shape(self):
            return self.component_a.event_shape
    return MixtureOfTwo


def class_wrapper(distribution_class, param_names, seed):
    _, seed = random.split(seed)

    class NewClass:
        def __init__(self, *args, **kwargs):
            for name, arg in zip(param_names, args):
                kwargs[name] = arg
            self._dist = distribution_class(**kwargs)
            self._seed = seed

        def __getattr__(self, name):
            if name in param_names:
                return getattr(self._dist, name)
            raise AttributeError(f'{distribution_class} object has no attribute {name}')

        def __setattr__(self, name, value):
            if name in param_names:
                setattr(self._dist, name, value)
            else:
                super().__setattr__(name, value)

        def log_prob(self, *args, **kwargs):
            return self._dist.log_prob(*args, **kwargs)
    
        def sample(self, shape, **kwargs):
            kwargs['seed'], self._seed = random.split(self._seed)
            kwargs['sample_shape'] = shape
            return self._dist.sample(**kwargs)
            
        @property
        def parameters(self):
            return tuple(getattr(self._dist, param_name)
                         for param_name in param_names)

        @classmethod
        def parameter_names(self):
            return param_names
        
        @property
        def event_shape(self):
            return self._dist.event_shape

    return NewClass


class Wrapper:
    def __init__(self, distribution, param_names):
        self._dist = distribution
        self._param_names = param_names

    def log_pmf(self, *args, **kwargs):
        return self._dist.log_pmf

    def params(self):
        return tuple(getattr(self._dist, param_name) for param_name in self._param_names)

    def sample(self, *args, **kwargs):
        return self._dist.sample(*args, **kwargs)


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


def _estimate_fisher_information(model: Distribution, n: int = 10000000, rng=None):
    n //= math.prod(model.event_shape)
    if rng is not None:
        x = model.sample(rng, (n,))
    else:
        x = model.sample((n,))
    f = get_log_likelihood_function(model.__class__, x)
    hessian = jax.hessian(f, argnums=list(range(len(model.parameters))))(*model.parameters)
    return hessian


def get_log_likelihood_function(distribution_class: type, data: np.ndarray):
    def log_likelihood_function(*args, **kwargs):
        return -np.mean(distribution_class(*args, **kwargs).log_prob(data))
    return log_likelihood_function


def estimate_fisher_information(model: Distribution, n: int = 10000000, rng=None):
    n //= math.prod(model.event_shape)
    if rng is not None:
        x = model.sample(rng, (n,))
    else:
        x = model.sample((n,))
    f = lambda params, data: -np.mean(model.__class__(*params).log_prob(data))
    print(model.parameters)
    hessian = jax.hessian(f)(model.parameters, x)
    return hessian


def linear_fisher_information(model: Distribution, n: int = 10000000, rng=None):
    h = estimate_fisher_information(model, n, rng)
    return tuple(np.atleast_2d(row[i]).diagonal() for i, row in enumerate(h))

# return tuple(tuple(-np.mean(h, axis=-1) for h in row)
#                  for row in hessian)


def estimate_gd(distribution, data, learning_rate=0.01, n_iterations=100):
    if not isinstance(data, tuple):
        data = (data, )
    for i in range(n_iterations):
        grad_func = jax.grad(lambda *params: -np.mean(distribution.__class__(*params).log_prob(*data)),
                             argnums=list(range(len(distribution.parameters))))
        grads = grad_func(*distribution.parameters)
        distribution = distribution.__class__(*(param-grad*learning_rate for param, grad in zip(distribution.parameters, grads)))
    return distribution


def estimate_sgd(distribution, data, learning_rate=0.01, n_iterations=100):
    logger.info(f'Running sgd on {distribution.__class__.__name__} with  {n_iterations} iterations')
    optimizer = optax.sgd(learning_rate)
    params = distribution.parameters
    opt_state = optimizer.init(params)
    if not isinstance(data, tuple):
        data = (data, )

    def loss_func(params, data):
        return -xp.mean(distribution.__class__(*params).log_prob(*data))

    @jax.jit
    def update_func(params, opt_state):
        grads = jax.grad(loss_func)(params, data)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state
        
    for _ in range(n_iterations):
        params, opt_state = update_func(params, opt_state)
        # grads = jax.grad(loss_func)(params, data)
        # updates, opt_state = optimizer.update(grads, opt_state)
        # params = optax.apply_updates(params, updates)
    return distribution.__class__(*params)

    
    # Initialize parameters of the model + optimizer.

    if not isinstance(data, tuple):
        data = (data, )
    for i in range(n_iterations):
        grad_func = jax.grad(lambda *params: -np.mean(distribution.__class__(*params).log_prob(*data)),
                             argnums=list(range(len(distribution.parameters))))
        grads = grad_func(*distribution.parameters)
        distribution = distribution.__class__(*(param-grad*learning_rate for param, grad in zip(distribution.parameters, grads)))
    return distribution


def init_like(model, rng):
    return model.__class__(
        *(-1*rng.random(param.shape) for param in model.parameters))
