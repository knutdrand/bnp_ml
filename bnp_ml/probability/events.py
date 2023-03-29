from abc import ABC, abstractmethod
from typing import Dict
import scipy.stats
import distrax
import numpy as np
from math import prod
import jax.numpy as jnp
from scipy.special import logsumexp
import operator
from numbers import Number
xp = jnp


class Probability:
    def __init__(self, p=None, log_p=None, log_odds=None):
        self._p = p
        self._log_p = log_p
        self._log_odds = None

    def prob(self):
        if self._p is not None:
            return self._p
        if self._log_p is not None:
            return xp.exp(self._log_p)

    def log_prob(self):
        if self._p is not None:
            return xp.log(self._p)
        if self._log_p is not None:
            return self._log_p

    def __getitem__(self, idx):
        if self._p is not None:
            return self.__class__(p=self._p[idx])
        if self._log_p is not None:
            return self.__class__(log_p=self._log_p[idx])

    def equals(self, p):
        t = self._p == p
        t |= self._log_p == xp.log(p)
        t |= self._log_odds == xp.log(p)-xp.log(1-p)
        return t

    def __mul__(self, other):
        if self._p is not None:
            return self.__class__(p=self._p*other._p)
        if self._log_p is not None:
            return self.__class__(log_p=self._log_p+other._log_p)

    def __sub__(self, other):
        if self._p is not None:
            return self.__class__(p=self._p-other._p)
        if self._log_p is not None:
            return self.__class__(log_p=logsumexp([self._log_p, other._log_p], b=[1, -1]))

    def sum(self, axis=None, **kwargs):
        if self._p is not None:
            return self.__class__(p=self._p.sum(axis=axis))
        if self._log_p is not None:
            return self.__class__(log_p=logsumexp(self._log_p, axis=axis))

    @property
    def shape(self):
        if self._p is not None:
            return self._p.shape
        if self._log_p is not None:
            return self._log_p.shape

    def __repr__(self):
        return f'Probability(p={self._p if self._p is not None else xp.exp(self._log_p)}, log_p={self._log_p})'

    @classmethod
    def apply_func(cls, op, elements):
        if op == operator.or_:
            return Probability(sum(element._p for element in elements))
        if op == operator.and_:
            return Probability(prod(element._p for element in elements))
        if op == operator.not_:
            assert len(elements) == 1
            return Probability(1-elements[0]._p)

        assert False


class RandomVariable(ABC):
    @property
    def event_shape(self):
        return ()

    @property
    def batch_shape(self):
        return ()

    def __eq__(self, value) -> 'Event':
        return Event(self, value)

    def __lt__(self, value):
        pass

    def __getitem__(self, idx):
        if isinstance(idx, RandomVariable):
            return ConvolutionVariable(idx, self)

        return IndexedVariable(self, idx)

    def __add__(self, other):
        if isinstance(other, Number):
            return TransformedVariable(self, lambda x: x+other, lambda z: z-other)

    @abstractmethod
    def probability(self, value) -> Probability:
        return NotImplemented

    @abstractmethod
    def sample(self, rng, shape=()) -> np.ndarray:
        return NotImplemented


class TransformedVariable(RandomVariable):
    def __init__(self, variable, f, inv_f):
        self._variable = variable
        self._f = f
        self._inv_f = inv_f

    def probability(self, value):
        return self._variable.probability(self._inv_f(value))

    def sample(self, *args, **kwargs):
        return self._f(self._variable.sample(*args, **kwargs))


class ConvolutionVariable(RandomVariable):
    # TODO: Maybe overwrite == with np.any
    def __init__(self, variable_a, variable_b):
        self._variable_a = variable_a
        self._variable_b = variable_b

    def probability(self, value):
        probs_b = self._variable_b.probability(value[..., np.newaxis])
        probs_a = self._variable_a.probability(xp.arange(self._variable_b.batch_shape[0]))
        return (probs_a*probs_b).sum(axis=-1)

    def sample(self, rng, shape=()):
        return self._variable_b[self._variable_a.sample(rng, shape)].sample(rng)


class IndexedVariable(RandomVariable):
    def __init__(self, random_variable, idx):
        self._random_variable = random_variable
        self._idx = idx

    def probability(self, value):
        return self._random_variable.probability(value)[..., self._idx]

    def sample(self, rng, shape=()):
        s = self._random_variable.sample(rng, shape)
        print(s.shape)
        return s[..., self._idx]


class ParameterizedDistribution(RandomVariable):
    def probability(self, value):
        pass

    def __eq__(self, value):
        return Event(self, value)


class Beta(ParameterizedDistribution):
    def __init__(self, a, b):
        self._a = a
        self._b = b
        self.event_shape = np.broadcast_shapes(np.asanyarray(a).shape, 
                                               np.asanyarray(b).shape)
        self._dist = scipy.stats.beta(a, b)

    def probability(self, value):
        return Probability(log_p=self._dist.logpdf(value))

    def sample(self, rng, shape=()):
        return self._dist.rvs(size=shape, random_state=rng)


def scipy_stats_wrapper(dist):
    class Wrapper(ParameterizedDistribution):
        def __init__(self, *args, **kwargs):
            self.event_shape = np.broadcast_shapes(*tuple(np.asanyarray(a).shape for a in args + tuple(kwargs.values())))
            self._dist = dist(*args, **kwargs)

        def probability(self, value) -> Probability:
            return Probability(log_p=self._dist.logpdf(value))

        def sample(self, rng, shape=()):
            print(shape, self.event_shape)
            shape = shape + self.event_shape
            return self._dist.rvs(size=shape, random_state=rng)

    return Wrapper


class SumVariable(RandomVariable):
    def __init__(self, a, b, a_domain):
        self._a = a
        self._b = b
        self._a_domain = a_domain

    def probability(self, value):
        if self._a_domain is None:
            return NotImplemented
        return np.sum(self._a.probability(self._a_domain)*self._b.probability(value-self._a_domain), axis=0)

    def sample(self, *args, **kwargs):
        return self._a.sample(*args, **kwargs) + self._b.sample(*args, **kwargs)


def jax_wrapper(dist, domain_func=None):
    class Wrapper(ParameterizedDistribution):
        def __init__(self, *args, **kwargs):
            self._dist = dist(*args, **kwargs)
            # self.batch_shape = self._dist.batch_shape
            # self.event_shape = self._dist.event_shape

        @property
        def batch_shape(self):
            return self._dist.batch_shape

        def probability(self, value) -> Probability:
            return Probability(log_p=self._dist.log_prob(value))

        def sample(self, rng, shape=()):
            # print(shape, self.event_shape)
            return self._dist.sample(seed=rng, sample_shape=shape)

        def __add__(self, other):
            if isinstance(other, Number):
                return super().__add__(other)
            domain = None
            if domain_func is not None:
                domain = domain_func(self._dist)
            else:
                return NotImplemented
            return SumVariable(self, other, domain)

        __radd__ = __add__

    return Wrapper


Categorical = jax_wrapper(distrax.Categorical, lambda d: xp.arange(d.num_categories))
Normal = jax_wrapper(distrax.Normal)
Geometric = scipy_stats_wrapper(scipy.stats.geom)

# Normal = scipy_stats_wrapper(scipy.stats.norm)


class Bernoulli(ParameterizedDistribution):
    def __init__(self, p: Probability):
        self._p = p

    def probability(self, value) -> Probability:
        return Probability(self._p**value*(1-self._p)**(1-value))

    def sample(self, rng, shape) -> Probability:
        return rng.choice([0, 1], size=shape, p=[1-self._p, self._p])


class DictRandomVariable(RandomVariable):
    def __init__(self, outcome_dict: Dict[Dict, Probability]):
        self._outcome_dict = outcome_dict

    def probability(self, value):
        return self._outcome_dict[value]

    def sample(self, rng, shape=()):
        return NotImplemented


class Event:
    def __init__(self, random_variable: RandomVariable, value):
        assert isinstance(random_variable , RandomVariable), random_variable
        self._random_variable = random_variable
        self._value = value

    def probability(self):
        return self._random_variable.probability(self._value)

    def __or__(self, other):
        return MultiEvent([self, other], operator.or_)

    def __and__(self, other):
        return MultiEvent([self, other], operator.and_)

    def __invert__(self):
        return MultiEvent([self], operator.not_)


class LTEvent(Event):
    pass


class MultiEvent(Event):
    def __init__(self, events, op):
        assert (isinstance(event, Event) for event in events)
        self._events = events
        self._op = op

    def probability(self):
        return Probability.apply_func(self._op, [event.probability() for event in self._events])


def P(event, **kwargs):
    p = event.probability()
    if 'given' in kwargs:
        return p-kwargs['given'].probability()
    return p
