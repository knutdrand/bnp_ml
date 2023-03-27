from abc import ABC, abstractmethod
from typing import Dict, Any, Protocol
import scipy.stats
import distrax
import numpy as np
from math import prod
from scipy.special import logsumexp
import operator


class Probability:
    def __init__(self, p=None, log_p=None, log_odds=None):
        self._p = p
        self._log_p = log_p
        self._log_odds = None

    def prob(self):
        if self._p is not None:
            return self._p
        if self._log_p is not None:
            return np.exp(self._log_p)

    def log_prob(self):
        if self._p is not None:
            return np.log(self._p)
        if self._log_p is not None:
            return self._log_p

    def __getitem__(self, idx):
        if self._p is not None:
            return self.__class__(p=self._p[idx])
        if self._log_p is not None:
            return self.__class__(log_p=self._log_p[idx])

    def equals(self, p):
        t = self._p == p
        t |= self._log_p == np.log(p)
        t |= self._log_odds == np.log(p)-np.log(1-p)
        return t

    def __mul__(self, other):
        if self._p is not None:
            return self.__class__(p=self._p*other._p)
        if self._log_p is not None:
            return self.__class__(log_p=self._log_p+other._log_p)

    def sum(self):
        if self._p is not None:
            return self.__class__(p=self._p.sum())
        if self._log_p is not None:
            return self.__class__(log_p=logsumexp(self._log_p))

    @property
    def shape(self):
        if self._p is not None:
            return self._p.shape
        if self._log_p is not None:
            return self._log_p.shape

    def __repr__(self):
        return f'Probability(p={self._p if self._p is not None else np.exp(self._log_p)}, log_p={self._log_p})'

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
    #TODO generic value
    def __eq__(self, value) -> 'Event':
        return Event(self, value)

    def __getitem__(self, idx):
        if isinstance(idx, RandomVariable):
            return ConvolutionVariable(idx, self)

        return IndexedVariable(self, idx)

    @abstractmethod
    def probability(self, value) -> Probability:
        return NotImplemented


class ConvolutionVariable(RandomVariable):
    # TODO: Maybe overwrite == with np.any
    def __init__(self, variable_a, variable_b):
        self._variable_a = variable_a
        self._variable_b = variable_b

    def probability(self, value):
        probs_b = self._variable_b.probability(value)
        probs_a = self._variable_a.probability(np.arange(probs_b.shape[0]))
        print(probs_a)
        print(probs_b)
        return (probs_a*probs_b).sum()


class IndexedVariable(RandomVariable):
    def __init__(self, random_variable, idx):
        self._random_variable = random_variable
        self._idx = idx

    def probability(self, value):
        return self._random_variable.probability(value)[self._idx]


class ParameterizedDistribution(RandomVariable):
    def probability(self, value):
        pass

    def __eq__(self, value):
        return Event(self, value)


class Beta(ParameterizedDistribution):
    def __init__(self, a, b):
        self._a = a
        self._b = b
        self._dist = scipy.stats.beta(a, b)

    def probability(self, value):
        return Probability(log_p=self._dist.logpdf(value))


def scipy_stats_wrapper(dist):
    class Wrapper(ParameterizedDistribution):
        def __init__(self, *args, **kwargs):
            self._dist = dist(*args, **kwargs)

        def probability(self, value) -> Probability:
            return Probability(log_p=self._dist.logpdf(value))

    return Wrapper


def jax_wrapper(dist):
    class Wrapper(ParameterizedDistribution):
        def __init__(self, *args, **kwargs):
            self._dist = dist(*args, **kwargs)

        def probability(self, value) -> Probability:
            return Probability(log_p=self._dist.log_prob(value))

    return Wrapper


Categorical = jax_wrapper(distrax.Categorical)
Normal = scipy_stats_wrapper(scipy.stats.norm)


class Bernoulli(ParameterizedDistribution):
    def __init__(self, p: Probability):
        self._p = p

    def probability(self, value) -> Probability:
        return Probability(self._p**value*(1-self._p)**(1-value))


class DictRandomVariable(RandomVariable):
    def __init__(self, outcome_dict: Dict[Dict, Probability]):
        self._outcome_dict = outcome_dict

    def probability(self, value):
        return self._outcome_dict[value]


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


class MultiEvent(Event):
    def __init__(self, events, op):
        assert (isinstance(event, Event) for event in events)
        self._events = events
        self._op = op

    def probability(self):
        return Probability.apply_func(self._op, [event.probability() for event in self._events])


def P(event):
    return event.probability()
