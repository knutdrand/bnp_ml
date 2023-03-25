from typing import Dict, Any, Protocol
from math import prod
import operator


class Probability:
    def __init__(self, p):
        self._p = p

    def equals(self, other):
        return self._p == other

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


class RandomVariable(Protocol):
    #TODO generic value
    def __eq__(self, value) -> 'Event':
        return NotImplemented

    def probability(self, value) -> Probability:
        return NotImplemented


class Bernoulli:
    def __init__(self, p: Probability):
        self._p = p

    def __eq__(self, value):
        return Event(self, value)

    def probability(self, value) -> Probability:
        return self._p**value*(1-self._p)**(1-value)


class DictRandomVariable:
    def __init__(self, outcome_dict: Dict[Dict, Probability]):
        self._outcome_dict = outcome_dict

    def __eq__(self, value):
        return Event(self, value)

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
