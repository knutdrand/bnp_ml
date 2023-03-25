from typing import Dict, Any


class Probability:
    def __init__(self, p):
        self._p = p
        
    def equals(self, other):
        return self._p == other


class RandomVariable:
    def __init__(self, outcome_dict: Dict[Dict, Probability]):
        self._outcome_dict = outcome_dict

    def __eq__(self, value):
        return Event(self, value)

    def probability(self, value):
        return self._outcome_dict[value]


class Event:
    def __init__(self, random_variable, value):
        self._random_variable = random_variable
        self._value = value

    def probability(self):
        return self._random_variable.probability(self._value)


def P(event):
    return event.probability()
