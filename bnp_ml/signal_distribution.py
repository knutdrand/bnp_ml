import numpy as np
from numpy.random import default_rng
from logarray.logarray import log_array
import dataclasses
from typing import Tuple


class RangeDomain:
    def __init__(self, window_size, max_fragment_length):
        pos_domain = {(pos, '+') for pos in range(max_fragment_length-1+window_size)}
        neg_domain = {(pos, '-') for pos in range(max_fragment_length-1,window_size+(max_fragment_length-1)*2)}
        self._domain = pos_domain | neg_domain

    def __contains__(self, elem):
        return elem in self._domain


@dataclasses.dataclass
class SignalModel:
    binding_affinity: np.ndarray
    fragment_length_distribution: np.ndarray
    signal_probability: float
    

class SignalProb:
    def __init__(self, params: SignalModel):
        self._params = params


class SignalSimulator:
    def __init__(self, params: SignalModel, rng: np.random.Generator):
        self._params = params
        self._rng = rng


class SignalModel:
    def __init__(self, binding_affinity, fragment_length_distribution, signal_probability):
        # self._w = np.log(binding_affinity)
        # self._f = np.log(fragment_length_distribution)
        # self._s = np.log(signal_probability)
        self.B = binding_affinity
        self.F = fragment_length_distribution
        self.S = signal_probability
        print(binding_affinity, fragment_length_distribution, signal_probability)
        self._binding_affinity = log_array(binding_affinity)
        self._fragment_length_distribution = log_array(fragment_length_distribution)
        self._signal_probability = log_array(signal_probability)
        self._max_fragment_length = len(self._fragment_length_distribution)-1
        self._padded_affinity = np.pad(self._binding_affinity, (self._max_fragment_length-1, ))
        self._domain = RangeDomain(len(self._binding_affinity), self._max_fragment_length)

        # self._param_names = ['_binding_affinity',
        #                      '_fragment_length_distribution',
        #                      '_signal_probability']
        self._param_names = ['B',
                             'F',
                             'S']

        self.arg_constraints = {name: None for name in self._param_names}

        self.event_shape = tuple()

    def log_prob(self, data: Tuple[int, str]):
        return [np.log(self.probability(p, s))
                for p, s in data]

    def probability(self, position: int, strand: str):
        assert (position, strand) in self._domain, (position, strand)
        return self._background_prob+self._foreground_prob(position, strand)

    def sample(self, n=(1,)):
        assert len(n) == 1
        return [self.simulate(np.random.default_rng()) for _ in range(n[0])]

    def simulate(self, rng: np.random.Generator):
        is_signal = rng.choice([False, True], p=[self._signal_probability.to_array(), 1-self._signal_probability.to_array()])
        if not is_signal:
            res = self._simulate_background(rng)
        else:
            res = self._simulate_foreground(rng)
        assert res in self._domain, (res, is_signal)
        return res

    @property
    def _cumulative_fragment_length_distribution(self):
        return np.cumsum(self._fragment_length_distribution)

    @property
    def _area_size(self):
        return self._max_fragment_length+len(self._binding_affinity)-1

    @property
    def _background_prob(self):
        return (1-self._signal_probability)*(1/(self._area_size*2))

    def _foreground_prob(self, position: int, strand: str):
        if strand == '+':
            index = slice(position, position+self._max_fragment_length)
        else:
            index = slice(position, position-self._max_fragment_length if position-self._max_fragment_length>=0 else None, -1)

        p = self._signal_probability*np.sum(self._fragment_length_distribution[1:] * 0.5*self._padded_affinity[index])
        return p

    def _simulate_background(self, rng):
        reverse = rng.choice([True, False])
        pos = rng.integers(self._area_size)
        if reverse:
            res = (pos+self._max_fragment_length-1, '-')
        else:
            res = (pos, '+')
        assert res in self._domain, (res, reverse, pos)
        return res

    def _simulate_foreground(self, rng):
        pos = rng.choice(np.arange(len(self._binding_affinity)), p=self._binding_affinity.to_array())
        fragment_length = rng.choice(np.arange(self._max_fragment_length+1), p=self._fragment_length_distribution.to_array())
        reverse = rng.choice([True, False])
        if not reverse:
            return pos+self._max_fragment_length-fragment_length, '+'
        if reverse:
            return pos+self._max_fragment_length-1 + fragment_length-1, '-'

