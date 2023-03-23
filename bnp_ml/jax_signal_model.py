import numpy as np
import math
from typing import Tuple, List, Union
from bionumpy import bnpdataclass
import jax.numpy as jnp

xp = jnp


# @bnpdataclass.bnpdataclass
# class ReadStart:
#     position: int
#     strand: str


class _JaxSignalModel:
    def __init__(self, binding_affinity, fragment_length_distribution):
        self.binding_affinity = binding_affinity
        self.fragment_length_distribution = fragment_length_distribution
        self._max_fragment_length = len(self.fragment_length_distribution)-1
        self._area_size = len(self.binding_affinity)
        self._padded_affinity = jnp.pad(self.binding_affinity, (self._max_fragment_length-1, ))

    @property
    def parameters(self):
        return (self.binding_affinity, self.fragment_length_distribution)

    @classmethod
    def parameter_names(cls):
        return ['binding_affinity', 'fragment_length_distribution']

    @property
    def event_shape(self):
        return (1, )

    def _sample_one(self, rng):
        pos = rng.choice(np.arange(self._area_size),
                         p=self.binding_affinity)
        fragment_length = rng.choice(np.arange(self._max_fragment_length+1),
                                     p=self.fragment_length_distribution)
        reverse = rng.choice([True, False])
        if not reverse:
            return pos + self._max_fragment_length-fragment_length, '+'
        if reverse:
            return pos + self._max_fragment_length-1 + fragment_length-1, '-'

    def _sample_n(self, rng,  n):
        return [self._sample_one(rng)
                for _ in range(n)]

    def sample(self, rng, shape):
        assert len(shape) == 1
        return self._sample_n(rng, math.prod(shape))

    def log_prob(self, X: Union[Tuple[int, str], List[Tuple[int, str]]]):
        return xp.log(self.probability(X))

    def probability(self, X: Union[Tuple[int, str], List[Tuple[int, str]]]):
        if isinstance(X, list):
            return xp.array([self.probability(x) for x in X])
        position, strand = X
        if strand == '+':
            index = slice(position, position+self._max_fragment_length)
        else:
            index = slice(
                position,
                position-self._max_fragment_length if position-self._max_fragment_length >= 0 else None,
                -1)
        return xp.sum(
            self.fragment_length_distribution[1:] * 0.5*self._padded_affinity[index])

    def domain(self):
        pos = {(self._max_fragment_length + binding_pos-fragment_length, '+')
               for fragment_length in range(1, self._max_fragment_length+1)
               for binding_pos in range(self._area_size)}
        neg = {(self._max_fragment_length + binding_pos+fragment_length, '-')
               for fragment_length in range(1, self._max_fragment_length+1)
               for binding_pos in range(self._area_size)}
        return pos | neg


class JaxSignalModel(_JaxSignalModel):
    ''' THis one is parameterized a bit wierdly. 
    binding affinity is the probability that a read coming a tf binding to a position,
    comes from a givn position, conditioned on the read falling in the desired windo. 
    Thus is should be lower on the edges
    '''

    def probability(self, X: Union[Tuple[int, str], List[Tuple[int, str]]]):
        if isinstance(X, list):
            return xp.array([self.probability(x) for x in X])
        position, strand = X
        if strand == '+':
            index = slice(position, position+self._max_fragment_length)
        else:
            index = slice(
                position,
                position-self._max_fragment_length if position-self._max_fragment_length >= 0 else None,
                -1)
        affinity = self.binding_affinity[index]
        length_prob = self.fragment_length_distribution[1:1+len(affinity)]
        return xp.sum(0.5*affinity*length_prob/length_prob.sum())

    def domain(self):
        return {(pos, strand) for pos in range(len(self.binding_affinity)) for strand in ('+', '-')}

    def _sample_one(self, rng):
        pos = rng.choice(np.arange(self._area_size),
                         p=self.binding_affinity)
        reverse = rng.choice([True, False])
        if not reverse:
            max_fragment_length = min(self._max_fragment_length, pos+1)
        else:
            max_fragment_length = min(self._max_fragment_length, self._area_size-pos)
        p = self.fragment_length_distribution[:max_fragment_length+1]
        p /= p.sum()
        fragment_length = rng.choice(np.arange(max_fragment_length+1),
                                     p=p)
        if reverse:
            pos = pos + fragment_length-1
        else:
            pos = pos - fragment_length + 1

        strand = '-' if reverse else '+'
        assert pos >= 0 and pos < self._area_size, (reverse, pos, fragment_length)
        return pos, strand


class MultiNomialReparametrization:
    @staticmethod
    def to_natural(probabilities):
        return xp.log(probabilities[..., :-1])-xp.log(probabilities[..., -1])

    @staticmethod
    def from_natural(etas):
        expd = np.append(xp.exp(etas), 1)
        s = xp.sum(expd)
        return expd/s


class NaturalSignalModel(JaxSignalModel):
    def __init__(self, eta, fragment_length_distribution):
        pass

    # return xp.sum(
    #         self.fragment_length_distribution[1:] * 0.5*self._padded_affinity[index])
