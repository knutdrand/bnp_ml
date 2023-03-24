import numpy as np
import math
from typing import Tuple, List, Union
from bionumpy import bnpdataclass
from functools import partial
import jax.numpy as jnp

xp = jnp


class _JaxSignalModel:

    def __init__(self, binding_affinity, fragment_length_distribution):
        self.binding_affinity = binding_affinity
        self.fragment_length_distribution = fragment_length_distribution
        self._max_fragment_length = len(self.fragment_length_distribution)-1
        self._area_size = len(self.binding_affinity)
        self._tot_probs = xp.array([self.fragment_length_distribution[1:position+2].sum() + self.fragment_length_distribution[1:(self._area_size-position+1)].sum()
                                    for position in range(self._area_size)])

    def tot_prob(self, position):
        return self.fragment_length_distribution[1:position+2].sum() + self.fragment_length_distribution[1:(self._area_size-position+1)]


    def __repr__(self):
        return f'{self.__class__.__name__}({self.binding_affinity}, {self.fragment_length_distribution})'

    @property
    def parameters(self):
        return (self.binding_affinity, self.fragment_length_distribution)

    @classmethod
    def parameter_names(cls):
        return ['binding_affinity', 'fragment_length_distribution']

    @property
    def event_shape(self):
        return (1, )

    def _sample_n(self, rng,  n):
        return [self._sample_one(rng)
                for _ in range(n)]

    def sample(self, rng, shape):
        assert len(shape) == 1
        return self._sample_n(rng, math.prod(shape))

    def log_prob(self, X: Union[Tuple[int, str], List[Tuple[int, str]]]):
        return xp.log(self.probability(X))

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
        tot_probs = self._tot_probs[index]
        length_prob = self.fragment_length_distribution[1:1+len(affinity)]/tot_probs
        return xp.sum(affinity*length_prob)

    def domain(self):
        return {(pos, strand) for pos in range(len(self.binding_affinity)) for strand in ('+', '-')}

    def _sample_one(self, rng):
        pos = rng.choice(np.arange(self._area_size),
                         p=np.array(self.binding_affinity))
        pre = self.fragment_length_distribution[1:pos+2]
        post = self.fragment_length_distribution[1:(self._area_size-pos+1)]
        s_pos = pre.sum()
        s_neg = post.sum()
        tot = s_pos + s_neg
        reverse = rng.choice([False, True], p=np.array((s_pos/tot, s_neg/tot)))
        if not reverse:
            pos = pos-rng.choice(np.arange(pre.size), p=np.array(pre/s_pos))
        else:
            pos = pos+rng.choice(np.arange(post.size), p=np.array(post/s_neg))
        strand = '-' if reverse else '+'
        assert pos >= 0 and pos < self._area_size, (reverse, pos, fragment_length)
        return pos, strand


class MultiNomialReparametrization:
    @staticmethod
    def to_natural(probabilities):
        return xp.log(probabilities[..., :-1])-xp.log(probabilities[..., -1])

    @staticmethod
    def from_natural(etas):
        expd = xp.append(xp.exp(etas), 1)
        s = xp.sum(expd)
        return expd/s


class NaturalSignalModel(JaxSignalModel):
    def __init__(self, eta, fragment_length_distribution):
        self.eta = eta
        super().__init__(MultiNomialReparametrization.from_natural(eta),
                         fragment_length_distribution)

    @classmethod
    def parameter_names(cls):
        return ['eta', 'fragment_length_distribution']

    @property
    def parameters(self):
        return (self.eta, self.fragment_length_distribution)


class NaturalSignalModelGeometricLength(NaturalSignalModel):
    is_natural = True

    def __init__(self, eta, log_p):
        self.log_p = log_p
        frag_dist = self._fill_fragment_length_dist(eta.size+1, log_p)
        super().__init__(eta, frag_dist)

    @staticmethod
    def _fill_fragment_length_dist(n, log_p):
        log1p = xp.log(1-xp.exp(log_p))
        dist = xp.exp(xp.arange(n)*log_p+log1p)
        return xp.insert(dist, 0, 0)/dist.sum()


    @classmethod
    def parameter_names(cls):
        return ['eta', 'log_p']

    @property
    def parameters(self):
        return (self.eta, self.log_p)
