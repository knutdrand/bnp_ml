import torch.distributions as dists
from logarray.logarray import log_array, LogArray
import numpy as np
import torch
import pytest

from bnp_ml.ml_check import estimate_fisher_information


@pytest.fixture
def standard_normal() -> dists.Distribution:
    return dists.Normal(torch.tensor([1.0]), torch.tensor([1.0]))


class NormalDistribution:
    def __init__(self,  loc, scale):
        self.loc = loc
        self.scale = scale
        self._loc = log_array(loc)
        self._scale = log_array(scale)

    def log_prob(self, data):
        return 1/(self._scale*np.sqrt(2*np.pi))*LogArray(-0.5*(data-self._loc)**2/self._scale**2)



def test_fisher_information(standard_normal):
    I = estimate_fisher_information(standard_normal)
    print(I)
