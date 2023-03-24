from typing import Protocol
from functools import partial


class Model(Protocol):
    def log_prob(self, *args, **kwargs):
        return NotImplemented

    def sample(self, rng, shape):
        return NotImplemented

    @property
    def parameters(self):
        return NotImplemented

    @classmethod
    def parameter_names(self):
        return NotImplemented


def curry(model_class, *args, **kwargs):
    idxs = [i for i, _ in enumerate(args)] + [model_class.parameter_names().index(key) for key in kwargs]
    init = partial(model_class, *args, **kwargs)

    class NewModel:
        is_natural = True
        def __init__(self, *args, **kwargs):
            self._model = init(*args, **kwargs)

        def log_prob(self, *args, **kwargs):
            return self._model.log_prob(*args, **kwargs)

        def sample(self, *args, **kwargs):
            return self._model.sample(*args, **kwargs)

        @property
        def event_shape(self):
            return self._model.event_shape

        @property
        def parameters(self):
            return [param for i, param in enumerate(self._model.parameters) if i not in idxs]

        @classmethod
        def parameter_names(cls):
            return [param for i, param in enumerate(model_class.parameter_names()) if i not in idxs]

    return NewModel
