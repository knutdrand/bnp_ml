from inspect import signature


def wrap_model_func(func):
    arg_names = list(signature(func).parameters.keys())

    class Model:
        def __init__(self, *args):
            self._rv = func(*args)
            self._args = args
            for name, arg in zip(arg_names, self._args):
                setattr(self, name, arg)

        @classmethod
        def parameter_names(cls):
            return arg_names

        @property
        def parameters(self):
            return self._args

        def log_prob(self, value):
            return self._rv.probability(value).log_prob()

        def sample(self, *args, **kwargs):
            return self._rv.sample(*args, **kwargs)

        @property
        def event_shape(self):
            return self._rv.event_shape

    return Model
