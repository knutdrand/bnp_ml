from bnp_ml.model import curry


class DummyModel:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def log_prob(self, *args, **kwargs):
        return

    def sample(self, rng, shape):
        return NotImplemented

    @property
    def parameters(self):
        return [self.a, self.b]

    @classmethod
    def parameter_names(self):
        return ['a', 'b']


def test_curry():
    new_model = curry(DummyModel, 10)
    assert new_model.parameter_names() == ['b']
    instance = new_model(20)
    assert instance.parameters == [20]
