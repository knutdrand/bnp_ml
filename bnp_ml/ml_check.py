import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import jax
import dataclasses
from typing import Protocol

class Distribution:
    def log_pmf(self, *data):
        return NotImplemented
    
    def sample(self, sample_shape):
        return NotImplemented


def get_var(information):
    return np.linalg.inv(information)


def get_func(self, *x):
    return lambda *params: torch.mean(self.log_likelihood(*x, *params))


def get_log_likelihood_function(distribution_class: type, data: torch.Tensor):
    def log_likelihood_function(*args, **kwargs):
        return np.mean(distribution_class(*args, **kwargs).log_prob(data))
    return log_likelihood_function


def estimate_fisher_information(model: Distribution, n=10000000, rng=None):
    n //= math.prod(model.event_shape)
    if rng is not None:
        x = model.sample(rng, (n,))
    else:
        x = model.sample((n,))
    f = get_log_likelihood_function(model.__class__, x)
    return jax.hessian(f)(*(getattr(model, key).numpy() for key in model.arg_constraints))
# H = torch.autograd.functional.hessian(f, tuple(getattr(model, key) for key in model.arg_constraints))
#     return [H[i][i].diag().numpy() for i in range(len(H))]


@dataclasses.dataclass
class Distribution(ABC):

    @abstractmethod
    def sample(self, n=100):
        pass

    @abstractmethod
    def log_likelihood(self, *args, **kwargs):
        pass

    def get_func(self, *x):
        return lambda *params: torch.mean(self.log_likelihood(*x, *params))

    @property
    def data_size(self):
        return 1

    def __post_init__(self):
        for field in dataclasses.fields(self):
            if field.type == torch.tensor:
                setattr(self, field.name, torch.as_tensor(getattr(self, field.name)))

    @property
    def params(self):
        return tuple(getattr(self, field.name) for field in dataclasses.fields(self))        

    def estimate_fisher_information(self, n=10000000):
        n = n//self.data_size
        x = self.sample(n)
        f = self.get_func(*x)
        H = torch.autograd.functional.hessian(f, self.params)
        dims = [p.numpy().size for p in self.params]
        H = np.vstack(
            [np.hstack([r.reshape((a_dim, b_dim)) for r, b_dim in zip(row, dims)])
             for a_dim, row in zip(dims, H)])
        return -np.array(H)

    def _flatten_params(self, params):
        return np.concatenate([p.ravel() for p in params])

    def plot_all_errors(self, color="red", n_params=None, n_iterations=200):
        params = self._flatten_params(self.params)
        if n_params is None:
            n_params = params.size
        name = self.__class__.__name__
        I = self.estimate_fisher_information()
        I = I[:n_params, :n_params]
        print(I)
        all_var = get_var(I)
        n_samples = [200*i for i in range(1, 10)]
        errors = (self.get_square_errors(n_samples=n, n_iterations=n_iterations, do_plot=False)
                  for n in n_samples)
        errors = [self._flatten_params(e) for e in errors]
        print(np.mean(errors, axis=0))
        fig, axes = plt.subplots((n_params+1)//2, 2)
        if (n_params+1)//2 == 1:
            axes = [axes]
        for i, param in enumerate(params[:n_params]):
            var = all_var[i, i]
            ax = axes[i//2][i % 2]
            ax.axline((0, 0), slope=1/var, color=color, label=name+" CRLB")
            ax.plot(n_samples, 1/np.array(errors)[:, i], color=color, label=name+" errors")
            ax.set_ylabel("1/sigma**2")
            ax.set_xlabel("n_samples")

    def get_square_errors(self, n_samples=1000, n_iterations=1000, do_plot=False):
        estimates = zip(*(self.estimate_parameters(n_samples)
                          for _ in range(n_iterations)))
        estimates = [np.array(e) for e in estimates]
        return [((e-np.array(p))**2).sum(axis=0)/n_iterations
                for e, p in zip(estimates, self.params)]
        """
        if do_plot:
            for i, param in enumerate(true_params):
                plt.hist(estimates[:, i])
                plt.axvline(x=param)
                plt.title(f"n={n_samples}")
                plt.show()
        print("E", estimates.mean(axis=0))
        print("T", true_params)
        return ((estimates-true_params)**2).sum(axis=0)/n_iterations
        """
