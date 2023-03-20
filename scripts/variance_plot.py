from bnp_ml.bernoulli import Bernoulli
from bnp_ml.jax_wrapper import estimate_fisher_information, estimate_sgd, linear_fisher_information, estimate_gd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def fisher_plot():
    dist = Bernoulli(0.4)
    print('getting fisher information')
    fisher_info = linear_fisher_information(dist, n=100000)
    sample_sizes = np.arange(1, 10)*10
    variances = []
    for sample_size in sample_sizes:
        print(f'running for sample_size {sample_size}')
        X = dist.sample((sample_size, ))
        estimated_ps = np.array([estimate_sgd(Bernoulli(0.6), X, n_iterations=100).p
                                 for _ in range(100)])
        variances.append(np.sum((estimated_ps-dist.p)**2)/100)
    plt.plot(sample_sizes, 1/np.array(variances))
    plt.show()


def param_diffs(dist_1, dist_2):
    return (dist_2.p-dist_1.p, )
    return tuple(p2-p1 for p2, p1 in zip(dist_1.parameters, dist_2.parameters))


def fisher_plot2(dist):
    dist = Bernoulli(0.4)
    print('getting fisher information')
    fisher_info = linear_fisher_information(dist, n=100000)
    print(fisher_info)
    sample_sizes = np.arange(1, 20)*10
    sds = np.sqrt(1/(sample_sizes*fisher_info))
    print(sds)
    estimates = [estimate_sgd(dist.__class__(*[0.6 for param in dist.parameters]),
                              dist.sample((sample_size, )),
                              n_iterations=1000)
                 for sample_size in sample_sizes]
    # print([estimate.p for estimate in estimates])
    # errors = [param_diffs(estimate, dist) for estimate in estimates]
    plt.scatter(sample_sizes, [(estimate_sgd(dist.__class__(*[0.6 for param in dist.parameters]), dist.sample((sample_size, )), n_iterations=100).p-dist.p) for sample_size, sd in zip(sample_sizes, sds)])
    plt.show()
    # errors = [(estimate_sgd(dist.__class__(*[0.6 for param in dist.parameters]), dist.sample((sample_size, )), n_iterations=100).p-dist.p) for sample_size, sd in zip(sample_sizes, sds)]
    print(np.abs(errors))
    print(np.array(errors).shape)
    plt.scatter(sample_sizes, np.abs(np.array(errors).ravel()))
    plt.show()
    for error_group, info_group in zip(zip(*errors), fisher_info):
        print('>>>', np.array(error_group))
        for error, info in zip(error_group, info_group):
            print('>>>>>>', error_group)
            plt.scatter(sample_sizes, error/np.sqrt(1/(sample_sizes*info)))

    plt.show()


fisher_plot2(Bernoulli(0.4))
