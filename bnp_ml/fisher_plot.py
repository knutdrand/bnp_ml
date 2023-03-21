from bnp_ml.bernoulli import Bernoulli
from bnp_ml.jax_wrapper import estimate_fisher_information, estimate_sgd, linear_fisher_information, estimate_gd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


def param_diffs(dist_1, dist_2):
    # return (dist_2.p-dist_1.p, )
    return tuple(p2-p1 for p2, p1 in zip(dist_2.parameters, dist_1.parameters))


def fisher_table(dist, estimator, sample_sizes=None, n_fisher=100000):
    if sample_sizes is None:
        sample_sizes = np.arange(1, 5)*10
    fisher_info = linear_fisher_information(dist, n=n_fisher)
    # sds = np.sqrt(1/(sample_sizes*fisher_info))

    estimates = [estimator(dist.__class__(*[0.6 for param in dist.parameters]),
                           dist.sample((sample_size, )))
                 for sample_size in sample_sizes]
    errors = [param_diffs(estimate, dist) for estimate in estimates]

    return {'sample_size': sample_sizes,
            'z_score': np.array(errors).ravel()/np.sqrt(1/(sample_sizes*fisher_info[0][0]))}


def fisher_plot2(dist, estimator):
    print('getting fisher information')
    fisher_info = linear_fisher_information(dist, n=100000)
    print(fisher_info)
    sample_sizes = np.repeat(np.arange(1, 200), 4)
    sds = np.sqrt(1/(sample_sizes*fisher_info))
    print(sds)
    estimates = [estimator(dist.__class__(*[0.6 for param in dist.parameters]),
                           dist.sample((sample_size, )))
                 for sample_size in sample_sizes]
    # print([estimate.p for estimate in estimates])
    print(np.array([e.p for e in estimates]))
    errors = [param_diffs(estimate, dist) for estimate in estimates]
    # plt.scatter(sample_sizes, [(estimate_sgd(dist.__class__(*[0.6 for param in dist.parameters]), dist.sample((sample_size, )), n_iterations=100).p-dist.p) for sample_size, sd in zip(sample_sizes, sds)])
    # plt.show()
    # errors = [(estimate_sgd(dist.__class__(*[0.6 for param in dist.parameters]), dist.sample((sample_size, )), n_iterations=100).p-dist.p) for sample_size, sd in zip(sample_sizes, sds)]
    print(np.array(errors).ravel())
    print(np.abs(errors))
    print(np.array(errors).shape)
    px.scatter(x=sample_sizes, y=np.array(errors).ravel()/np.sqrt(1/(sample_sizes*fisher_info[0][0])), marginal_y='histogram').show()
