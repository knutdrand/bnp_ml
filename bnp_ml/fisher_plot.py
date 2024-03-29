from bnp_ml.bernoulli import Bernoulli
import logging
from bnp_ml.jax_wrapper import estimate_fisher_information, estimate_sgd, linear_fisher_information, estimate_gd
import numpy as np
import plotly.express as px
from collections import defaultdict
from .jax_wrapper import init_like
logger = logging.getLogger(__name__)


def param_diffs(dist_1, dist_2):
    # return (dist_2.p-dist_1.p, )
    return tuple(p2-p1 for p2, p1 in zip(dist_2.parameters, dist_1.parameters))


def plot_table(table):
    return px.scatter(data_frame=table, x='sample_size', y='z_score', 
                      facet_row='param_name',
                      color='param_idx',
                      marginal_y='histogram')


def fisher_table(dist, estimator, sample_sizes=None, n_fisher=100000, rng=None):
    if sample_sizes is None:
        sample_sizes = np.arange(1, 5)*10
    fisher_info = linear_fisher_information(dist, n=n_fisher, rng=rng)
    table = defaultdict(list)
    for sample_size in sample_sizes:
        if hasattr(dist, 'is_natural') and dist.is_natural:
            init_dist = init_like(dist, rng)
        else:
            init_dist = dist.__class__(*[np.full_like(param, 0.6)+np.random.rand(*param.shape)/50
                                         for param in dist.parameters])
        s = dist.sample((sample_size, )) if rng is None else dist.sample(rng, (sample_size, ))
        estimate = estimator(init_dist, s)
        logger.info(estimate.parameters)
        for i, (errors, params) in enumerate(zip(param_diffs(estimate, dist), estimate.parameters)):
            for j, (error, param) in enumerate(zip(np.atleast_1d(errors), np.atleast_1d(params))):
                sd = 1/np.sqrt(sample_size*fisher_info[i][j])
                table['sample_size'].append(sample_size)
                table['param_name'].append(dist.parameter_names()[i])
                table['param_idx'].append(str(j))
                table['z_score'].append(error/sd)
                table['param'].append(param)
    return table
    return {name: np.array(l) for name, l in table.items()}
    estimates = [estimator(dist.__class__(*[0.6 for param in dist.parameters]),
                           dist.sample((sample_size, )))
                 for sample_size in sample_sizes]

    errors = [param_diffs(dist, estimate) for estimate in estimates]

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
