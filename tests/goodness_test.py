from collections import Counter
import scipy.stats as stats
import numpy as np
from numpy.testing import assert_approx_equal
# no of hours a student studies
# in a week vs expected no of hours
observed_data = [8, 6, 10, 7, 8, 11, 9]
expected_data = [9, 8, 11, 8, 10, 7, 6]


def assert_sample_logprob_fit(model, n_samples: int = 500, alpha: float=0.05):
    domain = model.domain()
    observed = Counter(model.sample(np.random.default_rng(), (n_samples,)))
    observed = [observed[key] for key in domain]
    assert_approx_equal(sum(np.exp(model.log_prob(key)) for key in domain), 1)
    expected = np.array([n_samples*np.exp(model.log_prob(key)) for key in domain])
    assert_approx_equal(sum(observed), np.sum(expected))
    expected *= sum(observed)/np.sum(expected)
    assert_goodness_of_fit(list(observed), expected, alpha)


def assert_goodness_of_fit(observed_data, expected_data, alpha=0.05):
    # Chi-Square Goodness of Fit Test
    chi_square_test_statistic, p_value = stats.chisquare(
    	observed_data, expected_data)
    assert p_value >= alpha
