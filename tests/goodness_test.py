from collections import Counter
import scipy.stats as stats
import numpy as np

# no of hours a student studies
# in a week vs expected no of hours
observed_data = [8, 6, 10, 7, 8, 11, 9]
expected_data = [9, 8, 11, 8, 10, 7, 6]


def assert_sample_logprob_fit(model, n_samples: int = 1000, alpha: float=0.05):
    observed = Counter(model.sample(np.random.default_rng(), (n_samples,)))
    expected = [n_samples*np.exp(model.log_prob(key)) for key in observed]
    assert_goodness_of_fit(list(observed.values()), expected, alpha)


def assert_goodness_of_fit(observed_data, expected_data, alpha=0.05):
    # Chi-Square Goodness of Fit Test
    chi_square_test_statistic, p_value = stats.chisquare(
    	observed_data, expected_data)
    assert p_value >= alpha
