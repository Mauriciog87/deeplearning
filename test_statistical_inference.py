import unittest

import numpy as np
from scipy.stats import binom, binomtest

from src.utils.randomness import _entropy_drift_test, _categorical_change_point_test
from src.utils.randomization import uniformity_test, permutation_pvalue, transition_statistic
from src.utils.multiple_testing import false_discovery_control
from src.utils.heatmaps import _maybe_anomaly


class StatisticalInferenceTests(unittest.TestCase):
    def test_sector_null_accounts_for_unequal_pocket_counts(self):
        from src.utils.bias_detection import sector_bias_test
        result = sector_bias_test(list(range(37)) * 100, n_sectors=8)
        self.assertAlmostEqual(result.statistic, 0)
        self.assertEqual(result.p_value, 1)

    def test_bias_report_exposes_adjusted_family(self):
        from src.utils.bias_detection import generate_bias_report
        report = generate_bias_report(list(range(37)) * 10)
        for result in (report.chi_square_result, report.sector_result, report.runs_result):
            self.assertGreaterEqual(result.details['q_value'], result.p_value)

    def test_identical_biased_windows_do_not_imply_drift(self):
        window = [0] * 50 + list(range(1, 26)) * 2
        numbers = window * 10
        entropy = _entropy_drift_test(numbers, 100, 100, .01, resamples=99)
        distribution = _categorical_change_point_test(numbers, 100, 100, .01, resamples=99)
        self.assertEqual(entropy.statistic, 0)
        self.assertEqual(entropy.p_value, 1)
        self.assertEqual(distribution.statistic, 0)
        self.assertEqual(distribution.p_value, 1)

    def test_iid_false_positives_stay_under_predeclared_binomial_limits(self):
        experiments, alpha = 100, .05
        maximum = int(binom.ppf(.995, experiments, alpha))
        rng = np.random.default_rng(512)
        uniform_rejections = transition_rejections = 0
        for seed in range(experiments):
            numbers = rng.integers(0, 37, size=100).tolist()
            _, uniform_p, method = uniformity_test(numbers, resamples=199, seed=seed)
            _, transition_p = permutation_pvalue(numbers, transition_statistic, resamples=199, seed=seed)
            self.assertEqual(method, 'multinomial Monte Carlo')
            uniform_rejections += uniform_p < alpha
            transition_rejections += transition_p < alpha
        self.assertLessEqual(uniform_rejections, maximum)
        self.assertLessEqual(transition_rejections, maximum)

    def test_by_is_no_less_conservative_than_bh(self):
        values = [.0001, .01, .04, .2, None]
        bh, _ = false_discovery_control(values, method='bh')
        by, _ = false_discovery_control(values, method='by')
        self.assertTrue(all(y >= h for h, y in zip(bh, by) if h is not None))

    def test_heatmap_uses_exact_binomial_probability(self):
        anomaly = _maybe_anomaly(7, 'number', 'all', .1, 1 / 37, 100, 0, True)[0]
        self.assertEqual(anomaly.p_value, binomtest(10, 100, 1 / 37).pvalue)


if __name__ == '__main__':
    unittest.main()
