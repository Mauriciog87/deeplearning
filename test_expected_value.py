import unittest

import numpy as np
from scipy.stats import beta

from src.expected_value import choose_expected_value_action, counterfactual_profits, expected_action_values
from src.settlement import settle_action
from src.utils.bias_detection import compute_probability_with_ci
from src.utils.evaluation_harness import EvaluationConfig, evaluate_walk_forward
from src.utils.sequential_inference import MultinomialMonitor
from src.utils.statistics import compute_probability_with_confidence, estimate_ornstein_uhlenbeck_params, filter_profitable_numbers


class ExpectedValueTests(unittest.TestCase):
    def test_all_counterfactual_actions_match_settlement(self):
        for actual in range(37):
            profits = counterfactual_profits(actual, 3)
            self.assertEqual(len(profits), 47)
            for action in range(47):
                self.assertEqual(profits[action], settle_action(actual, action, 3, 20).net_profit)
        values = expected_action_values(np.full(37, 1 / 37))
        np.testing.assert_allclose(values[:46], -1 / 37, atol=1e-14)
        self.assertEqual(values[46], 0)
        self.assertEqual(choose_expected_value_action(np.full(37, 1 / 37), 10).action, 46)

    def test_point_and_conservative_decisions_and_bankroll(self):
        probabilities = np.full(37, .9 / 36)
        probabilities[7] = .1
        choice = choose_expected_value_action(probabilities, 5, 2)
        self.assertEqual(choice.action, 7)
        self.assertAlmostEqual(choice.expected_profit, 5.2)
        self.assertEqual(choose_expected_value_action(probabilities, 1, 2).action, 46)
        monitor = MultinomialMonitor('test')
        self.assertEqual(choose_expected_value_action(probabilities, 5, probability_bounds=monitor.probability_bounds).action, 46)
        for index in range(200):
            monitor.update(7, event_id=str(index))
        choice = choose_expected_value_action(probabilities, 5, probability_bounds=monitor.probability_bounds)
        self.assertEqual(choice.action, 7)
        self.assertGreater(choice.lower_expected_profit, 0)

    def test_above_uniform_is_not_a_profitable_lower_bound(self):
        numbers = [7] * 345 + [number for number in range(37) if number != 7] * 309 + [0] * 31
        self.assertEqual(len(numbers), 11500)
        estimate = compute_probability_with_confidence(numbers, 7)
        self.assertGreater(estimate.confidence_interval_95[0], 1 / 37)
        self.assertLess(estimate.confidence_interval_95[0], 1 / 36)
        self.assertFalse(estimate.passes_threshold)
        self.assertNotIn(7, [item.number for item in filter_profitable_numbers(numbers)])
        self.assertFalse(compute_probability_with_ci(numbers, 7)['passes_3pct_threshold'])

    def test_exact_fixed_sample_family_interval_matches_beta_quantiles(self):
        sample = [7] * 12 + [0] * 88
        result = compute_probability_with_confidence(sample, 7, .9)
        tail = .1 / 37 / 2
        self.assertAlmostEqual(result.confidence_interval[0], beta.ppf(tail, 12, 89))
        self.assertAlmostEqual(result.confidence_interval[1], beta.ppf(1 - tail, 13, 88))
        pointwise = compute_probability_with_confidence(sample, 7, .9, family_size=1)
        self.assertLess(result.confidence_interval[0], pointwise.confidence_interval[0])
        self.assertGreater(result.confidence_interval[1], pointwise.confidence_interval[1])

    def test_iid_overlap_matches_descriptive_null(self):
        numbers = np.random.default_rng(20260912).integers(0, 37, 30000).tolist()
        fit = estimate_ornstein_uhlenbeck_params(numbers, 7, 100)
        self.assertAlmostEqual(fit.iid_overlap_lag1, .99)
        self.assertAlmostEqual(fit.iid_overlap_theta, .01)
        self.assertLess(abs(fit.raw_indicator_lag1), .02)
        self.assertLess(abs(fit.rolling_frequency_lag1 - fit.iid_overlap_lag1), .005)
        self.assertLess(abs(fit.theta - fit.iid_overlap_theta), .005)
        self.assertIn('not the next spin probability', fit.interpretation)
        constant = estimate_ornstein_uhlenbeck_params([0] * 200, 7)
        self.assertEqual(constant.theta, 0)
        self.assertIsNone(constant.rolling_frequency_lag1)

    def test_walk_forward_policy_uses_prefix_and_shared_outcomes(self):
        config = EvaluationConfig(training_window=40, testing_window=10, step_size=20, models=(),
                                  runs=2, compute_intervals=False, initial_bankroll=1)
        result = evaluate_walk_forward([7] * 50 + [0] * 20, config)
        policies = [row for row in result.rows if row.model == 'expected_value']
        first = policies[0]
        self.assertEqual(first.action, 7)
        self.assertEqual(first.total_stake, 1)
        self.assertAlmostEqual(first.expected_profit, 36 * 41 / 77 - 1)
        for model in ('expected_value', 'expected_value_cs'):
            by_run = [[(row.spin_id, row.action, row.profit) for row in result.rows if row.model == model and row.run_id == run]
                      for run in range(2)]
            self.assertEqual(by_run[0], by_run[1])
            self.assertIsNone(result.per_model_summaries[model].log_loss)
        self.assertTrue(all(row.balance_after >= 0 for row in result.rows))

    def test_invalid_inputs_do_not_create_decisions(self):
        for probabilities, balance, stake in (([1], 1, 1), (np.full(37, 1 / 37), -1, 1), (np.full(37, 1 / 37), 1, 0)):
            with self.assertRaises(ValueError):
                choose_expected_value_action(probabilities, balance, stake)
        with self.assertRaises(ValueError):
            compute_probability_with_confidence([], 7)
        with self.assertRaises(ValueError):
            filter_profitable_numbers([37])


if __name__ == '__main__':
    unittest.main()
