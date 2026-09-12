from math import log
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.special import gammaln, logsumexp

from src.utils.sequential_inference import MonitoringBudget, MultinomialMonitor


class SequentialInferenceTests(unittest.TestCase):
    def test_mixture_updates_equal_closed_form(self):
        monitor = MultinomialMonitor('table-a')
        numbers = np.random.default_rng(73).integers(0, 37, 370)
        for index, number in enumerate(numbers):
            monitor.update(int(number), event_id=str(index))
        counts = np.bincount(numbers, minlength=37)
        components = []
        for strength in monitor.prior_strengths:
            prior = np.full(37, strength / 37)
            components.append(gammaln(strength) - gammaln(strength + len(numbers))
                              + np.sum(gammaln(prior + counts) - gammaln(prior)))
        expected = logsumexp(components) - log(len(components)) + len(numbers) * log(37)
        self.assertAlmostEqual(monitor.log_evidence, expected, places=9)

    def test_repeated_event_is_idempotent_and_collision_fails(self):
        monitor = MultinomialMonitor('table-a')
        self.assertTrue(monitor.update(7, event_id='spin-1'))
        before = monitor.to_state()
        self.assertFalse(monitor.update(7, event_id='spin-1'))
        self.assertEqual(monitor.to_state(), before)
        with self.assertRaises(ValueError):
            monitor.update(8, event_id='spin-1')
        self.assertEqual(monitor.to_state(), before)

    def test_atomic_state_resume_matches_uninterrupted_updates(self):
        monitor = MultinomialMonitor('table-a', alpha=.013)
        numbers = [index % 37 for index in range(120)] + [7] * 150
        for index, number in enumerate(numbers[:120]):
            monitor.update(number, event_id=str(index))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'monitor.json'
            monitor.save(path)
            restored = MultinomialMonitor.load(path, stream_id='table-a')
            with self.assertRaises(ValueError):
                MultinomialMonitor.load(path, stream_id='other-table')
            for index, number in enumerate(numbers[120:], 120):
                monitor.update(number, event_id=str(index))
                restored.update(number, event_id=str(index))
            self.assertEqual(monitor.to_state(), restored.to_state())
            self.assertEqual(monitor.snapshot(), restored.snapshot())

    def test_projected_bounds_satisfy_likelihood_boundary(self):
        monitor = MultinomialMonitor('table-a')
        for index in range(370):
            monitor.update(index % 37, event_id=str(index))
        lower, upper = monitor.probability_bounds([0, 1, 2])
        self.assertLess(lower, 3 / 37)
        self.assertGreater(upper, 3 / 37)
        for mass in (lower, upper):
            probabilities = np.full(37, (1 - mass) / 34)
            probabilities[:3] = mass / 3
            log_ratio = monitor.log_marginal - float(np.dot(monitor.counts, np.log(probabilities)))
            self.assertAlmostEqual(log_ratio, -log(monitor.alpha), places=9)

    def test_zero_and_all_success_projection_endpoints(self):
        monitor = MultinomialMonitor('table-a')
        for index in range(200):
            monitor.update(7, event_id=str(index))
        self.assertEqual(monitor.probability_bounds([]), (0.0, 0.0))
        self.assertEqual(monitor.probability_bounds(range(37)), (1.0, 1.0))
        self.assertEqual(monitor.probability_bounds([0])[0], 0)
        self.assertEqual(monitor.probability_bounds([7])[1], 1)
        self.assertGreater(monitor.probability_bounds([7])[0], 1 / 36)
        self.assertTrue(monitor.snapshot()['rejected'])

    def test_stream_and_restart_allocations_fit_the_total_budget(self):
        budget = MonitoringBudget(.05, ('a', 'b', 'c'))
        allocations = [budget.allocation(stream, restart) for stream in budget.streams for restart in range(1000)]
        self.assertLess(sum(allocations), .05)
        self.assertGreater(sum(allocations), .049)
        self.assertLess(budget.allocation('a', 1), budget.allocation('a', 0))
        with self.assertRaises(ValueError):
            budget.allocation('undeclared')

    def test_sequential_null_error_is_bounded_and_large_bias_is_detected(self):
        from scipy.stats import binom
        rng = np.random.default_rng(842)
        rejected = 0
        for trial in range(100):
            monitor = MultinomialMonitor(str(trial), alpha=.05)
            for index, number in enumerate(rng.integers(0, 37, 1000)):
                monitor.update(int(number), event_id=str(index))
            rejected += monitor.snapshot()['rejected']
        self.assertLessEqual(rejected, int(binom.ppf(.995, 100, .05)))
        monitor = MultinomialMonitor('biased')
        for index in range(1000):
            monitor.update(7 if index % 5 == 0 else int(rng.integers(0, 37)), event_id=str(index))
        self.assertTrue(monitor.snapshot()['rejected'])

    def test_invalid_configuration_and_outcomes_are_rejected_before_mutation(self):
        for alpha in (0, 1, float('nan')):
            with self.assertRaises(ValueError):
                MultinomialMonitor('a', alpha=alpha)
        monitor = MultinomialMonitor('a')
        for value in (True, 37, -1, 1.5):
            with self.assertRaises(ValueError):
                monitor.update(value, event_id='bad')
        self.assertEqual(monitor.observations, 0)


if __name__ == '__main__':
    unittest.main()
