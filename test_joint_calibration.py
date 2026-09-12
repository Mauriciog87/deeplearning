from dataclasses import replace
from itertools import permutations
import json
from math import log
import unittest

import numpy as np
from scipy.special import gammaln
from scipy.stats import binom

from src.utils.joint_calibration import JointCalibrationMonitor
from src.utils.evaluation_harness import EvaluationConfig, _metric_values, _summarize_rows
from test_calibration_contract import forecast_row


class JointCalibrationTests(unittest.TestCase):
    def test_constant_vector_matches_dirichlet_likelihood(self):
        probabilities = np.full(37, 1 / 37)
        actuals = [7, 7, 2, 7, 4]
        monitor = JointCalibrationMonitor(prior_strength=37)
        for index, actual in enumerate(actuals):
            monitor.update([probabilities], actual, event_id=str(index))
        counts = np.bincount(actuals, minlength=37)
        expected = gammaln(37) - gammaln(37 + len(actuals)) + np.sum(gammaln(1 + counts)) + len(actuals) * log(37)
        self.assertAlmostEqual(monitor.log_evidence, expected, places=12)

    def test_joint_counterexample_has_zero_confidence_and_classwise_errors(self):
        rows = []
        for ordering in permutations(range(3)):
            inversions = sum(ordering[i] > ordering[j] for i in range(3) for j in range(i + 1, 3))
            probabilities = np.zeros(37)
            probabilities[:3] = np.array([.5, .3, .2])[list(ordering)]
            truth = probabilities.copy()
            truth[:3] += (-1 if inversions % 2 else 1) * np.array([.1, -.1, 0])
            for actual, count in enumerate(np.rint(truth * 1000).astype(int)):
                rows.extend((probabilities, actual) for _ in range(count))
        np.random.default_rng(1923).shuffle(rows)
        rows = [forecast_row(index, probabilities, actual) for index, (probabilities, actual) in enumerate(rows)]
        config = EvaluationConfig(runs=1, models=(), bootstrap_resamples=0)
        metrics = _metric_values(rows, config)
        self.assertAlmostEqual(metrics['ece'], 0, places=13)
        self.assertAlmostEqual(metrics['classwise_ece'], 0, places=13)
        summary = _summarize_rows(rows, config, len(rows))['example']
        self.assertAlmostEqual(summary.joint_calibration['joint_cell_total_variation'], .1, places=12)
        self.assertTrue(summary.joint_calibration['rejected'])
        self.assertEqual(summary.joint_calibration['observations'], 6000)

    def test_shared_seed_replicas_do_not_multiply_evidence(self):
        probability = np.full(37, .5 / 36)
        probability[7] = .5
        rows = [forecast_row(index, probability, 7) for index in range(80)]
        single = _summarize_rows(rows, EvaluationConfig(runs=1, bootstrap_resamples=0), 80)['example'].joint_calibration
        replicas = [replace(row, run_id=run) for run in range(5) for row in rows]
        repeated = _summarize_rows(replicas, EvaluationConfig(runs=5, bootstrap_resamples=0), 400)['example'].joint_calibration
        for key in ('log_evidence', 'max_log_evidence', 'anytime_p_value', 'allocated_alpha', 'observations', 'joint_cell_total_variation'):
            self.assertAlmostEqual(single[key], repeated[key], places=12)

    def test_sequential_null_uses_past_only_and_controls_false_alarms(self):
        experiments, alpha = 60, .05
        rng = np.random.default_rng(778)
        alarms = 0
        for trial in range(experiments):
            monitor = JointCalibrationMonitor(alpha=alpha)
            for index in range(600):
                probabilities = np.zeros(37)
                probabilities[:3] = [.5, .3, .2] if index % 2 else [.2, .5, .3]
                actual = int(rng.choice(37, p=probabilities))
                monitor.update([probabilities], actual, event_id=str(index))
            alarms += monitor.snapshot()['rejected']
        self.assertLessEqual(alarms, int(binom.ppf(.995, experiments, alpha)))

    def test_persistence_idempotency_and_support_violations(self):
        probabilities = np.full(37, 1 / 37)
        monitor = JointCalibrationMonitor()
        monitor.update([probabilities], 7, event_id='1')
        restored = JointCalibrationMonitor.from_state(json.loads(json.dumps(monitor.to_state())))
        self.assertFalse(restored.update([probabilities], 7, event_id='1'))
        with self.assertRaises(ValueError):
            restored.update([probabilities], 8, event_id='1')
        for item in (monitor, restored):
            item.update([probabilities], 7, event_id='2')
        self.assertEqual(restored.snapshot(), monitor.snapshot())
        probabilities[7] = 0
        probabilities /= probabilities.sum()
        restored.update([probabilities], 7, event_id='3')
        result = restored.snapshot()
        self.assertTrue(result['support_violation'])
        self.assertTrue(result['rejected'])
        self.assertIsNone(result['log_evidence'])
        json.dumps(result, allow_nan=False)

    def test_missing_seed_cohort_is_unavailable(self):
        rows = [forecast_row(index, np.full(37, 1 / 37), 7) for index in range(30)]
        summary = _summarize_rows(rows, EvaluationConfig(runs=2, bootstrap_resamples=0), 60)['example']
        self.assertEqual(summary.joint_calibration['status'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
