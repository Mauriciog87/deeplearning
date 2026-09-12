import json
from math import log
import unittest

import numpy as np
from scipy.stats import binom

from src.utils.change_detection import CategoricalEDetector, PITMonitor, ReferenceConditionalMonitor, categorical_pit


class ChangeDetectionTests(unittest.TestCase):
    def test_categorical_pit_respects_mass_order_and_randomized_ties(self):
        probabilities = np.zeros(37)
        probabilities[:3] = [.2, .5, .3]
        self.assertAlmostEqual(categorical_pit(probabilities, 1, .4), .4)
        self.assertAlmostEqual(categorical_pit(probabilities, 1, .4, order=tuple(reversed(range(37)))), .5)
        with self.assertRaises(ValueError):
            categorical_pit(probabilities, 1, 1)
        with self.assertRaises(ValueError):
            categorical_pit(probabilities, 37, .5)

    def test_e_sr_and_cusum_match_explicit_start_processes(self):
        actuals = [7, 7, 12, 7, 7, 0, 7]
        factors = np.asarray([[.9 + (3.7 if number == actual else 0) for number in range(37)] for actual in actuals])
        products = np.asarray([factors[start:].prod(axis=0) for start in range(len(actuals))])
        sr, cusum = CategoricalEDetector(), CategoricalEDetector(variant='cusum')
        for index, actual in enumerate(actuals):
            sr.update(actual, event_id=str(index))
            cusum.update(actual, event_id=str(index))
        self.assertAlmostEqual(sr.log_statistic, log(products.sum(axis=0).mean()))
        self.assertAlmostEqual(cusum.log_statistic, log(products.max(axis=0).mean()))
        statistics = np.exp(sr._statistics)
        expected_next = np.mean((statistics + 1)[:, None] * np.exp(sr._log_factors))
        self.assertAlmostEqual(expected_next, statistics.mean() + 1)
        self.assertIn('not probability of ever alarming', sr.snapshot()['interpretation'])

    def test_pit_full_mixture_includes_unstarted_processes(self):
        monitor = PITMonitor(bins=4, seed=52)
        self.assertEqual(monitor.snapshot()['log_evidence'], 0)
        for index, pit in enumerate([.2, .1, .2, .8, .9, .9, .6]):
            monitor.update(pit, event_id=str(index))
        counts = np.ones(4)
        factors = []
        for p_value in monitor._p_values:
            cell = int(p_value * 4)
            factors.append(4 * counts[cell] / counts.sum())
            counts[cell] += 1
        full = 1 / (len(factors) + 1) + sum(np.prod(factors[start:]) / ((start + 1) * (start + 2)) for start in range(len(factors)))
        self.assertAlmostEqual(monitor.snapshot()['log_evidence'], log(full), places=13)

    def test_reference_linear_bets_account_for_finite_reference(self):
        monitor = ReferenceConditionalMonitor(np.linspace(0, 1, 100), alpha=.1)
        monitor.update(2, event_id='first')
        expected = np.mean(1 + monitor._bets * .5 - abs(monitor._bets) * monitor.epsilon)
        self.assertAlmostEqual(np.exp(monitor._wealth).mean(), expected)
        self.assertLess(expected, 1)
        self.assertEqual(monitor.snapshot()['total_error_bound'], .1)
        self.assertAlmostEqual(monitor.snapshot()['conditional_monitoring_alpha'] + monitor.reference_delta, .1)

    def test_null_campaign_handles_discrete_scores_and_pit_atoms(self):
        experiments, horizon, alpha = 50, 500, .05
        rng = np.random.default_rng(771)
        alarms = {'reference': 0, 'pit': 0, 'sr': 0}
        for trial in range(experiments):
            reference = ReferenceConditionalMonitor(rng.integers(0, 2, 250), seed=trial, alpha=alpha)
            pit = PITMonitor(seed=trial, alpha=alpha)
            sr = CategoricalEDetector(average_run_length=horizon / alpha)
            for index in range(horizon):
                reference.update(int(rng.integers(0, 2)), event_id=str(index))
                pit.update(0, event_id=str(index))
                sr.update(int(rng.integers(0, 37)), event_id=str(index))
            for name, monitor in (('reference', reference), ('pit', pit), ('sr', sr)):
                alarms[name] += monitor.snapshot()['rejected']
        for name, count in alarms.items():
            with self.subTest(method=name):
                self.assertLessEqual(count, int(binom.ppf(.995, experiments, alpha)))

    def test_shift_detection_and_calibration_improvement_are_distinct_from_alarm_time(self):
        reference = ReferenceConditionalMonitor(np.linspace(0, 1, 500))
        saved_reference = reference.reference.copy()
        pit = PITMonitor(seed=92)
        sr = CategoricalEDetector(average_run_length=1000)
        rng = np.random.default_rng(41)
        for index in range(600):
            reference.update(float(rng.uniform()) if index < 300 else 2, event_id=str(index))
            pit.update(0 if index < 300 else float(rng.uniform()), event_id=str(index))
            sr.update(int(rng.integers(0, 37)) if index < 300 else 7, event_id=str(index))
        for monitor in (reference, pit, sr):
            self.assertTrue(monitor.snapshot()['rejected'])
            self.assertGreater(monitor.snapshot()['alarm_at'], 300)
        np.testing.assert_array_equal(reference.reference, saved_reference)
        location = pit.snapshot()['changepoint']
        self.assertLess(location['estimated_boundary_after'], pit.snapshot()['alarm_at'])
        self.assertIn('improvements', pit.snapshot()['interpretation'])

    def test_state_resume_and_duplicates_preserve_randomization_and_evidence(self):
        for monitor in (PITMonitor(seed=4), ReferenceConditionalMonitor([0] * 50 + [1] * 50, seed=4), CategoricalEDetector()):
            with self.subTest(method=type(monitor).__name__):
                for index in range(30):
                    monitor.update(index % 2, event_id=str(index))
                restored = type(monitor).from_state(json.loads(json.dumps(monitor.to_state(), allow_nan=False)))
                self.assertFalse(restored.update(0, event_id='0'))
                with self.assertRaises(ValueError):
                    restored.update(1, event_id='0')
                for index in range(30, 60):
                    for item in (monitor, restored):
                        item.update(index % 2, event_id=str(index))
                self.assertEqual(restored.snapshot(), monitor.snapshot())
                json.dumps(restored.snapshot(), allow_nan=False)


if __name__ == '__main__':
    unittest.main()
