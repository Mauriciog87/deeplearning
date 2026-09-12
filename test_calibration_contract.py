from dataclasses import replace
import unittest

import numpy as np

from src.utils.evaluation_harness import EvaluationConfig, PredictionEvaluationRow, _metric_values, _summarize_rows
from src.utils.temporal_statistics import adaptive_bins
from src.utils.calibration import fixed_bin_calibration_bound, hilbert_mean_radius


def forecast_row(index, probabilities, actual, run_id=0):
    ranking = sorted(enumerate(probabilities), key=lambda item: (-item[1], item[0]))
    return PredictionEvaluationRow(
        0, index, 'example', actual, ranking[0][0], ranking[0][1],
        dict(enumerate(probabilities)), ranking, 0, run_id=run_id, total_stake=0,
    )


class CalibrationContractTests(unittest.TestCase):
    def test_identical_seed_replicas_do_not_change_calibration(self):
        rows = []
        for index, confidence in enumerate(np.linspace(.05, .44, 40)):
            probabilities = np.full(37, (1 - confidence) / 36)
            probabilities[0] = confidence
            actual = 0 if 10 <= index < 20 or 30 <= index < 40 else 1
            rows.append(forecast_row(index, probabilities, actual))
        repeated = [replace(row, run_id=run) for run in range(5) for row in rows]
        config = EvaluationConfig()
        single_metrics = _metric_values(rows, config)
        repeated_metrics = _metric_values(repeated, config)
        for metric in ('ece', 'classwise_ece', 'top_3_ece', 'top_5_ece', 'top_10_ece'):
            with self.subTest(metric=metric):
                self.assertAlmostEqual(single_metrics[metric], repeated_metrics[metric], places=14)

    def test_different_run_errors_do_not_cancel(self):
        first = np.full(37, .5 / 36)
        first[0] = .5
        second = np.full(37, .5 / 36)
        second[1] = .5
        rows = [forecast_row(index, probabilities, 0, run)
                for run, probabilities in enumerate((first, second)) for index in range(40)]
        self.assertAlmostEqual(_metric_values(rows, EvaluationConfig())['ece'], .5)

    def test_two_full_bins_fit_exactly_forty_observations(self):
        bins = adaptive_bins(np.linspace(.05, .44, 40), np.zeros(40))
        self.assertEqual([entry[2] for entry in bins], [20, 20])

    def test_tied_confidences_are_not_split(self):
        bins = adaptive_bins([.1] * 39 + [.8] * 21, [0] * 60)
        self.assertEqual([entry[2] for entry in bins], [39, 21])

    def test_fair_predictor_bounds_include_zero_without_bootstrap(self):
        outcomes = np.random.default_rng(20260912).integers(0, 37, 200)
        rows = [forecast_row(index, np.full(37, 1 / 37), int(actual))
                for index, actual in enumerate(outcomes)]
        config = EvaluationConfig(runs=1, models=(), bootstrap_resamples=0)
        result = _summarize_rows(rows, config, len(rows))['example']
        for bound in result.calibration_bounds.values():
            self.assertEqual(bound.lower, 0)
            self.assertGreaterEqual(bound.upper, bound.point)
            self.assertEqual(bound.status, 'available')
            self.assertIn('conditional residual', bound.target)

    def test_replication_does_not_change_calibration_bounds(self):
        rows = [forecast_row(index, np.full(37, 1 / 37), index % 37) for index in range(200)]
        repeated = [replace(row, run_id=run) for run in range(5) for row in rows]
        def summarize(data, runs):
            return _summarize_rows(data, EvaluationConfig(runs=runs, models=(), bootstrap_resamples=0), len(data))['example']
        single, multiple = summarize(rows, 1), summarize(repeated, 5)
        for name, first in single.calibration_bounds.items():
            other = multiple.calibration_bounds[name]
            for field in ('point', 'lower', 'upper', 'allocated_alpha', 'observations'):
                self.assertAlmostEqual(getattr(first, field), getattr(other, field), places=14)

    def test_incomplete_seed_cohort_cannot_claim_bounds(self):
        rows = [forecast_row(index, np.full(37, 1 / 37), index % 37) for index in range(200)]
        result = _summarize_rows(rows, EvaluationConfig(runs=5, models=(), bootstrap_resamples=0), 200)['example']
        self.assertTrue(result.calibration_bounds)
        for bound in result.calibration_bounds.values():
            self.assertEqual(bound.status, 'unavailable')
            self.assertIsNone(bound.lower)

    def test_bounds_cover_known_conditional_targets_with_dependent_outcomes(self):
        rng = np.random.default_rng(2917)
        failures = 0
        for _ in range(100):
            truth, forecasts, outcomes = [], [], []
            previous = 0
            for index in range(1000):
                probability = .15 + .65 * previous
                prediction = .35 + .2 * previous
                actual = int(rng.random() < probability)
                truth.append(probability)
                forecasts.append(prediction)
                outcomes.append(actual)
                previous = actual
            for count in (100, 250, 500, 1000):
                predicted = np.array(forecasts[:count])[None, :, None]
                observed = np.array(outcomes[:count])[None, :, None]
                bound = fixed_bin_calibration_bound(predicted, observed, bins=10, alpha=.05)
                groups = (predicted[0, :, 0] * 10).astype(int)
                residual = np.array(truth[:count]) - predicted[0, :, 0]
                target = np.abs(np.bincount(groups, weights=residual, minlength=10)).sum() / count
                if not bound.lower <= target <= bound.upper:
                    failures += 1
                    break
        from scipy.stats import binom
        self.assertLessEqual(failures, int(binom.ppf(.995, 100, .05)))

    def test_bounds_detect_large_known_error_and_contract_with_data(self):
        widths = []
        for count in (4096, 16384):
            bound = fixed_bin_calibration_bound(np.full((1, count, 1), .1), np.ones((1, count, 1)))
            self.assertLessEqual(bound.lower, .9)
            self.assertGreaterEqual(bound.upper, .9)
            self.assertGreater(bound.lower, .5)
            widths.append(bound.upper - bound.lower)
        self.assertLess(widths[1], widths[0])

    def test_categorical_contract_rejects_non_distribution_channels(self):
        with self.assertRaises(ValueError):
            fixed_bin_calibration_bound(np.full((1, 20, 37), .5), np.zeros((1, 20, 37)), categorical=True)

    def test_radius_rejects_invalid_configuration(self):
        for count, alpha in ((0, .05), (1.5, .05), (True, .05), (20, 0), (20, 1)):
            with self.subTest(count=count, alpha=alpha), self.assertRaises(ValueError):
                hilbert_mean_radius(count, alpha)


if __name__ == '__main__':
    unittest.main()
