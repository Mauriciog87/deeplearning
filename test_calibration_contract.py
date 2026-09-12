from dataclasses import replace
import unittest

import numpy as np

from src.utils.evaluation_harness import EvaluationConfig, PredictionEvaluationRow, _metric_values
from src.utils.temporal_statistics import adaptive_bins


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


if __name__ == '__main__':
    unittest.main()
