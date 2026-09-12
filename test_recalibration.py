import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import check_grad, linprog
from scipy.special import softmax

from src.utils.recalibration import CALIBRATORS, ProbabilityCalibrator, monotone_log_gap, normalized_isotonic_objective
from src.utils.evaluation_harness import EvaluationConfig, evaluate_walk_forward
from src.utils.predictor import ExtraTreesPredictor


class RecalibrationTests(unittest.TestCase):
    def test_log_parameterization_matches_paper_objective_and_gradient(self):
        rng = np.random.default_rng(178)
        assignments = rng.integers(0, 6, (30, 37))
        actual = rng.integers(0, 37, 30)
        log_values = np.linspace(-6, 0, 6)
        objective, gradient = normalized_isotonic_objective(log_values, assignments, actual)
        values = np.exp(log_values)[assignments]
        original = -np.log(values[np.arange(30), actual] / values.sum(axis=1)).mean()
        self.assertAlmostEqual(objective, original, places=13)
        error = check_grad(lambda z: normalized_isotonic_objective(z, assignments, actual)[0],
                           lambda z: normalized_isotonic_objective(z, assignments, actual)[1], log_values)
        self.assertLess(error, 1e-6)
        other = np.array([-9, -7, -4, -2, -1, 0])
        midpoint = normalized_isotonic_objective((log_values + other) / 2, assignments, actual)[0]
        self.assertLessEqual(midpoint, (objective + normalized_isotonic_objective(other, assignments, actual)[0]) / 2 + 1e-12)
        gap = monotone_log_gap(log_values, gradient)
        linear = linprog(gradient, A_ub=-np.diff(np.eye(6), axis=0), b_ub=np.zeros(5),
                         bounds=[(-30, 0)] * 5 + [(0, 0)], method='highs')
        self.assertTrue(linear.success)
        self.assertAlmostEqual(gap, np.dot(gradient, log_values) - linear.fun, places=12)

    def test_calibrators_improve_a_separate_overconfidence_sample(self):
        rng = np.random.default_rng(456)
        truth = rng.dirichlet(np.ones(37) * .3, size=1800)
        forecasts = softmax(np.log(truth) * 3, axis=1)
        actual = np.array([rng.choice(37, p=p) for p in truth])
        before = -np.log(np.maximum(forecasts[1200:][np.arange(600), actual[1200:]], 1e-12)).mean()
        for method in CALIBRATORS:
            with self.subTest(method=method):
                calibrator = ProbabilityCalibrator(method)
                report = calibrator.fit(forecasts[:1200], actual[:1200])
                transformed = calibrator.transform(forecasts[1200:])
                after = -np.log(np.maximum(transformed[np.arange(600), actual[1200:]], 1e-12)).mean()
                self.assertLess(after, before)
                self.assertEqual(report['samples'], 1200)
                self.assertLessEqual(report['fitted_objective'], report['initial_objective'] + 1e-10)
                np.testing.assert_allclose(transformed.sum(axis=1), 1, atol=1e-14)
                restored = ProbabilityCalibrator.from_state(json.loads(json.dumps(calibrator.to_state(), allow_nan=False)))
                np.testing.assert_array_equal(restored.transform(forecasts[1200:]), transformed)
                if method == 'normalized_isotonic':
                    self.assertLess(report['objective_suboptimality_upper_bound'], 1e-4)

    def test_mcllo_uses_reference_log_odds_not_vector_scaling(self):
        calibrator = ProbabilityCalibrator('mcllo', reference_class=36)
        calibrator.is_fitted = True
        offsets, slopes = np.linspace(-.5, .5, 36), np.linspace(.3, 1.4, 36)
        calibrator.parameters = {'offsets': offsets.tolist(), 'slopes': slopes.tolist()}
        probability = np.arange(1, 38, dtype=float)
        probability /= probability.sum()
        transformed = calibrator.transform([probability])[0]
        np.testing.assert_allclose(np.log(transformed[:36] / transformed[36]),
                                   offsets + slopes * np.log(probability[:36] / probability[36]), atol=1e-13)

    def test_normalized_isotonic_preserves_order_and_class_permutations(self):
        rng = np.random.default_rng(21)
        forecasts = rng.dirichlet(np.ones(37), size=80)
        actual = rng.integers(0, 37, 80)
        calibrator = ProbabilityCalibrator('normalized_isotonic')
        calibrator.fit(forecasts, actual)
        order = rng.permutation(37)
        transformed = calibrator.transform(forecasts)
        np.testing.assert_allclose(calibrator.transform(forecasts[:, order]), transformed[:, order])
        sorted_outputs = np.take_along_axis(transformed, np.argsort(forecasts, axis=1), axis=1)
        self.assertTrue(np.all(np.diff(sorted_outputs, axis=1) >= -1e-14))

    def test_absent_classes_and_zero_predictions_remain_valid(self):
        forecasts = np.zeros((40, 37))
        forecasts[:, :2] = [.9, .1]
        actual = [0] * 20 + [1] * 20
        for method in CALIBRATORS:
            with self.subTest(method=method):
                calibrator = ProbabilityCalibrator(method)
                calibrator.fit(forecasts, actual)
                transformed = calibrator.transform(forecasts)
                self.assertTrue(np.isfinite(transformed).all())
                self.assertEqual(transformed.shape, forecasts.shape)
                np.testing.assert_allclose(transformed.sum(axis=1), 1)
                json.dumps(calibrator.to_state(), allow_nan=False)

    def test_invalid_or_unfitted_calibrators_are_rejected(self):
        with self.assertRaises(ValueError):
            ProbabilityCalibrator('unknown')
        calibrator = ProbabilityCalibrator()
        with self.assertRaises(RuntimeError):
            calibrator.transform([np.full(37, 1 / 37)])
        with self.assertRaises(ValueError):
            calibrator.fit([[1, 0]], [0])
        with self.assertRaises(ValueError):
            calibrator.fit([np.full(37, 1 / 37)], [37])

    def test_chronological_partitions_keep_test_labels_out_of_every_fit(self):
        training, calibration = [], []
        original_train, original_calibrate = ExtraTreesPredictor.fit, ProbabilityCalibrator.fit

        def train(model, history, *args, **kwargs):
            training.append(list(history))
            return original_train(model, history, *args, **kwargs)

        def calibrate(model, probabilities, actual):
            calibration.append(list(actual))
            return original_calibrate(model, probabilities, actual)

        config = EvaluationConfig(training_window=60, testing_window=5, step_size=5, models=('extra_trees',),
                                  runs=1, compute_intervals=False, recalibrators=CALIBRATORS, calibration_window=20)
        with patch.object(ExtraTreesPredictor, 'fit', train), patch.object(ProbabilityCalibrator, 'fit', calibrate):
            result = evaluate_walk_forward([7] * 40 + [8] * 20 + [9] * 5, config)
        self.assertEqual(training, [[7] * 40])
        self.assertGreaterEqual(len(calibration), 3)
        self.assertTrue(all(actual == [8] * 20 for actual in calibration))
        partition = result.manifest['partitions'][0]
        sets = [set(partition[name]) for name in ('train_spin_ids', 'calibration_spin_ids', 'test_spin_ids')]
        self.assertEqual([len(part) for part in sets], [40, 20, 5])
        self.assertEqual(sum(map(len, sets)), len(set.union(*sets)))
        for method in CALIBRATORS:
            rows = [row for row in result.rows if row.model == f'extra_trees__{method}']
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(row.actual == 9 for row in rows))
            self.assertTrue(all(len(row.number_probs) == 37 for row in rows))
        for record in result.manifest['calibration_fits']:
            if record['status'] == 'ready':
                self.assertEqual(record['fit']['samples'], 20)
        with self.assertRaises(ValueError):
            EvaluationConfig(training_window=100, recalibrators=CALIBRATORS, calibration_window=90)


if __name__ == '__main__':
    unittest.main()
