import math
import unittest
from unittest.mock import patch

from src.datasets import EvaluationSession
from src.utils.predictor import ExtraTreesPredictor

from src.utils.evaluation_harness import EvaluationConfig, evaluate_walk_forward, _row_from_probs, _classwise_ece


class TemporalEvaluationTests(unittest.TestCase):
    def test_overlapping_test_folds_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_walk_forward(list(range(37)) * 8,
                                  EvaluationConfig(runs=1, models=(), training_window=200, testing_window=37, step_size=10,
                                                   compute_intervals=False))

    def test_fair_metrics_use_full_ranking(self):
        result = evaluate_walk_forward(list(range(37)) * 8,
                                       EvaluationConfig(runs=1, models=(), training_window=200, testing_window=37, step_size=37,
                                                        compute_intervals=False))
        fair = result.per_model_summaries['fair']
        self.assertAlmostEqual(fair.top_k_hit_rates[5], 5 / 37)
        self.assertAlmostEqual(fair.top_k_hit_rates[10], 10 / 37)
        self.assertAlmostEqual(fair.log_loss, math.log(37))
        self.assertAlmostEqual(fair.brier, 36 / 37)

    def test_classwise_calibration_does_not_cancel_classes(self):
        probabilities = {number: .91 / 36 for number in range(37)}
        probabilities[0] = .09
        config = EvaluationConfig(runs=1, models=(), compute_intervals=False)
        rows = [_row_from_probs(0, index, 'example', 36, probabilities, config) for index in range(100)]
        self.assertGreater(_classwise_ece(rows, 10), .05)

    def test_training_uses_only_its_session_prefix(self):
        sessions = [EvaluationSession('one', tuple([1] * 20 + [2] * 10)),
                    EvaluationSession('two', tuple([3] * 20 + [4] * 10))]
        seen = []
        original = ExtraTreesPredictor.fit
        def recording_fit(model, history, *args, **kwargs):
            seen.append(list(history))
            return original(model, history, *args, **kwargs)
        config = EvaluationConfig(training_window=20, testing_window=5, step_size=5, runs=1,
                                  models=('extra_trees',), compute_intervals=False)
        with patch.object(ExtraTreesPredictor, 'fit', recording_fit):
            result = evaluate_walk_forward(sessions, config)
        self.assertEqual(seen, [[1] * 20, [1] * 15 + [2] * 5, [3] * 20, [3] * 15 + [4] * 5])
        self.assertEqual(result.per_model_summaries['extra_trees'].spins, 20)
        self.assertEqual(len(result.manifest['partitions']), 4)
        fair_rows = [row for row in result.rows if row.model == 'fair' and row.session_id == 'one']
        self.assertEqual(fair_rows[5].balance_before, fair_rows[4].balance_after)
        self.assertIsNone(result.per_model_summaries['pass'].roi)


if __name__ == '__main__':
    unittest.main()
