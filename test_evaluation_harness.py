import math
import unittest

from src.utils.evaluation_harness import (
    EvaluationConfig,
    evaluate_walk_forward,
    format_evaluation_report,
)


class EvaluationHarnessTest(unittest.TestCase):
    def test_uniform_dataset_has_fair_log_loss(self):
        numbers = [index % 37 for index in range(370)]
        config = EvaluationConfig(
            training_window=100,
            testing_window=37,
            step_size=37,
            bet_top_n=5,
        )

        result = evaluate_walk_forward(numbers, config)

        self.assertGreater(result.folds, 0)
        fair = result.per_model_summaries["fair"]
        self.assertAlmostEqual(fair.log_loss, math.log(37), places=6)

    def test_biased_dataset_improves_rolling_probability(self):
        numbers = [7] * 260 + [index % 37 for index in range(260)]
        config = EvaluationConfig(
            training_window=120,
            testing_window=40,
            step_size=40,
            bet_top_n=5,
        )

        result = evaluate_walk_forward(numbers, config)

        fair = result.per_model_summaries["fair"]
        rolling = result.per_model_summaries["rolling_frequency"]
        self.assertLess(rolling.log_loss, fair.log_loss)

    def test_short_dataset_returns_warning_without_folds(self):
        numbers = [index % 37 for index in range(50)]
        config = EvaluationConfig(training_window=100, testing_window=50)

        result = evaluate_walk_forward(numbers, config)

        self.assertEqual(result.folds, 0)
        self.assertEqual(result.per_model_summaries, {})
        self.assertTrue(result.warnings)

    def test_metrics_are_finite(self):
        numbers = [index % 37 for index in range(260)]
        config = EvaluationConfig(
            training_window=100,
            testing_window=50,
            step_size=50,
        )

        result = evaluate_walk_forward(numbers, config)

        for summary in result.per_model_summaries.values():
            self.assertTrue(math.isfinite(summary.log_loss))
            self.assertTrue(math.isfinite(summary.brier))
            self.assertTrue(math.isfinite(summary.ece))
            self.assertTrue(math.isfinite(summary.roi))

    def test_report_formats_result(self):
        numbers = [index % 37 for index in range(220)]
        config = EvaluationConfig(
            training_window=100,
            testing_window=50,
            step_size=50,
        )

        result = evaluate_walk_forward(numbers, config)
        report = format_evaluation_report(result)

        self.assertIn("WALK-FORWARD ENGINE EVALUATION", report)
        self.assertIn("fair", report)


if __name__ == "__main__":
    unittest.main()
