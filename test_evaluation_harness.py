import math
import unittest

from src.utils.evaluation_harness import (
    EvaluationConfig,
    PredictionEvaluationRow,
    evaluate_walk_forward,
    format_evaluation_report,
    _summarize_rows,
)


class EvaluationHarnessTest(unittest.TestCase):
    def test_uniform_dataset_has_fair_log_loss(self):
        numbers = [index % 37 for index in range(370)]
        config = EvaluationConfig(runs=1, models=(),
            training_window=100,
            testing_window=37,
            step_size=37,
            bet_top_n=5,
            bootstrap_resamples=50,
        )

        result = evaluate_walk_forward(numbers, config)

        self.assertGreater(result.folds, 0)
        fair = result.per_model_summaries["fair"]
        self.assertAlmostEqual(fair.log_loss, math.log(37), places=6)

    def test_biased_dataset_improves_rolling_probability(self):
        numbers = [7] * 260 + [index % 37 for index in range(260)]
        config = EvaluationConfig(runs=1, models=(),
            training_window=120,
            testing_window=40,
            step_size=40,
            bet_top_n=5,
            bootstrap_resamples=50,
        )

        result = evaluate_walk_forward(numbers, config)

        fair = result.per_model_summaries["fair"]
        rolling = result.per_model_summaries["rolling_frequency"]
        self.assertLess(rolling.log_loss, fair.log_loss)

    def test_short_dataset_returns_warning_without_folds(self):
        numbers = [index % 37 for index in range(50)]
        config = EvaluationConfig(runs=1, models=(),
            training_window=100,
            testing_window=50,
            bootstrap_resamples=50,
        )

        result = evaluate_walk_forward(numbers, config)

        self.assertEqual(result.folds, 0)
        self.assertEqual(result.per_model_summaries, {})
        self.assertTrue(result.warnings)

    def test_metrics_are_finite(self):
        numbers = [index % 37 for index in range(260)]
        config = EvaluationConfig(runs=1, models=(),
            training_window=100,
            testing_window=50,
            step_size=50,
            bootstrap_resamples=50,
        )

        result = evaluate_walk_forward(numbers, config)

        for summary in result.per_model_summaries.values():
            if summary.log_loss is None:
                self.assertIsNone(summary.brier)
                self.assertIsNone(summary.ece)
                continue
            self.assertTrue(math.isfinite(summary.log_loss))
            self.assertTrue(math.isfinite(summary.brier))
            self.assertTrue(math.isfinite(summary.ece))
            self.assertTrue(math.isfinite(summary.classwise_ece))
            self.assertTrue(math.isfinite(summary.roi))

    def test_report_formats_result(self):
        numbers = [index % 37 for index in range(220)]
        config = EvaluationConfig(runs=1, models=(),
            training_window=100,
            testing_window=50,
            step_size=50,
            bootstrap_resamples=50,
        )

        result = evaluate_walk_forward(numbers, config)
        report = format_evaluation_report(result)

        self.assertIn("WALK-FORWARD ENGINE EVALUATION", report)
        self.assertIn("fair", report)
        self.assertIn("BOOTSTRAP INTERVALS", report)
        self.assertIn("CALIBRATION RELIABILITY", report)
        self.assertIn("PAIRED IMPROVEMENT VS fair", report)
        self.assertIn("OPE READINESS", report)
        self.assertIn("Propensities: no", report)

    def test_bootstrap_intervals_are_deterministic_and_contain_mean(self):
        numbers = [index % 37 for index in range(260)]
        config = EvaluationConfig(runs=1, models=(),
            training_window=100,
            testing_window=50,
            step_size=50,
            bootstrap_resamples=80,
            seed=7,
        )

        first = evaluate_walk_forward(numbers, config)
        second = evaluate_walk_forward(numbers, config)

        first_interval = first.per_model_summaries["fair"].log_loss_interval
        second_interval = second.per_model_summaries["fair"].log_loss_interval
        self.assertEqual(first_interval, second_interval)
        self.assertLessEqual(first_interval.lower, first_interval.mean)
        self.assertGreaterEqual(first_interval.upper, first_interval.mean)

    def test_calibration_bins_cover_model_rows(self):
        numbers = [index % 37 for index in range(260)]
        config = EvaluationConfig(runs=1, models=(),
            training_window=100,
            testing_window=50,
            step_size=50,
            ece_bins=5,
            bootstrap_resamples=50,
        )

        result = evaluate_walk_forward(numbers, config)
        fair = result.per_model_summaries["fair"]

        self.assertEqual(sum(item.count for item in fair.calibration_bins), fair.spins)
        self.assertEqual(len(fair.calibration_bins), 1)
        self.assertGreaterEqual(fair.ece_interval.lower, 0)
        self.assertLessEqual(fair.ece_interval.lower, fair.ece_interval.upper)
        self.assertIn(5, fair.top_k_ece)

    def test_full_and_top_k_calibration_are_zero_for_exactly_fair_rows(self):
        fair_probs = {number: 1.0 / 37.0 for number in range(37)}
        fair_top = [(number, fair_probs[number]) for number in range(37)]
        rows = [
            PredictionEvaluationRow(
                fold=0,
                index=index,
                model="fair",
                actual=actual,
                predicted=0,
                confidence=fair_probs[0],
                number_probs=fair_probs,
                top_numbers=fair_top,
                profit=35.0 if actual == 0 else -1.0,
            )
            for index, actual in enumerate(range(37))
        ]
        config = EvaluationConfig(runs=1, models=(),
            bet_top_n=5,
            top_k=(1, 5),
            bootstrap_resamples=20,
            seed=17,
        )

        summaries = _summarize_rows(rows, config, total_evaluation_spins=len(rows))
        fair = summaries["fair"]

        self.assertAlmostEqual(fair.ece, 0.0, places=12)
        self.assertAlmostEqual(fair.classwise_ece, 0.0, places=12)
        self.assertAlmostEqual(fair.top_k_ece[5], 0.0, places=12)

    def test_paired_comparison_prefers_perfect_model(self):
        rows = []
        fair_probs = {number: 1.0 / 37.0 for number in range(37)}
        fair_top = [(0, fair_probs[0])]
        numbers = [index % 37 for index in range(74)]

        for index, actual in enumerate(numbers):
            perfect_probs = {
                number: 0.1 / 36.0
                for number in range(37)
            }
            perfect_probs[actual] = 0.9
            rows.append(PredictionEvaluationRow(
                fold=0,
                index=index,
                model="perfect",
                actual=actual,
                predicted=actual,
                confidence=0.9,
                number_probs=perfect_probs,
                top_numbers=[(actual, 0.9)],
                profit=35.0,
            ))
            rows.append(PredictionEvaluationRow(
                fold=0,
                index=index,
                model="fair",
                actual=actual,
                predicted=0,
                confidence=fair_probs[0],
                number_probs=fair_probs,
                top_numbers=fair_top,
                profit=35.0 if actual == 0 else -1.0,
            ))

        config = EvaluationConfig(runs=1, models=(),
            bet_top_n=1,
            bootstrap_resamples=80,
            seed=11,
        )

        summaries = _summarize_rows(rows, config, total_evaluation_spins=len(numbers))
        comparisons = summaries["perfect"].comparisons

        self.assertGreater(comparisons["exact_accuracy"].delta_mean, 0.9)
        self.assertGreater(comparisons["log_loss"].delta_mean, 2.0)
        self.assertGreater(comparisons["log_loss"].positive_bootstrap_fraction, 0.99)

    def test_ope_readiness_reports_missing_logged_policy_fields(self):
        numbers = [index % 37 for index in range(220)]
        config = EvaluationConfig(runs=1, models=(),
            training_window=100,
            testing_window=50,
            step_size=50,
            bootstrap_resamples=20,
        )

        result = evaluate_walk_forward(numbers, config)

        self.assertEqual(result.ope_readiness.supported_level, "paired_walk_forward_backtest")
        self.assertFalse(result.ope_readiness.propensities_available)
        self.assertIn("action_probability", result.ope_readiness.missing_fields)


if __name__ == "__main__":
    unittest.main()
