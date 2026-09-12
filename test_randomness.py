import subprocess
import sys
import unittest
from random import Random

from src.utils.randomness import analyze_randomness, format_randomness_report


class RandomnessTest(unittest.TestCase):
    def test_seeded_fair_sequence_has_no_rejections_at_strict_alpha(self):
        rng = Random(13)
        numbers = [rng.randrange(37) for _ in range(4000)]

        report = analyze_randomness(numbers, window_size=400, step_size=200, alpha=0.001, resamples=999)

        rejected = [result.name for result in report.tests if result.status == "reject"]
        self.assertEqual(rejected, [])

    def test_biased_sequence_rejects_uniformity(self):
        rng = Random(21)
        numbers = [rng.randrange(37) for _ in range(1000)] + [7] * 350

        report = analyze_randomness(numbers, window_size=250, step_size=125, alpha=0.01, resamples=999)
        uniformity = next(result for result in report.tests if result.name == "uniformity_chi_square")

        self.assertEqual(uniformity.status, "reject")
        self.assertLess(uniformity.p_value, 0.01)
        self.assertLess(uniformity.q_value, 0.01)
        self.assertTrue(uniformity.fdr_significant)

    def test_repeating_cycle_rejects_transition_independence(self):
        numbers = list(range(37)) * 80

        report = analyze_randomness(numbers, window_size=370, step_size=185, alpha=0.01, resamples=999)
        transition = next(result for result in report.tests if result.name == "transition_chi_square")

        self.assertEqual(transition.status, "reject")
        self.assertLess(transition.p_value, 0.01)

    def test_regime_change_rejects_categorical_change_point(self):
        rng = Random(31)
        fair_prefix = [rng.randrange(37) for _ in range(600)]
        biased_suffix = [7 if index % 2 == 0 else rng.randrange(37) for index in range(600)]

        report = analyze_randomness(
            fair_prefix + biased_suffix,
            window_size=200,
            step_size=50,
            alpha=0.01, resamples=999,
        )
        change_point = next(
            result for result in report.tests
            if result.name == "categorical_change_point"
        )

        self.assertIn(change_point.status, ('reject', 'exploratory'))
        if change_point.status == 'exploratory':
            self.assertIsNone(change_point.p_value)
            self.assertFalse(change_point.fdr_significant)
        else:
            self.assertTrue(change_point.fdr_significant)
        self.assertGreater(change_point.details["best_split"], 400.0)

    def test_report_format_is_deterministic(self):
        numbers = [index % 37 for index in range(370)]

        first = format_randomness_report(analyze_randomness(numbers, resamples=999))
        second = format_randomness_report(analyze_randomness(numbers, resamples=999))

        self.assertEqual(first, second)
        self.assertIn("ROULETTE RANDOMNESS TEST REPORT", first)
        self.assertIn("transition_chi_square", first)
        self.assertIn("q-value", first)

    def test_cli_help_exposes_randomness_flags(self):
        completed = subprocess.run(
            [sys.executable, "roulette_cli.py", "randomness", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("--window-size", completed.stdout)
        self.assertIn("--alpha", completed.stdout)
        self.assertIn("--min-spins", completed.stdout)
        self.assertIn("--markov-max-lag", completed.stdout)


if __name__ == "__main__":
    unittest.main()
