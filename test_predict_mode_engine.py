import unittest

from roulette_cli import _format_live_predictions
from src.utils.evaluation_harness import (
    LiveModelPrediction,
    make_fair_prediction,
    make_last_n_prediction,
    make_rolling_frequency_prediction,
)


class PredictModeEngineTest(unittest.TestCase):
    def test_public_helpers_return_normalized_distributions(self):
        history = [1, 2, 3, 3, 3, 7, 9]
        predictions = [
            make_fair_prediction(history),
            make_rolling_frequency_prediction(history),
            make_last_n_prediction(history),
        ]

        for prediction in predictions:
            self.assertEqual(len(prediction.number_probs), 37)
            self.assertAlmostEqual(sum(prediction.number_probs.values()), 1.0)
            self.assertTrue(prediction.top_numbers)

    def test_fair_prediction_is_uniform(self):
        prediction = make_fair_prediction([1, 2, 3])

        self.assertAlmostEqual(prediction.number_probs[0], prediction.number_probs[36])

    def test_rolling_frequency_increases_repeated_number(self):
        prediction = make_rolling_frequency_prediction([7] * 20 + [1, 2, 3])

        self.assertGreater(prediction.number_probs[7], prediction.number_probs[1])

    def test_last_n_prediction_includes_recent_numbers(self):
        prediction = make_last_n_prediction([1, 2, 3, 4, 5], last_n=3, bet_top_n=3)
        top_numbers = [number for number, _ in prediction.top_numbers]

        self.assertIn(3, top_numbers)
        self.assertIn(4, top_numbers)
        self.assertIn(5, top_numbers)

    def test_live_formatter_handles_active_and_inactive_models(self):
        prediction = LiveModelPrediction(
            model="fair",
            predicted=0,
            confidence=1 / 37,
            top_numbers=[(0, 1 / 37)],
            number_probs={number: 1 / 37 for number in range(37)},
        )

        output = _format_live_predictions({"consensus": None, "fair": prediction}, 5)

        self.assertIn("inactive", output)
        self.assertIn("fair", output)


if __name__ == "__main__":
    unittest.main()
