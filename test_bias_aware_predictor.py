import unittest

from src.engine.prediction_engine import PredictionEngine
from src.utils.bias_aware_predictor import BiasAwarePredictor
from src.utils.backtesting import (
    create_bias_strategy,
    create_hot_numbers_strategy,
    create_last_n_bunching_strategy,
    flat_bet_backtest,
)


class BiasAwarePredictorTest(unittest.TestCase):
    def test_uniform_wheel_is_not_significant(self):
        history = [number % 37 for number in range(10000)]

        result = BiasAwarePredictor().fit_predict(history)

        self.assertFalse(result.is_significant)
        self.assertEqual(result.selected_numbers, [])

    def test_biased_wheel_selects_high_probability_numbers(self):
        history = []
        for number in range(5):
            history.extend([number] * 350)
        for number in range(5, 37):
            history.extend([number] * 258)

        result = BiasAwarePredictor().fit_predict(history)

        self.assertTrue(result.is_significant)
        for number in range(5):
            self.assertIn(number, result.selected_numbers)
            self.assertGreater(result.number_probs[number], result.number_probs[36])

    def test_prediction_engine_bias_returns_none_for_short_history(self):
        engine = PredictionEngine()
        engine.history = [number % 37 for number in range(50)]

        prediction = engine._predict_bias()

        self.assertIsNone(prediction)

    def test_last_n_backtest_strategy_runs_with_existing_strategies(self):
        numbers = [number % 37 for number in range(300)]
        test_data = [number % 37 for number in range(100)]
        strategies = [
            create_bias_strategy(0.03),
            create_hot_numbers_strategy(5),
            create_last_n_bunching_strategy(),
        ]

        for strategy in strategies:
            bet_numbers = strategy(numbers)
            self.assertTrue(bet_numbers)
            result = flat_bet_backtest(test_data, bet_numbers)
            self.assertGreater(result.total_bets, 0)


if __name__ == "__main__":
    unittest.main()
