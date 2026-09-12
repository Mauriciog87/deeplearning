import unittest
from unittest.mock import Mock

import numpy as np

from src.engine.prediction_engine import PredictionEngine, PredictorType
from src.probabilities import validate_probabilities
from src.utils.predictor import ExtraTreesPredictor
from src.utils.bias_aware_predictor import BiasAwarePredictor


class PredictionContractTests(unittest.TestCase):
    def test_extra_trees_maps_absent_classes_to_zero(self):
        predictor = ExtraTreesPredictor()
        predictor.fit([7, 23] * 30)
        probabilities = predictor.predict_proba([7, 23] * 5)
        self.assertEqual(len(probabilities), 37)
        self.assertEqual(probabilities[0], 0)
        self.assertAlmostEqual(probabilities[7] + probabilities[23], 1)

    def test_probabilities_drive_every_category_including_zero(self):
        probabilities = np.full(37, .1 / 36)
        probabilities[0] = .9
        result = PredictionEngine().prediction_from_probabilities(PredictorType.LSTM, probabilities)
        for name in ('number', 'color', 'parity', 'high_low', 'dozen', 'column'):
            category = getattr(result, name)
            self.assertAlmostEqual(sum(category.all_probabilities.values()), 1)
            self.assertEqual(category.probability, category.all_probabilities[category.value])
        self.assertEqual(result.dozen.value, 0)
        self.assertEqual(result.parity.value, 'zero')
        self.assertEqual(len(result.top_numbers), 37)

    def test_consensus_reuses_snapshot_and_excludes_policy(self):
        engine = PredictionEngine()
        engine.load_history(list(range(37)))
        engine.lstm_predictor = Mock()
        engine.lstm_predictor.predict_proba.return_value = np.full(37, 1 / 37)
        engine.is_lstm_trained = True
        first = engine.predict_all()
        engine.get_consensus_prediction()
        self.assertEqual(engine.lstm_predictor.predict_proba.call_count, 1)
        self.assertNotIn(PredictorType.DQN, first)
        engine.load_history([])
        self.assertEqual(engine.get_consensus_prediction().predictor, PredictorType.FAIR)

    def test_invalid_distribution_is_rejected(self):
        for vector in ([0] * 37, [float('nan')] * 37, [-1] + [2 / 36] * 36, [1 / 37] * 36):
            with self.assertRaises(ValueError):
                validate_probabilities(vector)

    def test_bias_probabilities_are_smoothed_counts(self):
        history = [7] * 200 + list(range(37))
        probabilities = BiasAwarePredictor().predict_proba(history)
        self.assertAlmostEqual(probabilities[7], 202 / (len(history) + 37))


if __name__ == '__main__':
    unittest.main()
