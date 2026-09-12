from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.checkpoints import seed_everything
from src.utils.lstm_predictor import LSTMPredictor


class LSTMRepresentationTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_categorical_distances_do_not_impose_numeric_proximity(self):
        predictor = LSTMPredictor(sequence_length=1, hidden_size=8, device='cpu')
        inputs, labels = predictor._prepare_sequences([0, 1, 36, 7])
        self.assertEqual(inputs.shape, (3, 1, 37))
        self.assertEqual(labels.tolist(), [1, 36, 7])
        self.assertEqual(inputs.argmax(dim=2).flatten().tolist(), [0, 1, 36])
        self.assertEqual(torch.dist(inputs[0], inputs[1]), torch.dist(inputs[0], inputs[2]))
        ordinal = LSTMPredictor(sequence_length=1, hidden_size=8, device='cpu', representation='ordinal')
        inputs, _ = ordinal._prepare_sequences([0, 1, 36, 7])
        self.assertEqual(inputs.shape, (3, 1, 1))
        self.assertLess(torch.dist(inputs[0], inputs[1]), torch.dist(inputs[0], inputs[2]))

    def test_both_representations_train_save_and_resume_exactly_on_cpu(self):
        history = [index % 37 for index in range(55)]
        for representation in ('one_hot', 'ordinal'):
            with self.subTest(representation=representation), tempfile.TemporaryDirectory() as folder:
                seed_everything(42, 'cpu')
                predictor = LSTMPredictor(sequence_length=4, hidden_size=8, device='cpu', representation=representation)
                self.assertEqual(predictor.fit(history, epochs=1)['status'], 'success')
                before = predictor.predict_proba(history)
                self.assertEqual(before.shape, (37,))
                self.assertAlmostEqual(before.sum(), 1)
                path = Path(folder) / 'predictor.pt'
                predictor.save(path)
                predictor.fit(history, epochs=1)
                expected = predictor.predict_proba(history)
                restored = LSTMPredictor(device='cpu')
                restored.load(path)
                self.assertEqual(restored.device.type, 'cpu')
                self.assertEqual(restored.representation, representation)
                np.testing.assert_array_equal(restored.predict_proba(history), before)
                restored.fit(history, epochs=1)
                np.testing.assert_array_equal(restored.predict_proba(history), expected)

    def test_legacy_scalar_checkpoint_loads_as_explicit_ordinal_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'legacy.pt'
            predictor = LSTMPredictor(hidden_size=8, device='cpu', representation='ordinal')
            predictor.save(path)
            state = torch.load(path, weights_only=True)
            del state['representation']
            del state['input_contract_version']
            torch.save(state, path)
            restored = LSTMPredictor(device='cpu')
            restored.load(path)
            self.assertEqual(restored.representation, 'ordinal')
            self.assertEqual(restored.model.lstm1.input_size, 1)

    def test_invalid_numbers_and_representations_are_rejected(self):
        with self.assertRaises(ValueError):
            LSTMPredictor(representation='embedding', device='cpu')
        predictor = LSTMPredictor(device='cpu')
        with self.assertRaises(ValueError):
            predictor._prepare_sequences([0] * 30 + [37])


if __name__ == '__main__':
    unittest.main()
