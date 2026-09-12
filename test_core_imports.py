import subprocess
import os
import sys
import unittest
from pathlib import Path


class CoreImportTests(unittest.TestCase):
    def test_cli_help_works_with_redirected_windows_encoding(self):
        result = subprocess.run([sys.executable, 'roulette_cli.py', '--help'],
                                cwd=Path(__file__).resolve().parent, capture_output=True,
                                env=dict(os.environ, PYTHONIOENCODING='cp1252'), timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode('utf-8', errors='replace'))
        self.assertIn('usage:', result.stdout.decode('utf-8'))

    def test_core_commands_do_not_load_torch_or_gui(self):
        script = """
import sys
from src.database import RouletteRepository
from src.engine.prediction_engine import PredictionEngine
from src.utils.evaluation_harness import EvaluationConfig, evaluate_walk_forward
from src.utils.randomness import analyze_randomness
import roulette_cli
engine = PredictionEngine()
engine.load_history([1, 2, 3])
assert engine.get_consensus_prediction().predictor.value == 'fair'
assert 'torch' not in sys.modules
assert 'customtkinter' not in sys.modules
assert 'easyocr' not in sys.modules
"""
        result = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parent,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
