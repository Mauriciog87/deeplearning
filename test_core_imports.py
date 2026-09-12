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

    def test_core_commands_work_with_optional_dependencies_blocked(self):
        script = """
import importlib.abc
import sys

class WithoutOptionalDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'sklearn', 'customtkinter', 'easyocr'}:
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, WithoutOptionalDependencies())
from src.database import RouletteRepository
from src.engine.prediction_engine import PredictionEngine
from src.utils.evaluation_harness import EvaluationConfig, evaluate_walk_forward
from src.utils.randomness import analyze_randomness
from src.utils.research_benchmark import ResearchConfig, run_research_benchmark
import roulette_cli
engine = PredictionEngine()
engine.load_history([1, 2, 3])
assert engine.get_consensus_prediction().predictor.value == 'fair'
result = run_research_benchmark(ResearchConfig(trials=1, observations=20,
    reference_size=20, calibration_size=20, scenarios=('uniform',)))
assert result['summary']['uniform']['trials'] == 1
assert 'torch' not in sys.modules
assert 'sklearn' not in sys.modules
assert 'customtkinter' not in sys.modules
assert 'easyocr' not in sys.modules
"""
        result = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parent,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
