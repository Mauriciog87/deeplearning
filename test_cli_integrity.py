from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import roulette_cli
from src.database import RouletteRepository


class CliIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.repo = RouletteRepository(str(Path(self.folder.name) / 'cli.db'))
        for offset in (0, 1):
            self.repo.create_session_with_numbers([(index + offset) % 37 for index in range(50)],
                                                 name=f'Wheel {offset}')

    def tearDown(self):
        self.folder.cleanup()

    def run_command(self, *args, inputs=()):
        output = io.StringIO()
        with patch('src.database.RouletteRepository', return_value=self.repo), \
                patch('sys.argv', ['roulette_cli.py', *args]), \
                patch('builtins.input', side_effect=inputs), redirect_stdout(output):
            code = roulette_cli.main()
        return code, output.getvalue()

    def test_evaluate_export_preserves_sessions_and_cli_configuration(self):
        output = Path(self.folder.name) / 'evaluation.json'
        code, _ = self.run_command('evaluate', '--models', '--runs', '1', '--no-intervals',
                                   '--train-window', '30', '--test-window', '10', '--step', '10',
                                   '--device', 'cpu', '--output', str(output))
        self.assertEqual(code, 0)
        result = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(result['folds'], 4)
        self.assertEqual(len(result['manifest']['dataset']['sessions']), 2)
        self.assertEqual(result['config']['unit_stake'], 1)
        for fold in result['manifest']['partitions']:
            self.assertFalse(set(fold['train_spin_ids']) & set(fold['test_spin_ids']))

    def test_predict_records_displayed_forecasts_before_outcome(self):
        code, output = self.run_command('predict', '--session', '1', inputs=('7', 'q'))
        self.assertEqual(code, 0)
        stats = self.repo.db.get_predictor_stats(1)
        self.assertEqual({item['predictor_type'] for item in stats}, {'fair', 'rolling_frequency', 'last_n'})
        self.assertTrue(all(item['total'] == 1 for item in stats))
        self.assertEqual(self.repo.get_numbers_by_session(1)[-1], 7)

    def test_randomness_does_not_join_independent_sessions(self):
        code, output = self.run_command('randomness', '--resamples', '9')
        self.assertEqual(code, 0)
        self.assertIn('Sesion 1: 50', output)
        self.assertIn('Sesion 2: 50', output)


if __name__ == '__main__':
    unittest.main()
