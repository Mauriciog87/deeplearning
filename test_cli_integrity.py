from contextlib import redirect_stdout
from dataclasses import replace
from importlib.util import find_spec
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

    def test_monitor_preserves_family_state_and_only_counts_new_spins(self):
        state_path = Path(self.folder.name) / 'monitor.json'
        output_path = Path(self.folder.name) / 'monitor-report.json'
        args = ('monitor', '--sessions', '1', '2', '--state', str(state_path), '--output', str(output_path))
        self.assertEqual(self.run_command(*args)[0], 0)
        first_state = state_path.read_bytes()
        self.assertEqual(self.run_command(*args)[0], 0)
        self.assertEqual(state_path.read_bytes(), first_state)
        self.repo.add_number_to_session(1, 7)
        self.assertEqual(self.run_command(*args)[0], 0)
        result = json.loads(output_path.read_text(encoding='utf-8'))
        self.assertEqual(result['sessions']['1']['observations'], 51)
        self.assertEqual(result['sessions']['2']['observations'], 50)
        self.assertEqual(result['sessions']['1']['allocated_alpha'], .0125)
        self.assertEqual(self.run_command(*args, '--reset')[0], 0)
        result = json.loads(output_path.read_text(encoding='utf-8'))
        self.assertEqual(result['sessions']['1']['observations'], 0)
        self.assertLess(result['sessions']['1']['allocated_alpha'], .0125)

    def test_monitor_rejects_changed_configuration_without_overwriting_state(self):
        state_path = Path(self.folder.name) / 'monitor.json'
        args = ('monitor', '--sessions', '1', '2', '--state', str(state_path))
        self.assertEqual(self.run_command(*args)[0], 0)
        before = state_path.read_bytes()
        code, message = self.run_command(*args, '--alpha', '.1')
        self.assertEqual(code, 1)
        self.assertIn('presupuesto', message)
        self.assertEqual(state_path.read_bytes(), before)

    def test_monitor_report_cannot_overwrite_its_state_or_database(self):
        state_path = Path(self.folder.name) / 'monitor.json'
        args = ('monitor', '--sessions', '1', '--state', str(state_path))
        for output_path in (state_path, Path(self.repo.db.db_path)):
            with self.subTest(path=output_path):
                code, message = self.run_command(*args, '--output', str(output_path))
                self.assertEqual(code, 1)
                self.assertIn('archivo distinto', message)
                self.assertFalse(state_path.exists())
                self.assertEqual(len(self.repo.get_numbers_by_session(1)), 50)

    def test_monitor_rejects_edited_or_deleted_history_without_changing_state(self):
        state_path = Path(self.folder.name) / 'monitor.json'
        args = ('monitor', '--sessions', '1', '--state', str(state_path))
        self.assertEqual(self.run_command(*args)[0], 0)
        before = state_path.read_bytes()
        original = self.repo.get_evaluation_sessions()[0]
        variants = [replace(original, numbers=(7,) + original.numbers[1:]),
                    replace(original, numbers=original.numbers[:-1], spin_ids=original.spin_ids[:-1])]
        for session in variants:
            with self.subTest(length=len(session.numbers)), patch.object(self.repo, 'get_evaluation_sessions', return_value=[session]):
                code, message = self.run_command(*args)
                self.assertEqual(code, 1)
                self.assertIn('historial que solo crezca', message)
                self.assertEqual(state_path.read_bytes(), before)

    @unittest.skipUnless(find_spec('sklearn') is not None, 'Recalibration comparison requires optional scikit-learn')
    def test_recalibration_cli_exports_disjoint_partitions_and_fitted_state(self):
        output = Path(self.folder.name) / 'calibrated.json'
        code, _ = self.run_command('evaluate', '--session', '1', '--models', 'extra_trees', '--runs', '1',
                                   '--train-window', '40', '--calibration-window', '20', '--test-window', '5', '--step', '5',
                                   '--recalibrators', 'temperature', 'mcllo', 'normalized_isotonic', '--lstm-representation', 'ordinal',
                                   '--no-intervals', '--device', 'cpu', '--output', str(output))
        self.assertEqual(code, 0)
        result = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(result['config']['lstm_representation'], 'ordinal')
        self.assertEqual(len(result['config']['recalibrators']), 3)
        for partition in result['manifest']['partitions']:
            self.assertEqual(len(partition['train_spin_ids']), 20)
            self.assertEqual(len(partition['calibration_spin_ids']), 20)
            self.assertFalse(set(partition['calibration_spin_ids']) & set(partition['test_spin_ids']))
        fitted = [item for item in result['manifest']['calibration_fits'] if item['status'] == 'ready']
        self.assertGreaterEqual(len(fitted), 6)
        self.assertTrue(all(item['state']['schema_version'] == 1 for item in fitted))


if __name__ == '__main__':
    unittest.main()
