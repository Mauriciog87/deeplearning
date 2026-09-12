from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import roulette_cli
from src.utils.recalibration import ProbabilityCalibrator
from src.utils.research_benchmark import ResearchConfig, SCENARIOS, generate_scenario, run_research_benchmark, run_research_trial


class ResearchBenchmarkTests(unittest.TestCase):
    def test_scenarios_reproduce_without_reusing_labels_or_confusing_nulls(self):
        config = ResearchConfig(trials=1, observations=40, reference_size=20, calibration_size=20)
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                first, second = generate_scenario(config, scenario, 0), generate_scenario(config, scenario, 0)
                self.assertEqual(first['sha256'], second['sha256'])
                self.assertNotEqual(first['sha256'], generate_scenario(config, scenario, 1)['sha256'])
                self.assertEqual(first['truth'].shape, (80, 37))
                np.testing.assert_allclose(first['truth'].sum(axis=1), 1)
                np.testing.assert_allclose(first['forecasts'].sum(axis=1), 1)
                if scenario == 'calibration_improvement':
                    np.testing.assert_allclose(first['truth'], 1 / 37)
                    self.assertGreater(first['forecasts'][59, 7], first['forecasts'][60, 7])
                elif scenario == 'dependence':
                    self.assertEqual(first['truth'][1].argmax(), first['outcomes'][0])

    def test_trials_are_replayable_and_calibration_never_sees_test_labels(self):
        config = ResearchConfig(trials=1, observations=30, reference_size=20, calibration_size=20,
                                scenarios=('overconfidence',), recalibrators=('temperature',), include_online=True, online_iterations=2)
        data = generate_scenario(config, 'overconfidence', 0)
        seen = []
        original = ProbabilityCalibrator.fit

        def fit(calibrator, probabilities, actual):
            seen.append((np.asarray(probabilities), np.asarray(actual)))
            return original(calibrator, probabilities, actual)

        with patch.object(ProbabilityCalibrator, 'fit', fit):
            first = run_research_trial(config, 'overconfidence', 0)
        second = run_research_trial(config, 'overconfidence', 0)
        self.assertEqual(first, second)
        np.testing.assert_array_equal(seen[0][0], data['forecasts'][20:40])
        np.testing.assert_array_equal(seen[0][1], data['outcomes'][20:40])
        self.assertEqual(first['online']['observations'], 30)
        self.assertEqual(first['partitions']['test'], [40, 70])
        self.assertEqual(first['policies']['pass']['profit'], 0)
        self.assertEqual(first['policies']['pass']['stake'], 0)
        self.assertTrue(all(sum(policy['action_counts']) == 30 for policy in first['policies'].values()))
        self.assertEqual(first['counterfactual_unit_profit_totals'][46], 0)
        self.assertEqual(first['confidence_sequences']['status'], 'not_applicable_fixed_probability_assumption')
        json.dumps(first, allow_nan=False)

    def test_summary_reports_uncertainty_assumptions_and_censoring(self):
        config = ResearchConfig(trials=2, observations=20, reference_size=20, calibration_size=20,
                                scenarios=('uniform', 'dependence', 'abrupt_bias'))
        result = run_research_benchmark(config)
        self.assertEqual(len(result['trials']), 6)
        uniform = result['summary']['uniform']
        self.assertEqual(uniform['detectors']['pit']['alarm_rate']['trials'], 2)
        self.assertEqual(uniform['confidence_sequences']['kt']['coverage']['trials'], 2)
        self.assertEqual(result['summary']['dependence']['detectors']['pit']['null_status'], 'assumption_violated')
        self.assertIsNone(result['summary']['dependence']['confidence_sequences']['kt']['coverage']['rate'])
        self.assertIn('censors', result['summary']['abrupt_bias']['detectors']['pit']['delay_interpretation'])
        self.assertIsNone(uniform['policies']['pass']['roi']['mean'])
        self.assertIn('src/utils/research_benchmark.py', result['manifest']['source_sha256'])
        json.dumps(result, allow_nan=False)

    def test_cli_exports_results_without_accessing_the_database(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'benchmark.json'
            args = ['roulette_cli.py', 'research', '--trials', '1', '--observations', '20', '--reference-size', '20',
                    '--calibration-size', '20', '--scenarios', 'uniform', '--output', str(output)]
            with patch('sys.argv', args), patch('src.database.RouletteRepository', side_effect=AssertionError('Unexpected database access')), redirect_stdout(io.StringIO()):
                code = roulette_cli.main()
            self.assertEqual(code, 0)
            result = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(result['config']['observations'], 20)
            self.assertEqual(result['manifest']['schema_version'], 1)

    def test_invalid_research_configuration_is_rejected(self):
        for kwargs in ({'trials': 0}, {'observations': 5}, {'scenarios': ('uniform', 'uniform')}, {'alpha': 2}, {'seed': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ResearchConfig(**kwargs)


if __name__ == '__main__':
    unittest.main()
