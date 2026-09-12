import json
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from src.utils.bias_detection import BiasLevel, WheelBiasAnalyzer


class SequentialAnalyzerTests(unittest.TestCase):
    def test_reset_spends_budget_and_preserves_event_deduplication(self):
        analyzer = WheelBiasAnalyzer(200, stream_id='a', alpha=.05, streams=('a', 'b'))
        analyzer.add_spin(7, event_id='first')
        first_alpha = analyzer.monitor.alpha
        analyzer.reset()
        analyzer.add_spin(7, event_id='first')
        self.assertEqual(analyzer.total_spins, 0)
        self.assertLess(analyzer.monitor.alpha, first_alpha)
        analyzer.add_spin(7, event_id='second')
        self.assertEqual(analyzer.total_spins, 1)
        self.assertEqual(analyzer.sequential_evidence()['restart_index'], 1)

    def test_analyzer_state_roundtrip_preserves_budget_and_history(self):
        analyzer = WheelBiasAnalyzer(200, stream_id='a')
        for index in range(50):
            analyzer.add_spin(index % 37, event_id=str(index))
        analyzer.reset()
        for index in range(50, 100):
            analyzer.add_spin(index % 37, event_id=str(index))
        restored = WheelBiasAnalyzer.from_state(json.loads(json.dumps(analyzer.to_state())))
        for index in range(100, 200):
            analyzer.add_spin(7, event_id=str(index))
            restored.add_spin(7, event_id=str(index))
        self.assertEqual(analyzer.to_state(), restored.to_state())
        self.assertEqual(analyzer.sequential_evidence(), restored.sequential_evidence())
        self.assertEqual(analyzer.spin_history, restored.spin_history)
        restored.add_spin(1, event_id='1')
        self.assertEqual(restored.total_spins, 150)

    def test_candidates_require_a_confidence_bound_above_break_even(self):
        analyzer = WheelBiasAnalyzer(200, stream_id='a')
        for index in range(200):
            analyzer.add_spin(7, event_id=str(index))
        self.assertEqual(analyzer.get_recommended_bets(), [7])
        report = analyzer.get_report()
        self.assertIsNotNone(report.sequential_evidence)
        self.assertTrue(report.sequential_evidence['rejected'])
        self.assertGreater(analyzer.monitor.probability_bounds([7])[0], 1 / 36)

    def test_hh_consumes_sequential_evidence_instead_of_repeated_fixed_tests(self):
        from train_hh import train_episode
        agent = MagicMock(current_llh=None, bankroll=1000)
        agent.select_action.return_value = (46, 0)
        env = MagicMock()
        env.reset.return_value = ({}, {})
        env.step.return_value = ({}, 0, True, False, {'winning_number': 0})
        analyzer = MagicMock(total_spins=50)
        analyzer.get_report.return_value = SimpleNamespace(overall_bias=BiasLevel.STRONG)
        analyzer.sequential_evidence.return_value = {'rejected': False}
        analyzer.get_recommended_bets.return_value = []
        train_episode(agent, env, 1, analyzer, seed=1)
        agent.set_bias_detected.assert_called_once_with(False, [])


if __name__ == '__main__':
    unittest.main()
