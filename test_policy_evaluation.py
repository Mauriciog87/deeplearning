import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agents.dqn_agent import DQNAgent
from src.datasets import EvaluationSession, training_partition
from src.environment.roulette_env import RouletteEnv
from src.settlement import action_bet, settle_action
from src.utils.evaluation_harness import EvaluationConfig, evaluate_walk_forward


class PolicyEvaluationTests(unittest.TestCase):
    def test_all_action_payoffs_match_environment_and_house_edge(self):
        env = RouletteEnv()
        for action, info in env.actions.items():
            numbers, payout = action_bet(action)
            self.assertEqual(set(numbers), info.winning_numbers)
            self.assertEqual(payout, info.payout)
            profit = sum(settle_action(actual, action, 1, 100).net_profit for actual in range(37))
            self.assertEqual(profit, 0 if action == 46 else -1)

    def test_dqn_is_scored_as_policy_and_unverifiable_training_is_rejected(self):
        session = EvaluationSession('sample', tuple(range(37)) * 2)
        _, partition = training_partition([session])
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'agent.pt')
            agent = DQNAgent(hidden_size=16, embedding_dim=4, device='cpu')
            config = EvaluationConfig(training_window=30, testing_window=10, step_size=50,
                                      runs=1, models=('dqn',), device='cpu', model_path=path,
                                      compute_intervals=False, comparison_baseline='pass')
            for provenance in ({}, {'data_source': 'real', 'partition': partition},
                               {'data_source': 'simulated'}):
                agent.save(path, {'training_config': provenance})
                with patch.object(DQNAgent, 'act', return_value=37):
                    result = evaluate_walk_forward([session], config)
                summary = result.per_model_summaries['dqn']
                if provenance.get('data_source') != 'simulated':
                    self.assertEqual(summary.spins, 0)
                    self.assertTrue(result.warnings)
                else:
                    rows = [row for row in result.rows if row.model == 'dqn']
                    self.assertEqual(len(rows), 10)
                    self.assertIsNone(summary.log_loss)
                    self.assertIsNone(summary.ece)
                    self.assertEqual(summary.total_stake, 10)
                    self.assertIn('profit_per_spin', summary.comparisons)
                    for row in rows:
                        self.assertEqual(row.kind, 'policy')
                        self.assertEqual(row.number_probs, {})
                        self.assertEqual(row.profit, settle_action(row.actual, 37, 1, row.balance_before).net_profit)


if __name__ == '__main__':
    unittest.main()
