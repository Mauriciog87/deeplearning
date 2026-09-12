import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.agents.dqn_agent import DQNAgent
from src.agents.fuzzy_adaptive import FuzzyAdaptiveDQN
from src.agents.hyper_heuristic import DQNHyperHeuristic, HyperHeuristicAgent, LLHType
from src.checkpoints import seed_everything


class AgentStateTests(unittest.TestCase):
    def test_hh_replay_credits_executed_heuristic(self):
        seed_everything(0, 'cpu')
        agent = DQNHyperHeuristic(seed=0)
        agent.total_steps = 4
        executed = agent.select_llh()
        agent.update(1, 0, -10)
        self.assertEqual(agent.replay_buffer[-1][1], list(LLHType).index(executed))

    def test_evaluation_updates_progression_without_learning(self):
        agent = HyperHeuristicAgent(seed=0)
        agent.current_llh = LLHType.MARTINGALE
        before = copy.deepcopy(dict(agent.q_table))
        epsilon = agent.epsilon
        agent.update(1, 0, -10, learn=False)
        self.assertEqual(agent.llhs[LLHType.MARTINGALE].current_multiplier, 2)
        self.assertEqual(dict(agent.q_table), before)
        self.assertEqual(agent.epsilon, epsilon)

    def test_dqn_resume_matches_next_learning_step(self):
        seed_everything(12, 'cpu')
        agent = DQNAgent(hidden_size=16, embedding_dim=4, batch_size=4, device='cpu')
        for i in range(8):
            history = np.full(20, i)
            agent.remember(history, 1, i, -1, history + 1, .999, False,
                           next_action_mask=np.ones(47, dtype=bool), truncated=i == 7)
        agent.train()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'nested' / 'agent.pt'
            agent.save(str(path))
            expected_loss = agent.train()
            expected = copy.deepcopy(agent.q_network.state_dict())
            restored = DQNAgent(device='cpu')
            restored.load(str(path))
            self.assertEqual(restored.train(), expected_loss)
            for key, value in expected.items():
                torch.testing.assert_close(restored.q_network.state_dict()[key], value, rtol=0, atol=0)

    def test_target_respects_mask_and_bootstraps_truncation_only(self):
        agent = DQNAgent(hidden_size=16, embedding_dim=4, batch_size=4, device='cpu')
        with torch.no_grad():
            for network in (agent.q_network, agent.target_network):
                for parameter in network.parameters():
                    parameter.zero_()
            online = [layer for layer in agent.q_network.modules() if isinstance(layer, torch.nn.Linear)][-1]
            target = [layer for layer in agent.target_network.modules() if isinstance(layer, torch.nn.Linear)][-1]
            online.bias[0], online.bias[46] = 100, 1
            target.bias[0], target.bias[46] = 999, 5
        targets = []
        def criterion(current, expected):
            targets.extend(expected.detach().tolist())
            return torch.nn.functional.mse_loss(current, expected)
        agent.criterion = criterion
        mask = np.zeros(47, dtype=bool)
        mask[46] = True
        for terminal in (True, False, True, False):
            agent.remember(np.zeros(20), 1, 46, -1, np.ones(20), 1,
                           terminal, mask, truncated=not terminal)
        agent.train()
        np.testing.assert_allclose(sorted(targets), [-1, -1, -1 + agent.gamma * 5, -1 + agent.gamma * 5])

    def test_hh_round_trip_includes_network_and_progression(self):
        for cls, suffix in ((HyperHeuristicAgent, '.json'), (DQNHyperHeuristic, '.pt')):
            agent = cls(seed=13)
            agent.select_action()
            agent.update(1, 0, -10)
            with tempfile.TemporaryDirectory() as folder:
                path = str(Path(folder) / ('agent' + suffix))
                agent.save(path)
                restored = cls.load(path)
                self.assertEqual(restored.select_action(), agent.select_action())
                self.assertEqual(restored.bankroll, agent.bankroll)
                if cls is DQNHyperHeuristic:
                    for key, value in agent.q_network.state_dict().items():
                        torch.testing.assert_close(restored.q_network.state_dict()[key], value)

    def test_dqn_hh_resume_matches_next_learning_step(self):
        seed_everything(13, 'cpu')
        agent = DQNHyperHeuristic(seed=13, device='cpu')
        agent.batch_size = 4
        for number in range(8):
            action, stake = agent.select_action()
            agent.update(number, action, -stake)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'hh.pt')
            agent.save(path)
            agent._train_step()
            expected = copy.deepcopy(agent.q_network.state_dict())
            restored = DQNHyperHeuristic.load(path, device='cpu')
            restored._train_step()
            for key, value in expected.items():
                torch.testing.assert_close(restored.q_network.state_dict()[key], value, rtol=0, atol=0)

    def test_fuzzy_resume_preserves_estimators(self):
        seed_everything(21, 'cpu')
        agent = FuzzyAdaptiveDQN(DQNAgent(hidden_size=16, embedding_dim=4, batch_size=4, device='cpu'))
        for i in range(25):
            history = np.full(20, i)
            agent.remember(history, 1, i, -1, history + 1, .999, False, np.arange(47))
        agent.end_episode(-25)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'fuzzy.pt')
            agent.save(path)
            restored = FuzzyAdaptiveDQN(DQNAgent(device='cpu'))
            restored.load(path)
            self.assertEqual(agent.get_metrics(), restored.get_metrics())
            self.assertEqual(agent.fuzzy_controller.compute_epsilon(True),
                             restored.fuzzy_controller.compute_epsilon(True))


if __name__ == '__main__':
    unittest.main()
