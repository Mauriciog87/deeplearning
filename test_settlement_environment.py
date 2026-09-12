import unittest

import numpy as np

from src.environment.roulette_env import RouletteEnv, RouletteEnvFlat, RouletteEnvNearMissFlat
from src.utils.backtesting import flat_bet_backtest


class SettlementEnvironmentTests(unittest.TestCase):
    def test_five_straights_over_complete_wheel(self):
        result = flat_bet_backtest(list(range(37)), list(range(5)), 1, 1000)
        self.assertEqual(result.total_profit, -5)
        self.assertEqual(result.bets[0].profit, 31)
        self.assertEqual(result.bets[5].profit, -5)

    def test_seed_controls_all_spins(self):
        env = RouletteEnv()
        first, _ = env.reset(seed=42)
        second, _ = env.reset(seed=42)
        np.testing.assert_array_equal(first['history'], second['history'])

    def test_flat_observation_contains_large_win(self):
        for cls in (RouletteEnv, RouletteEnvFlat, RouletteEnvNearMissFlat):
            env = cls(real_data=[1] * 22, bet_size=50)
            env.reset(seed=42)
            obs, _, _, _, _ = env.step(1)
            self.assertTrue(env.observation_space.contains(obs), cls.__name__)
            gain = obs['gain'][0] if isinstance(obs, dict) else obs[20]
            self.assertAlmostEqual(float(gain), 2.75)

    def test_default_stake_is_one_unit(self):
        self.assertEqual(RouletteEnv().bet_size, 1)

    def test_pass_and_exhausted_bankroll_do_not_count_as_losing_bets(self):
        for selected, balance in (([], 1000), ([1], 0)):
            result = flat_bet_backtest(list(range(37)), selected, 1, balance)
            self.assertEqual(result.total_bets, 0)
            self.assertEqual(result.total_profit, 0)
            self.assertEqual(result.final_balance, balance)

    def test_stake_is_used_and_rejected_before_consuming_spin(self):
        env = RouletteEnv(real_data=[1] * 24, bet_size=10)
        env.reset()
        for stake in (10, 20, 40):
            _, reward, _, _, info = env.step(0, stake=stake)
            self.assertEqual(reward, -stake)
            self.assertEqual(info['total_stake'], stake)
        index = env.real_data_index
        with self.assertRaises(ValueError):
            env.step(0, stake=10000)
        self.assertEqual(env.real_data_index, index)
        _, reward, _, _, info = env.step(46)
        self.assertEqual(reward, 0)
        self.assertEqual(info['total_stake'], 0)

    def test_real_data_ends_without_wrapping(self):
        env = RouletteEnv(real_data=[1] * 21)
        env.reset()
        _, _, terminated, truncated, _ = env.step(46)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        with self.assertRaises(RuntimeError):
            env.step(46)
        self.assertEqual(env.real_data_index, 21)


if __name__ == '__main__':
    unittest.main()
