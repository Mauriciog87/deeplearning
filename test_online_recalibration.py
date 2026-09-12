import json
import unittest

import numpy as np
from scipy.optimize import check_grad

from src.utils.online_recalibration import OnlineRecalibrator


class OnlineRecalibrationTests(unittest.TestCase):
    def test_halfspace_gradient_and_adversary_are_checked_independently(self):
        rng = np.random.default_rng(451)
        recalibrator = OnlineRecalibrator(max_iterations=5)
        baseline = rng.dirichlet(np.ones(37))
        recalibrator.forecast(baseline, event_id='first')
        recalibrator.update(7, event_id='first')
        recalibrator._regret_sum = .2
        candidate = rng.dirichlet(np.ones(37))
        values, gradients = recalibrator._halfspace(candidate, baseline)
        for actual in (0, 7, 36):
            error = check_grad(lambda p: recalibrator._halfspace(p, baseline)[0][actual],
                               lambda p: recalibrator._halfspace(p, baseline)[1][actual], candidate)
            self.assertLess(error, 1e-6)
            outcome = np.eye(37)[actual]
            payoff = recalibrator._features(candidate)[0][:, None] * (outcome - candidate)[None, :]
            expected = np.sum(recalibrator._calibration_sum * payoff)
            regret = max(0, recalibrator._regret_sum)
            expected += regret * (np.sum((candidate - outcome) ** 2) - np.sum((baseline - outcome) ** 2)) / 2
            self.assertAlmostEqual(values[actual], expected, places=13)

    def test_adversarial_outcomes_satisfy_the_bound_with_actual_oracle_residuals(self):
        recalibrator = OnlineRecalibrator(max_iterations=1)
        baseline = np.full(37, .2 / 36)
        baseline[7] = .8
        for index in range(25):
            decision = recalibrator.forecast(baseline, event_id=str(index))
            probabilities = np.asarray(decision['probabilities'])
            self.assertEqual(recalibrator.snapshot()['observations'], index)
            np.testing.assert_allclose(probabilities.sum(), 1, atol=1e-14)
            self.assertTrue(np.all(probabilities >= 0))
            actual = int(np.argmax(recalibrator._halfspace(probabilities, baseline)[0]))
            self.assertAlmostEqual(decision['oracle_upper_bound'], recalibrator._halfspace(probabilities, baseline)[0][actual])
            recalibrator.update(actual, event_id=str(index))
            report = recalibrator.snapshot()
            self.assertLessEqual(report['cone_distance'], report['residual_bound'] + 1e-12)
        self.assertGreater(recalibrator.snapshot()['maximum_positive_oracle_residual'], 0)
        self.assertIn('No unconditional full-calibration', recalibrator.snapshot()['interpretation'])

    def test_pending_forecasts_precede_labels_and_survive_resume(self):
        baseline = np.full(37, 1 / 37)
        recalibrator = OnlineRecalibrator(max_iterations=10)
        with self.assertRaises(ValueError):
            recalibrator.update(7, event_id='0')
        decision = recalibrator.forecast(baseline, event_id='0')
        self.assertEqual(recalibrator.forecast(baseline, event_id='0'), decision)
        with self.assertRaises(ValueError):
            recalibrator.forecast(baseline, event_id='1')
        restored = OnlineRecalibrator.from_state(json.loads(json.dumps(recalibrator.to_state(), allow_nan=False)))
        for item in (recalibrator, restored):
            item.update(7, event_id='0')
            self.assertFalse(item.update(7, event_id='0'))
        self.assertEqual(restored.forecast(baseline, event_id='1'), recalibrator.forecast(baseline, event_id='1'))
        for item in (recalibrator, restored):
            item.update(0, event_id='1')
        self.assertEqual(restored.snapshot(), recalibrator.snapshot())
        again = OnlineRecalibrator.from_state(json.loads(json.dumps(restored.to_state())))
        self.assertEqual(again.snapshot(), restored.snapshot())

    def test_saved_forecasts_and_oracle_residuals_cannot_disagree(self):
        recalibrator = OnlineRecalibrator()
        recalibrator.forecast(np.full(37, 1 / 37), event_id='first')
        state = recalibrator.to_state()
        state['pending']['oracle_upper_bound'] = -1
        state['pending']['oracle_nonpositive'] = True
        with self.assertRaises(ValueError):
            OnlineRecalibrator.from_state(state)


if __name__ == '__main__':
    unittest.main()
