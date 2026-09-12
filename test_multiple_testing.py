import unittest

from src.utils.multiple_testing import benjamini_hochberg


class MultipleTestingTest(unittest.TestCase):
    def test_benjamini_hochberg_returns_monotone_q_values(self):
        q_values, rejected = benjamini_hochberg([0.001, 0.02, 0.03, 0.9], alpha=0.05)

        self.assertEqual(len(q_values), 4)
        self.assertTrue(rejected[0])
        self.assertTrue(rejected[1])
        self.assertTrue(rejected[2])
        self.assertFalse(rejected[3])
        self.assertLessEqual(q_values[0], q_values[1])
        self.assertLessEqual(q_values[1], q_values[2])

    def test_benjamini_hochberg_ignores_missing_p_values(self):
        q_values, rejected = benjamini_hochberg([None, 0.01, None], alpha=0.05)

        self.assertIsNone(q_values[0])
        self.assertIsNotNone(q_values[1])
        self.assertIsNone(q_values[2])
        self.assertFalse(rejected[0])
        self.assertTrue(rejected[1])
        self.assertFalse(rejected[2])


if __name__ == "__main__":
    unittest.main()
