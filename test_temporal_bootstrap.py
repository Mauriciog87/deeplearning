from types import SimpleNamespace
import unittest

import numpy as np

from src.utils.temporal_statistics import block_bootstrap_indices


class TemporalBootstrapTests(unittest.TestCase):
    def test_repeating_same_outcomes_across_runs_does_not_shrink_uncertainty(self):
        single = [SimpleNamespace(run_id=0, session_id='a', index=index, profit=index % 7)
                  for index in range(40)]
        repeated = [SimpleNamespace(run_id=run, session_id='a', index=row.index, profit=row.profit)
                    for run in range(5) for row in single]
        def means(rows):
            return [np.mean([rows[index].profit for index in sample])
                    for sample in block_bootstrap_indices(rows, 100, 42, block_length=4)]
        np.testing.assert_array_equal(means(single), means(repeated))


if __name__ == '__main__':
    unittest.main()
