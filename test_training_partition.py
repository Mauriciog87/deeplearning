from pathlib import Path
import tempfile
import unittest

from src.datasets import EvaluationSession, training_partition, held_out_segments
from train import train_agent
from test import test_agent, test_random_agent


class TrainingPartitionTests(unittest.TestCase):
    def test_test_partition_rejects_changed_data_and_overlap(self):
        numbers = list(range(37)) * 2
        segments, partition = training_partition(numbers)
        self.assertEqual(len(segments[0]), 59)
        self.assertEqual(held_out_segments(numbers, partition)[0][20:], numbers[59:])
        with self.assertRaises(ValueError):
            held_out_segments(numbers + [1], partition)
        partition['partitions'][0]['test_spin_ids'].append(partition['partitions'][0]['train_spin_ids'][0])
        with self.assertRaises(ValueError):
            held_out_segments(numbers, partition)

    def test_training_and_testing_use_reserved_partition(self):
        sessions = [EvaluationSession('a', tuple(list(range(37)) * 2))]
        with tempfile.TemporaryDirectory() as folder:
            result = train_agent(episodes=1, max_spins=4, batch_size=4, save_dir=folder,
                                 real_numbers=sessions, use_real_data=True, device='cpu', verbose=False)
            evaluated = test_agent(model_path=result['model_path'], use_real_data=True, real_numbers=sessions,
                                   max_spins=15, device='cpu', verbose=False)
            self.assertEqual(evaluated['episodes'], 1)
            self.assertEqual(evaluated['avg_steps'], 15)
            random = test_random_agent(use_real_data=True, real_numbers=sessions, max_spins=15,
                                       partition=evaluated['partition'], verbose=False)
            self.assertEqual(random['episodes'], 1)
            with self.assertRaises(FileNotFoundError):
                test_agent(model_path=str(Path(folder) / 'missing.pt'), verbose=False)


if __name__ == '__main__':
    unittest.main()
