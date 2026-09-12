from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.database.models import Database
from src.database.repository import RouletteRepository
from src.engine.prediction_engine import PredictionEngine, PredictorType


class DatabaseIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.db = Database(str(Path(self.folder.name) / 'roulette.db'))

    def test_delete_session_cascades_to_spins(self):
        session = self.db.create_session('temporary')
        self.db.add_spins_bulk(session.id, [1, 2, 3])
        self.db.delete_session(session.id)
        self.assertEqual(self.db.get_total_spins(), 0)

    def test_orphan_spin_is_rejected(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.add_spin(12345, 1)

    def test_predictions_settle_once_in_the_correct_session(self):
        first = self.db.create_session('first')
        second = self.db.create_session('second')
        engine = PredictionEngine()
        prediction = engine.get_consensus_prediction()
        self.db.emit_predictions(first.id, {PredictorType.FAIR: prediction}, expected_count=0)
        self.db.add_spin(second.id, 0)
        self.assertEqual(self.db.get_predictor_stats(first.id), [])
        spin = self.db.add_spin(first.id, 0, event_id='event-1')
        duplicate = self.db.add_spin(first.id, 0, event_id='event-1')
        self.assertEqual(spin.id, duplicate.id)
        self.db.add_spin(first.id, 0, event_id='event-2')
        self.assertEqual(self.db.get_total_spins(first.id), 2)
        self.assertEqual(self.db.get_predictor_stats(first.id)[0]['total'], 1)
        self.assertEqual(self.db.get_predictor_stats(first.id)[0]['number_correct'], 1)
        with self.assertRaises(ValueError):
            self.db.emit_predictions(first.id, {PredictorType.FAIR: prediction}, expected_count=0)

    def test_sequences_and_transitions_never_cross_sessions(self):
        repo = RouletteRepository(str(self.db.db_path))
        first = repo.create_session_with_numbers([1] * 5, 'first')
        second = repo.create_session_with_numbers([2] * 5, 'second')
        sequences, targets = repo.get_training_sequences(sequence_length=3)
        self.assertEqual(len(sequences), 4)
        self.assertTrue(all(len(set(sequence)) == 1 for sequence in sequences))
        self.assertEqual(repo.compute_transition_matrix()[1, 2], 0)
        self.assertEqual([item.session_id for item in repo.get_evaluation_sessions()], [str(first.id), str(second.id)])

    def test_bulk_insert_rolls_back_on_invalid_input(self):
        session = self.db.create_session('first')
        with self.assertRaises(ValueError):
            self.db.add_spins_bulk(session.id, [1, 2, 3.5])
        self.assertEqual(self.db.get_total_spins(session.id), 0)


if __name__ == '__main__':
    unittest.main()
