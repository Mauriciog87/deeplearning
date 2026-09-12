import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from src.engine.prediction_engine import PredictionEngine, PredictorType
from src.gui.app import RouletteGUI
from src.utils.predictor import ExtraTreesPredictor


class GuiJobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.app = RouletteGUI(db_path=str(Path(cls.folder.name) / 'gui.db'))
        cls.app.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.app._close()
        cls.folder.cleanup()

    def setUp(self):
        self.app._jobs.clear()
        self.app.engine.load_history([])

    def test_stale_and_failed_jobs_restore_buttons_without_publishing(self):
        token, cancel = self.app._register_job('backtest')
        self.app.engine.load_history([1], session_id='new')
        with patch.object(self.app, '_backtest_finished') as publish:
            self.app._finish_job('backtest', token, object(), None)
            publish.assert_not_called()
        self.assertEqual(self.app.backtest_btn.cget('state'), 'normal')
        token, cancel = self.app._register_job('backtest')
        self.app._finish_job('backtest', token, None, 'deliberate failure')
        self.assertIn('deliberate failure', self.app.status_bar.status_label.cget('text'))
        self.assertEqual(self.app._jobs, {})

    def test_cancelled_result_is_discarded(self):
        token, cancel = self.app._register_job('backtest')
        self.app._cancel_jobs()
        self.assertTrue(cancel.is_set())
        with patch.object(self.app, '_backtest_finished') as publish:
            self.app._finish_job('backtest', token, object(), None)
            publish.assert_not_called()
        self.assertEqual(self.app.status_bar.status_label.cget('text'), 'Cancelled')

    def test_number_entry_settles_only_previously_emitted_predictions(self):
        session = self.app.db.create_session('GUI ledger')
        self.app.current_session_id = session.id
        self.app._load_session_data(session.id)
        self.assertEqual(self.app.db.get_predictor_stats(session.id), [])
        self.app._on_add_number(0)
        stats = self.app.db.get_predictor_stats(session.id)
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]['predictor_type'], 'fair')
        self.assertEqual(stats[0]['total'], 1)
        self.assertEqual(stats[0]['accuracies']['number'], 100)

    def test_training_failure_and_session_change_release_state(self):
        engine = PredictionEngine(device='cpu')
        engine.load_history(list(range(37)))
        with patch.object(ExtraTreesPredictor, 'fit', side_effect=RuntimeError('fit failed')):
            engine.train_sync(models=('extra_trees',))
        self.assertFalse(engine.training_in_progress)
        self.assertEqual(engine.statuses[PredictorType.EXTRA_TREES].state.value, 'failed')
        started, release = threading.Event(), threading.Event()
        def delayed(model, history):
            started.set()
            if not release.wait(5):
                raise TimeoutError('Test did not release training')
            model.is_fitted = True
            return {'status': 'ready'}
        with patch.object(ExtraTreesPredictor, 'fit', delayed):
            worker = threading.Thread(target=lambda: engine.train_sync(models=('extra_trees',)))
            worker.start()
            try:
                self.assertTrue(started.wait(5))
                engine.load_history([2, 2], session_id='replacement')
            finally:
                release.set()
                worker.join(10)
        self.assertFalse(worker.is_alive())
        self.assertFalse(engine.training_in_progress)
        self.assertFalse(engine.is_extra_trees_trained)
        self.assertEqual(engine.history, [2, 2])


if __name__ == '__main__':
    unittest.main()
