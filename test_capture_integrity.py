from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from PIL import Image

from src.capture.reconciliation import CaptureObservation, reconcile_history
from src.capture.roulette_monitor import RouletteMonitor
from src.capture.ocr_reader import RouletteOCR


class CaptureIntegrityTests(unittest.TestCase):
    def test_restart_does_not_reenable_old_capture_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            observation_sink = Mock()
            monitor = RouletteMonitor(config_path=str(Path(folder) / 'capture.json'),
                                      on_observation=observation_sink)
            monitor.set_region(0, 0, 20, 20)
            monitor.stop_event.set()
            worker = Mock()
            worker.is_alive.return_value = True
            monitor.screen_capture.monitor_thread = worker
            with self.assertRaises(RuntimeError):
                monitor.start()
            self.assertTrue(monitor.stop_event.is_set())
            self.assertFalse(monitor.is_running)
            monitor._record_observation(CaptureObservation([7], .9))
            observation_sink.assert_not_called()

    def test_sequence_alignment_preserves_repeated_outcomes(self):
        self.assertEqual(reconcile_history([3, 7], [7, 7, 3]), ([7], ''))
        self.assertIsNone(reconcile_history([1, 2, 1, 2], [3, 2, 1, 2, 1])[0])
        self.assertIsNone(reconcile_history([3], [4, 3])[0])

    def test_persistence_must_acknowledge_before_advancing_history(self):
        with tempfile.TemporaryDirectory() as folder:
            monitor = RouletteMonitor(config_path=str(Path(folder) / 'config.json'),
                                      on_number_detected=Mock(side_effect=RuntimeError('disk failure')))
            observation = CaptureObservation([7], .9)
            self.assertFalse(monitor._deliver(observation))
            self.assertEqual(monitor.session_numbers, [])
            self.assertIn(observation.event_id, monitor.pending_observations)
            monitor.on_number_detected = Mock(return_value=True)
            self.assertTrue(monitor._deliver(observation))
            self.assertTrue(monitor._deliver(observation))
            self.assertEqual(monitor.session_numbers, [7])
            self.assertTrue(monitor._deliver(CaptureObservation([7], .9)))
            self.assertEqual(monitor.session_numbers, [7, 7])
            monitor.stop_event.set()
            self.assertFalse(monitor._deliver(CaptureObservation([8], .9)))

    def test_green_background_is_not_zero_without_ocr_evidence(self):
        ocr = RouletteOCR(use_gpu=False)
        ocr._initialized = True
        ocr.reader = Mock()
        ocr.reader.readtext.return_value = []
        image = Image.new('RGB', (20, 20), 'green')
        self.assertIsNone(ocr.read_number(image))
        self.assertEqual(ocr.read_all_numbers(image), [])


if __name__ == '__main__':
    unittest.main()
