import json
import time
import threading
from pathlib import Path
from typing import Optional, Callable, List
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

from .screen_capture import ScreenCapture, CaptureRegion, RegionSelector
from .ocr_reader import RouletteOCR
from .reconciliation import CaptureObservation, reconcile_history


@dataclass
class CaptureConfig:
    region_x: int = 0
    region_y: int = 0
    region_width: int = 200
    region_height: int = 50
    check_interval: float = 0.5
    change_threshold: float = 0.05
    post_detection_pause: float = 20.0
    min_confidence: float = 0.4
    use_gpu: bool = True
    reconnect_x: int = 0
    reconnect_y: int = 0
    reconnect_width: int = 0
    reconnect_height: int = 0
    inactivity_timeout: float = 60.0
    history_x: int = 0
    history_y: int = 0
    history_width: int = 0
    history_height: int = 0
    reconcile_after_reconnect: bool = True
    reconcile_count: int = 5
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'CaptureConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class RouletteMonitor:
    
    def __init__(
        self,
        on_number_detected: Optional[Callable[[int], None]] = None,
        config_path: Optional[str] = None,
        on_observation: Optional[Callable] = None
    ):
        self.screen_capture = ScreenCapture()
        self.ocr = RouletteOCR(use_gpu=True)
        self.on_number_detected = on_number_detected
        self.on_observation = on_observation
        self.pending_observations = {}
        self.accepted_events = set()
        self.stop_event = threading.Event()
        self.state_lock = threading.RLock()
        self.inactivity_thread = None
        self.last_history_confidence = 0.0
        
        self.config_path = config_path or str(Path(__file__).parent.parent.parent / "data" / "capture_config.json")
        self.config = self._load_or_create_config()
        self.ocr.use_gpu = self.config.use_gpu
        
        self.is_running = False
        self.numbers_detected: List[int] = []
        self.last_detection_time: Optional[datetime] = None
        self.last_activity_time: float = time.time()
        self.detection_count = 0
        self.reconnect_count = 0
        self.reconciled_count = 0
        self.auto_reconnect_enabled = True
        self.session_numbers: List[int] = []
        
        self._apply_config()
    
    def _load_or_create_config(self) -> CaptureConfig:
        try:
            return CaptureConfig.load(self.config_path)
        except (FileNotFoundError, json.JSONDecodeError):
            config = CaptureConfig()
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
            config.save(self.config_path)
            return config
    
    def _apply_config(self):
        if self.config.region_width > 0 and self.config.region_height > 0:
            self.screen_capture.set_region(
                self.config.region_x,
                self.config.region_y,
                self.config.region_width,
                self.config.region_height
            )
    
    def select_region(self) -> bool:
        region = RegionSelector.select_region_interactive()
        if region:
            self.config.region_x = region.x
            self.config.region_y = region.y
            self.config.region_width = region.width
            self.config.region_height = region.height
            self.config.save(self.config_path)
            self._apply_config()
            return True
        return False
    
    def set_region(self, x: int, y: int, width: int, height: int):
        self.config.region_x = x
        self.config.region_y = y
        self.config.region_width = width
        self.config.region_height = height
        self.config.save(self.config_path)
        self._apply_config()
    
    def _record_observation(self, observation):
        with self.state_lock:
            if self.stop_event.is_set():
                return False
            self.pending_observations[observation.event_id] = observation
            if self.on_observation:
                self.on_observation(observation)
            return True

    def _deliver(self, observation):
        with self.state_lock:
            if observation.event_id in self.accepted_events:
                return True
            if self.stop_event.is_set():
                return False
            self._record_observation(observation)
            try:
                if not self.on_number_detected:
                    raise RuntimeError('No persistence callback is configured')
                acknowledged = self.on_number_detected(
                    observation.numbers[0], event_id=observation.event_id, observed_at=observation.observed_at)
                if not acknowledged:
                    raise RuntimeError('Persistence was not acknowledged')
            except Exception as error:
                observation.reason = str(error)
                self._record_observation(observation)
                return False
            number = observation.numbers[0]
            self.session_numbers.append(number)
            self.numbers_detected.append(number)
            self.last_detection_time = observation.observed_at
            self.last_activity_time = time.time()
            self.detection_count += 1
            self.accepted_events.add(observation.event_id)
            observation.status = 'accepted'
            if self.on_observation:
                self.on_observation(observation)
            self.pending_observations.pop(observation.event_id, None)
            return True

    def _on_screen_change(self, image):
        if self.stop_event.wait(.3):
            return
        fresh_image = self.screen_capture.capture()
        result = self.ocr.read_first_number(fresh_image if fresh_image is not None else image)
        if self.stop_event.is_set():
            return
        if not result or self.stop_event.is_set():
            return
        number, confidence = result
        observation = CaptureObservation([number], confidence)
        if confidence < self.config.min_confidence:
            observation.reason = 'OCR confidence is below the configured threshold'
            self._record_observation(observation)
            return
        self._deliver(observation)
        if self.stop_event.wait(self.config.post_detection_pause):
            return
        if self.has_history_region():
            self._reconcile_numbers()

    def start(self):
        if self.screen_capture.region is None:
            raise ValueError("Region not configured. Call select_region() or set_region() first.")
        
        if self.is_running or any(thread and thread.is_alive() for thread in
                                  (self.screen_capture.monitor_thread, self.inactivity_thread)):
            raise RuntimeError('Previous monitoring work must finish before restarting')
        self.stop_event.clear()
        self.is_running = True
        self.last_activity_time = time.time()
        print(f"[RouletteMonitor] Starting monitoring...")
        print(f"[RouletteMonitor] Region: ({self.config.region_x}, {self.config.region_y}) {self.config.region_width}x{self.config.region_height}")
        
        try:
            self.screen_capture.start_monitoring(
                on_change=self._on_screen_change,
                check_interval=self.config.check_interval,
                change_threshold=self.config.change_threshold
            )
            if self.has_reconnect_region() and self.auto_reconnect_enabled:
                self._start_inactivity_checker()
        except Exception:
            self.is_running = False
            self.stop_event.set()
            raise

    def stop(self):
        with self.state_lock:
            self.is_running = False
            self.stop_event.set()
        self.screen_capture.stop_monitoring()
        if self.inactivity_thread and self.inactivity_thread is not threading.current_thread():
            self.inactivity_thread.join(timeout=2)
        print(f"[RouletteMonitor] Stopped. Total detections: {self.detection_count}")
    
    def test_capture(self) -> Optional[int]:
        if self.screen_capture.region is None:
            print("[RouletteMonitor] No region set")
            return None
        
        image = self.screen_capture.capture()
        if image is None:
            print("[RouletteMonitor] Failed to capture")
            return None
        
        result = self.ocr.read_first_number(image)
        if result:
            number, confidence = result
            print(f"[RouletteMonitor] Test read: {number} (confidence: {confidence:.2f})")
            return number
        else:
            print("[RouletteMonitor] No number detected in test")
            return None
    
    def get_stats(self) -> dict:
        return {
            "is_running": self.is_running,
            "detection_count": self.detection_count,
            "last_detection": self.last_detection_time.isoformat() if self.last_detection_time else None,
            "reconnect_count": self.reconnect_count,
            "reconciled_count": self.reconciled_count,
            "region": {
                "x": self.config.region_x,
                "y": self.config.region_y,
                "width": self.config.region_width,
                "height": self.config.region_height
            },
            "history_region": {
                "x": self.config.history_x,
                "y": self.config.history_y,
                "width": self.config.history_width,
                "height": self.config.history_height
            }
        }
    
    def select_reconnect_region(self) -> bool:
        print("\n[RouletteMonitor] Select the reconnect button region")
        print("Draw a rectangle around the 'Reconectar' button.")
        region = RegionSelector.select_region_interactive()
        if region:
            self.config.reconnect_x = region.x
            self.config.reconnect_y = region.y
            self.config.reconnect_width = region.width
            self.config.reconnect_height = region.height
            self.config.save(self.config_path)
            print(f"[RouletteMonitor] Reconnect region saved: ({region.x}, {region.y}) {region.width}x{region.height}")
            return True
        return False
    
    def select_history_region(self) -> bool:
        print("\n[RouletteMonitor] Select the history region")
        print("Draw a rectangle around the last numbers display.")
        print("(Newest number on the LEFT, oldest on the RIGHT)")
        region = RegionSelector.select_region_interactive()
        if region:
            self.config.history_x = region.x
            self.config.history_y = region.y
            self.config.history_width = region.width
            self.config.history_height = region.height
            self.config.save(self.config_path)
            print(f"[RouletteMonitor] History region saved: ({region.x}, {region.y}) {region.width}x{region.height}")
            return True
        return False
    
    def has_history_region(self) -> bool:
        return self.config.history_width > 0 and self.config.history_height > 0
    
    def set_session_numbers(self, numbers: List[int]):
        from src.settlement import validate_number
        with self.state_lock:
            if self.is_running:
                raise RuntimeError('Stop capture before switching sessions')
            self.session_numbers = [validate_number(number) for number in numbers]
            self.pending_observations.clear()
            self.accepted_events.clear()
    
    def _read_history_numbers(self) -> List[int]:
        if not self.has_history_region():
            return []
        
        history_region = CaptureRegion(
            x=self.config.history_x,
            y=self.config.history_y,
            width=self.config.history_width,
            height=self.config.history_height
        )
        
        temp_capture = ScreenCapture()
        temp_capture.set_region(
            history_region.x,
            history_region.y,
            history_region.width,
            history_region.height
        )
        
        image = temp_capture.capture()
        if image is None:
            return []
        
        results = self.ocr.read_all_numbers(image)
        results = results[:self.config.reconcile_count]
        self.last_history_confidence = min((confidence for _, confidence in results), default=0.0)
        if results and self.last_history_confidence < self.config.min_confidence:
            self._record_observation(CaptureObservation([number for number, _ in results], self.last_history_confidence,
                                                       'History OCR contains uncertain numbers'))
            return []
        return [number for number, _ in results]
    
    def _reconcile_numbers(self) -> int:
        if not self.has_history_region() or not self.config.reconcile_after_reconnect or self.stop_event.is_set():
            return 0
        observed = self._read_history_numbers()
        if not observed or self.stop_event.is_set():
            return 0
        with self.state_lock:
            missing, reason = reconcile_history(self.session_numbers, observed)
            if missing is None:
                self._record_observation(CaptureObservation(observed, self.last_history_confidence, reason))
                return 0
            reconciled = 0
            for number in missing:
                if not self._deliver(CaptureObservation([number], self.last_history_confidence)):
                    break
                reconciled += 1
                self.reconciled_count += 1
            return reconciled

    def has_reconnect_region(self) -> bool:
        return self.config.reconnect_width > 0 and self.config.reconnect_height > 0
    
    def _start_inactivity_checker(self):
        def check_loop():
            while self.is_running:
                if self.stop_event.wait(10):
                    break
                if not self.is_running:
                    break
                
                inactive_time = time.time() - self.last_activity_time
                
                if inactive_time >= self.config.inactivity_timeout:
                    print(f"[RouletteMonitor] Inactivity detected ({inactive_time:.0f}s). Clicking reconnect...")
                    self._click_reconnect()
                    self.last_activity_time = time.time()
        
        self.inactivity_thread = threading.Thread(target=check_loop, daemon=True)
        self.inactivity_thread.start()
    
    def _click_reconnect(self):
        if self.stop_event.is_set():
            return False
        if not PYAUTOGUI_AVAILABLE:
            print("[RouletteMonitor] pyautogui not available for auto-reconnect")
            return False
        
        if not self.has_reconnect_region():
            print("[RouletteMonitor] Reconnect region not configured")
            return False
        
        center_x = self.config.reconnect_x + self.config.reconnect_width // 2
        center_y = self.config.reconnect_y + self.config.reconnect_height // 2
        
        try:
            pyautogui.click(center_x, center_y)
            self.reconnect_count += 1
            print(f"[RouletteMonitor] Reconnect clicked at ({center_x}, {center_y}) - Total: {self.reconnect_count}")
            
            if self.stop_event.wait(3):
                return False
            self._reconcile_numbers()
            
            return True
        except Exception as e:
            print(f"[RouletteMonitor] Click error: {e}")
            return False
    
    def set_inactivity_timeout(self, seconds: float):
        self.config.inactivity_timeout = seconds
        self.config.save(self.config_path)
        print(f"[RouletteMonitor] Inactivity timeout set to {seconds}s")
