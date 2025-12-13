import json
import time
import threading
from pathlib import Path
from typing import Optional, Callable, List
from dataclasses import dataclass, asdict, field
from datetime import datetime

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

from .screen_capture import ScreenCapture, CaptureRegion, RegionSelector
from .ocr_reader import RouletteOCR


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
        config_path: Optional[str] = None
    ):
        self.screen_capture = ScreenCapture()
        self.ocr = RouletteOCR(use_gpu=True)
        self.on_number_detected = on_number_detected
        
        self.config_path = config_path or str(Path(__file__).parent.parent.parent / "data" / "capture_config.json")
        self.config = self._load_or_create_config()
        
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
    
    def _on_screen_change(self, image):
        time.sleep(0.3)
        
        fresh_image = self.screen_capture.capture()
        if fresh_image is None:
            fresh_image = image
        
        result = self.ocr.read_first_number(fresh_image)
        
        if result:
            number, confidence = result
            
            if confidence < self.config.min_confidence:
                print(f"[RouletteMonitor] Low confidence: {number} ({confidence:.2f}) - IGNORED")
                return
            
            self.numbers_detected.append(number)
            self.session_numbers.append(number)
            self.last_detection_time = datetime.now()
            self.last_activity_time = time.time()
            self.detection_count += 1
            
            print(f"[RouletteMonitor] Detected: {number} (confidence: {confidence:.2f})")
            
            if self.on_number_detected:
                try:
                    self.on_number_detected(number)
                    print(f">>> Number {number} saved <<<")
                except Exception as e:
                    print(f"[RouletteMonitor] Callback error: {e}")
            
            if self.config.post_detection_pause > 0:
                print(f"[RouletteMonitor] Pausing {self.config.post_detection_pause}s...")
                time.sleep(self.config.post_detection_pause)
                self.screen_capture.last_image = None
                print("[RouletteMonitor] Resuming...")
        else:
            print("[RouletteMonitor] Change detected but no number found")
    
    def start(self):
        if self.screen_capture.region is None:
            raise ValueError("Region not configured. Call select_region() or set_region() first.")
        
        self.is_running = True
        self.last_activity_time = time.time()
        print(f"[RouletteMonitor] Starting monitoring...")
        print(f"[RouletteMonitor] Region: ({self.config.region_x}, {self.config.region_y}) {self.config.region_width}x{self.config.region_height}")
        
        if self.has_reconnect_region() and self.auto_reconnect_enabled:
            print(f"[RouletteMonitor] Auto-reconnect enabled (timeout: {self.config.inactivity_timeout}s)")
            self._start_inactivity_checker()
        
        self.screen_capture.start_monitoring(
            on_change=self._on_screen_change,
            check_interval=self.config.check_interval,
            change_threshold=self.config.change_threshold
        )
    
    def stop(self):
        self.is_running = False
        self.screen_capture.stop_monitoring()
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
        self.session_numbers = list(numbers)
    
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
        numbers = [n for n, conf in results if conf >= self.config.min_confidence]
        
        return numbers[:self.config.reconcile_count]
    
    def _reconcile_numbers(self) -> int:
        if not self.has_history_region() or not self.config.reconcile_after_reconnect:
            return 0
        
        print("[RouletteMonitor] Reading history for reconciliation...")
        history_numbers = self._read_history_numbers()
        
        if not history_numbers:
            print("[RouletteMonitor] No numbers found in history region")
            return 0
        
        print(f"[RouletteMonitor] History numbers (L->R): {history_numbers}")
        
        last_known = self.session_numbers[-1] if self.session_numbers else None
        print(f"[RouletteMonitor] Last known number: {last_known}")
        
        if last_known is None:
            return 0
        
        try:
            sync_point = history_numbers.index(last_known)
            print(f"[RouletteMonitor] Found sync point at index {sync_point}")
        except ValueError:
            print(f"[RouletteMonitor] Last known number {last_known} not in history - cannot reconcile")
            return 0
        
        missing_numbers = history_numbers[:sync_point]
        
        if not missing_numbers:
            print("[RouletteMonitor] No new numbers to reconcile")
            return 0
        
        print(f"[RouletteMonitor] Missing numbers to add: {missing_numbers}")
        
        reconciled = 0
        for number in reversed(missing_numbers):
            print(f"[RouletteMonitor] Reconciling missing number: {number}")
            self.session_numbers.append(number)
            self.reconciled_count += 1
            reconciled += 1
            
            if self.on_number_detected:
                try:
                    self.on_number_detected(number)
                    print(f">>> Reconciled number {number} saved <<<")
                except Exception as e:
                    print(f"[RouletteMonitor] Callback error: {e}")
        
        print(f"[RouletteMonitor] Reconciled {reconciled} number(s)")
        return reconciled
    
    def has_reconnect_region(self) -> bool:
        return self.config.reconnect_width > 0 and self.config.reconnect_height > 0
    
    def _start_inactivity_checker(self):
        def check_loop():
            while self.is_running:
                time.sleep(10)
                if not self.is_running:
                    break
                
                inactive_time = time.time() - self.last_activity_time
                
                if inactive_time >= self.config.inactivity_timeout:
                    print(f"[RouletteMonitor] Inactivity detected ({inactive_time:.0f}s). Clicking reconnect...")
                    self._click_reconnect()
                    self.last_activity_time = time.time()
        
        thread = threading.Thread(target=check_loop, daemon=True)
        thread.start()
    
    def _click_reconnect(self):
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
            
            time.sleep(3)
            self._reconcile_numbers()
            
            return True
        except Exception as e:
            print(f"[RouletteMonitor] Click error: {e}")
            return False
    
    def set_inactivity_timeout(self, seconds: float):
        self.config.inactivity_timeout = seconds
        self.config.save(self.config_path)
        print(f"[RouletteMonitor] Inactivity timeout set to {seconds}s")
