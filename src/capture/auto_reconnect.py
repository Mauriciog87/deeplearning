import time
import threading
from typing import Optional, Tuple
from PIL import Image
import numpy as np

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

from .screen_capture import _capture_region, CaptureRegion


class AutoReconnect:
    
    def __init__(self):
        self.button_region: Optional[CaptureRegion] = None
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.check_interval = 3.0
        self.reconnect_count = 0
        self.last_reconnect_time: Optional[float] = None
        self.baseline_image: Optional[np.ndarray] = None
        self.popup_detected = False
    
    def set_button_region(self, x: int, y: int, width: int, height: int):
        self.button_region = CaptureRegion(x, y, width, height)
        self.baseline_image = None
    
    def capture_baseline(self):
        if self.button_region is None:
            return
        img = _capture_region(self.button_region.to_mss_monitor())
        if img:
            self.baseline_image = np.array(img.convert('L'))
            print("[AutoReconnect] Baseline captured (normal state)")
    
    def _detect_button(self) -> bool:
        if self.button_region is None:
            return False
        
        img = _capture_region(self.button_region.to_mss_monitor())
        if img is None:
            return False
        
        pixels = np.array(img)
        gray = np.array(img.convert('L'))
        
        mean_brightness = np.mean(gray)
        
        has_teal = False
        r, g, b = pixels[:,:,0], pixels[:,:,1], pixels[:,:,2]
        teal_mask = (g > r + 20) & (b > r + 20) & (g > 100) & (b > 100)
        teal_ratio = np.sum(teal_mask) / teal_mask.size
        if teal_ratio > 0.05:
            has_teal = True
        
        if self.baseline_image is not None and self.baseline_image.shape == gray.shape:
            diff = np.abs(gray.astype(float) - self.baseline_image.astype(float))
            change_ratio = np.mean(diff) / 255.0
            if change_ratio > 0.15 and mean_brightness > 80:
                return True
        
        if has_teal and mean_brightness > 100:
            return True
        
        if mean_brightness > 150:
            return True
        
        return False
    
    def _click_button(self):
        if not PYAUTOGUI_AVAILABLE or self.button_region is None:
            return False
        
        center_x = self.button_region.x + self.button_region.width // 2
        center_y = self.button_region.y + self.button_region.height // 2
        
        try:
            pyautogui.click(center_x, center_y)
            self.reconnect_count += 1
            self.last_reconnect_time = time.time()
            print(f"[AutoReconnect] Clicked at ({center_x}, {center_y}) - Total: {self.reconnect_count}")
            return True
        except Exception as e:
            print(f"[AutoReconnect] Click error: {e}")
            return False
    
    def check_and_reconnect(self) -> bool:
        if self._detect_button():
            print("[AutoReconnect] Disconnect popup detected!")
            time.sleep(0.5)
            return self._click_button()
        return False
    
    def start_monitoring(self, check_interval: float = 2.0):
        if not PYAUTOGUI_AVAILABLE:
            print("[AutoReconnect] pyautogui not available")
            return
        
        if self.button_region is None:
            print("[AutoReconnect] Button region not set")
            return
        
        self.check_interval = check_interval
        self.is_monitoring = True
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"[AutoReconnect] Monitoring started (checking every {check_interval}s)")
    
    def stop_monitoring(self):
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
            self.monitor_thread = None
        print(f"[AutoReconnect] Stopped. Total reconnects: {self.reconnect_count}")
    
    def _monitor_loop(self):
        while self.is_monitoring:
            self.check_and_reconnect()
            time.sleep(self.check_interval)


def select_reconnect_button() -> Optional[CaptureRegion]:
    from .screen_capture import RegionSelector
    print("\n[AutoReconnect] Select the 'Reconectar' button region")
    print("Draw a rectangle around the button.")
    return RegionSelector.select_region_interactive()
