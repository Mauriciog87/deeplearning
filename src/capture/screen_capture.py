import time
import threading
from typing import Optional, Callable, Tuple, List
from dataclasses import dataclass
import numpy as np
from PIL import Image
import mss

_mss_lock = threading.Lock()


@dataclass
class CaptureRegion:
    x: int
    y: int
    width: int
    height: int
    
    def to_mss_monitor(self) -> dict:
        return {
            "left": self.x,
            "top": self.y,
            "width": self.width,
            "height": self.height
        }


def _capture_region(region: dict) -> Optional[Image.Image]:
    with _mss_lock:
        try:
            sct = mss.mss()
            screenshot = sct.grab(region)
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            sct.close()
            return img
        except Exception as e:
            print(f"[ScreenCapture] Error: {e}")
            return None


def _capture_full_screen() -> Optional[Image.Image]:
    with _mss_lock:
        try:
            sct = mss.mss()
            monitor = sct.monitors[0]
            screenshot = sct.grab(monitor)
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            sct.close()
            return img
        except Exception as e:
            print(f"[ScreenCapture] Full screen error: {e}")
            return None


class ScreenCapture:
    
    def __init__(self):
        self.region: Optional[CaptureRegion] = None
        self.last_image: Optional[np.ndarray] = None
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.on_change_callback: Optional[Callable[[Image.Image], None]] = None
        self.change_threshold = 0.05
        self.check_interval = 0.5
    
    def set_region(self, x: int, y: int, width: int, height: int):
        self.region = CaptureRegion(x, y, width, height)
        self.last_image = None
    
    def capture(self) -> Optional[Image.Image]:
        if self.region is None:
            return None
        return _capture_region(self.region.to_mss_monitor())
    
    def capture_full_screen(self) -> Optional[Image.Image]:
        return _capture_full_screen()
    
    def _image_to_array(self, img: Image.Image) -> np.ndarray:
        return np.array(img.convert("L"))
    
    def _calculate_difference(self, img1: np.ndarray, img2: np.ndarray) -> float:
        if img1.shape != img2.shape:
            return 1.0
        diff = np.abs(img1.astype(float) - img2.astype(float))
        return np.mean(diff) / 255.0
    
    def has_changed(self, current_img: Image.Image) -> bool:
        current_array = self._image_to_array(current_img)
        
        if self.last_image is None:
            self.last_image = current_array
            return True
        
        diff = self._calculate_difference(self.last_image, current_array)
        
        if diff > self.change_threshold:
            self.last_image = current_array
            return True
        
        return False
    
    def start_monitoring(
        self, 
        on_change: Callable[[Image.Image], None],
        check_interval: float = 0.5,
        change_threshold: float = 0.05
    ):
        if self.region is None:
            raise ValueError("Region not set. Call set_region first.")
        
        self.on_change_callback = on_change
        self.check_interval = check_interval
        self.change_threshold = change_threshold
        self.is_monitoring = True
        self.last_image = None
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
            self.monitor_thread = None
    
    def _monitor_loop(self):
        stable_count = 0
        last_change_time = time.time()
        waiting_for_stable = False
        
        while self.is_monitoring:
            img = self.capture()
            if img is None:
                time.sleep(self.check_interval)
                continue
            
            current_array = self._image_to_array(img)
            
            if self.last_image is None:
                self.last_image = current_array
                time.sleep(self.check_interval)
                continue
            
            diff = self._calculate_difference(self.last_image, current_array)
            
            if diff > self.change_threshold:
                waiting_for_stable = True
                stable_count = 0
                last_change_time = time.time()
                self.last_image = current_array
            elif waiting_for_stable:
                stable_count += 1
                if stable_count >= 3:
                    waiting_for_stable = False
                    if self.on_change_callback:
                        try:
                            self.on_change_callback(img)
                        except Exception as e:
                            print(f"[ScreenCapture] Callback error: {e}")
            
            time.sleep(self.check_interval)
    
    def get_monitors_info(self) -> List[dict]:
        with _mss_lock:
            sct = mss.mss()
            monitors = sct.monitors
            sct.close()
            return monitors


class RegionSelector:
    
    @staticmethod
    def select_region_interactive() -> Optional[CaptureRegion]:
        try:
            import tkinter as tk
            from PIL import ImageTk
        except ImportError:
            print("tkinter not available")
            return None
        
        with _mss_lock:
            sct = mss.mss()
            all_monitors = sct.monitors[0]
            screenshot_raw = sct.grab(all_monitors)
            screenshot = Image.frombytes("RGB", screenshot_raw.size, screenshot_raw.bgra, "raw", "BGRX")
            offset_x = all_monitors["left"]
            offset_y = all_monitors["top"]
            sct.close()
        
        root = tk.Tk()
        root.title("Select Region - Click and drag to select, ESC to cancel")
        root.overrideredirect(True)
        root.geometry(f"{all_monitors['width']}x{all_monitors['height']}+{offset_x}+{offset_y}")
        root.attributes("-topmost", True)
        
        canvas = tk.Canvas(root, cursor="cross", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        
        img_tk = ImageTk.PhotoImage(screenshot)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        
        canvas.create_text(
            screenshot.width // 2, 30,
            text="Click and drag to select the capture region. Press ESC to cancel.",
            fill="yellow", font=("Arial", 16, "bold")
        )
        
        selection = {"start": None, "rect": None, "region": None}
        
        def on_press(event):
            selection["start"] = (event.x, event.y)
            if selection["rect"]:
                canvas.delete(selection["rect"])
            selection["rect"] = canvas.create_rectangle(
                event.x, event.y, event.x, event.y,
                outline="red", width=3
            )
        
        def on_drag(event):
            if selection["start"] and selection["rect"]:
                canvas.coords(
                    selection["rect"],
                    selection["start"][0], selection["start"][1],
                    event.x, event.y
                )
        
        def on_release(event):
            if selection["start"]:
                x1, y1 = selection["start"]
                x2, y2 = event.x, event.y
                
                x = min(x1, x2) + offset_x
                y = min(y1, y2) + offset_y
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                
                if w > 10 and h > 10:
                    selection["region"] = CaptureRegion(x, y, w, h)
                
                root.quit()
        
        def on_escape(event):
            root.quit()
        
        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        root.bind("<Escape>", on_escape)
        
        root.mainloop()
        root.destroy()
        
        return selection["region"]
