import re
from typing import Optional, Tuple, List
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


class RouletteOCR:
    
    def __init__(self, use_gpu: bool = True):
        self.reader = None
        self.use_gpu = use_gpu
        self._initialized = False
    
    def _ensure_initialized(self):
        if not self._initialized:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=self.use_gpu, verbose=False)
            self._initialized = True
    
    def _is_green_dominant(self, image: Image.Image) -> bool:
        img = image.convert('RGB')
        pixels = np.array(img)
        
        r = pixels[:, :, 0].astype(float)
        g = pixels[:, :, 1].astype(float)
        b = pixels[:, :, 2].astype(float)
        
        green_mask = (g > r * 1.2) & (g > b * 1.2) & (g > 80)
        green_ratio = np.sum(green_mask) / green_mask.size
        
        return green_ratio > 0.15
    
    def _preprocess(self, image: Image.Image) -> Image.Image:
        img = image.convert('RGB')
        
        scale = 3
        new_size = (img.width * scale, img.height * scale)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)
        
        return img
    
    def read_number(self, image: Image.Image) -> Optional[int]:
        if self._is_green_dominant(image):
            return 0
        
        self._ensure_initialized()
        
        processed = self._preprocess(image)
        img_array = np.array(processed)
        
        results = self.reader.readtext(img_array, allowlist='0123456789')
        
        for (bbox, text, confidence) in results:
            number = self._parse_roulette_number(text)
            if number is not None:
                return number
        
        return None
    
    def read_all_numbers(self, image: Image.Image) -> List[Tuple[int, float]]:
        if self._is_green_dominant(image):
            return [(0, 1.0)]
        
        self._ensure_initialized()
        
        processed = self._preprocess(image)
        img_array = np.array(processed)
        
        results = self.reader.readtext(img_array, allowlist='0123456789')
        
        numbers = []
        for (bbox, text, confidence) in results:
            number = self._parse_roulette_number(text)
            if number is not None:
                x_pos = bbox[0][0]
                numbers.append((number, confidence, x_pos))
        
        numbers.sort(key=lambda x: x[2])
        
        return [(n, c) for n, c, _ in numbers]
    
    def read_first_number(self, image: Image.Image) -> Optional[Tuple[int, float]]:
        numbers = self.read_all_numbers(image)
        if numbers:
            return numbers[0]
        return None
    
    def _parse_roulette_number(self, text: str) -> Optional[int]:
        text = text.strip()
        text = re.sub(r'[^0-9]', '', text)
        
        if not text:
            return None
        
        try:
            number = int(text)
            if 0 <= number <= 36:
                return number
            if text.startswith('0') and len(text) > 1:
                number = int(text[1:])
                if 0 <= number <= 36:
                    return number
        except ValueError:
            pass
        
        return None
