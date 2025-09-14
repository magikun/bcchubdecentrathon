from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image

# Optional OpenCV import with fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False


@dataclass
class PreprocessResult:
    image: Image.Image
    info: Dict[str, Any]


def pil_to_cv(image: Image.Image) -> np.ndarray:
    if not CV2_AVAILABLE:
        # Fallback: return RGB array without BGR conversion
        return np.array(image.convert("RGB"))
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv_to_pil(mat: np.ndarray) -> Image.Image:
    if not CV2_AVAILABLE:
        # Fallback: assume RGB format
        return Image.fromarray(mat)
    return Image.fromarray(cv2.cvtColor(mat, cv2.COLOR_BGR2RGB))


def detect_noise_level(image: Image.Image) -> float:
    if not CV2_AVAILABLE:
        # Fallback: simple variance-based noise estimation
        gray = np.array(image.convert("L"))
        variance = np.var(gray)
        score = max(0.0, min(1.0, 1.0 - (variance / 500.0)))
        return float(score)
    
    mat = pil_to_cv(image)
    gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    variance = lap.var()
    # Lower variance -> blur; return noise score inversely
    score = max(0.0, min(1.0, 1.0 - (variance / 500.0)))
    return float(score)


def enhance_for_ocr(image: Image.Image) -> PreprocessResult:
    if not CV2_AVAILABLE:
        # Fallback: simple PIL-based enhancement
        orig_w, orig_h = image.size
        scale = 2.0 if max(orig_h, orig_w) < 1200 else 1.0
        
        # Simple enhancement: convert to grayscale, enhance contrast
        enhanced = image.convert("L").convert("RGB")
        
        # Upscale if small
        if scale != 1.0:
            new_size = (int(orig_w * scale), int(orig_h * scale))
            enhanced = enhanced.resize(new_size, Image.Resampling.LANCZOS)
        
        return PreprocessResult(image=enhanced, info={"scale": scale, "fallback": True})
    
    mat = pil_to_cv(image)
    orig_h, orig_w = mat.shape[:2]

    # Denoise
    denoised = cv2.fastNlMeansDenoisingColored(mat, None, 10, 10, 7, 21)

    # Grayscale and adaptive threshold
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 15)

    # Morphological open to remove specks
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

    # Upscale if small
    scale = 2.0 if max(orig_h, orig_w) < 1200 else 1.0
    if scale != 1.0:
        opened = cv2.resize(opened, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Convert back to RGB
    res_rgb = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)
    out = cv_to_pil(res_rgb)
    return PreprocessResult(image=out, info={"scale": scale})


def prepare_for_transformer(image: Image.Image) -> Image.Image:
    """Prepare an image for transformer OCR models.

    - Keep RGB, avoid hard binarization
    - Light denoise
    - Apply CLAHE on luminance for contrast
    - Upscale small images to ~1280px on the long side for better legibility
    """
    if not CV2_AVAILABLE:
        # Fallback: simple PIL-based preparation
        w, h = image.size
        long_side = max(h, w)
        target = 1280
        if long_side < target:
            scale = target / long_side
            new_size = (int(w * scale), int(h * scale))
            return image.resize(new_size, Image.Resampling.LANCZOS)
        return image
    
    mat = pil_to_cv(image)
    den = cv2.fastNlMeansDenoisingColored(mat, None, 3, 3, 7, 21)
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    rgb = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    h, w = rgb.shape[:2]
    long_side = max(h, w)
    target = 1280
    if long_side < target:
        scale = target / long_side
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    return cv_to_pil(rgb)


# Stronger variant for very noisy scans
def enhance_for_ocr_strong(image: Image.Image) -> PreprocessResult:
    if not CV2_AVAILABLE:
        # Fallback: simple PIL-based enhancement with more aggressive scaling
        orig_w, orig_h = image.size
        scale = 3.0 if max(orig_h, orig_w) < 1000 else (2.0 if max(orig_h, orig_w) < 1500 else 1.5)
        
        # Simple enhancement: convert to grayscale, enhance contrast
        enhanced = image.convert("L").convert("RGB")
        
        # Upscale more aggressively if small
        if scale != 1.0:
            new_size = (int(orig_w * scale), int(orig_h * scale))
            enhanced = enhanced.resize(new_size, Image.Resampling.LANCZOS)
        
        return PreprocessResult(image=enhanced, info={"scale": scale, "strong": True, "fallback": True})
    
    mat = pil_to_cv(image)
    orig_h, orig_w = mat.shape[:2]

    # Strong denoise and sharpening
    denoised = cv2.fastNlMeansDenoisingColored(mat, None, 15, 15, 7, 21)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold with smaller block size to better separate text
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 25, 10)

    # Morphology: open then close to remove specks and bridge gaps
    kernel_open = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel_open, iterations=1)
    kernel_close = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # Upscale more aggressively if small
    scale = 3.0 if max(orig_h, orig_w) < 1000 else (2.0 if max(orig_h, orig_w) < 1500 else 1.5)
    if scale != 1.0:
        closed = cv2.resize(closed, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    res_rgb = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    out = cv_to_pil(res_rgb)
    return PreprocessResult(image=out, info={"scale": scale, "strong": True})

