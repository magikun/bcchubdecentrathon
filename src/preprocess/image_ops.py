from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image


@dataclass
class PreprocessResult:
    image: Image.Image
    info: Dict[str, Any]


def pil_to_cv(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv_to_pil(mat: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(mat, cv2.COLOR_BGR2RGB))


def detect_noise_level(image: Image.Image) -> float:
    mat = pil_to_cv(image)
    gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    variance = lap.var()
    # Lower variance -> blur; return noise score inversely
    score = max(0.0, min(1.0, 1.0 - (variance / 500.0)))
    return float(score)


def enhance_for_ocr(image: Image.Image) -> PreprocessResult:
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


