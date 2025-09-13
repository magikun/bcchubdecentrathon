from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from paddleocr import PaddleOCR
import numpy as np
from PIL import Image


@dataclass
class OcrResult:
    text: str
    raw: Dict[str, Any]
    engine: str


class PaddleOCREngine:
    def __init__(self, lang: str = "en", det: bool = True, rec: bool = True, cls: bool = True):
        self.ocr = PaddleOCR(use_angle_cls=cls, lang=lang, show_log=False, det=det, rec=rec)

    def run(self, image: Image.Image) -> OcrResult:
        img = np.array(image.convert("RGB"))
        result = self.ocr.ocr(img, cls=True)
        lines = []
        for page in result:
            for line in page:
                lines.append(line[1][0])
        text = "\n".join(lines)
        return OcrResult(text=text, raw={"result": result}, engine="paddleocr")


