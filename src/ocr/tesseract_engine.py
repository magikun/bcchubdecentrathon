from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import pytesseract
from PIL import Image


@dataclass
class OcrResult:
    text: str
    raw: Dict[str, Any]
    engine: str


class TesseractEngine:
    def __init__(self, lang: str = "eng+rus", oem: int = 3, psm: int = 3, tesseract_cmd: Optional[str] = None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.lang = lang
        self.oem = oem
        self.psm = psm

    def run(self, image: Image.Image) -> OcrResult:
        config = f"--oem {self.oem} --psm {self.psm}"
        text = pytesseract.image_to_string(image, lang=self.lang, config=config)
        data = pytesseract.image_to_data(image, lang=self.lang, config=config, output_type=pytesseract.Output.DICT)
        return OcrResult(text=text, raw={"data": data, "config": config}, engine="tesseract")


