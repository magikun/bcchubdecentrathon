from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import os
from pathlib import Path
import pytesseract
from PIL import Image


@dataclass
class OcrResult:
    text: str
    raw: Dict[str, Any]
    engine: str


class TesseractEngine:
    def __init__(self, lang: str = "eng+rus", oem: int = 3, psm: int = 3, tesseract_cmd: Optional[str] = None):
        # Best-effort auto-configure Windows installation path and tessdata
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        else:
            # Common Windows path
            default_win = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
            if default_win.exists():
                pytesseract.pytesseract.tesseract_cmd = str(default_win)
                tessdata = default_win.parent / "tessdata"
                # Respect existing env var if already set
                os.environ.setdefault("TESSDATA_PREFIX", str(tessdata))

        self.lang = lang
        self.oem = oem
        self.psm = psm

    def run(self, image: Image.Image) -> OcrResult:
        # If Russian in requested lang, add configs to improve Cyrillic accuracy
        extra = ""
        if "rus" in (self.lang or ""):
            extra = " -c preserve_interword_spaces=1 -c tessedit_write_images=0"
        config = f"--oem {self.oem} --psm {self.psm}{extra}"
        try:
            text = pytesseract.image_to_string(image, lang=self.lang, config=config)
            data = pytesseract.image_to_data(image, lang=self.lang, config=config, output_type=pytesseract.Output.DICT)
        except Exception as e:
            # Surface a helpful message if rus.traineddata is missing
            tcmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "")
            tessdata_prefix = os.environ.get("TESSDATA_PREFIX")
            hint = {
                "tesseract_cmd": tcmd,
                "TESSDATA_PREFIX": tessdata_prefix,
                "expected_rus_traineddata": str(Path(tcmd).parent / "tessdata" / "rus.traineddata") if tcmd else None,
                "error": str(e),
            }
            return OcrResult(text="", raw={"error": hint}, engine="tesseract")
        return OcrResult(text=text, raw={"data": data, "config": config}, engine="tesseract")


