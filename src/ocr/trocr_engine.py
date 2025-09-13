from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


@dataclass
class OcrResult:
    text: str
    raw: Dict[str, Any]
    engine: str


class TrOCREngine:
    def __init__(self, model_name: str = "microsoft/trocr-base-printed", device: Optional[str] = None):
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def run(self, image: Image.Image, max_new_tokens: int = 256) -> OcrResult:
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = self.model.generate(pixel_values, max_new_tokens=max_new_tokens)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return OcrResult(text=text, raw={"ids": generated_ids.tolist()}, engine="tr_ocr")


