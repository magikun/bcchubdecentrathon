from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel


@dataclass
class OcrResult:
    text: str
    raw: Dict[str, Any]
    engine: str


class DonutEngine:
    def __init__(self, model_name: str = "naver-clova-ix/donut-base", device: Optional[str] = None):
        self.processor = DonutProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def run(self, image: Image.Image, prompt: Optional[str] = None, max_new_tokens: int = 512) -> OcrResult:
        if prompt is None:
            prompt = "<s_cord-v2>"
        task_prompt = self.processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        decoder_input_ids = task_prompt.to(self.device)
        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        sequence = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return OcrResult(text=sequence, raw={"ids": outputs.tolist()}, engine="donut")


