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
        
        # Optimized prompts for Russian banking documents
        self.russian_prompts = {
            "general": "<s_cord-v2>",
            "banking": "<s_cord-v2> Extract all text from this Russian banking document, preserving line breaks and structure.",
            "structured": "<s_cord-v2> Extract text maintaining original layout, columns, and line structure from this Russian document."
        }

    @torch.inference_mode()
    def run(self, image: Image.Image, prompt: Optional[str] = None, max_new_tokens: int = 512) -> OcrResult:
        # Use structured prompt for better layout preservation
        if prompt is None:
            prompt = self.russian_prompts["structured"]
        
        task_prompt = self.processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        decoder_input_ids = task_prompt.to(self.device)
        
        # Optimized generation parameters for speed and accuracy
        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            do_sample=False,   # Deterministic output
            num_beams=1,       # Faster generation (no beam search)
        )
        
        sequence = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Clean up the output - remove the prompt prefix
        if sequence.startswith(prompt):
            sequence = sequence[len(prompt):].strip()
        
        return OcrResult(text=sequence, raw={"ids": outputs.tolist(), "prompt": prompt}, engine="donut")


