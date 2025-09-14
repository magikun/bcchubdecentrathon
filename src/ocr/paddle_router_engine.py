from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from src.utils.text_normalize import detect_cyrillic_ratio, normalize_homoglyphs_to_cyrillic
from src.ocr.paddle_safe import make_detector, make_recognizer


@dataclass
class OcrResult:
    text: str
    raw: Dict[str, Any]
    engine: str


class PaddleOCRRouterEngine:
    """Route each detected line to RU or EN recognizer to avoid cross-alphabet corruption."""

    def __init__(self, cyr_threshold: float = 0.3, arbitration: str = "heuristic", batch_size: int = 8):
        # Build via safe factory to avoid partial init under Streamlit cache
        self.det = make_detector()
        self.rec_ru = make_recognizer("ru")
        self.rec_en = make_recognizer("en")
        self.cyr_threshold = float(cyr_threshold)
        self.arbitration = arbitration
        self.batch_size = int(batch_size)

    def _pil_to_np(self, image: Image.Image) -> np.ndarray:
        return np.array(image.convert("RGB"))

    def _crop_from_box(self, img: np.ndarray, box: List[List[float]]) -> np.ndarray:
        xs = [int(p[0]) for p in box]
        ys = [int(p[1]) for p in box]
        x0, x1 = max(min(xs), 0), min(max(xs), img.shape[1])
        y0, y1 = max(min(ys), 0), min(max(ys), img.shape[0])
        if y1 <= y0 or x1 <= x0:
            return img[0:1, 0:1, :]
        return img[y0:y1, x0:x1, :]

    def _route_lang_heuristic(self, crop: np.ndarray) -> str:
        # Downscale for speed
        h, w = crop.shape[:2]
        scale = 256 / max(h, w) if max(h, w) > 256 else 1.0
        if scale < 1.0:
            crop = np.ascontiguousarray(Image.fromarray(crop).resize((int(w*scale), int(h*scale))).convert("RGB"))
        # Use RU recognizer quickly to get text preview
        preview = self.rec_ru.ocr(crop, cls=True)
        lines = []
        for page in (preview if isinstance(preview, list) else [preview]):
            if page:
                for line in page:
                    try:
                        lines.append(str(line[1][0]))
                    except Exception:
                        pass
        text = " ".join(lines)
        ratio = detect_cyrillic_ratio(text)
        return "ru" if ratio >= self.cyr_threshold else ("en" if ratio <= (1 - self.cyr_threshold) else "ambiguous")

    def _recognize_batch(self, crops: List[np.ndarray], lang: str) -> List[Tuple[str, float]]:
        if not crops:
            return []
        recog = self.rec_ru if lang == "ru" else self.rec_en
        out: List[Tuple[str, float]] = []
        for i in range(0, len(crops), self.batch_size):
            batch = crops[i:i+self.batch_size]
            res = recog.ocr(batch, cls=True)
            pages = res if isinstance(res, list) else [res]
            for page in pages:
                if not page:
                    out.append(("", 0.0))
                    continue
                # Paddle returns per image results when input is list
                # Grab concatenated text and average confidence
                texts, probs = [], []
                for line in page:
                    try:
                        texts.append(str(line[1][0]))
                        probs.append(float(line[1][1]))
                    except Exception:
                        pass
                avgp = float(sum(probs)/len(probs)) if probs else 0.0
                out.append((" ".join(texts), avgp))
        return out

    def run(self, image: Image.Image) -> OcrResult:
        img = self._pil_to_np(image)
        det = self.det.ocr(img, cls=True)
        lines: List[Dict[str, Any]] = []
        boxes: List[List[List[float]]] = []
        for page in (det if isinstance(det, list) else [det]):
            for line in (page or []):
                try:
                    boxes.append(line[0])
                except Exception:
                    pass

        crops = [self._crop_from_box(img, b) for b in boxes]
        routes = [self._route_lang_heuristic(c) for c in crops]

        # First pass: heuristic RU/EN
        ru_idx = [i for i, r in enumerate(routes) if r == "ru"]
        en_idx = [i for i, r in enumerate(routes) if r == "en"]
        amb_idx = [i for i, r in enumerate(routes) if r == "ambiguous"]

        ru_texts = self._recognize_batch([crops[i] for i in ru_idx], "ru")
        en_texts = self._recognize_batch([crops[i] for i in en_idx], "en")

        # Ambiguous: run both and pick higher confidence
        amb_ru = self._recognize_batch([crops[i] for i in amb_idx], "ru")
        amb_en = self._recognize_batch([crops[i] for i in amb_idx], "en")

        texts: List[str] = [""] * len(crops)
        langs: List[str] = ["?"] * len(crops)

        for slot, (t, p) in zip(ru_idx, ru_texts):
            langs[slot] = "ru"
            texts[slot] = normalize_homoglyphs_to_cyrillic(t)
        for slot, (t, p) in zip(en_idx, en_texts):
            langs[slot] = "en"
            texts[slot] = t
        for k, slot in enumerate(amb_idx):
            t_ru, p_ru = amb_ru[k] if k < len(amb_ru) else ("", 0.0)
            t_en, p_en = amb_en[k] if k < len(amb_en) else ("", 0.0)
            if p_ru >= p_en:
                langs[slot] = "ru"
                texts[slot] = normalize_homoglyphs_to_cyrillic(t_ru)
            else:
                langs[slot] = "en"
                texts[slot] = t_en

        merged_text = "\n".join([t.strip() for t in texts if t])
        raw = {
            "boxes": boxes,
            "langs": langs,
            "counts": {"ru": langs.count("ru"), "en": langs.count("en"), "amb": len(amb_idx)},
        }
        return OcrResult(text=merged_text, raw=raw, engine="paddleocr_router")


