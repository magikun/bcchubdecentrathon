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
    def __init__(self, lang: str = "ru", det: bool = True, rec: bool = True, cls: bool = True):
        """Initialize PaddleOCR for Russian-only documents with speed optimizations."""
        l = "ru"
        try:
            # Optimized for speed: disable angle classification, use ONNX if available
            self.ocr = PaddleOCR(
                use_angle_cls=False,  # Disable angle classification for speed
                lang=l, 
                det=det, 
                rec=rec, 
                show_log=False,
                use_onnx=True,  # Use ONNX for faster inference
                use_gpu=False   # Disable GPU for stability
            )
        except Exception:
            try:
                # Fallback without ONNX
                self.ocr = PaddleOCR(
                    use_angle_cls=False,
                    lang=l, 
                    det=det, 
                    rec=rec, 
                    show_log=False,
                    use_gpu=False
                )
            except Exception:
                # Last resort
                self.ocr = PaddleOCR(lang=l, use_angle_cls=False)

    def run(self, image: Image.Image) -> OcrResult:
        img_pil = image.convert("RGB")
        img = np.array(img_pil)
        W, H = img_pil.size
        try:
            # Optimized OCR call: disable angle classification for speed
            result = self.ocr.ocr(img, cls=False)
        except Exception as e:
            return OcrResult(text="", raw={"error": str(e), "result": []}, engine="paddleocr")

        # Collect boxes (x_min, y_min, x_max, y_max), text
        all_lines = []
        pages = result if isinstance(result, list) else [result]
        for page in (pages or []):
            if not page:
                continue

            boxes = []
            heights = []
            widths = []
            for item in page:
                try:
                    coords = item[0]  # 4 points
                    text = str(item[1][0])
                    xs = [p[0] for p in coords]
                    ys = [p[1] for p in coords]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    h = max(1.0, y_max - y_min)
                    w = max(1.0, x_max - x_min)
                    heights.append(h)
                    widths.append(w)
                    boxes.append((x_min, y_min, x_max, y_max, text))
                except Exception:
                    continue

            if not boxes:
                continue

            # Dynamic thresholds
            median_h = float(np.median(heights)) if heights else 20.0
            median_w = float(np.median(widths)) if widths else 40.0
            same_line_thresh = max(8.0, 0.6 * median_h)
            column_gap_thresh = max(W * 0.12, 2.5 * median_w)
            word_gap_thresh = max(2.0, 0.3 * median_w)  # Small gaps between words

            # Sort by x center to detect columns
            boxes_with_centers = []
            for (x0, y0, x1, y1, t) in boxes:
                x_center = (x0 + x1) / 2.0
                boxes_with_centers.append((x_center, x0, y0, x1, y1, t))
            boxes_with_centers.sort(key=lambda x: x[0])

            # Split into columns based on large x gaps
            columns: list[list[tuple]] = []
            current: list[tuple] = []
            last_right = None
            for (_, x0, y0, x1, y1, t) in boxes_with_centers:
                if last_right is None or (x0 - last_right) <= column_gap_thresh:
                    current.append((x0, y0, x1, y1, t))
                    last_right = max(x1, last_right or x1)
                else:
                    columns.append(current)
                    current = [(x0, y0, x1, y1, t)]
                    last_right = x1
            if current:
                columns.append(current)

                # Within each column, group by y into lines, then sort by x
                column_text_lines: list[str] = []
                for col in columns:
                    # sort by y then x
                    col.sort(key=lambda b: (b[1], b[0]))
                    current_line: list[tuple] = []
                    last_y = None
                    for (x0, y0, x1, y1, t) in col:
                        y_center = (y0 + y1) / 2.0
                        if last_y is None or abs(y_center - last_y) <= same_line_thresh:
                            current_line.append((x0, x1, t))  # Store x0, x1, text
                            last_y = y_center if last_y is None else (last_y + y_center) / 2.0
                        else:
                            # flush previous line
                            if current_line:
                                current_line.sort(key=lambda it: it[0])
                                # Join words with proper spacing
                                words = [word for (_, _, word) in current_line]
                                column_text_lines.append(" ".join(words))
                            current_line = [(x0, x1, t)]
                            last_y = y_center
                    if current_line:
                        current_line.sort(key=lambda it: it[0])
                        # Join words with proper spacing
                        words = [word for (_, _, word) in current_line]
                        column_text_lines.append(" ".join(words))

            # Reading order: left-to-right columns, each top-to-bottom lines
            # Handle hyphenation: merge lines ending with "-" and starting with lowercase Cyrillic
            processed_lines = []
            for i, line in enumerate(column_text_lines):
                if i > 0 and processed_lines and processed_lines[-1].endswith("-"):
                    # Check if current line starts with lowercase Cyrillic
                    if line and any(ord(c) >= 0x0430 and ord(c) <= 0x044F for c in line[:1]):
                        # Merge with previous line (remove the "-")
                        processed_lines[-1] = processed_lines[-1][:-1] + line
                        continue
                processed_lines.append(line)
            all_lines.extend(processed_lines)

        # Join lines with spaces for natural text flow
        text = " ".join(all_lines)
        
        # Clean up multiple spaces and normalize formatting
        import re
        text = re.sub(r'\s+', ' ', text).strip()
        return OcrResult(text=text, raw={"result": result}, engine="paddleocr")


