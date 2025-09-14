from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Dict, Any

from PIL import Image

from src.ocr.tesseract_engine import TesseractEngine
from src.preprocess.image_ops import enhance_for_ocr, detect_noise_level
from src.metrics.text import compute_cer, compute_wer, normalized_levenshtein
from src.metrics.fields import field_level_scores, exact_match
from src.llm.postprocess import llm_extract_json
from src.schema.bank_docs import BankingDoc


def load_ground_truth(gt_path: Path) -> Dict[str, Any]:
    if not gt_path.exists():
        return {}
    with gt_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_engine(name: str):
    if name == "tesseract":
        return TesseractEngine()
    if name == "paddleocr":
        # Import lazily to avoid hard failure if paddleocr is missing
        try:
            from src.ocr.paddle_engine import PaddleOCREngine
            return PaddleOCREngine()
        except Exception:
            return TesseractEngine()
    if name == "trocr":
        try:
            from src.ocr.trocr_engine import TrOCREngine
            return TrOCREngine()
        except Exception:
            return TesseractEngine()
    if name == "donut":
        try:
            from src.ocr.donut_engine import DonutEngine
            return DonutEngine()
        except Exception:
            return TesseractEngine()
    raise ValueError(f"Unknown engine {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, required=True, help="Folder with images")
    parser.add_argument("--gts", type=str, required=False, help="Folder with ground-truth JSONs")
    parser.add_argument("--engine", type=str, default="paddleocr", choices=["tesseract", "paddleocr", "trocr", "donut"])
    parser.add_argument("--compare_tesseract", action="store_true")
    parser.add_argument("--out", type=str, default="eval_results.json")
    args = parser.parse_args()

    images_dir = Path(args.images)
    gts_dir = Path(args.gts) if args.gts else None
    out_path = Path(args.out)

    engine = get_engine(args.engine)
    tess_engine = get_engine("tesseract") if args.compare_tesseract else None

    rows = []
    for img_path in sorted(images_dir.glob("*")):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
            continue

        gt = load_ground_truth(gts_dir / (img_path.stem + ".json")) if gts_dir else {}
        image = Image.open(img_path)
        noise = detect_noise_level(image)
        pre = enhance_for_ocr(image)

        # Candidate engine
        t0 = time.time()
        res = engine.run(pre.image)
        text = res.text
        t_engine = time.time() - t0

        # LLM extraction
        t1 = time.time()
        pp = llm_extract_json(text)
        t_llm = time.time() - t1

        pred_json = pp.json_data

        # JSON validity and key presence
        try:
            BankingDoc.model_validate(pred_json)
            json_valid = True
        except Exception:
            json_valid = False
        top_keys = ["schema_version", "ocr_engine", "ocr_text", "check", "statement", "contract"]
        present_top_keys = [k for k in top_keys if isinstance(pred_json, dict) and k in pred_json]

        # Metrics vs GT text if provided
        gt_text = gt.get("ocr_text", "") if isinstance(gt, dict) else ""
        m = {
            "cer": compute_cer(gt_text, text) if gt_text else None,
            "wer": compute_wer(gt_text, text) if gt_text else None,
            "norm_lev": normalized_levenshtein(gt_text, text) if gt_text else None,
        }

        # Field metrics if GT JSON provided
        field_scores = field_level_scores(gt, pred_json) if gt else {}
        exact = exact_match(gt, pred_json) if gt else None

        row = {
            "file": img_path.name,
            "engine": args.engine,
            "time_engine_s": t_engine,
            "time_llm_s": t_llm,
            "noise_score": noise,
            "metrics": m,
            "exact_match": exact,
            "field_scores": field_scores,
            "json_valid": json_valid,
            "present_top_keys": present_top_keys,
        }

        # Compare with tesseract baseline (optional)
        if tess_engine is not None:
            try:
                t0b = time.time()
                base = tess_engine.run(pre.image)
                base_text = base.text
                t_base = time.time() - t0b
                row["tesseract"] = {
                    "time_engine_s": t_base,
                    "cer": compute_cer(gt_text, base_text) if gt_text else None,
                    "wer": compute_wer(gt_text, base_text) if gt_text else None,
                    "norm_lev": normalized_levenshtein(gt_text, base_text) if gt_text else None,
                }
            except Exception as e:
                row["tesseract_error"] = str(e)

        rows.append(row)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()


