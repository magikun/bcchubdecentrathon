from __future__ import annotations

import json
import time
from io import BytesIO
from typing import Optional

import streamlit as st
from PIL import Image

from src.preprocess.image_ops import enhance_for_ocr, detect_noise_level
from src.llm.postprocess import llm_extract_json
from src.schema.bank_docs import BankingDoc
from src.metrics.text import compute_cer, compute_wer, normalized_levenshtein
from src.metrics.fields import field_level_scores, exact_match
from src.utils.logging_utils import append_jsonl
from pathlib import Path


st.set_page_config(page_title="Intelligent Banking OCR", layout="wide")
st.title("Intelligent OCR for Banking Documents")
st.caption("Donut / Layout-aware + PaddleOCR/TrOCR with LLM post-processing")


def get_engine(name: str):
    if name == "Tesseract":
        try:
            from src.ocr.tesseract_engine import TesseractEngine
            return TesseractEngine()
        except ImportError:
            st.error("Tesseract backend not available. Install 'pytesseract' and the Tesseract OCR binary.")
            st.stop()
    if name == "PaddleOCR":
        try:
            from src.ocr.paddle_engine import PaddleOCREngine
            return PaddleOCREngine()
        except ImportError:
            st.error("PaddleOCR backend not available. Install 'paddlepaddle' and 'paddleocr' for your platform.")
            st.stop()
    if name == "TrOCR":
        try:
            from src.ocr.trocr_engine import TrOCREngine
            return TrOCREngine()
        except ImportError:
            st.error("TrOCR backend not available. Ensure 'transformers' and 'torch' are installed.")
            st.stop()
    if name == "Donut":
        try:
            from src.ocr.donut_engine import DonutEngine
            return DonutEngine()
        except ImportError:
            st.error("Donut backend not available. Ensure 'transformers', 'timm', and 'torch' are installed.")
            st.stop()
    raise ValueError(name)


with st.sidebar:
    st.header("Settings")
    engine_name = st.selectbox("OCR Engine", ["PaddleOCR", "TrOCR", "Donut", "Tesseract"], index=0)
    do_llm = st.checkbox("LLM post-processing", value=True)


uploaded = st.file_uploader("Upload a document (image or PDF)", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "pdf"])

col1, col2, col3 = st.columns([1, 1, 1])

if uploaded is not None:
    raw = uploaded.read()
    if uploaded.type == "application/pdf" or uploaded.name.lower().endswith(".pdf"):
        try:
            from src.utils.pdf_loader import pdf_bytes_to_images
            pages = pdf_bytes_to_images(raw)
            img = pages[0]
            st.info("PDF detected. Using first page for demo.")
        except Exception as e:
            st.error(f"Failed to render PDF: {e}")
            st.stop()
    else:
        img = Image.open(BytesIO(raw))
    col1.subheader("Original")
    col1.image(img, use_column_width=True)

    noise = detect_noise_level(img)
    pre = enhance_for_ocr(img)
    col2.subheader("Preprocessed")
    col2.image(pre.image, use_column_width=True)
    col2.caption(f"Noise score: {noise:.3f}; scale: {pre.info.get('scale', 1.0)}")

    engine = get_engine(engine_name)
    t0 = time.time()
    ocr_res = engine.run(pre.image)
    t_engine = time.time() - t0

    col3.subheader("OCR Text")
    col3.code((ocr_res.text or "").strip()[:5000], language="text")
    col3.caption(f"Engine: {engine_name} | Time: {t_engine:.2f}s")

    json_data = None
    if do_llm:
        t1 = time.time()
        pp = llm_extract_json(ocr_res.text)
        t_llm = time.time() - t1
        json_data = pp.json_data
        col3.caption(f"LLM: {pp.model_name} | Time: {t_llm:.2f}s")

    st.divider()

    left, right = st.columns([1, 1])
    left.subheader("Structured JSON")
    if json_data is None:
        json_data = {"schema_version": "1.0", "ocr_text": ocr_res.text[:10000]}
    # Try validation
    is_valid = True
    try:
        validated = BankingDoc.model_validate(json_data)
        json_data = json.loads(validated.model_dump_json(exclude_none=True, indent=2))
    except Exception:
        is_valid = False
    left.json(json_data)
    left.caption("Valid JSON schema" if is_valid else "Invalid JSON schema")

    # Export buttons
    st.download_button("Export JSON", data=json.dumps(json_data, ensure_ascii=False, indent=2), file_name="extracted.json", mime="application/json")
    st.download_button("Export Text", data=(ocr_res.text or ""), file_name="ocr.txt", mime="text/plain")

    # Log run
    try:
        append_jsonl(Path("logs/runs.jsonl"), {
            "filename": uploaded.name,
            "engine": engine_name,
            "time_engine_s": round(t_engine, 3),
            "time_llm_s": round(t_llm, 3) if do_llm else None,
            "noise_score": noise,
            "json_valid": is_valid,
        })
    except Exception:
        pass

    # Placeholder for GT upload/compare in UI
    with st.expander("Compare with Ground Truth (optional)"):
        gt_file = st.file_uploader("Upload ground-truth JSON", type=["json"], key="gt")
        if gt_file is not None:
            try:
                gt_json = json.loads(gt_file.read().decode("utf-8"))
                st.json(gt_json)
                # Text metrics if GT provides ocr_text
                gt_text = gt_json.get("ocr_text", "") if isinstance(gt_json, dict) else ""
                if gt_text:
                    st.write("Text metrics vs. GT")
                    st.write({
                        "CER": compute_cer(gt_text, ocr_res.text),
                        "WER": compute_wer(gt_text, ocr_res.text),
                        "Norm Levenshtein": normalized_levenshtein(gt_text, ocr_res.text),
                    })

                # Field metrics
                st.write("Field-level metrics")
                st.json(field_level_scores(gt_json, json_data))
                st.write({"Exact Match": exact_match(gt_json, json_data)})

                # Optional compare with Tesseract baseline
                if st.checkbox("Compare with Tesseract baseline", value=False):
                    try:
                        from src.ocr.tesseract_engine import TesseractEngine as Tess
                        t0b = time.time()
                        base = Tess().run(pre.image)
                        tb = time.time() - t0b
                        st.caption(f"Tesseract time: {tb:.2f}s")
                        if gt_text:
                            st.write("Tesseract vs GT")
                            st.write({
                                "CER": compute_cer(gt_text, base.text),
                                "WER": compute_wer(gt_text, base.text),
                                "Norm Levenshtein": normalized_levenshtein(gt_text, base.text),
                            })
                    except ImportError:
                        st.warning("Tesseract backend not available in this environment.")
            except Exception as e:
                st.error(f"Failed to parse ground truth: {e}")


