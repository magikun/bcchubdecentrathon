from __future__ import annotations

import json
import sys
import time
from io import BytesIO
from typing import Optional

import streamlit as st
from PIL import Image

# Set page config FIRST - must be the first Streamlit command
st.set_page_config(page_title="Intelligent Banking OCR", layout="wide")

# Check Python version and show compatibility info
python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
if sys.version_info >= (3, 13):
    st.info(f"🐍 **Python {python_version} detected** - PaddleOCR may not be available. Tesseract and TrOCR engines will work fine.")
else:
    st.success(f"🐍 **Python {python_version} detected** - All OCR engines including PaddleOCR are available!")

from src.preprocess.image_ops import enhance_for_ocr, detect_noise_level, prepare_for_transformer, enhance_for_ocr_strong
from src.llm.postprocess import (
    llm_extract_json,
    llm_extract_json_iterative,
    ai_normalize_pseudocyrillic,
    ai_normalize_pseudocyrillic_bulk,
)
from src.schema.bank_docs import BankingDoc
from src.metrics.text import compute_cer, compute_wer, normalized_levenshtein
from src.metrics.fields import field_level_scores, exact_match
from src.utils.logging_utils import append_jsonl
from pathlib import Path
from src.utils.pdf_loader import pdf_bytes_to_images, pdf_to_images
from src.utils.text_normalize import detect_cyrillic_ratio, normalize_homoglyphs_to_cyrillic
from src.utils.xlsx_export import export_standard_xlsx

st.title("Intelligent OCR for Banking Documents")
st.caption("Donut / Layout-aware + PaddleOCR/TrOCR with LLM post-processing")


@st.cache_resource(show_spinner=False)
def get_engine(name: str, lang_choice: str):
    if name == "Tesseract":
        try:
            from src.ocr.tesseract_engine import TesseractEngine
            # Force Russian only
            lang = "rus"
            return TesseractEngine(lang=lang)
        except ImportError:
            st.error("Tesseract backend not available. Install 'pytesseract' and the Tesseract OCR binary.")
            st.stop()
    if name == "PaddleOCR":
        try:
            from src.ocr.paddle_engine import PaddleOCREngine
            # Force Russian-only model per your request
            lang = "ru"
            return PaddleOCREngine(lang=lang)
        except ImportError:
            st.error("PaddleOCR backend not available. Install 'paddlepaddle' and 'paddleocr' for your platform.")
            st.info("💡 **Note**: PaddlePaddle may not support Python 3.13+. Use Tesseract or TrOCR engines instead.")
            st.stop()
        except Exception as e:
            # Show the actual PaddleOCR error instead of silently falling back
            st.error(f"PaddleOCR initialization failed: {e}")
            st.error("Please check PaddleOCR installation. Falling back to Tesseract.")
            try:
                from src.ocr.tesseract_engine import TesseractEngine
                lang = "rus"  # Force Russian only
                return TesseractEngine(lang=lang)
            except Exception:
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
            # Use CPU for faster startup and better compatibility
            return DonutEngine(device="cpu")
        except ImportError:
            st.error("Donut backend not available. Ensure 'transformers', 'timm', and 'torch' are installed.")
            st.stop()
        except Exception as e:
            st.error(f"Donut initialization failed: {e}")
            st.error("Falling back to Tesseract.")
            try:
                from src.ocr.tesseract_engine import TesseractEngine
                return TesseractEngine(lang="rus")
            except Exception:
                st.stop()
    raise ValueError(name)


@st.cache_resource(show_spinner=False)
def get_engine_resolved(name: str, resolved_lang: str):
    """Return an engine instance pinned to a resolved language.

    resolved_lang: 'ru' or 'en' (others ignored)
    """
    if name == "Tesseract":
        try:
            from src.ocr.tesseract_engine import TesseractEngine
            lang = "rus"
            return TesseractEngine(lang=lang)
        except Exception:
            st.stop()
    if name == "Donut":
        try:
            from src.ocr.donut_engine import DonutEngine
            # Use CPU for faster startup and better compatibility
            return DonutEngine(device="cpu")
        except Exception:
            st.stop()
    if name == "PaddleOCR":
        try:
            from src.ocr.paddle_engine import PaddleOCREngine
            # Always Russian-only engine in resolved path
            lang = "ru"
            return PaddleOCREngine(lang=lang)
        except Exception:
            # silent fallback to Tesseract specific lang
            from src.ocr.tesseract_engine import TesseractEngine
            lang = "rus"
            return TesseractEngine(lang=lang)
    if name == "TrOCR":
        from src.ocr.trocr_engine import TrOCREngine
        return TrOCREngine()
    if name == "Donut":
        from src.ocr.donut_engine import DonutEngine
        return DonutEngine()
    raise ValueError(name)


def resolve_language_setting(lang_choice: str, preview_image: Image.Image) -> str:
    """Resolve 'ru' or 'en' for this document.

    If lang_choice is Auto, run a fast Tesseract (eng+rus) pass on the preview image
    and compute Cyrillic ratio. Threshold 0.3 -> 'ru', else 'en'.
    """
    if lang_choice.startswith("Auto"):
        try:
            from src.ocr.tesseract_engine import TesseractEngine as Tess
            # quick sample on preview image
            sample = Tess(lang="eng+rus", psm=6).run(preview_image).text
            return "ru" if detect_cyrillic_ratio(sample) >= 0.3 else "en"
        except Exception:
            return "ru"  # default bias
    if lang_choice == "Russian":
        return "ru"
    return "en"


with st.sidebar:
    st.header("Settings")
    engine_name = st.selectbox("OCR Engine", ["PaddleOCR", "Donut", "TrOCR", "Tesseract"], index=0)
    if engine_name == "Donut":
        st.info("Donut: Better layout preservation but slower. Good for complex documents.")
    elif engine_name == "PaddleOCR":
        st.info("PaddleOCR: Fast Russian OCR with good accuracy.")
    # Language is fixed to Russian only
    lang_choice = "Russian"
    do_llm = st.checkbox("LLM post-processing", value=True)
    do_ai_norm = st.checkbox("AI normalize Cyrillic (fix mixed alphabet)", value=True)
    fast_mode = st.checkbox("Fast mode (larger paragraphs only)", value=False)
    st.divider()
    batch_mode = st.checkbox("Batch processing (folder or multi-file)", value=False)
    folder_path = ""
    if batch_mode:
        folder_path = st.text_input("Folder path (optional)", value="")
    st.divider()
    time_budget_s = st.slider("Per-file time budget (s)", min_value=2, max_value=30, value=5, step=1)
    max_pages = st.number_input("Max pages per file", min_value=1, max_value=200, value=20, step=1)


uploads = st.file_uploader("Upload document(s) (image or PDF)", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "pdf"], accept_multiple_files=True)

col1, col2 = st.columns([1, 1])

if uploads or ("batch_mode" in locals() and batch_mode and folder_path):
    uploaded_files = uploads if isinstance(uploads, list) else ([] if uploads is None else [uploads])
    if not ("batch_mode" in locals() and batch_mode) and len(uploaded_files) > 1:
        st.warning("Multiple files uploaded. Enable 'Batch processing' to process them all. Showing the first only.")
        uploaded_files = uploaded_files[:1]

    # Build documents with all pages
    documents = []
    # From uploader
    for f in uploaded_files:
        try:
            raw = f.read()
            if (getattr(f, "type", "") == "application/pdf") or f.name.lower().endswith(".pdf"):
                pages = pdf_bytes_to_images(raw)
                if pages:
                    documents.append({"name": f.name, "pages": pages})
            else:
                documents.append({"name": f.name, "pages": [Image.open(BytesIO(raw))]})
        except Exception as e:
            st.warning(f"Skipping {getattr(f, 'name', 'file')}: {e}")

    # From folder path
    if ("batch_mode" in locals() and batch_mode) and folder_path:
        try:
            p = Path(folder_path)
            if p.exists() and p.is_dir():
                for ext in ["*.pdf", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp", "*.webp"]:
                    for fp in p.glob(ext):
                        try:
                            if fp.suffix.lower() == ".pdf":
                                pages = pdf_to_images(fp)
                                if pages:
                                    documents.append({"name": fp.name, "pages": pages})
                            else:
                                documents.append({"name": fp.name, "pages": [Image.open(fp)]})
                        except Exception as e:
                            st.warning(f"Skipping {fp.name}: {e}")
            else:
                st.warning("Folder path does not exist or is not a directory.")
        except Exception as e:
            st.warning(f"Folder scan failed: {e}")

    if not documents:
        st.stop()

    def process_document(pages, engine_label: str, resolved_lang: Optional[str] = None):
        if resolved_lang and engine_label in {"Tesseract", "PaddleOCR"}:
            engine = get_engine_resolved(engine_label, resolved_lang)
        else:
            engine = get_engine(engine_label, lang_choice)
        combined = []
        total_time = 0.0
        noises = []
        pages_done = 0
        paddle_lines_all = []
        for page in pages:
            if pages_done >= max_pages or total_time >= float(time_budget_s):
                break
            noise = detect_noise_level(page)
            noises.append(noise)
            pre = enhance_for_ocr(page)
            if noise > 0.6:
                pre = enhance_for_ocr_strong(page)
            # PaddleOCR prefers RGB (not binarized); use transformer prep
            use_raw = engine_label in {"TrOCR", "Donut", "PaddleOCR"}
            engine_input = (prepare_for_transformer(page) if use_raw else pre.image)
            t0 = time.time()
            res = engine.run(engine_input)
            t_engine = time.time() - t0
            total_time += t_engine
            text_here = (res.text or "").strip()
            # Collect Paddle per-line texts if available for later per-line normalization
            try:
                raw = getattr(res, "raw", None)
                if isinstance(raw, dict) and "result" in raw:
                    rp = raw.get("result")
                    for pg in (rp if isinstance(rp, list) else [rp]):
                        if pg:
                            for ln in pg:
                                try:
                                    paddle_lines_all.append(str(ln[1][0]))
                                except Exception:
                                    pass
            except Exception:
                pass
            if not text_here and engine_label != "Tesseract":
                try:
                    from src.ocr.tesseract_engine import TesseractEngine as Tess
                    t0f = time.time()
                    res = Tess().run(pre.image)
                    t_engine = time.time() - t0f
                    total_time += t_engine
                    text_here = (res.text or "").strip()
                except Exception:
                    pass
            combined.append(text_here)
            pages_done += 1
        avg_noise = (sum(noises) / len(noises)) if noises else 0.0
        return "\n\f\n".join(combined), total_time, avg_noise, pages_done, paddle_lines_all

    # Single-file interactive view
    if not ("batch_mode" in locals() and batch_mode) or len(documents) == 1:
        doc = documents[0]
        name = doc["name"] if documents else "uploaded"
        img = doc["pages"][0]
        col1.subheader("Original")
        col1.image(img, use_column_width=True)

        noise = detect_noise_level(img)
        pre_preview = enhance_for_ocr(img)
        if noise > 0.6:
            pre_preview = enhance_for_ocr_strong(img)
        col2.subheader("Preprocessed")
        col2.image(pre_preview.image, use_column_width=True)
        col2.caption(f"Noise score: {noise:.3f}; scale: {pre_preview.info.get('scale', 1.0)}")

        # Resolve language per document based on preview
        resolved = resolve_language_setting(lang_choice, img)
        engine_label = engine_name
        ocr_text, t_engine, avg_noise, pages_done, paddle_lines = process_document(doc["pages"], engine_label, resolved_lang=resolved)
        # LLM-based normalization (preferred): convert pseudo-Cyrillic to correct Russian
        ai_used = None
        t_ai = None
        if do_ai_norm and ocr_text.strip():
            st.info("🔧 AI Normalization: ON")
            t2 = time.time()
            try:
                # Normalize only Russian/mixed segments; keep Latin-only intact
                lines_source = paddle_lines if paddle_lines else ocr_text.splitlines()
                def is_latin_only(s: str) -> bool:
                    has_cyr = detect_cyrillic_ratio(s) > 0.0
                    has_lat = any(ch.isalpha() and ('A' <= ch <= 'Z' or 'a' <= ch <= 'z') for ch in s)
                    return has_lat and not has_cyr

                # Build contiguous Cyrillic/mixed blocks to minimize tokens
                blocks = []  # list of tuples: (start_idx, end_idx_exclusive, text_or_line, needs_ai)
                i = 0
                n = len(lines_source)
                while i < n:
                    ln = lines_source[i]
                    if not ln or is_latin_only(ln) or any(tag in ln for tag in ["IBAN", "SWIFT", "http://", "https://", "E-mail", "Email"]):
                        blocks.append((i, i+1, ln, False))
                        i += 1
                        continue
                    # Use higher threshold in fast mode to reduce AI calls
                    cyrillic_threshold = 0.4 if fast_mode else 0.2
                    if detect_cyrillic_ratio(ln) >= cyrillic_threshold:
                        j = i
                        buf = []
                        # Grow paragraph while lines look Cyrillic/mixed
                        while j < n:
                            lnj = lines_source[j]
                            if lnj and not is_latin_only(lnj) and detect_cyrillic_ratio(lnj) >= cyrillic_threshold and not any(tag in lnj for tag in ["IBAN", "SWIFT", "http://", "https://", "E-mail", "Email"]):
                                buf.append(lnj)
                                j += 1
                            else:
                                break
                        blocks.append((i, j, "\n".join(buf), True))
                        i = j
                    else:
                        blocks.append((i, i+1, ln, False))
                        i += 1

                # Collect texts for AI and normalize in bulk
                to_norm = [b[2] for b in blocks if b[3]]
                if to_norm:
                    normed_blocks, ai_used = ai_normalize_pseudocyrillic_bulk(to_norm, fast_mode=fast_mode)
                else:
                    normed_blocks, ai_used = [], None

                # Rebuild lines
                out_lines = []
                it = iter(normed_blocks)
                for (s, e, content, needs_ai) in blocks:
                    if needs_ai:
                        norm_text = next(it)
                        out_lines.extend(norm_text.splitlines())
                    else:
                        out_lines.extend(content.splitlines())
                ocr_text = "\n".join(out_lines)
            except Exception:
                pass
            t_ai = time.time() - t2
        else:
            st.info("🔧 AI Normalization: OFF")
            # Fallback: light homoglyph normalization only if clearly Cyrillic-heavy
            if detect_cyrillic_ratio(ocr_text) >= 0.5:
                ocr_text = normalize_homoglyphs_to_cyrillic(ocr_text)
        engine_used = engine_name if ocr_text.strip() else ("Tesseract (fallback)" if engine_name != "Tesseract" else "Tesseract")

        # OCR Text column removed - text is only available in JSON and exports
        if ai_used:
            extra = f" | AI normalize: {ai_used} {t_ai:.2f}s"
        else:
            extra = " | AI normalize: OFF"
        st.caption(f"Engine: {engine_used} | Pages processed: {pages_done} | Time: {t_engine:.2f}s (budget {time_budget_s}s){extra}")

        if lang_choice == "Russian" and engine_name in {"TrOCR", "Donut"}:
            st.warning("Selected engine may have limited Cyrillic support. For Russian, prefer PaddleOCR or Tesseract.")

        json_data = None
        t_llm = None
        if do_llm:
            t1 = time.time()
            # Use the normalized OCR text (after AI normalization if enabled)
            max_iters = 1 if fast_mode else 3
            pp_iter, iters = llm_extract_json_iterative(ocr_text, max_iters=max_iters, fast_mode=fast_mode)
            t_llm = time.time() - t1
            json_data = pp_iter.json_data
            st.caption(f"LLM: {pp_iter.model_name} | Iters: {iters} | Time: {t_llm:.2f}s")

        st.divider()

        left, right = st.columns([1, 1])
        left.subheader("Structured JSON")
        if json_data is None:
            json_data = {"schema_version": "1.0", "ocr_text": ocr_text[:10000]}
        json_data.setdefault("schema_version", "1.0")
        json_data.setdefault("ocr_text", ocr_text[:10000])
        json_data.setdefault("source_filename", name)
        json_data.setdefault("ocr_engine", engine_used)
        json_data.setdefault("noise_score", round(avg_noise, 3))
        is_valid = True
        try:
            validated = BankingDoc.model_validate(json_data)
            json_data = json.loads(validated.model_dump_json(exclude_none=True, indent=2))
        except Exception:
            is_valid = False
        left.json(json_data)
        left.caption("Valid JSON schema" if is_valid else "Invalid JSON schema")

        st.download_button("Export JSON", data=json.dumps(json_data, ensure_ascii=False, indent=2), file_name=f"{Path(name).stem}_extracted.json", mime="application/json")
        # Export Text - clean up the formatting
        clean_text = ocr_text.replace('\n', ' ').replace('  ', ' ').strip()
        st.download_button("Export Text", data=clean_text, file_name=f"{Path(name).stem}_ocr.txt", mime="text/plain")
        
        # Always try to export XLSX - function handles errors internally
        xlsx_bytes = export_standard_xlsx(json_data)
        st.download_button("Export XLSX (standard)", data=xlsx_bytes, file_name=f"{Path(name).stem}_standard.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
        # Show XLSX preview
        st.subheader("📊 XLSX Preview")
        try:
            import pandas as pd
            from io import BytesIO
            df = pd.read_excel(BytesIO(xlsx_bytes))
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.warning(f"Предварительный просмотр недоступен: {e}")
            st.info("Но файл XLSX можно скачать!")

        try:
            append_jsonl(Path("logs/runs.jsonl"), {
                "filename": name,
                "engine": engine_used,
                "language": lang_choice,
                "time_engine_s": round(t_engine, 3),
                "time_llm_s": round(t_llm, 3) if (t_llm is not None) else None,
                "noise_score": avg_noise,
                "json_valid": is_valid,
            })
        except Exception:
            pass

        with st.expander("Compare with Ground Truth (optional)"):
            gt_file = st.file_uploader("Upload ground-truth JSON", type=["json"], key="gt")
            if gt_file is not None:
                try:
                    gt_json = json.loads(gt_file.read().decode("utf-8"))
                    st.json(gt_json)
                    gt_text = gt_json.get("ocr_text", "") if isinstance(gt_json, dict) else ""
                    if gt_text:
                        st.write("Text metrics vs. GT")
                        st.write({
                            "CER": compute_cer(gt_text, ocr_text),
                            "WER": compute_wer(gt_text, ocr_text),
                            "Norm Levenshtein": normalized_levenshtein(gt_text, ocr_text),
                        })

                    st.write("Field-level metrics")
                    st.json(field_level_scores(gt_json, json_data))
                    st.write({"Exact Match": exact_match(gt_json, json_data)})

                    # Auto-compare with Tesseract baseline (no toggle needed)
                    try:
                        from src.ocr.tesseract_engine import TesseractEngine as Tess
                        t0b = time.time()
                        base = Tess().run(pre_preview.image)
                        tb = time.time() - t0b
                        st.caption(f"Tesseract baseline time: {tb:.2f}s")
                        if gt_text:
                            st.write("Baseline (Tesseract) vs GT")
                            st.write({
                                "CER": compute_cer(gt_text, base.text),
                                "WER": compute_wer(gt_text, base.text),
                                "Norm Levenshtein": normalized_levenshtein(gt_text, base.text),
                            })
                    except ImportError:
                        st.warning("Tesseract backend not available for baseline.")
                except Exception as e:
                    st.error(f"Failed to parse ground truth: {e}")
    else:
        # Batch view: iterate all and show a compact table + download
        import pandas as pd
        st.subheader(f"Batch processing {len(documents)} file(s)")
        progress = st.progress(0)
        rows = []
        jsonl_lines = []
        json_rows = []
        for idx, doc in enumerate(documents):
            try:
                name = doc["name"]
                resolved = resolve_language_setting(lang_choice, doc["pages"][0])
                engine_label = engine_name
                ocr_text, t_engine, avg_noise, pages_done, paddle_lines = process_document(doc["pages"], engine_label, resolved_lang=resolved)
                
                # Apply AI normalization if enabled (same logic as single-file processing)
                if do_ai_norm and ocr_text.strip():
                    st.info("🔧 AI Normalization: ON")
                    try:
                        lines_source = paddle_lines if paddle_lines else ocr_text.splitlines()
                        def is_latin_only(s: str) -> bool:
                            has_cyr = detect_cyrillic_ratio(s) > 0.0
                            has_lat = any(ch.isalpha() and ('A' <= ch <= 'Z' or 'a' <= ch <= 'z') for ch in s)
                            return has_lat and not has_cyr

                        blocks = []
                        i2 = 0
                        n2 = len(lines_source)
                        while i2 < n2:
                            ln = lines_source[i2]
                            if not ln or is_latin_only(ln) or any(tag in ln for tag in ["IBAN", "SWIFT", "http://", "https://", "E-mail", "Email"]):
                                blocks.append((i2, i2+1, ln, False))
                                i2 += 1
                                continue
                            # Use higher threshold in fast mode to reduce AI calls
                            cyrillic_threshold = 0.4 if fast_mode else 0.2
                            if detect_cyrillic_ratio(ln) >= cyrillic_threshold:
                                j2 = i2
                                buf = []
                                while j2 < n2:
                                    lnj = lines_source[j2]
                                    if lnj and not is_latin_only(lnj) and detect_cyrillic_ratio(lnj) >= cyrillic_threshold and not any(tag in lnj for tag in ["IBAN", "SWIFT", "http://", "https://", "E-mail", "Email"]):
                                        buf.append(lnj)
                                        j2 += 1
                                    else:
                                        break
                                blocks.append((i2, j2, "\n".join(buf), True))
                                i2 = j2
                            else:
                                blocks.append((i2, i2+1, ln, False))
                                i2 += 1

                        to_norm = [b[2] for b in blocks if b[3]]
                        if to_norm:
                            normed_blocks, _ = ai_normalize_pseudocyrillic_bulk(to_norm, fast_mode=fast_mode)
                        else:
                            normed_blocks = []

                        out_lines = []
                        it2 = iter(normed_blocks)
                        for (_s, _e, content, needs_ai) in blocks:
                            if needs_ai:
                                norm_text = next(it2)
                                out_lines.extend(norm_text.splitlines())
                            else:
                                out_lines.extend(content.splitlines())
                        ocr_text = "\n".join(out_lines)
                    except Exception:
                        pass
                else:
                    st.info("🔧 AI Normalization: OFF")
                    # Fallback: light homoglyph normalization only if clearly Cyrillic-heavy
                    if detect_cyrillic_ratio(ocr_text) >= 0.5:
                        ocr_text = normalize_homoglyphs_to_cyrillic(ocr_text)
                engine_used = engine_name if ocr_text.strip() else ("Tesseract (fallback)" if engine_name != "Tesseract" else "Tesseract")

                t_llm = None
                json_data = None
                if do_llm:
                    t1 = time.time()
                    max_iters = 1 if fast_mode else 2
                    pp_iter, iters = llm_extract_json_iterative(ocr_text, max_iters=max_iters, fast_mode=fast_mode)
                    t_llm = time.time() - t1
                    json_data = pp_iter.json_data
                if json_data is None:
                    json_data = {"schema_version": "1.0", "ocr_text": ocr_text[:10000]}
                json_data.setdefault("schema_version", "1.0")
                json_data.setdefault("ocr_text", ocr_text[:10000])
                json_data.setdefault("source_filename", name)
                json_data.setdefault("ocr_engine", engine_used)
                json_data.setdefault("noise_score", round(avg_noise, 3))
                is_valid = True
                try:
                    _validated = BankingDoc.model_validate(json_data)
                except Exception:
                    is_valid = False

                rows.append({
                    "filename": name,
                    "engine": engine_used,
                    "noise": round(avg_noise, 3),
                    "time_engine_s": round(t_engine, 3),
                    "time_llm_s": round(t_llm, 3) if (t_llm is not None) else None,
                    "json_valid": is_valid,
                    "text_preview": ocr_text.strip()[:120],
                })

                json_rows.append({
                    "filename": name,
                    "json": json_data,
                })

                rec = {
                    "filename": name,
                    "engine": engine_used,
                    "language": lang_choice,
                    "time_engine_s": round(t_engine, 3),
                    "time_llm_s": round(t_llm, 3) if (t_llm is not None) else None,
                    "noise_score": avg_noise,
                    "json_valid": is_valid,
                }
                jsonl_lines.append(json.dumps(rec, ensure_ascii=False))
            except Exception as e:
                rows.append({
                    "filename": name,
                    "engine": engine_name,
                    "noise": None,
                    "time_engine_s": None,
                    "time_llm_s": None,
                    "json_valid": False,
                    "text_preview": f"ERROR: {e}",
                })
            progress.progress((idx + 1) / len(documents))

        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        # Show JSON outputs per processed file
        st.subheader("Extracted JSON per file")
        json_table = [{"filename": r["filename"], "json": json.dumps(r["json"], ensure_ascii=False)} for r in json_rows]
        st.dataframe(pd.DataFrame(json_table), use_container_width=True)
        aggregated = {r["filename"]: r["json"] for r in json_rows}
        st.download_button(
            "Download run summary (JSONL)",
            data="\n".join(jsonl_lines),
            file_name="batch_runs.jsonl",
            mime="application/json",
        )
        st.download_button(
            "Download extracted JSONs",
            data=json.dumps(aggregated, ensure_ascii=False, indent=2),
            file_name="batch_extracted.json",
            mime="application/json",
        )


