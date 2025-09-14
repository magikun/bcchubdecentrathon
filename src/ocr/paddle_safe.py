from __future__ import annotations

# Centralized factories with lazy import to avoid partial init under Streamlit cache


def make_detector():  # type: ignore
    from paddleocr import PaddleOCR  # type: ignore
    return PaddleOCR(use_angle_cls=True, det=True, rec=False, lang="ru", show_log=False)


def make_recognizer(lang: str):  # type: ignore
    from paddleocr import PaddleOCR  # type: ignore
    if lang not in {"ru", "en"}:
        lang = "ru"
    return PaddleOCR(use_angle_cls=True, det=False, rec=True, lang=lang, show_log=False)


