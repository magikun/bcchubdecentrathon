from __future__ import annotations

from typing import List, Union, Optional
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path, convert_from_bytes
import os
import shutil
from glob import glob
from typing import cast

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None  # type: ignore


def _resolve_poppler_path(explicit: Optional[str] = None) -> Optional[str]:
    # 1) explicit
    if explicit and Path(explicit).exists():
        return explicit
    # 2) env var
    env_path = os.getenv("POPPLER_PATH")
    if env_path and Path(env_path).exists():
        return env_path
    # 3) PATH lookup
    exe = shutil.which("pdftoppm")
    if exe:
        return str(Path(exe).parent)
    # 4) common install locations (Windows)
    candidates = []
    for base in [os.getenv("LOCALAPPDATA"), os.getenv("PROGRAMFILES"), os.getenv("PROGRAMFILES(X86)")]:
        if base:
            candidates.append(Path(base) / "poppler")
    for root in candidates:
        pattern = str(root / "**" / "pdftoppm.exe")
        matches = glob(pattern, recursive=True)
        if matches:
            return str(Path(matches[0]).parent)
    return None


def pdf_to_images(path: Union[str, Path], dpi: int = 300, poppler_path: Optional[str] = None) -> List[Image.Image]:
    poppler = _resolve_poppler_path(poppler_path)
    try:
        if poppler:
            images = convert_from_path(str(path), dpi=dpi, poppler_path=poppler)
        else:
            images = convert_from_path(str(path), dpi=dpi)
        return images
    except Exception:
        # Fallback to PyMuPDF if available
        if fitz is None:
            raise
        doc = fitz.open(str(path))
        if doc.page_count == 0:
            raise RuntimeError("PDF has no pages")
        page = doc.load_page(0)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return [img]


def pdf_bytes_to_images(data: bytes, dpi: int = 300, poppler_path: Optional[str] = None) -> List[Image.Image]:
    poppler = _resolve_poppler_path(poppler_path)
    try:
        if poppler:
            images = convert_from_bytes(data, dpi=dpi, poppler_path=poppler)
        else:
            images = convert_from_bytes(data, dpi=dpi)
        return images
    except Exception:
        if fitz is None:
            raise
        doc = fitz.open(stream=data, filetype="pdf")
        if doc.page_count == 0:
            raise RuntimeError("PDF has no pages")
        page = doc.load_page(0)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return [img]


