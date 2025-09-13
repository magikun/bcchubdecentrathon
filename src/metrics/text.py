from __future__ import annotations

from typing import List, Dict, Tuple
from rapidfuzz.distance import Levenshtein
from jiwer import wer, cer


def compute_cer(ref: str, hyp: str) -> float:
    try:
        return float(cer(ref, hyp))
    except Exception:
        return 1.0


def compute_wer(ref: str, hyp: str) -> float:
    try:
        return float(wer(ref, hyp))
    except Exception:
        return 1.0


def normalized_levenshtein(ref: str, hyp: str) -> float:
    if not ref:
        return 1.0 if hyp else 0.0
    dist = Levenshtein.distance(ref, hyp)
    return dist / max(1, len(ref))


