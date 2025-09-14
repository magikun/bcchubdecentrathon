from __future__ import annotations

import re


CYRILLIC_RANGE = re.compile(r"[\u0400-\u04FF]")
LATIN_RANGE = re.compile(r"[A-Za-z]")


def detect_cyrillic_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    cyr = sum(1 for ch in letters if CYRILLIC_RANGE.match(ch) is not None)
    return cyr / len(letters)


_LATIN_TO_CYR = {
    # Uppercase homoglyphs / close transliterations
    "A": "А", "B": "Б", "C": "С", "D": "Д", "E": "Е", "F": "Ф", "G": "Г",
    "H": "Н", "I": "И", "J": "Й", "K": "К", "L": "Л", "M": "М", "N": "Н",
    "O": "О", "P": "Р", "Q": "К", "R": "Р", "S": "С", "T": "Т", "U": "У",
    "V": "В", "W": "В", "X": "Х", "Y": "У", "Z": "З",
    # Lowercase
    "a": "а", "b": "б", "c": "с", "d": "д", "e": "е", "f": "ф", "g": "г",
    "h": "н", "i": "и", "j": "й", "k": "к", "l": "л", "m": "м", "n": "н",
    "o": "о", "p": "р", "q": "к", "r": "р", "s": "с", "t": "т", "u": "у",
    "v": "в", "w": "в", "x": "х", "y": "у", "z": "з",
}


def normalize_homoglyphs_to_cyrillic(text: str) -> str:
    return "".join(_LATIN_TO_CYR.get(ch, ch) for ch in text)



def normalize_prefer_cyrillic(text: str, threshold: float = 0.2) -> str:
    """Prefer Cyrillic everywhere, but preserve Latin tokens that are clearly Latin.

    - If a token contains any Cyrillic, convert Latin homoglyphs inside it to Cyrillic.
    - If a token is Latin-dominant (latin_ratio >= 1 - threshold) and has no Cyrillic, keep as is.
    - Digits/punctuation are preserved.
    """
    if not text:
        return text

    parts = re.split(r"(\s+)", text)
    out = []
    for part in parts:
        if not part or part.isspace():
            out.append(part)
            continue

        has_cyr = CYRILLIC_RANGE.search(part) is not None
        has_lat = LATIN_RANGE.search(part) is not None

        if has_cyr:
            out.append(normalize_homoglyphs_to_cyrillic(part))
            continue

        if has_lat:
            # Latin-only token: keep as-is
            out.append(part)
            continue

        # No letters or mixed symbols: safe to keep
        out.append(part)
    return "".join(out)


def normalize_force_cyrillic(text: str) -> str:
    """Convert ALL Latin letters to Cyrillic using approximate one-to-one mapping.

    This aggressively forces Cyrillic output even for Latin-only tokens.
    """
    if not text:
        return text
    return "".join(_LATIN_TO_CYR.get(ch, ch) for ch in text)

