from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Dict, Any

from pydantic import ValidationError
from dotenv import load_dotenv
import os

from src.schema.bank_docs import BankingDoc


load_dotenv()


@dataclass
class PostprocessResult:
    json_data: Dict[str, Any]
    model_name: str


SYSTEM_PROMPT = (
    "You are a precise information extraction assistant for banking documents (checks, contracts, statements). "
    "Given raw OCR text, extract a JSON object strictly matching the schema. "
    "If a field is unknown, omit it. Use ISO dates (YYYY-MM-DD) and floats with dot decimals."
)


def build_user_prompt(ocr_text: str) -> str:
    return (
        "Extract JSON with top-level keys: schema_version, source_filename, ocr_engine, ocr_text, "
        "noise_score, and one of: check, statement, contract. "
        "The nested structures must follow the banking schema.\n\n"
        f"OCR TEXT:\n{ocr_text}\n"
    )


def llm_extract_json(ocr_text: str, client: Any = None, model: Optional[str] = None) -> PostprocessResult:
    """Calls an LLM (OpenAI-compatible) to extract structured JSON. Fallback to heuristic JSON if unavailable."""
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")

    if client is None and api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
        except Exception:
            client = None

    # Default skeleton if no LLM available
    skeleton = {
        "schema_version": "1.0",
        "ocr_text": ocr_text[:20000],
    }

    if client is None:
        return PostprocessResult(json_data=skeleton, model_name="heuristic")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(ocr_text)},
    ]

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
    except Exception:
        data = skeleton

    # Validate
    try:
        validated = BankingDoc.model_validate(data)
        data = json.loads(validated.model_dump_json(exclude_none=True))
    except ValidationError:
        # Return unvalidated, UI will mark invalid
        pass

    return PostprocessResult(json_data=data, model_name=model_name)


