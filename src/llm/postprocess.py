from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

from pydantic import ValidationError
from dotenv import load_dotenv
import os

from src.schema.bank_docs import BankingDoc
from src.utils.text_normalize import normalize_force_cyrillic
from typing import List


load_dotenv()


@dataclass
class PostprocessResult:
    json_data: Dict[str, Any]
    model_name: str


SYSTEM_PROMPT = (
    "You are a precise information extraction assistant for Russian banking documents (checks, contracts, statements). "
    "Extract ALL available information from the OCR text into a comprehensive JSON object. "
    "Be thorough - extract company names, INN/KPP numbers, addresses, bank details, amounts, dates, etc. "
    "Use ISO dates (YYYY-MM-DD) and floats with dot decimals. "
    "\n\nCRITICAL: If the OCR text contains mixed Latin/Cyrillic characters (pseudo-Cyrillic), "
    "normalize it to proper Russian Cyrillic before extracting JSON. For example: "
    "'TopapuujectBO C OfpaHMYeHHOM OTBETCTBEHHOCTbIO' should become "
    "'Товарищество с ограниченной ответственностью'. "
    "Preserve Latin codes like IBAN, SWIFT, URLs, and email addresses. "
    "\n\nEXTRACTION PRIORITIES: "
    "1. Company/organization names (наименование) "
    "2. INN numbers (ИНН) "
    "3. KPP numbers (КПП) "
    "4. Bank names and BIK codes "
    "5. Account numbers (расчетный счет) "
    "6. Amounts and currencies "
    "7. Dates and document numbers "
    "8. Addresses and contact information "
    "\n\nFOR CONTRACTS, extract: "
    "- contract_number (номер контракта) "
    "- date (дата заключения) "
    "- parties (участники: продавец, покупатель, контрагент) "
    "- total_amount (сумма контракта) "
    "- currency (валюта контракта) "
    "- terms (условия контракта) "
    "- subject (предмет контракта) "
    "\n\nAlways fill the 'contract' object with all available contract details."
)


def build_user_prompt(ocr_text: str) -> str:
    return (
        "Extract comprehensive JSON with top-level keys: schema_version, source_filename, ocr_engine, ocr_text, "
        "noise_score, and one of: check, statement, contract. "
        "Fill ALL available fields in the nested structures according to the banking schema. "
        "If you find company names, INN/KPP numbers, bank details, amounts, or addresses in the text, "
        "include them in the appropriate fields. Be thorough and extract everything you can identify.\n\n"
        "For CONTRACTS, ensure the 'contract' object contains:\n"
        "- doc_type: 'contract'\n"
        "- parties: dictionary with keys like 'seller', 'buyer', 'contractor' containing name, inn, kpp, address\n"
        "- date: contract signing date in YYYY-MM-DD format\n"
        "- contract_number: contract number/identifier\n"
        "- subject: what the contract is about\n"
        "- total_amount: contract amount as number\n"
        "- currency: currency code (RUB, USD, EUR, etc.)\n"
        "- terms: contract terms and conditions\n\n"
        f"OCR TEXT:\n{ocr_text}\n"
    )


def llm_extract_json(ocr_text: str, client: Any = None, model: Optional[str] = None, fast_mode: bool = False) -> PostprocessResult:
    """Calls an LLM (OpenAI-compatible) to extract structured JSON. Fallback to heuristic JSON if unavailable."""
    # Use faster model in fast mode
    if fast_mode and not model:
        model_name = "gpt-4o-mini"  # Always use fastest model in fast mode
    else:
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
    # Read API credentials from environment only
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_ALT") or ""
    base_url = os.getenv("OPENAI_BASE_URL")
    organization = os.getenv("OPENAI_ORG")

    if client is None and api_key:
        try:
            from openai import OpenAI
            if base_url and organization:
                client = OpenAI(api_key=api_key, base_url=base_url, organization=organization)
            elif base_url:
                client = OpenAI(api_key=api_key, base_url=base_url)
            elif organization:
                client = OpenAI(api_key=api_key, organization=organization)
            else:
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


def llm_extract_json_iterative(ocr_text: str, max_iters: int = 2, client: Any = None, model: Optional[str] = None, fast_mode: bool = False) -> Tuple[PostprocessResult, int]:
    """Try to obtain a schema-valid JSON by calling the LLM up to max_iters times.

    Returns (final_result, iterations_used).
    """
    iters = 0
    last = llm_extract_json(ocr_text, client=client, model=model, fast_mode=fast_mode)
    iters += 1
    try:
        BankingDoc.model_validate(last.json_data)
        return last, iters
    except Exception:
        pass

    while iters < max_iters:
        iters += 1
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")
        if client is None and api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
            except Exception:
                client = None
        if client is None:
            # Heuristic fallback only; break as we cannot improve
            break

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + " Ensure the JSON strictly validates against the schema."},
            {"role": "user", "content": build_user_prompt(ocr_text) + "\nIf previous JSON was invalid, fix keys/types and return a valid JSON only."},
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
            data = last.json_data

        try:
            validated = BankingDoc.model_validate(data)
            data = json.loads(validated.model_dump_json(exclude_none=True))
            return PostprocessResult(json_data=data, model_name=model_name), iters
        except Exception:
            last = PostprocessResult(json_data=data, model_name=model_name)
            continue

    return last, iters



# --- AI-powered normalization for pseudo-Cyrillic / mixed-alphabet OCR text ---
def ai_normalize_pseudocyrillic(
    text: str,
    client: Any = None,
    model: Optional[str] = None,
) -> Tuple[str, str]:
    """Normalize OCR text that mixes Latin lookalikes with Cyrillic to clean Russian.

    Returns (normalized_text, model_name). Falls back to a deterministic
    normalization if the LLM is unavailable.
    """
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_ALT") or ""
    base_url = os.getenv("OPENAI_BASE_URL")
    organization = os.getenv("OPENAI_ORG")

    if client is None and api_key:
        try:
            from openai import OpenAI
            if base_url and organization:
                client = OpenAI(api_key=api_key, base_url=base_url, organization=organization)
            elif base_url:
                client = OpenAI(api_key=api_key, base_url=base_url)
            elif organization:
                client = OpenAI(api_key=api_key, organization=organization)
            else:
                client = OpenAI(api_key=api_key)
        except Exception:
            client = None

    # If no client, apply deterministic Cyrillic-forcing normalization
    if client is None:
        return normalize_force_cyrillic(text), "heuristic"

    system = (
        "Ты помощник по нормализации текста. Твоя задача — превратить строку,\n"
        "в которой русские слова написаны смесью латинских и кириллических букв,\n"
        "в корректный русский текст на кириллице. Ничего не переводить и не\n"
        "переформулировать. Сохраняй цифры и пунктуацию, регистр букв и пробелы.\n"
        "Если встречаются реальные латинские аббревиатуры/коды, оставляй их на латинице."
    )
    user = f"Текст:\n{text}\n\nВерни только нормализованный текст."

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
        )
        content = resp.choices[0].message.content or ""
        return content.strip(), model_name
    except Exception:
        # Fallback to deterministic Cyrillic mapping
        return normalize_force_cyrillic(text), "heuristic"


_AI_NORM_CACHE: Dict[str, str] = {}

def ai_normalize_pseudocyrillic_bulk(
    texts: List[str],
    client: Any = None,
    model: Optional[str] = None,
    fast_mode: bool = False,
) -> Tuple[List[str], str]:
    """Normalize multiple lines in a single request to reduce latency.

    - Preserves order; returns exactly one output per input.
    - Uses a small in-memory cache to avoid re-normalizing identical lines.
    - Falls back to heuristic conversion if LLM is unavailable.
    """
    if not texts:
        return [], "heuristic"

    # Serve from cache where possible
    pending_indices: List[int] = []
    outputs: List[Optional[str]] = [None] * len(texts)
    for i, t in enumerate(texts):
        if t in _AI_NORM_CACHE:
            outputs[i] = _AI_NORM_CACHE[t]
        else:
            pending_indices.append(i)

    # If everything was cached, return immediately
    if not pending_indices:
        return [o or "" for o in outputs], "cache"

    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_ALT") or ""
    base_url = os.getenv("OPENAI_BASE_URL")
    organization = os.getenv("OPENAI_ORG")

    local_client = client
    if local_client is None and api_key:
        try:
            from openai import OpenAI
            if base_url and organization:
                local_client = OpenAI(api_key=api_key, base_url=base_url, organization=organization)
            elif base_url:
                local_client = OpenAI(api_key=api_key, base_url=base_url)
            elif organization:
                local_client = OpenAI(api_key=api_key, organization=organization)
            else:
                local_client = OpenAI(api_key=api_key)
        except Exception:
            local_client = None

    # If no client, fallback heuristically for pending items
    if local_client is None:
        for idx in pending_indices:
            t = texts[idx]
            out = normalize_force_cyrillic(t)
            outputs[idx] = out
            _AI_NORM_CACHE[t] = out
        return [o or "" for o in outputs], "heuristic"

    # Limit request size to prevent timeouts - larger chunks in fast mode
    max_chars = 3000 if fast_mode else 1200
    current_chars = 0
    batches = []
    current_batch = []
    
    for idx in pending_indices:
        text = texts[idx]
        if current_chars + len(text) > max_chars and current_batch:
            batches.append(current_batch)
            current_batch = [idx]
            current_chars = len(text)
        else:
            current_batch.append(idx)
            current_chars += len(text)
    
    if current_batch:
        batches.append(current_batch)
    
    # Process each batch
    all_results = [None] * len(texts)
    for batch_indices in batches:
        numbered_lines = "\n".join(f"{i+1}. {texts[i]}" for i in batch_indices)
        system = (
            "Ты помощник по нормализации. Для КАЖДОЙ строки замени смешанные латинские/кириллические"
            " буквы на корректную кириллицу. Сохраняй пробелы, пунктуацию и регистр."
            " Реальные латинские коды (IBAN, SWIFT, URL, email) оставляй на латинице."
            " Верни строго столько строк, сколько пришло, по одной на строку, без нумерации."
        )
        user = f"Строки для нормализации (не меняй порядок):\n{numbered_lines}"

        try:
            resp = local_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
            )
            content = (resp.choices[0].message.content or "").strip()
            # Split by lines; if counts mismatch, fallback heuristics for safety
            returned = [ln.rstrip("\r") for ln in content.split("\n")]
            if len(returned) != len(batch_indices):
                for idx in batch_indices:
                    t = texts[idx]
                    out = normalize_force_cyrillic(t)
                    all_results[idx] = out
                    _AI_NORM_CACHE[t] = out
            else:
                # Fill results and cache
                for pos, idx in enumerate(batch_indices):
                    out = returned[pos].strip()
                    all_results[idx] = out
                    _AI_NORM_CACHE[texts[idx]] = out
        except Exception:
            # Fallback: heuristic for this batch
            for idx in batch_indices:
                t = texts[idx]
                out = normalize_force_cyrillic(t)
                all_results[idx] = out
                _AI_NORM_CACHE[t] = out
    
    # Fill final outputs
    for i, result in enumerate(all_results):
        if result is not None:
            outputs[i] = result
    
    return [o or "" for o in outputs], model_name
