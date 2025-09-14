from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def extract_standard_fields(doc: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """Map BankingDoc JSON to a standardized RU field list suitable for XLSX export.

    This is a pragmatic mapping for contracts/checks commonly found in banking flows.
    """
    check = doc.get("check") or {}
    contract = doc.get("contract") or {}
    account = _get(check, "bank_account") or _get(contract, "account") or {}
    payer = _get(check, "payer") or {}
    payee = _get(check, "payee") or {}

    fields: List[Tuple[str, Any]] = []
    fields.append(("Номер документа", doc.get("document_number") or contract.get("contract_number")))
    fields.append(("Дата документа", check.get("date") or contract.get("date")))
    fields.append(("Валюта договора", check.get("currency") or contract.get("currency")))
    fields.append(("Сумма договора", check.get("amount") or contract.get("total_amount")))
    fields.append(("Предмет/Описание", check.get("memo") or contract.get("subject")))

    # Стороны
    fields.append(("Плательщик (наименование)", payer.get("name")))
    fields.append(("Плательщик ИНН", payer.get("inn")))
    fields.append(("Плательщик КПП", payer.get("kpp")))
    fields.append(("Плательщик адрес", payer.get("address")))

    fields.append(("Получатель (наименование)", payee.get("name")))
    fields.append(("Получатель ИНН", payee.get("inn")))
    fields.append(("Получатель КПП", payee.get("kpp")))
    fields.append(("Получатель адрес", payee.get("address")))

    # Банковские реквизиты
    fields.append(("Банк получателя", account.get("bank_name")))
    fields.append(("БИК", account.get("bik")))
    fields.append(("Расчетный счет", account.get("account_number")))
    fields.append(("Корреспондентский счет", account.get("correspondent_account")))

    # Технические метки
    fields.append(("OCR движок", doc.get("ocr_engine")))
    fields.append(("Шум (0-1)", doc.get("noise_score")))

    return fields


def export_standard_xlsx(doc: Dict[str, Any]) -> bytes:
    rows = extract_standard_fields(doc)
    df = pd.DataFrame(rows, columns=["Показатель", "Значение"])
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Стандарт")
    return bio.getvalue()


