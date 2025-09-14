from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def _get(d: Any, path: str, default: Any = None) -> Any:
    """Safely get nested value from dict/list/any structure."""
    if not isinstance(d, (dict, list)):
        return default
    
    cur: Any = d
    for key in path.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        elif isinstance(cur, list) and key.isdigit() and int(key) < len(cur):
            cur = cur[int(key)]
        else:
            return default
    return cur


def extract_standard_fields(doc: Any) -> List[Tuple[str, Any]]:
    """Map BankingDoc JSON to a standardized RU field list suitable for XLSX export.

    This is a comprehensive mapping for all document types (contracts/checks/statements).
    Handles any input gracefully - if not a dict, returns basic fields.
    """
    # Ensure doc is a dictionary
    if not isinstance(doc, dict):
        # If it's not a dict, create a basic structure
        doc = {"ocr_text": str(doc) if doc else ""}
    
    check = doc.get("check") or {}
    contract = doc.get("contract") or {}
    statement = doc.get("statement") or {}
    
    # Банковские счета из разных источников
    check_account = _get(check, "bank_account") or {}
    contract_account = _get(contract, "account") or {}
    statement_account = _get(statement, "account") or {}
    
    # Объединяем данные из всех источников
    account = {**check_account, **contract_account, **statement_account}
    
    # Стороны из разных источников
    payer = _get(check, "payer") or {}
    payee = _get(check, "payee") or {}
    
    # Для контрактов берем стороны из parties
    if contract.get("parties"):
        parties = contract["parties"]
        # Ищем стороны по ключевым словам
        for party_key, party_data in parties.items():
            if isinstance(party_data, dict):
                # Более гибкий поиск сторон
                party_key_lower = party_key.lower()
                if any(keyword in party_key_lower for keyword in ["продавец", "seller", "плательщик", "payer", "поставщик", "supplier"]):
                    payer = {**payer, **party_data}
                elif any(keyword in party_key_lower for keyword in ["покупатель", "buyer", "получатель", "payee", "заказчик", "customer"]):
                    payee = {**payee, **party_data}
                else:
                    # Если не можем определить роль, добавляем к обеим сторонам
                    payer = {**payer, **party_data}
                    payee = {**payee, **party_data}

    fields: List[Tuple[str, Any]] = []
    
    # Основные поля документа
    fields.append(("Номер документа", 
                  doc.get("document_number") or 
                  contract.get("contract_number") or 
                  check.get("doc_type")))
    
    fields.append(("Дата документа", 
                  check.get("date") or 
                  contract.get("date") or 
                  statement.get("period_start")))
    
    fields.append(("Валюта документа", 
                  check.get("currency") or 
                  contract.get("currency")))
    
    fields.append(("Сумма документа", 
                  check.get("amount") or 
                  contract.get("total_amount") or 
                  statement.get("opening_balance")))
    
    fields.append(("Предмет/Описание", 
                  check.get("memo") or 
                  contract.get("subject") or 
                  contract.get("terms")))
    
    # Дополнительные поля контракта
    if contract:
        fields.append(("Тип контракта", contract.get("doc_type")))
        fields.append(("Условия контракта", contract.get("terms")))
        fields.append(("Все стороны контракта", str(contract.get("parties", {}))))

    # Стороны - Плательщик
    fields.append(("Плательщик (наименование)", payer.get("name")))
    fields.append(("Плательщик ИНН", payer.get("inn")))
    fields.append(("Плательщик КПП", payer.get("kpp")))
    fields.append(("Плательщик адрес", payer.get("address")))

    # Стороны - Получатель
    fields.append(("Получатель (наименование)", payee.get("name")))
    fields.append(("Получатель ИНН", payee.get("inn")))
    fields.append(("Получатель КПП", payee.get("kpp")))
    fields.append(("Получатель адрес", payee.get("address")))

    # Банковские реквизиты
    fields.append(("Банк получателя", account.get("bank_name")))
    fields.append(("БИК", account.get("bik")))
    fields.append(("Расчетный счет", account.get("account_number")))
    fields.append(("Корреспондентский счет", account.get("correspondent_account")))

    # Дополнительные поля для выписок
    if statement:
        fields.append(("Период с", statement.get("period_start")))
        fields.append(("Период по", statement.get("period_end")))
        fields.append(("Остаток на начало", statement.get("opening_balance")))
        fields.append(("Остаток на конец", statement.get("closing_balance")))
        if statement.get("transactions"):
            fields.append(("Количество операций", len(statement["transactions"])))

    # Технические метки
    fields.append(("OCR движок", doc.get("ocr_engine")))
    fields.append(("Шум (0-1)", doc.get("noise_score")))
    fields.append(("Исходный файл", doc.get("source_filename")))

    return fields


def export_standard_xlsx(doc: Any) -> bytes:
    """Export any data to XLSX format, handling errors gracefully."""
    try:
        rows = extract_standard_fields(doc)
        df = pd.DataFrame(rows, columns=["Показатель", "Значение"])
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Стандарт")
        return bio.getvalue()
    except Exception as e:
        # Fallback: create minimal XLSX with error info
        fallback_data = [
            ("Ошибка экспорта", str(e)),
            ("Исходные данные", str(doc)[:500] if doc else "Нет данных"),
            ("Тип данных", str(type(doc).__name__))
        ]
        df = pd.DataFrame(fallback_data, columns=["Показатель", "Значение"])
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Ошибка")
        return bio.getvalue()


