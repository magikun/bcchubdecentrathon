from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd


def heuristic_map(record: Dict[str, Any]) -> Dict[str, Any]:
    """Map common Russian banking column names into BankingDoc-like JSON.

    This is heuristic and may be adapted as needed.
    """
    lower = {str(k).strip().lower(): v for k, v in record.items()}

    def pick(*names):
        for n in names:
            if n in lower and pd.notna(lower[n]) and str(lower[n]).strip() != "":
                return lower[n]
        return None

    data: Dict[str, Any] = {
        "schema_version": "1.0",
        "check": None,
        "statement": None,
        "contract": None,
    }

    # Try to infer a generic document with bank account and parties
    bank = {
        "bank_name": pick("банк", "наименование банка", "bank", "bank name"),
        "bik": pick("бик", "bik"),
        "account_number": pick("номер счета", "счет", "account", "account number"),
        "correspondent_account": pick("корр. счет", "корреспондентский счет", "corr account"),
    }
    payer = {
        "name": pick("плательщик", "payer", "клиент", "client"),
        "inn": pick("инн плательщика", "инн", "inn"),
        "kpp": pick("кпп", "kpp"),
        "address": pick("адрес", "address"),
    }
    payee = {
        "name": pick("получатель", "payee", "банк", "beneficiary"),
        "inn": pick("инн получателя", "инн", "inn"),
        "kpp": pick("кпп", "kpp"),
        "address": pick("адрес получателя", "address"),
    }

    amount = pick("сумма", "amount")
    currency = pick("валюта", "currency")
    date = pick("дата", "date")
    doc_number = pick("номер документа", "номер", "doc number", "document number")
    purpose = pick("назначение платежа", "описание", "description")

    # Prefer check-like document when amount/date/parties are present
    check: Dict[str, Any] = {
        "doc_type": "check",
        "bank_account": bank if any(bank.values()) else None,
        "amount": amount,
        "currency": currency,
        "date": date,
        "payee": payee if any(payee.values()) else None,
        "payer": payer if any(payer.values()) else None,
        "memo": purpose,
    }

    if any(v is not None for v in check.values()):
        data["check"] = check

    # Attach some convenience aliases at top-level if present in the sheet
    if doc_number:
        data["document_number"] = doc_number

    # Try to provide GT OCR text if available
    ocr_text = pick("ocr_text", "текст", "full text", "text", "контент")
    if ocr_text is None:
        # Fallback: join all string-like cells to approximate textual GT
        try:
            parts = []
            for k, v in record.items():
                if isinstance(v, str) and len(v.strip()) >= 3:
                    parts.append(v.strip())
            if parts:
                ocr_text = "\n".join(parts)
        except Exception:
            ocr_text = None
    if ocr_text:
        data["ocr_text"] = ocr_text

    return {k: v for k, v in data.items() if v is not None}


def build_gt_from_xlsx(xlsx_path: Path, out_dir: Path) -> None:
    xl = pd.ExcelFile(xlsx_path)
    # Use first sheet
    df = xl.parse(xl.sheet_names[0])
    if df.empty:
        return
    # Use first row as single-record GT (common for per-document XLSX)
    record = df.iloc[0].to_dict()
    data = heuristic_map(record)
    out = out_dir / f"{xlsx_path.stem}.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx_dir", type=str, required=True)
    parser.add_argument("--out", type=str, default="eval_gts")
    args = parser.parse_args()

    src = Path(args.xlsx_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for x in src.glob("*.xlsx"):
        try:
            build_gt_from_xlsx(x, out)
            print("built", x.name)
        except Exception as e:
            print("error", x.name, e)


if __name__ == "__main__":
    main()


