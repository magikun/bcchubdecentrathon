from __future__ import annotations

from typing import Optional, List, Dict
from pydantic import BaseModel, Field, field_validator


class Party(BaseModel):
    name: Optional[str] = Field(None, description="Legal name of the party")
    inn: Optional[str] = Field(None, description="Tax ID (INN)")
    kpp: Optional[str] = Field(None, description="KPP (if applicable)")
    address: Optional[str] = None


class BankAccount(BaseModel):
    bank_name: Optional[str] = None
    bik: Optional[str] = None
    account_number: Optional[str] = None
    correspondent_account: Optional[str] = None


class CheckDocument(BaseModel):
    doc_type: str = Field("check", frozen=True)
    bank_account: Optional[BankAccount] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    date: Optional[str] = None
    payee: Optional[Party] = None
    payer: Optional[Party] = None
    memo: Optional[str] = None


class StatementTransaction(BaseModel):
    date: Optional[str] = None
    description: Optional[str] = None
    amount: Optional[float] = None
    balance: Optional[float] = None


class StatementDocument(BaseModel):
    doc_type: str = Field("statement", frozen=True)
    account: Optional[BankAccount] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    opening_balance: Optional[float] = None
    closing_balance: Optional[float] = None
    transactions: List[StatementTransaction] = Field(default_factory=list)


class ContractDocument(BaseModel):
    doc_type: str = Field("contract", frozen=True)
    parties: Dict[str, Party] = Field(default_factory=dict)
    date: Optional[str] = None
    contract_number: Optional[str] = None
    subject: Optional[str] = None
    total_amount: Optional[float] = None
    currency: Optional[str] = None
    terms: Optional[str] = None


class BankingDoc(BaseModel):
    """Unified JSON schema with mandatory top-level keys."""

    schema_version: str = Field("1.0")
    source_filename: Optional[str] = None
    ocr_engine: Optional[str] = None
    ocr_text: Optional[str] = None
    noise_score: Optional[float] = None

    check: Optional[CheckDocument] = None
    statement: Optional[StatementDocument] = None
    contract: Optional[ContractDocument] = None

    @field_validator("schema_version")
    @classmethod
    def version_nonempty(cls, v: str) -> str:
        assert v and isinstance(v, str)
        return v


