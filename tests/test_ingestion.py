"""
Test suite for the Phase 2 Ingestion Agent.

Covers: deterministic Pass 1 extraction across all four formats, the Pass 1->Pass 2
confidence gate, the self-correction retry loop, graceful failure after exhausted
retries, and the merge rule that protects HIGH-confidence Pass 1 fields.

All LLM interaction is mocked via tools.llm.MockLLMClient — no live API calls.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from agents.ingestion import (
    _extract_from_txt,
    _extract_with_llm,
    _needs_llm,
    extract,
    ExtractedField,
)
from models.invoice_data import Confidence, InvoiceData, LineItem
from tools.llm import MockLLMClient

INVOICES = Path(__file__).resolve().parent.parent / "data" / "invoices"

# Valid JSON the mock can return that satisfies the InvoiceData schema.
VALID_LLM_JSON = (
    '{"vendor": "LLM Vendor", "amount": 1234.0, '
    '"items": [{"item": "X", "quantity": 1, "unit_price": 1234.0}], '
    '"due_date": "2026-05-05", "invoice_id": "LLM-1"}'
)


# ---------------------------------------------------------------------------
# (a) Clean extraction for each of the four formats — Pass 1 only, no LLM.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "filename, vendor, amount, due, n_items",
    [
        ("invoice_1001.txt", "Widgets Inc.", 5000.0, date(2026, 2, 1), 2),
        ("invoice_1004.json", "Precision Parts Ltd.", 1890.0, date(2026, 2, 22), 2),
        ("invoice_1006.csv", "Acme Industrial Supplies", 2750.0, date(2026, 2, 10), 2),
        ("invoice_1007.csv", "MegaWidgets Corp", 15525.0, date(2026, 2, 28), 3),
        ("invoice_1011.pdf", "Summit Manufacturing Co.", 3000.0, date(2026, 2, 20), 2),
    ],
)
def test_clean_extraction_per_format(filename, vendor, amount, due, n_items):
    mock = MockLLMClient()
    data = extract(INVOICES / filename, llm_client=mock)

    assert data.vendor == vendor
    assert data.amount == amount
    assert data.due_date == due
    assert len(data.items) == n_items
    assert data.overall_confidence is Confidence.HIGH
    # A clean invoice must never pay the LLM tax.
    assert mock.call_count == 0


# ---------------------------------------------------------------------------
# (b) The confidence gate routes incomplete Pass 1 results to Pass 2.
# ---------------------------------------------------------------------------
def test_gate_routes_messy_invoice_to_llm():
    # invoice_1002.txt is deliberately mangled ("INVOCE", "Vndr:", "Amt:"),
    # so the vendor label is unrecognisable -> Pass 1 is incomplete.
    _, fields = _extract_from_txt(INVOICES / "invoice_1002.txt")
    assert _needs_llm(fields) is True

    mock = MockLLMClient()  # returns its default valid payload
    data = extract(INVOICES / "invoice_1002.txt", llm_client=mock)
    assert mock.call_count == 1
    assert data.vendor == "Mock Vendor"  # came from the LLM, not Pass 1


def test_gate_keeps_clean_invoice_on_fast_path():
    _, fields = _extract_from_txt(INVOICES / "invoice_1001.txt")
    assert _needs_llm(fields) is False


# ---------------------------------------------------------------------------
# (c) Self-correction loop recovers on attempt 3 of 3.
# ---------------------------------------------------------------------------
def test_self_correction_recovers_on_third_attempt():
    mock = MockLLMClient(responses=["this is not json", '{"vendor": "X"}', VALID_LLM_JSON])
    data = _extract_with_llm("raw invoice text", fields={}, llm=mock)

    assert isinstance(data, InvoiceData)
    assert data.vendor == "LLM Vendor"
    assert data.due_date == date(2026, 5, 5)
    assert mock.call_count == 3  # two failures, then success


# ---------------------------------------------------------------------------
# (d) Graceful failure after three exhausted retries.
# ---------------------------------------------------------------------------
def test_exhausted_retries_raise_with_full_context():
    mock = MockLLMClient(responses=["bad-1", "bad-2", "bad-3"])
    with pytest.raises(ValueError) as exc_info:
        _extract_with_llm("raw invoice text", fields={}, llm=mock)

    message = str(exc_info.value)
    assert mock.call_count == 3
    # Every attempt's broken output is preserved for the human-review queue.
    for marker in ("bad-1", "bad-2", "bad-3"):
        assert marker in message
    assert "raw invoice text" in message


# ---------------------------------------------------------------------------
# (e) Merge logic: a HIGH-confidence Pass 1 field is not overwritten by the LLM.
# ---------------------------------------------------------------------------
def test_merge_preserves_high_confidence_pass1_fields():
    # Pass 1 confidently found vendor/amount/items but missed due_date, so we
    # fall to the LLM. The mock "hallucinates" a different vendor and amount.
    fields = {
        "vendor": ExtractedField("RealVendor", Confidence.HIGH),
        "amount": ExtractedField(999.0, Confidence.HIGH),
        "items": ExtractedField([LineItem(item="A", quantity=1, unit_price=999.0)], Confidence.HIGH),
    }
    mock = MockLLMClient(responses=[VALID_LLM_JSON])  # vendor "LLM Vendor", amount 1234.0
    data = _extract_with_llm("raw", fields=fields, llm=mock)

    # HIGH Pass 1 values win; the missing field is filled by the LLM at MEDIUM.
    assert data.vendor == "RealVendor"
    assert data.amount == 999.0
    assert data.due_date == date(2026, 5, 5)
    assert data.field_confidence["vendor"] is Confidence.HIGH
    assert data.field_confidence["amount"] is Confidence.HIGH
    assert data.field_confidence["due_date"] is Confidence.MEDIUM


# ---------------------------------------------------------------------------
# Model-level checks: the overall-confidence majority rule and audit serialisation.
# ---------------------------------------------------------------------------
def _make(**confidences) -> InvoiceData:
    return InvoiceData(
        vendor="V",
        amount=1.0,
        items=[LineItem(item="A", quantity=1, unit_price=1.0)],
        due_date=date(2026, 1, 1),
        invoice_id="INV-1",
        field_confidence=confidences,
    )


def test_overall_confidence_majority_rule():
    high = {f: Confidence.HIGH for f in ("vendor", "amount", "items", "due_date")}
    assert _make(**high).overall_confidence is Confidence.HIGH

    two_low = {"vendor": Confidence.LOW, "amount": Confidence.LOW,
               "items": Confidence.HIGH, "due_date": Confidence.MEDIUM}
    assert _make(**two_low).overall_confidence is Confidence.LOW

    mixed = {"vendor": Confidence.HIGH, "amount": Confidence.MEDIUM,
             "items": Confidence.MEDIUM, "due_date": Confidence.HIGH}
    assert _make(**mixed).overall_confidence is Confidence.MEDIUM


def test_model_dump_log_is_json_serialisable():
    import json

    log = _make(**{f: Confidence.HIGH for f in ("vendor", "amount", "items", "due_date")}).model_dump_log()
    json.dumps(log)  # must not raise
    assert log["due_date"] == "2026-01-01"
    assert log["overall_confidence"] == "high"
    assert log["field_confidence"]["vendor"] == "high"


# ---------------------------------------------------------------------------
# Robustness: an unsupported format must route to the LLM, not crash.
# ---------------------------------------------------------------------------
def test_unsupported_format_routes_to_llm():
    mock = MockLLMClient()
    data = extract(INVOICES / "invoice_1014.xml", llm_client=mock)
    assert isinstance(data, InvoiceData)
    assert mock.call_count == 1
