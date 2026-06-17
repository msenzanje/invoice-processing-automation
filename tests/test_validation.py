"""
Test suite for the Phase 3 Validation Agent.

Covers: the four item-level checks (UNKNOWN_ITEM, ZERO_STOCK, STOCK_EXCEEDED,
NEGATIVE_QUANTITY), the three invoice-level checks (VENDOR_UNRECOGNIZED,
ZERO_AMOUNT, LOW_CONFIDENCE_EXTRACTION), the fail-forward contract (validation
never raises and never hard-rejects), and the graph node's skip-on-no-data path.

No live API calls and no contact with the real inventory.db — every test runs
against a fresh temporary database seeded by the ``db`` fixture.
"""

from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from agents.validation import run_validation, validate_invoice
from models.invoice_data import Confidence, InvoiceData, LineItem
from models.results import FlagCode, ValidationResult
from tools.db import get_connection, init_db

# Inventory the fixture seeds — mirrors setup_db.py's catalog.
_INVENTORY = (
    ("WidgetA", 15, 250.00, "widgets"),
    ("WidgetB", 10, 500.00, "widgets"),
    ("GadgetX", 5, 750.00, "gadgets"),
    ("FakeItem", 0, 0.00, "unknown"),
)
_APPROVED_VENDORS = ("Widgets Inc.", "Precision Parts Ltd.")


@pytest.fixture()
def db(tmp_path):
    """A temporary, fully seeded inventory database; yields a live connection."""
    db_path = tmp_path / "inventory.db"
    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.executemany(
            "INSERT INTO inventory (item, stock, unit_price, category) VALUES (?, ?, ?, ?)",
            _INVENTORY,
        )
        conn.executemany(
            "INSERT INTO vendors (vendor_name, approved, added_date) VALUES (?, 1, '2026-01-01')",
            [(name,) for name in _APPROVED_VENDORS],
        )
        conn.commit()
        yield conn


def _invoice(
    *,
    vendor: str = "Widgets Inc.",
    amount: float = 5000.0,
    items: list[LineItem] | None = None,
    confidence: dict[str, Confidence] | None = None,
    invoice_id: str = "INV-TEST",
) -> InvoiceData:
    """Build an InvoiceData with sensible, all-clean defaults."""
    return InvoiceData(
        vendor=vendor,
        amount=amount,
        items=items if items is not None else [LineItem(item="WidgetA", quantity=5, unit_price=250.0)],
        due_date=date(2026, 2, 1),
        invoice_id=invoice_id,
        field_confidence=confidence
        or {name: Confidence.HIGH for name in ("vendor", "amount", "items", "due_date")},
    )


def _codes(result: ValidationResult) -> set[FlagCode]:
    return {flag.code for flag in result.all_flags()}


# ---------------------------------------------------------------------------
# (a) The happy path — a fully valid invoice passes with zero flags.
# ---------------------------------------------------------------------------
def test_clean_invoice_passes(db):
    result = run_validation(_invoice(), db)

    assert result.passes is True
    assert result.has_errors is False
    assert result.all_flags() == []


# ---------------------------------------------------------------------------
# (b) Item-level checks.
# ---------------------------------------------------------------------------
def test_unknown_item_is_flagged(db):
    # WidgetC is not in the catalog (mirrors invoice_1016).
    data = _invoice(items=[LineItem(item="WidgetC", quantity=3, unit_price=350.0)])
    result = run_validation(data, db)

    assert FlagCode.UNKNOWN_ITEM in _codes(result)
    assert result.passes is False
    assert result.has_errors is True


def test_stock_exceeded_is_flagged(db):
    # GadgetX has stock 5; requesting 8 exceeds it (mirrors invoice_1005).
    data = _invoice(items=[LineItem(item="GadgetX", quantity=8, unit_price=750.0)])
    result = run_validation(data, db)

    assert FlagCode.STOCK_EXCEEDED in _codes(result)
    assert result.has_errors is True


def test_zero_stock_is_flagged_not_stock_exceeded(db):
    # FakeItem exists but has stock 0 -> ZERO_STOCK, and NOT STOCK_EXCEEDED.
    data = _invoice(items=[LineItem(item="FakeItem", quantity=100, unit_price=1000.0)])
    result = run_validation(data, db)

    codes = _codes(result)
    assert FlagCode.ZERO_STOCK in codes
    assert FlagCode.STOCK_EXCEEDED not in codes


def test_negative_quantity_is_flagged(db):
    # Mirrors invoice_1009's first line item (quantity -5).
    data = _invoice(items=[LineItem(item="WidgetA", quantity=-5, unit_price=250.0)])
    result = run_validation(data, db)

    assert FlagCode.NEGATIVE_QUANTITY in _codes(result)
    assert result.has_errors is True


def test_sufficient_stock_does_not_flag(db):
    # WidgetA stock is 15; requesting exactly 15 is allowed (boundary).
    data = _invoice(items=[LineItem(item="WidgetA", quantity=15, unit_price=250.0)])
    result = run_validation(data, db)

    assert result.passes is True


def test_flags_are_attached_to_the_right_item(db):
    data = _invoice(
        items=[
            LineItem(item="WidgetA", quantity=5, unit_price=250.0),  # clean
            LineItem(item="WidgetC", quantity=1, unit_price=10.0),  # unknown
        ]
    )
    result = run_validation(data, db)

    by_item = {iv.item: iv for iv in result.item_validations}
    assert by_item["WidgetA"].flags == []
    assert by_item["WidgetC"].flags[0].code is FlagCode.UNKNOWN_ITEM
    assert by_item["WidgetC"].flags[0].item == "WidgetC"


# ---------------------------------------------------------------------------
# (c) Invoice-level checks.
# ---------------------------------------------------------------------------
def test_unrecognized_vendor_is_a_warning(db):
    # "Fraudster LLC" is not an approved vendor (mirrors invoice_1003).
    result = run_validation(_invoice(vendor="Fraudster LLC"), db)

    flag = next(f for f in result.all_flags() if f.code is FlagCode.VENDOR_UNRECOGNIZED)
    assert flag.severity == "warning"
    assert result.has_errors is False  # a lone warning is not an error
    assert result.passes is False  # ...but it still does not "pass"


def test_approved_vendor_is_not_flagged(db):
    result = run_validation(_invoice(vendor="Widgets Inc."), db)
    assert FlagCode.VENDOR_UNRECOGNIZED not in _codes(result)


def test_blank_vendor_is_unrecognized(db):
    # invoice_1009 has an empty vendor name.
    result = run_validation(_invoice(vendor=""), db)
    assert FlagCode.VENDOR_UNRECOGNIZED in _codes(result)


@pytest.mark.parametrize("amount", [0.0, -250.0])
def test_non_positive_amount_is_flagged(db, amount):
    # invoice_1009 has total -250.00.
    result = run_validation(_invoice(amount=amount), db)
    assert FlagCode.ZERO_AMOUNT in _codes(result)


def test_low_confidence_extraction_is_flagged(db):
    data = _invoice(
        confidence={
            "vendor": Confidence.LOW,
            "amount": Confidence.HIGH,
            "items": Confidence.HIGH,
            "due_date": Confidence.MEDIUM,
        }
    )
    result = run_validation(data, db)
    assert FlagCode.LOW_CONFIDENCE_EXTRACTION in _codes(result)


# ---------------------------------------------------------------------------
# (d) Fail-forward: even a worst-case invoice yields a clean result, no raise.
# ---------------------------------------------------------------------------
def test_heavily_flagged_invoice_does_not_raise(db):
    data = _invoice(
        vendor="Fraudster LLC",
        amount=-100.0,
        items=[
            LineItem(item="FakeItem", quantity=-100, unit_price=1000.0),
            LineItem(item="GhostItem", quantity=999, unit_price=1.0),
            LineItem(item="GadgetX", quantity=8, unit_price=750.0),
        ],
        confidence={"vendor": Confidence.LOW, "amount": Confidence.LOW},
    )
    result = run_validation(data, db)

    assert isinstance(result, ValidationResult)
    assert result.passes is False
    assert result.has_errors is True
    # Sanity: a broad spread of distinct codes was raised.
    assert len(_codes(result)) >= 4


def test_empty_items_list_does_not_raise(db):
    result = run_validation(_invoice(items=[]), db)

    assert isinstance(result, ValidationResult)
    assert result.item_validations == []
    assert result.passes is True  # clean vendor + positive amount, no items


# ---------------------------------------------------------------------------
# (e) Serialisation contract (audit trail).
# ---------------------------------------------------------------------------
def test_to_dict_is_json_serialisable(db):
    import json

    data = _invoice(items=[LineItem(item="WidgetC", quantity=1, unit_price=10.0)])
    payload = run_validation(data, db).to_dict()

    encoded = json.dumps(payload)  # raises if anything is non-serialisable
    assert '"UNKNOWN_ITEM"' in encoded
    assert payload["passes"] is False


# ---------------------------------------------------------------------------
# (f) Graph node behaviour (state in -> state out).
# ---------------------------------------------------------------------------
def _base_state(invoice_data) -> dict:
    return {
        "invoice_id": "INV-TEST",
        "filename": "t.txt",
        "file_format": "txt",
        "file_path": "t.txt",
        "raw_content": "",
        "parsed_data": {},
        "invoice_data": invoice_data,
        "processing_stage": "extracted",
        "is_valid": None,
        "validation_errors": [],
        "validation_result": None,
        "inventory_updated": False,
        "db_record_id": None,
        "extraction_confidence": 0.9,
        "processing_log": ["ingestion: ok"],
        "error_message": None,
        "timestamp": "2026-01-01T00:00:00+00:00",
    }


def test_node_skips_when_no_invoice_data():
    # Mirrors a failed ingestion: invoice_data is None -> validation no-ops.
    out = validate_invoice(_base_state(None))

    assert out.get("validation_result") is None
    assert "validation: skipped (no invoice_data)" in out["processing_log"]


def test_node_embeds_result_and_sets_is_valid(monkeypatch, tmp_path, db):
    # Point the agent's DB at the seeded temp database for this run.
    db_path = tmp_path / "inventory.db"
    monkeypatch.setattr("agents.validation.init_db", lambda: None)
    monkeypatch.setattr(
        "agents.validation.get_connection", lambda: get_connection(db_path)
    )

    out = validate_invoice(_base_state(_invoice()))

    assert isinstance(out["validation_result"], ValidationResult)
    assert out["is_valid"] is True
    assert out["processing_stage"] == "validated"


def test_node_db_error_is_failed_forward(monkeypatch):
    def _boom():
        raise sqlite3.OperationalError("disk gone")

    monkeypatch.setattr("agents.validation.init_db", lambda: None)
    monkeypatch.setattr("agents.validation.get_connection", _boom)

    out = validate_invoice(_base_state(_invoice()))

    assert out["error_message"].startswith("VALIDATION_DB_ERROR")
    assert "validation: DB_ERROR" in out["processing_log"]
