"""
Validation Agent — checks extracted invoice data against inventory.db and emits a
typed :class:`~models.results.ValidationResult`.

Philosophy is *flag, don't reject* (PROJECT_CONTEXT §1, §3). Validation never
hard-fails an invoice and never raises into the graph: even a heavily flagged
invoice (unknown items, negative quantities, unrecognised vendor) produces a clean
result object that the approval agent reasons over. The checks performed:

* per line item — UNKNOWN_ITEM, ZERO_STOCK, STOCK_EXCEEDED, NEGATIVE_QUANTITY
* per invoice   — VENDOR_UNRECOGNIZED, ZERO_AMOUNT, LOW_CONFIDENCE_EXTRACTION

The public graph node is :func:`validate_invoice`; the pure, DB-injected core is
:func:`run_validation` (tested directly against a temp database).
"""

from __future__ import annotations

import logging
import sqlite3

from models.invoice_data import CONFIDENCE_FIELDS, Confidence, InvoiceData, LineItem
from models.results import FlagCode, ItemValidation, ValidationFlag, ValidationResult
from models.state import InvoiceState
from tools.db import get_connection, get_inventory_item, init_db, is_vendor_approved

logger = logging.getLogger(__name__)


# =============================================================================
# Per-item checks
# =============================================================================
def _validate_item(item: LineItem, conn: sqlite3.Connection, invoice_id: str) -> ItemValidation:
    """Run every item-scoped check and collect the resulting flags."""
    flags: list[ValidationFlag] = []
    _check_negative_quantity(item, flags)
    _check_against_inventory(item, conn, invoice_id, flags)
    return ItemValidation(item=item.item, quantity=item.quantity, flags=flags)


def _check_negative_quantity(item: LineItem, flags: list[ValidationFlag]) -> None:
    """A negative line quantity is impossible on a real purchase invoice."""
    if item.quantity < 0:
        flags.append(
            ValidationFlag.of(
                FlagCode.NEGATIVE_QUANTITY,
                f"Line item '{item.item}' has negative quantity {item.quantity}.",
                item=item.item,
            )
        )


def _check_against_inventory(
    item: LineItem, conn: sqlite3.Connection, invoice_id: str, flags: list[ValidationFlag]
) -> None:
    """Inventory existence and stock-sufficiency checks for one line item."""
    record = get_inventory_item(conn, item.item)
    logger.debug("validation[%s]: item=%s inventory=%s", invoice_id, item.item, record)

    if record is None:
        flags.append(
            ValidationFlag.of(
                FlagCode.UNKNOWN_ITEM,
                f"Item '{item.item}' is not in the inventory catalog.",
                item=item.item,
            )
        )
        return  # No stock to compare against once the item is unknown.

    if record.stock == 0:
        flags.append(
            ValidationFlag.of(
                FlagCode.ZERO_STOCK,
                f"Item '{item.item}' is cataloged but has zero stock.",
                item=item.item,
            )
        )
        return  # ZERO_STOCK supersedes STOCK_EXCEEDED — don't double-flag.

    if item.quantity > record.stock:
        flags.append(
            ValidationFlag.of(
                FlagCode.STOCK_EXCEEDED,
                f"Requested {item.quantity} of '{item.item}' but only {record.stock} in stock.",
                item=item.item,
            )
        )


# =============================================================================
# Per-invoice checks
# =============================================================================
def _validate_invoice_level(
    data: InvoiceData, conn: sqlite3.Connection
) -> list[ValidationFlag]:
    """Run every invoice-scoped check and collect the resulting flags."""
    flags: list[ValidationFlag] = []
    _check_vendor(data, conn, flags)
    _check_amount(data, flags)
    _check_extraction_confidence(data, flags)
    return flags


def _check_vendor(data: InvoiceData, conn: sqlite3.Connection, flags: list[ValidationFlag]) -> None:
    """Vendor must be a known, approved vendor (warning if not)."""
    if not is_vendor_approved(conn, data.vendor):
        vendor_label = data.vendor.strip() if data.vendor and data.vendor.strip() else "(blank)"
        flags.append(
            ValidationFlag.of(
                FlagCode.VENDOR_UNRECOGNIZED,
                f"Vendor '{vendor_label}' is not in the approved-vendor list.",
            )
        )


def _check_amount(data: InvoiceData, flags: list[ValidationFlag]) -> None:
    """A zero or negative invoice total is suspect (warning)."""
    if data.amount <= 0:
        flags.append(
            ValidationFlag.of(
                FlagCode.ZERO_AMOUNT,
                f"Invoice total is non-positive (${data.amount:,.2f}).",
            )
        )


def _check_extraction_confidence(data: InvoiceData, flags: list[ValidationFlag]) -> None:
    """Surface any required field the ingestion agent extracted with low trust."""
    low_fields = [
        name
        for name in CONFIDENCE_FIELDS
        if data.field_confidence.get(name) is Confidence.LOW
    ]
    if low_fields:
        flags.append(
            ValidationFlag.of(
                FlagCode.LOW_CONFIDENCE_EXTRACTION,
                f"Low-confidence extraction on: {', '.join(low_fields)}.",
            )
        )


# =============================================================================
# Pure core (DB injected) — directly unit-testable
# =============================================================================
def run_validation(data: InvoiceData, conn: sqlite3.Connection) -> ValidationResult:
    """Validate ``data`` against the catalog in ``conn``; never raises.

    Returns a :class:`ValidationResult` with per-item and invoice-level flags.
    The connection is injected so tests can point it at a temp database and so
    the graph node owns connection lifetime.
    """
    invoice_id = data.invoice_id or "unknown"
    item_validations = [_validate_item(item, conn, invoice_id) for item in data.items]
    invoice_flags = _validate_invoice_level(data, conn)
    result = ValidationResult(
        invoice_id=data.invoice_id,
        item_validations=item_validations,
        invoice_flags=invoice_flags,
    )
    logger.info(
        "validation[%s]: %d flag(s), passes=%s, errors=%s",
        invoice_id, len(result.all_flags()), result.passes, result.has_errors,
    )
    return result


# =============================================================================
# Graph node (state in, state out) — fail-forward
# =============================================================================
def validate_invoice(state: InvoiceState) -> InvoiceState:
    """LangGraph node: validate ``state['invoice_data']`` and embed the result.

    Fail-forward: if ingestion produced no invoice_data, or
    the database is unreachable, the node records the situation in the state and
    returns rather than raising into the graph.
    """
    log = list(state.get("processing_log") or [])
    data = state.get("invoice_data")
    if data is None:
        log.append("validation: skipped (no invoice_data)")
        return {**state, "processing_log": log}

    try:
        init_db()
        with get_connection() as conn:
            result = run_validation(data, conn)
    except sqlite3.Error as exc:
        logger.error("Validation DB error for %s: %s", state.get("invoice_id"), exc)
        log.append("validation: DB_ERROR")
        return {
            **state,
            "processing_stage": "validated",
            "error_message": f"VALIDATION_DB_ERROR: {exc}",
            "processing_log": log,
        }

    log.append(f"validation: {len(result.all_flags())} flag(s) (errors={result.has_errors})")
    return {
        **state,
        "validation_result": result,
        "is_valid": result.passes,
        "validation_errors": [flag.message for flag in result.error_flags()],
        "processing_stage": "validated",
        "processing_log": log,
    }
