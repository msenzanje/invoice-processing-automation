"""
Test suite for the Phase 5 Payment Agent & Audit Log.

Covers, with no live I/O beyond ``tmp_path`` and no live API calls:

* (a) the writer — appends exactly one parseable JSON line, and *appends* (does not
      overwrite) on a second call;
* (b) each terminal node — driven with a constructed state and a monkeypatched audit
      path, the line is read back and its schema fields asserted per decision type
      (approved / rejected / needs_review);
* (c) the dead-invoice path — failed ingestion (no invoice_data, no approval_result)
      still writes a valid rejected record sourced from error_message;
* (d) the serialisation contract — ``AuditRecord.to_dict()`` is json.dumps-safe and
      carries the PROJECT_CONTEXT §5 fields;
* (e) payment_status is "success" only on the approved path and null elsewhere;
* (f) the swallow-but-signal failure path — a failed write returns an error string
      (never raises) and the node still terminates cleanly with WRITE_FAILED logged.

Timing assertions only check ``processing_time_ms`` is a non-negative int, never an
exact value.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from agents.payment import escalation_node, log_node, payment_node
from models.invoice_data import Confidence, InvoiceData, LineItem
from models.results import (
    ApprovalDecision,
    ApprovalResult,
    AuditRecord,
    FlagCode,
    ItemValidation,
    ReflectionTrace,
    ValidationFlag,
    ValidationResult,
)
from tools import audit
from tools.audit import elapsed_ms, write_audit_record


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _invoice(
    *,
    vendor: str = "Widgets Inc.",
    amount: float = 5000.0,
    invoice_id: str = "INV-TEST",
) -> InvoiceData:
    return InvoiceData(
        vendor=vendor,
        amount=amount,
        items=[LineItem(item="WidgetA", quantity=5, unit_price=250.0)],
        due_date=date(2026, 2, 1),
        invoice_id=invoice_id,
        field_confidence={n: Confidence.HIGH for n in ("vendor", "amount", "items", "due_date")},
    )


def _approval(
    decision: ApprovalDecision,
    *,
    reasoning: str = "looks fine",
    critique: str | None = None,
) -> ApprovalResult:
    trace = ReflectionTrace(critique=critique) if critique is not None else ReflectionTrace()
    return ApprovalResult(
        invoice_id="INV-TEST",
        decision=decision,
        reasoning=reasoning,
        route="full_loop",
        trace=trace,
    )


def _validation_with_error() -> ValidationResult:
    return ValidationResult(
        invoice_id="INV-TEST",
        item_validations=[
            ItemValidation(
                item="WidgetC",
                quantity=1,
                flags=[ValidationFlag.of(FlagCode.UNKNOWN_ITEM, "not stocked", item="WidgetC")],
            )
        ],
        invoice_flags=[ValidationFlag.of(FlagCode.ZERO_AMOUNT, "total is zero")],
    )


def _state(invoice_data, *, approval=None, validation=None, error_message=None) -> dict:
    return {
        "invoice_id": "INV-TEST",
        "filename": "t.txt",
        "file_format": "txt",
        "file_path": "t.txt",
        "raw_content": "",
        "parsed_data": {},
        "invoice_data": invoice_data,
        "processing_stage": "validated",
        "is_valid": True,
        "validation_errors": [],
        "validation_result": validation,
        "approval_result": approval,
        "inventory_updated": False,
        "db_record_id": None,
        "extraction_confidence": 0.9,
        "processing_log": ["validation: ok"],
        "error_message": error_message,
        "timestamp": "2026-01-01T00:00:00+00:00",
    }


def _read_lines(path) -> list[dict]:
    """Parse every JSON line in the audit file."""
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


@pytest.fixture
def audit_path(tmp_path, monkeypatch):
    """Point the default audit path at a temp file for the duration of a test."""
    target = tmp_path / "logs" / "audit.jsonl"
    monkeypatch.setattr(audit, "AUDIT_PATH", target)
    return target


# ---------------------------------------------------------------------------
# (a) The writer.
# ---------------------------------------------------------------------------
def test_writer_appends_one_parseable_line(tmp_path):
    target = tmp_path / "logs" / "audit.jsonl"
    record = AuditRecord(
        invoice_id="INV-TEST",
        timestamp="2026-01-01T00:00:00+00:00",
        vendor="Widgets Inc.",
        amount=5000.0,
        decision="approved",
        reasoning="ok",
        processing_time_ms=12,
    )
    assert write_audit_record(record, path=target) is None
    lines = _read_lines(target)
    assert len(lines) == 1
    assert lines[0]["invoice_id"] == "INV-TEST"


def test_writer_appends_rather_than_overwrites(tmp_path):
    target = tmp_path / "logs" / "audit.jsonl"
    record = AuditRecord(
        invoice_id="INV-TEST", timestamp="t", amount=1.0, decision="rejected",
        reasoning="x", processing_time_ms=0,
    )
    write_audit_record(record, path=target)
    write_audit_record(record, path=target)
    assert len(_read_lines(target)) == 2


def test_writer_creates_missing_parent_dir(tmp_path):
    target = tmp_path / "deep" / "nested" / "audit.jsonl"
    record = AuditRecord(
        invoice_id="X", timestamp="t", amount=0.0, decision="rejected",
        reasoning="x", processing_time_ms=0,
    )
    assert write_audit_record(record, path=target) is None
    assert target.exists()


# ---------------------------------------------------------------------------
# (b) Terminal nodes write the right schema per decision.
# ---------------------------------------------------------------------------
def test_payment_node_writes_approved_record(audit_path):
    state = _state(_invoice(), approval=_approval(ApprovalDecision.APPROVED, reasoning="pay it"))
    out = payment_node(state)

    assert out["processing_stage"] == "approved"
    [line] = _read_lines(audit_path)
    assert line["decision"] == "approved"
    assert line["vendor"] == "Widgets Inc."
    assert line["amount"] == 5000.0
    assert line["reasoning"] == "pay it"
    assert line["payment_status"] == "success"
    assert isinstance(line["processing_time_ms"], int) and line["processing_time_ms"] >= 0


def test_log_node_writes_rejected_record_with_flags_and_critique(audit_path):
    approval = _approval(ApprovalDecision.REJECTED, reasoning="stock short", critique="draft was too lenient")
    state = _state(_invoice(), approval=approval, validation=_validation_with_error())
    out = log_node(state)

    assert out["processing_stage"] == "rejected"
    [line] = _read_lines(audit_path)
    assert line["decision"] == "rejected"
    assert line["reasoning"] == "stock short"
    assert line["critique"] == "draft was too lenient"
    assert line["payment_status"] is None
    # flags is the flat list of flag codes (item-level then invoice-level).
    assert set(line["flags"]) == {"UNKNOWN_ITEM", "ZERO_AMOUNT"}


def test_escalation_node_writes_needs_review_record(audit_path):
    approval = _approval(ApprovalDecision.NEEDS_REVIEW, reasoning="draft and revised disagreed")
    state = _state(_invoice(), approval=approval)
    out = escalation_node(state)

    assert out["processing_stage"] == "needs_review"
    [line] = _read_lines(audit_path)
    assert line["decision"] == "needs_review"
    assert line["reasoning"] == "draft and revised disagreed"
    assert line["payment_status"] is None


def test_critique_is_null_when_loop_did_not_run(audit_path):
    # A fast-tracked approval has an empty ReflectionTrace -> critique null.
    state = _state(_invoice(), approval=_approval(ApprovalDecision.APPROVED))
    payment_node(state)
    [line] = _read_lines(audit_path)
    assert line["critique"] is None


# ---------------------------------------------------------------------------
# (c) The dead-invoice path.
# ---------------------------------------------------------------------------
def test_dead_invoice_writes_rejected_record_from_error(audit_path):
    # Failed ingestion: no invoice_data, no approval_result, error carried in state.
    state = _state(None, approval=None, error_message="EXTRACTION_FAILED: bad file")
    out = log_node(state)

    assert out["processing_stage"] == "rejected"
    [line] = _read_lines(audit_path)
    assert line["decision"] == "rejected"
    assert line["vendor"] is None          # honest null, not the invoice_id
    assert line["amount"] == 0.0
    assert line["critique"] is None
    assert line["flags"] == []
    assert "EXTRACTION_FAILED" in line["reasoning"]
    assert line["invoice_id"] == "INV-TEST"  # identity preserved here


# ---------------------------------------------------------------------------
# (d) Serialisation contract.
# ---------------------------------------------------------------------------
def test_audit_record_to_dict_is_json_serialisable():
    record = AuditRecord(
        invoice_id="INV-TEST",
        timestamp="2026-01-01T00:00:00+00:00",
        vendor="Widgets Inc.",
        amount=5000.0,
        decision="approved",
        reasoning="ok",
        critique=None,
        flags=["ZERO_AMOUNT"],
        payment_status="success",
        processing_time_ms=7,
    )
    encoded = json.dumps(record.to_dict())
    restored = json.loads(encoded)
    assert restored["flags"] == ["ZERO_AMOUNT"]
    # Every §5 field is present.
    assert set(restored) == {
        "invoice_id", "timestamp", "vendor", "amount", "decision",
        "reasoning", "critique", "flags", "payment_status", "processing_time_ms",
    }


# ---------------------------------------------------------------------------
# (e) payment_status is "success" only on the approved path.
# ---------------------------------------------------------------------------
def test_payment_status_only_on_approved_path(audit_path):
    payment_node(_state(_invoice(), approval=_approval(ApprovalDecision.APPROVED)))
    log_node(_state(_invoice(), approval=_approval(ApprovalDecision.REJECTED)))
    escalation_node(_state(_invoice(), approval=_approval(ApprovalDecision.NEEDS_REVIEW)))

    statuses = {line["decision"]: line["payment_status"] for line in _read_lines(audit_path)}
    assert statuses == {"approved": "success", "rejected": None, "needs_review": None}


# ---------------------------------------------------------------------------
# (f) Swallow-but-signal: a failed write never crashes the node.
# ---------------------------------------------------------------------------
def test_writer_returns_error_string_instead_of_raising(tmp_path):
    # A path whose parent is a *file*, not a directory, makes mkdir/open raise OSError.
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    target = blocker / "audit.jsonl"
    result = write_audit_record(
        AuditRecord(invoice_id="X", timestamp="t", amount=0.0, decision="rejected",
                    reasoning="x", processing_time_ms=0),
        path=target,
    )
    assert isinstance(result, str) and result  # error message, not None, and not raised


def test_node_logs_write_failed_and_still_terminates(tmp_path, monkeypatch):
    # Force the default-path write to fail; the node must survive and signal it.
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    monkeypatch.setattr(audit, "AUDIT_PATH", blocker / "audit.jsonl")

    out = payment_node(_state(_invoice(), approval=_approval(ApprovalDecision.APPROVED)))

    assert out["processing_stage"] == "approved"  # payment still completed
    assert any("audit: WRITE_FAILED" in entry for entry in out["processing_log"])


# ---------------------------------------------------------------------------
# Timing helper.
# ---------------------------------------------------------------------------
def test_elapsed_ms_is_non_negative_int():
    assert isinstance(elapsed_ms("2026-01-01T00:00:00+00:00"), int)
    assert elapsed_ms("2026-01-01T00:00:00+00:00") >= 0


def test_elapsed_ms_zero_on_bad_or_missing_timestamp():
    assert elapsed_ms("") == 0
    assert elapsed_ms("not-a-timestamp") == 0


def test_elapsed_ms_handles_naive_timestamp():
    # A timestamp with no tzinfo is treated as UTC rather than raising.
    assert elapsed_ms("2026-01-01T00:00:00") >= 0
