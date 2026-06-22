"""
Terminal nodes — the three branches an approved/rejected/escalated invoice lands in.

Each is a fail-forward graph node (state in, state out, never raises) that writes
exactly one final audit record to ``logs/audit.jsonl``:

* :func:`payment_node`     — APPROVED -> call ``mock_payment``, record the result with
                             ``payment_status``.
* :func:`log_node`         — REJECTED (or a dead upstream invoice) -> record the
                             rejection and its reasoning, ``payment_status`` null.
* :func:`escalation_node`  — NEEDS_REVIEW -> record the invoice plus its reflection
                             reasoning for the human-review queue, ``payment_status`` null.

Record assembly is shared via :func:`_build_audit_record` so all three branches emit
one schema (PROJECT_CONTEXT §5) and there are no three drifting copies. The write
itself never raises: :func:`tools.audit.write_audit_record` returns an error string on
I/O failure, which the node surfaces into ``processing_log`` (swallow-but-signal) so a
failed write to the legally-significant trail is visible without crashing a payment.
"""

from __future__ import annotations

import logging
from typing import Optional

from models.results import AuditRecord
from models.state import InvoiceState
from tools.audit import elapsed_ms, write_audit_record
from tools.payment_api import mock_payment

logger = logging.getLogger(__name__)


def _build_audit_record(
    state: InvoiceState, decision: str, payment_status: Optional[str]
) -> AuditRecord:
    """Assemble the single audit record for a terminal branch.

    ``decision`` and ``payment_status`` are supplied by the calling node (never read
    from ``approval_result``) because the dead-invoice path has no approval decision
    at all. Everything else is pulled defensively from state: a missing
    ``approval_result`` (failed ingestion) falls back to ``error_message`` for
    reasoning and leaves ``critique`` null; a missing ``invoice_data`` leaves
    ``vendor`` null (identity stays in ``invoice_id``) and ``amount`` 0.0; a missing
    ``validation_result`` yields no flags.
    """
    data = state.get("invoice_data")
    approval = state.get("approval_result")
    validation = state.get("validation_result")

    if approval is not None:
        reasoning = approval.reasoning
        critique = approval.trace.critique
    else:
        reasoning = state.get("error_message") or "no approval decision was produced"
        critique = None

    flags = [flag.code.value for flag in validation.all_flags()] if validation is not None else []

    return AuditRecord(
        invoice_id=state.get("invoice_id"),
        timestamp=state.get("timestamp") or "",
        vendor=data.vendor if data is not None else None,
        amount=data.amount if data is not None else 0.0,
        decision=decision,
        reasoning=reasoning,
        critique=critique,
        flags=flags,
        payment_status=payment_status,
        processing_time_ms=elapsed_ms(state.get("timestamp") or ""),
    )


def _record_audit(state: InvoiceState, log: list[str], decision: str, payment_status: Optional[str]) -> None:
    """Build and write the audit record, appending a WRITE_FAILED note on I/O error.

    Mutates ``log`` in place. The write is swallow-but-signal: an error never
    propagates, but it lands in ``processing_log`` so the operator can see that the
    durable record for this invoice did not land.
    """
    record = _build_audit_record(state, decision, payment_status)
    error = write_audit_record(record)
    if error is not None:
        log.append(f"audit: WRITE_FAILED ({error})")


def payment_node(state: InvoiceState) -> InvoiceState:
    """Pay an approved invoice via ``mock_payment`` and record the outcome."""
    log = list(state.get("processing_log") or [])
    data = state.get("invoice_data")
    if data is None:  # defensive: router should never send a dead invoice here
        log.append("payment: skipped (no invoice_data)")
        return {**state, "processing_log": log}

    receipt = mock_payment(data.vendor, data.amount)
    logger.info("payment[%s]: %s txn=%s", data.invoice_id, receipt["status"], receipt["transaction_id"])
    log.append(f"payment: {receipt['status']} (txn={receipt['transaction_id']})")
    _record_audit(state, log, decision="approved", payment_status=receipt["status"])
    return {
        **state,
        "processing_stage": "approved",
        "db_record_id": None,
        "processing_log": log,
    }


def log_node(state: InvoiceState) -> InvoiceState:
    """Record a rejection (or a dead upstream invoice) with its reasoning."""
    log = list(state.get("processing_log") or [])
    result = state.get("approval_result")
    if result is None:
        # An invoice that never reached an approval decision (failed ingestion).
        reason = state.get("error_message") or "no approval decision was produced"
        logger.info("log[%s]: terminated without decision (%s)", state.get("invoice_id"), reason)
        log.append(f"rejection_log: terminated ({reason})")
    else:
        logger.info("log[%s]: rejected — %s", state.get("invoice_id"), result.reasoning)
        log.append(f"rejection_log: rejected — {result.reasoning}")
    _record_audit(state, log, decision="rejected", payment_status=None)
    return {**state, "processing_stage": "rejected", "processing_log": log}


def escalation_node(state: InvoiceState) -> InvoiceState:
    """Record a needs_review invoice plus its reflection transcript for a human."""
    log = list(state.get("processing_log") or [])
    result = state.get("approval_result")
    reasoning = result.reasoning if result is not None else "(no reasoning recorded)"
    logger.info("escalation[%s]: queued for human review — %s", state.get("invoice_id"), reasoning)

    if result is not None and result.trace.ran:
        log.append(
            "escalation: needs_review "
            f"(draft={_value(result.trace.draft_decision)}, "
            f"revised={_value(result.trace.revised_decision)})"
        )
    else:
        log.append(f"escalation: needs_review — {reasoning}")
    _record_audit(state, log, decision="needs_review", payment_status=None)
    return {**state, "processing_stage": "needs_review", "processing_log": log}


def _value(decision) -> str:
    """Render an optional ApprovalDecision as its string value for the log."""
    return decision.value if decision is not None else "none"
