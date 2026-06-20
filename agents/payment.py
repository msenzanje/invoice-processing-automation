"""
Terminal nodes — the three branches an approved/rejected/escalated invoice lands in.

Each is a fail-forward graph node (state in, state out, never raises) that writes
one final audit record:

* :func:`payment_node`     — APPROVED -> call ``mock_payment`` and record the result.
* :func:`log_node`         — REJECTED (or a dead upstream invoice) -> record the
                             rejection and its reasoning.
* :func:`escalation_node`  — NEEDS_REVIEW -> record the invoice plus the full
                             reflection transcript to the human-review queue.

"""

from __future__ import annotations

import logging

from models.state import InvoiceState
from tools.payment_api import mock_payment

logger = logging.getLogger(__name__)


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
        return {**state, "processing_stage": "rejected", "processing_log": log}

    logger.info("log[%s]: rejected — %s", state.get("invoice_id"), result.reasoning)
    log.append(f"rejection_log: rejected — {result.reasoning}")
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
    return {**state, "processing_stage": "needs_review", "processing_log": log}


def _value(decision) -> str:
    """Render an optional ApprovalDecision as its string value for the log."""
    return decision.value if decision is not None else "none"
