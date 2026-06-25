"""
Test suite for the Phase 4 Approval Agent.

Covers, all with MockLLMClient (zero live API calls, consistent with the other
suites):

* (a) the deterministic gate's four routes — fast-approve, auto-review (vendor),
      full-loop (error flag / over-threshold / warning-only / mid-range);
* (b) the reflection loop with scripted responses — exactly three LLM calls,
      transcript captured, malformed-JSON self-correction;
* (c) the resolver — agree->commit (approve and reject), disagree->escalate;
* (d) the node's fail-forward guards — no invoice_data, LLM transport failure,
      and a clean fast-approve succeeding with no client at all;
* (e) ApprovalResult serialisation for the audit trail.
"""

from __future__ import annotations

import json
from datetime import date

import httpx
import pytest

from agents.approval import (
    APPROVAL_THRESHOLD_HIGH,
    APPROVAL_THRESHOLD_LOW,
    Route,
    _classify,
    _resolve,
    _run_reflection,
    approve_invoice,
    run_approval,
)
from models.invoice_data import Confidence, InvoiceData, LineItem
from models.results import (
    ApprovalDecision,
    ApprovalResult,
    FlagCode,
    ItemValidation,
    ReflectionTrace,
    ValidationFlag,
    ValidationResult,
)
from tools.llm import MockLLMClient


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _invoice(
    *,
    vendor: str = "Widgets Inc.",
    amount: float = 5000.0,
    items: list[LineItem] | None = None,
    confidence: dict[str, Confidence] | None = None,
    invoice_id: str = "INV-TEST",
) -> InvoiceData:
    """An InvoiceData with all-clean, high-confidence defaults."""
    return InvoiceData(
        vendor=vendor,
        amount=amount,
        items=items if items is not None else [LineItem(item="WidgetA", quantity=5, unit_price=250.0)],
        due_date=date(2026, 2, 1),
        invoice_id=invoice_id,
        field_confidence=confidence
        or {name: Confidence.HIGH for name in ("vendor", "amount", "items", "due_date")},
    )


def _clean() -> ValidationResult:
    return ValidationResult(invoice_id="INV-TEST")


def _with_flags(*flags: ValidationFlag) -> ValidationResult:
    """A validation result carrying the given invoice-level flags."""
    return ValidationResult(invoice_id="INV-TEST", invoice_flags=list(flags))


def _error_result() -> ValidationResult:
    """A result with an error-severity item flag (UNKNOWN_ITEM)."""
    return ValidationResult(
        invoice_id="INV-TEST",
        item_validations=[
            ItemValidation(
                item="WidgetC",
                quantity=1,
                flags=[ValidationFlag.of(FlagCode.UNKNOWN_ITEM, "not stocked", item="WidgetC")],
            )
        ],
    )


def _decision_json(decision: str, reasoning: str = "because") -> str:
    return json.dumps({"decision": decision, "reasoning": reasoning})


def _loop_script(draft: str, revised: str, critique: str = "A skeptical note.") -> list[str]:
    """The three scripted responses one full loop consumes, in call order."""
    return [_decision_json(draft), critique, _decision_json(revised)]


def _base_state(invoice_data, validation=None) -> dict:
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
        "approval_result": None,
        "inventory_updated": False,
        "db_record_id": None,
        "extraction_confidence": 0.9,
        "processing_log": ["validation: ok"],
        "error_message": None,
        "timestamp": "2026-01-01T00:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# (a) The deterministic gate.
# ---------------------------------------------------------------------------
def test_gate_fast_approves_clean_small_high_confidence():
    route = _classify(_invoice(amount=APPROVAL_THRESHOLD_LOW - 1), _clean())
    assert route is Route.FAST_APPROVE


def test_gate_does_not_fast_approve_at_the_low_threshold():
    # Boundary: exactly the low threshold is NOT "below" it.
    route = _classify(_invoice(amount=APPROVAL_THRESHOLD_LOW), _clean())
    assert route is Route.FULL_LOOP


def test_gate_unrecognized_vendor_runs_the_loop():
    # An unknown vendor is only a warning: it no longer auto-escalates. It carries a
    # flag (so it can't fast-track) and falls through to the loop, where the LLM
    # judges the invoice's legitimacy.
    result = _with_flags(ValidationFlag.of(FlagCode.VENDOR_UNRECOGNIZED, "unknown vendor"))
    route = _classify(_invoice(amount=500.0, vendor="New Vendor LLC"), result)
    assert route is Route.FULL_LOOP


def test_gate_unrecognized_vendor_does_not_short_circuit_error_or_threshold():
    # With the vendor escalation gone, an unknown vendor alongside an error flag and a
    # huge amount still simply routes to the loop — nothing overrides it.
    result = ValidationResult(
        invoice_id="INV-TEST",
        item_validations=_error_result().item_validations,
        invoice_flags=[ValidationFlag.of(FlagCode.VENDOR_UNRECOGNIZED, "unknown")],
    )
    route = _classify(_invoice(amount=99999.0, vendor="New Vendor LLC"), result)
    assert route is Route.FULL_LOOP


def test_gate_error_flag_forces_full_loop():
    route = _classify(_invoice(amount=500.0), _error_result())
    assert route is Route.FULL_LOOP


def test_gate_over_high_threshold_forces_full_loop():
    route = _classify(_invoice(amount=APPROVAL_THRESHOLD_HIGH + 1), _clean())
    assert route is Route.FULL_LOOP


def test_gate_warning_only_goes_to_full_loop():
    # A lone warning (ZERO_AMOUNT) is not clean -> not fast-track, no vendor flag.
    result = _with_flags(ValidationFlag.of(FlagCode.ZERO_AMOUNT, "total is zero"))
    route = _classify(_invoice(amount=500.0), result)
    assert route is Route.FULL_LOOP


def test_gate_small_clean_but_low_confidence_goes_to_full_loop():
    data = _invoice(amount=500.0, confidence={"vendor": Confidence.LOW, "amount": Confidence.LOW})
    assert _classify(data, _clean()) is Route.FULL_LOOP


# ---------------------------------------------------------------------------
# (b) The reflection loop.
# ---------------------------------------------------------------------------
def test_loop_makes_exactly_three_calls_and_captures_transcript():
    llm = MockLLMClient(responses=_loop_script("approved", "approved", critique="Looks risky."))
    trace = _run_reflection(_invoice(amount=50000.0), _clean(), llm)

    assert llm.call_count == 3
    assert trace.draft_decision is ApprovalDecision.APPROVED
    assert trace.revised_decision is ApprovalDecision.APPROVED
    assert trace.critique == "Looks risky."
    assert trace.ran is True


def test_loop_self_corrects_on_malformed_draft_json():
    # First draft response is junk; the retry returns valid JSON. The extra call
    # shifts the critique/revise responses by one, so script accordingly.
    llm = MockLLMClient(
        responses=[
            "not json at all",  # draft attempt 1 -> fails, re-prompts
            _decision_json("rejected", "bad invoice"),  # draft attempt 2 -> ok
            "A critique.",  # critique pass
            _decision_json("rejected", "still bad"),  # revise pass
        ]
    )
    trace = _run_reflection(_invoice(amount=50000.0), _error_result(), llm)
    assert trace.draft_decision is ApprovalDecision.REJECTED
    assert trace.revised_decision is ApprovalDecision.REJECTED


def test_loop_raises_when_draft_never_parses():
    llm = MockLLMClient(responses=["nope"])  # repeats forever -> never parses
    with pytest.raises(ValueError):
        _run_reflection(_invoice(amount=50000.0), _clean(), llm)


# ---------------------------------------------------------------------------
# (c) The resolver.
# ---------------------------------------------------------------------------
def test_resolver_agree_approve_commits_approved():
    trace = ReflectionTrace(
        draft_decision=ApprovalDecision.APPROVED, draft_reasoning="ok",
        critique="c", revised_decision=ApprovalDecision.APPROVED, revised_reasoning="still ok",
    )
    result = _resolve(Route.FULL_LOOP, _invoice(), _clean(), trace)
    assert result.decision is ApprovalDecision.APPROVED
    assert result.reasoning == "still ok"


def test_resolver_agree_reject_commits_rejected():
    trace = ReflectionTrace(
        draft_decision=ApprovalDecision.REJECTED, draft_reasoning="bad",
        critique="c", revised_decision=ApprovalDecision.REJECTED, revised_reasoning="still bad",
    )
    result = _resolve(Route.FULL_LOOP, _invoice(), _clean(), trace)
    assert result.decision is ApprovalDecision.REJECTED


def test_resolver_disagreement_escalates_with_both_passes_visible():
    trace = ReflectionTrace(
        draft_decision=ApprovalDecision.APPROVED, draft_reasoning="looked fine",
        critique="The VP flagged the vendor.",
        revised_decision=ApprovalDecision.REJECTED, revised_reasoning="reversing on fraud",
    )
    result = _resolve(Route.FULL_LOOP, _invoice(), _clean(), trace)
    assert result.decision is ApprovalDecision.NEEDS_REVIEW
    # Both passes and the critique are surfaced for the human reviewer.
    assert "approved" in result.reasoning and "rejected" in result.reasoning
    assert "The VP flagged the vendor." in result.reasoning


def test_resolver_fast_approve_short_circuits_without_trace():
    result = _resolve(Route.FAST_APPROVE, _invoice(amount=500.0), _clean(), ReflectionTrace())
    assert result.decision is ApprovalDecision.APPROVED
    assert result.trace.ran is False


# ---------------------------------------------------------------------------
# run_approval — the pure core glued together.
# ---------------------------------------------------------------------------
def test_run_approval_full_loop_reject_end_to_end():
    llm = MockLLMClient(responses=_loop_script("rejected", "rejected"))
    result = run_approval(_invoice(amount=50000.0), _error_result(), llm)
    assert result.decision is ApprovalDecision.REJECTED
    assert result.route == Route.FULL_LOOP.value
    assert result.trace.ran is True


def test_run_approval_fast_approve_never_calls_llm():
    llm = MockLLMClient(responses=["should not be used"])
    result = run_approval(_invoice(amount=500.0), _clean(), llm)
    assert result.decision is ApprovalDecision.APPROVED
    assert llm.call_count == 0


# ---------------------------------------------------------------------------
# (d) Graph node behaviour (state in -> state out), fail-forward.
# ---------------------------------------------------------------------------
def test_node_skips_when_no_invoice_data():
    out = approve_invoice(_base_state(None))
    assert out.get("approval_result") is None
    assert "approval: skipped (no invoice_data)" in out["processing_log"]


def test_node_fast_approves_with_no_client_and_no_key(monkeypatch):
    # No injected client; the gate fast-approves, so get_llm_client must never run.
    def _boom():
        raise AssertionError("get_llm_client should not be called on a fast-approve")

    monkeypatch.setattr("agents.approval.get_llm_client", _boom)
    out = approve_invoice(_base_state(_invoice(amount=500.0), _clean()))

    assert out["approval_result"].decision is ApprovalDecision.APPROVED
    assert out["processing_stage"] == "approved"


def test_node_full_loop_uses_injected_client():
    llm = MockLLMClient(responses=_loop_script("approved", "approved"))
    out = approve_invoice(_base_state(_invoice(amount=50000.0), _clean()), llm_client=llm)

    assert out["approval_result"].decision is ApprovalDecision.APPROVED
    assert out["processing_stage"] == "approved"
    assert llm.call_count == 3


def test_node_llm_transport_failure_fails_forward_to_review():
    class _Boom:
        def complete(self, prompt: str) -> str:
            raise httpx.ConnectError("model unreachable")

    out = approve_invoice(_base_state(_invoice(amount=50000.0), _error_result()), llm_client=_Boom())

    assert out["approval_result"].decision is ApprovalDecision.NEEDS_REVIEW
    assert out["processing_stage"] == "needs_review"
    assert "approval: LLM_ERROR -> needs_review" in out["processing_log"]


def test_node_missing_validation_result_is_tolerated():
    # validation_result None -> treated as empty; a clean small invoice still routes.
    out = approve_invoice(_base_state(_invoice(amount=500.0), None))
    assert out["approval_result"].decision is ApprovalDecision.APPROVED


# ---------------------------------------------------------------------------
# (e) Serialisation contract (audit trail).
# ---------------------------------------------------------------------------
def test_to_dict_is_json_serialisable():
    llm = MockLLMClient(responses=_loop_script("approved", "rejected"))  # disagree
    result = run_approval(_invoice(amount=50000.0), _clean(), llm)
    payload = result.to_dict()

    encoded = json.dumps(payload)  # raises if anything is non-serialisable
    assert payload["decision"] == "needs_review"
    assert payload["trace"]["draft_decision"] == "approved"
    assert payload["trace"]["revised_decision"] == "rejected"
    assert '"needs_review"' in encoded
