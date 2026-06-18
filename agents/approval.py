"""
Approval Agent — turns a validated invoice into a committed, defensible decision.

This is the most agentic part of the pipeline. Its design has two layers:

* A **deterministic gate** (:func:`_classify`) applies the hard policy rules 
  *before* a token is spent: an unrecognised vendor
  always escalates; any error-severity flag or an amount over $10K always earns the
  full critique loop; a clean sub-$1K high-confidence invoice fast-tracks. Policy is
  code, not LLM judgment — a hallucinating model can never override these.

* A **reflection loop** (:func:`_run_reflection`) runs only for the genuinely
  ambiguous invoices the gate routes to ``FULL_LOOP``. The LLM drafts a decision,
  then critiques its own draft from a skeptical-VP stance, then revises. If the
  draft and the revision agree, that decision is committed; if they disagree, the
  invoice escalates to a human — the system admits uncertainty rather than guessing.

The public graph node is :func:`approve_invoice`; it is fail-forward like every
other node — an unreachable LLM yields ``NEEDS_REVIEW`` (escalate to a human), never
a crash and never a silent auto-approve.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Optional

import httpx
from pydantic import BaseModel, ValidationError

from models.invoice_data import Confidence, InvoiceData
from models.results import (
    ApprovalDecision,
    ApprovalResult,
    ReflectionTrace,
    ValidationResult,
)
from models.state import InvoiceState
from tools.llm import LLMProtocol, get_llm_client, strip_json

logger = logging.getLogger(__name__)

# A draft/revision pass may emit malformed JSON; re-prompt this many times before
# giving up on that pass. Mirrors the ingestion self-correction loop.
MAX_PARSE_RETRIES = 2


# --- Policy thresholds --------------------------------------------------------
# Env-driven with the documented defaults (PROJECT_CONTEXT §9), mirroring how
# tools/llm.py reads its configuration. A policy change is a one-line env override.
APPROVAL_THRESHOLD_HIGH = float(os.environ.get("APPROVAL_THRESHOLD_HIGH", "10000"))
APPROVAL_THRESHOLD_LOW = float(os.environ.get("APPROVAL_THRESHOLD_LOW", "1000"))


class Route(str, Enum):
    """The deterministic gate's verdict — *which* path resolves the invoice.

    Only ``FULL_LOOP`` invokes the LLM. The other two are resolved by policy alone
    (and recorded on the :class:`~models.results.ApprovalResult` for the audit log).
    Note there is no ``AUTO_REJECT``: no flag in the taxonomy means "fraud", so
    rejection is never a deterministic verdict — it can only emerge from the loop
    when the draft and revised passes agree to reject.
    """

    FAST_APPROVE = "fast_approve"  # clean, small, high-confidence -> approve, no LLM
    AUTO_REVIEW = "auto_review"  # unrecognised vendor -> escalate, no LLM
    FULL_LOOP = "full_loop"  # ambiguous -> run the draft->critique->revise loop


# Deterministic rules gate (pre-LLM router)
def _classify(data: InvoiceData, validation: ValidationResult) -> Route:
    """Apply the hard routing rules (PROJECT_CONTEXT §5) in priority order.

    Pure and I/O-free, so every branch is unit-testable with constructed fixtures.
    Priority matters: the vendor-escalation rule is checked before the
    error/threshold rules so an unrecognised vendor always lands in human review,
    even on an otherwise loop-worthy invoice.
    """
    from models.results import FlagCode

    flags = validation.all_flags()

    # Rule 1 — any VENDOR_UNRECOGNIZED flag -> always needs_review (escalate).
    if any(flag.code is FlagCode.VENDOR_UNRECOGNIZED for flag in flags):
        return Route.AUTO_REVIEW

    # Rule 2 — any error-severity flag -> full critique loop, regardless of amount.
    if validation.has_errors:
        return Route.FULL_LOOP

    # Rule 3 — amount over the high threshold -> always the full critique loop.
    if data.amount > APPROVAL_THRESHOLD_HIGH:
        return Route.FULL_LOOP

    # Rule 4 — clean, small, high-confidence -> fast-track approval (no LLM).
    if (
        not flags
        and data.amount < APPROVAL_THRESHOLD_LOW
        and data.overall_confidence is Confidence.HIGH
    ):
        return Route.FAST_APPROVE

    # Everything else (e.g. warning-only, or mid-range amount) -> full loop.
    return Route.FULL_LOOP


# Draft -> critique -> revise reflection loop
class _DecisionPass(BaseModel):
    """The structured shape the draft and revise passes must return.

    Only ``approved`` / ``rejected`` / ``needs_review`` are accepted for
    ``decision`` (enforced by the :class:`ApprovalDecision` enum), so a chatty model
    cannot smuggle a fourth verdict through. ``reasoning`` is free text.
    """

    decision: ApprovalDecision
    reasoning: str


def _summarize_invoice(data: InvoiceData, validation: ValidationResult) -> str:
    """Render the full invoice context — fields, flags, confidences — as prompt text.

    This is what every pass reasons over. Built with explicit loops (not
    comprehensions) for readability, per the project style guide.
    """
    lines = [
        f"Invoice ID: {data.invoice_id or '(none)'}",
        f"Vendor: {data.vendor or '(blank)'}",
        f"Amount: ${data.amount:,.2f}",
        f"Due date: {data.due_date.isoformat()}",
        f"Overall extraction confidence: {data.overall_confidence.value}",
        "",
        "Line items:",
    ]
    if data.items:
        for item in data.items:
            lines.append(
                f"  - {item.item}: qty={item.quantity} @ ${item.unit_price:,.2f}"
            )
    else:
        lines.append("  (none extracted)")

    lines.append("")
    lines.append("Per-field extraction confidence:")
    for name, rating in data.field_confidence.items():
        lines.append(f"  - {name}: {rating.value}")

    lines.append("")
    flags = validation.all_flags()
    if flags:
        lines.append("Validation flags raised:")
        for flag in flags:
            scope = f" (item: {flag.item})" if flag.item else ""
            lines.append(f"  - [{flag.severity.upper()}] {flag.code.value}{scope}: {flag.message}")
    else:
        lines.append("Validation flags raised: none")
    return "\n".join(lines)


_DRAFT_INSTRUCTIONS = """You are an accounts-payable analyst reviewing an invoice for payment.
Decide whether to APPROVE it for payment, REJECT it, or send it for human REVIEW.

Weigh the validation flags (error-severity flags are serious), the extraction
confidence, the vendor, and whether the amounts and quantities are plausible. Be
alert to fraud signals: unrecognised vendors, urgent payment pressure, impossible
quantities, or items that do not exist in inventory.

Return ONLY a single JSON object — no markdown, no commentary:
{
  "decision": "approved" | "rejected" | "needs_review",
  "reasoning": "<2-4 sentences justifying the decision, citing specific flags or values>"
}

Invoice under review:
---
{invoice}
---"""


_CRITIQUE_INSTRUCTIONS = """You are a skeptical VP of Finance reviewing an analyst's recommendation
on an invoice. Your job is to find what the analyst got wrong or under-weighted.

Identify unaddressed risks, edge cases, or validation flags the analyst may have
discounted. Be adversarial — assume the invoice could be fraudulent or erroneous
until proven otherwise. Do NOT make a decision yourself; only surface the risks the
analyst should reconsider. Respond in 2-5 sentences of plain prose.

The invoice:
---
{invoice}
---

The analyst's draft recommendation:
Decision: {draft_decision}
Reasoning: {draft_reasoning}"""


_REVISE_INSTRUCTIONS = """You are the same accounts-payable analyst. A skeptical VP has critiqued
your draft recommendation. Reconsider in light of the critique and produce your
FINAL decision. You may keep your original decision if the critique does not change
your assessment, or change it if the VP raised a valid risk.

Return ONLY a single JSON object — no markdown, no commentary:
{
  "decision": "approved" | "rejected" | "needs_review",
  "reasoning": "<2-4 sentences; if you changed your decision, say explicitly why>"
}

Invoice under review:
---
{invoice}
---

Your draft recommendation:
Decision: {draft_decision}
Reasoning: {draft_reasoning}

The VP's critique:
{critique}"""


def _parse_decision_pass(llm: LLMProtocol, prompt: str, pass_name: str) -> _DecisionPass:
    """Call the LLM and parse a :class:`_DecisionPass`, self-correcting on bad JSON.

    Re-prompts up to :data:`MAX_PARSE_RETRIES` times with the validation error
    appended (the ingestion-agent pattern). Raises ``ValueError`` if every attempt
    fails to produce parseable, schema-valid JSON — the caller (the node) converts
    that into a fail-forward ``NEEDS_REVIEW``.
    """
    current = prompt
    last_error = ""
    for attempt in range(MAX_PARSE_RETRIES + 1):
        logger.debug("approval %s pass: LLM attempt %d", pass_name, attempt + 1)
        response = llm.complete(current)
        try:
            return _DecisionPass.model_validate_json(strip_json(response))
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            logger.warning("approval %s pass attempt %d failed: %s", pass_name, attempt + 1, last_error)
            current = (
                f"{prompt}\n\nYour previous response was:\n{response}\n\n"
                f"It failed validation with this error:\n{last_error}\n\n"
                "Return a corrected JSON object. Output ONLY the JSON."
            )
    raise ValueError(f"approval {pass_name} pass: unparseable after {MAX_PARSE_RETRIES + 1} attempts: {last_error}")


def _run_reflection(
    data: InvoiceData, validation: ValidationResult, llm: LLMProtocol
) -> ReflectionTrace:
    """Run the three-pass draft -> critique -> revise cycle and capture the transcript.

    Each pass is recorded on the returned :class:`ReflectionTrace` so the audit log
    shows exactly how the model reasoned and where the revision diverged from the
    draft. The critique pass returns free text (no decision), so it is not parsed.
    """
    invoice = _summarize_invoice(data, validation)

    draft = _parse_decision_pass(
        llm, _DRAFT_INSTRUCTIONS.replace("{invoice}", invoice), "draft"
    )
    logger.info("approval[%s]: draft=%s", data.invoice_id, draft.decision.value)

    critique = llm.complete(
        _CRITIQUE_INSTRUCTIONS.replace("{invoice}", invoice)
        .replace("{draft_decision}", draft.decision.value)
        .replace("{draft_reasoning}", draft.reasoning)
    ).strip()

    revised = _parse_decision_pass(
        llm,
        _REVISE_INSTRUCTIONS.replace("{invoice}", invoice)
        .replace("{draft_decision}", draft.decision.value)
        .replace("{draft_reasoning}", draft.reasoning)
        .replace("{critique}", critique),
        "revise",
    )
    logger.info("approval[%s]: revised=%s", data.invoice_id, revised.decision.value)

    return ReflectionTrace(
        draft_decision=draft.decision,
        draft_reasoning=draft.reasoning,
        critique=critique,
        revised_decision=revised.decision,
        revised_reasoning=revised.reasoning,
    )


# =============================================================================
# Step 5 — agreement / disagreement resolution
# =============================================================================
def _resolve(
    route: Route,
    data: InvoiceData,
    validation: ValidationResult,
    trace: ReflectionTrace,
) -> ApprovalResult:
    """Map (route, reflection trace) -> a committed :class:`ApprovalResult`.

    The gate's deterministic verdicts short-circuit without ever consulting the
    trace. For ``FULL_LOOP`` runs the rule is: draft and revised agree -> commit that
    decision; they disagree -> escalate to ``NEEDS_REVIEW`` with both passes plus the
    critique surfaced so a human sees exactly why the model split with itself.
    """
    invoice_id = data.invoice_id

    if route is Route.FAST_APPROVE:
        return ApprovalResult(
            invoice_id=invoice_id,
            decision=ApprovalDecision.APPROVED,
            reasoning=(
                f"Fast-tracked: clean validation, amount ${data.amount:,.2f} is below "
                f"${APPROVAL_THRESHOLD_LOW:,.0f}, and extraction confidence is high."
            ),
            route=route.value,
        )

    if route is Route.AUTO_REVIEW:
        return ApprovalResult(
            invoice_id=invoice_id,
            decision=ApprovalDecision.NEEDS_REVIEW,
            reasoning=(
                "Escalated by policy: the vendor is not on the approved-vendor list "
                "(VENDOR_UNRECOGNIZED), which always requires human review."
            ),
            route=route.value,
        )

    # FULL_LOOP: adjudicate the draft vs. the revised decision.
    if trace.draft_decision == trace.revised_decision:
        return ApprovalResult(
            invoice_id=invoice_id,
            decision=trace.revised_decision,
            reasoning=trace.revised_reasoning or "",
            route=route.value,
            trace=trace,
        )

    return ApprovalResult(
        invoice_id=invoice_id,
        decision=ApprovalDecision.NEEDS_REVIEW,
        reasoning=(
            "Escalated: the draft and revised passes disagreed "
            f"(draft={_value(trace.draft_decision)}, revised={_value(trace.revised_decision)}). "
            f"Draft reasoning: {trace.draft_reasoning} "
            f"VP critique: {trace.critique} "
            f"Revised reasoning: {trace.revised_reasoning}"
        ),
        route=route.value,
        trace=trace,
    )


def _value(decision: Optional[ApprovalDecision]) -> str:
    """Render an optional decision as its string value for inclusion in reasoning."""
    return decision.value if decision is not None else "none"



# Pure core (LLM injected) — directly unit-testable
def run_approval(
    data: InvoiceData, validation: ValidationResult, llm: LLMProtocol
) -> ApprovalResult:
    """Gate -> (loop if needed) -> resolve. Never raises on LLM *content* problems.

    The LLM is injected so tests drive it with :class:`MockLLMClient`. Transport
    errors are *not* caught here — the graph node owns that fail-forward conversion,
    keeping this core honest about whether the loop actually completed. A pass that
    cannot be parsed after retries surfaces as a ``ValueError`` for the node to map.

    The LLM is touched only when the gate routes to ``FULL_LOOP``; a fast-approve or
    auto-review never calls ``llm.complete``, so a stub client is harmless there.
    """
    route = _classify(data, validation)
    logger.info("approval[%s]: gate route=%s", data.invoice_id, route.value)

    trace = ReflectionTrace()
    if route is Route.FULL_LOOP:
        trace = _run_reflection(data, validation, llm)

    result = _resolve(route, data, validation, trace)
    logger.info("approval[%s]: decision=%s", data.invoice_id, result.decision.value)
    return result


# Graph node (state in, state out) — fail-forward
class _UnusedLLM:
    """A stand-in client for routes that never call the model (fast-approve / auto-review).

    Returning this instead of constructing a real client means a clean fast-track
    invoice processes with no API key configured. If it is ever actually called it
    raises, which would be a routing bug surfaced loudly in tests, not in production.
    """

    def complete(self, prompt: str) -> str:  # pragma: no cover - must never run
        raise RuntimeError("LLM called on a route that should not need it")


def _client_for(
    state: InvoiceState,
    validation: ValidationResult,
    data: InvoiceData,
    llm_client: Optional[LLMProtocol],
) -> LLMProtocol:
    """Provide an LLM only when the gate's route actually requires one.

    An injected client (tests) always wins. Otherwise the real client is built lazily
    and only for ``FULL_LOOP``; non-loop routes get a stub so a missing API key never
    blocks a fast-approve. Construction errors propagate to the node's fail-forward
    handler, which converts them to ``NEEDS_REVIEW``.
    """
    if llm_client is not None:
        return llm_client
    if _classify(data, validation) is Route.FULL_LOOP:
        return get_llm_client()
    return _UnusedLLM()


# Maps the committed decision onto the matching terminal processing_stage value.
_DECISION_TO_STAGE = {
    ApprovalDecision.APPROVED: "approved",
    ApprovalDecision.REJECTED: "rejected",
    ApprovalDecision.NEEDS_REVIEW: "needs_review",
}


def approve_invoice(
    state: InvoiceState, llm_client: Optional[LLMProtocol] = None
) -> InvoiceState:
    """LangGraph node: decide the invoice and embed an :class:`ApprovalResult`.

    Fail-forward (PROJECT_CONTEXT §3):

    * No ``invoice_data`` (ingestion failed) -> no-op with a logged note, exactly
      like the validation node's guard. The router sends it to the log node.
    * No ``validation_result`` (validation skipped/errored) -> treated as empty so
      the gate still routes; a missing result never crashes the node.
    * Any LLM transport failure -> :class:`ApprovalDecision.NEEDS_REVIEW`. When the
      model is unreachable, escalating to a human is the only safe failure: never a
      crash, and never a silent auto-approve.
    """
    log = list(state.get("processing_log") or [])
    data = state.get("invoice_data")
    if data is None:
        log.append("approval: skipped (no invoice_data)")
        return {**state, "processing_log": log}

    validation = state.get("validation_result") or ValidationResult(invoice_id=data.invoice_id)

    try:
        # Resolve the LLM client lazily and only when the gate actually needs it: a
        # fast-approve / auto-review invoice never touches the model, so it must not
        # require an API key. Any transport failure (or an unconfigured key during a
        # full loop) escalates to review rather than crashing the graph.
        llm = _client_for(state, validation, data, llm_client)
        result = run_approval(data, validation, llm)
    except (httpx.HTTPError, ValueError) as exc:
        logger.error("Approval LLM failure for %s: %s", state.get("invoice_id"), exc)
        result = ApprovalResult(
            invoice_id=data.invoice_id,
            decision=ApprovalDecision.NEEDS_REVIEW,
            reasoning=f"Escalated: the approval model was unavailable or unparseable ({exc}).",
            route=Route.FULL_LOOP.value,
        )
        log.append("approval: LLM_ERROR -> needs_review")

    log.append(f"approval: {result.decision.value} (route={result.route})")
    return {
        **state,
        "approval_result": result,
        "processing_stage": _DECISION_TO_STAGE[result.decision],
        "processing_log": log,
    }
