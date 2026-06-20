"""
Agent output models — the Pydantic contracts that cross agent boundaries. 
Phase 3 adds the validation contract here; later phases
(approval) add their result models alongside it.

The validation contract is the interface between the validation agent and the
approval agent. Its central type is :class:`ValidationResult`, a typed list of
:class:`ValidationFlag` objects. The
approval agent routes on these flags — specifically on flag ``severity`` — so the
codes and their severities are a fixed, non-negotiable contract, not free strings.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# Flag severity. ``error`` flags force an invoice through the full approval
# critique loop; ``warning`` flags are advisory context for the approval LLM.
Severity = Literal["warning", "error"]


class FlagCode(str, Enum):
    """The complete validation flag taxonomy (PROJECT_CONTEXT §5).

    No validation flag exists outside this enum — there are no string-named
    flags floating through the system.
    """

    UNKNOWN_ITEM = "UNKNOWN_ITEM"  # item not found in inventory
    STOCK_EXCEEDED = "STOCK_EXCEEDED"  # requested qty > available stock
    ZERO_STOCK = "ZERO_STOCK"  # item exists but stock = 0
    VENDOR_UNRECOGNIZED = "VENDOR_UNRECOGNIZED"  # vendor not an approved vendor
    NEGATIVE_QUANTITY = "NEGATIVE_QUANTITY"  # line item qty is negative
    ZERO_AMOUNT = "ZERO_AMOUNT"  # invoice total is 0 (or non-positive)
    LOW_CONFIDENCE_EXTRACTION = "LOW_CONFIDENCE_EXTRACTION"  # a field is low-confidence


# The severity each code carries. Centralised so the validation agent never has
# to remember a code's severity at the call site and so the approval agent reads
# the same mapping. Matches the PROJECT_CONTEXT §5 taxonomy table exactly.
FLAG_SEVERITY: dict[FlagCode, Severity] = {
    FlagCode.UNKNOWN_ITEM: "error",
    FlagCode.STOCK_EXCEEDED: "error",
    FlagCode.ZERO_STOCK: "error",
    FlagCode.VENDOR_UNRECOGNIZED: "warning",
    FlagCode.NEGATIVE_QUANTITY: "error",
    FlagCode.ZERO_AMOUNT: "warning",
    FlagCode.LOW_CONFIDENCE_EXTRACTION: "warning",
}


class ValidationFlag(BaseModel):
    """One issue found during validation.

    ``item`` is set for item-scoped flags (UNKNOWN_ITEM, STOCK_EXCEEDED,
    ZERO_STOCK, NEGATIVE_QUANTITY) and left None for invoice-level flags
    (VENDOR_UNRECOGNIZED, ZERO_AMOUNT, LOW_CONFIDENCE_EXTRACTION).
    """

    code: FlagCode
    severity: Severity
    message: str
    item: Optional[str] = None

    @classmethod
    def of(cls, code: FlagCode, message: str, item: Optional[str] = None) -> "ValidationFlag":
        """Build a flag, deriving severity from :data:`FLAG_SEVERITY`."""
        return cls(code=code, severity=FLAG_SEVERITY[code], message=message, item=item)


class ItemValidation(BaseModel):
    """Validation outcome for a single invoice line item."""

    item: str
    quantity: int
    flags: list[ValidationFlag] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """The complete validation outcome for one invoice.

    ``passes`` is True only when no flags of any severity were raised. Note this
    is stricter than "no errors": a warning-only invoice does not ``pass``, but
    the approval agent still distinguishes warnings from errors via flag
    ``severity`` when it routes.
    """

    invoice_id: Optional[str] = None
    item_validations: list[ItemValidation] = Field(default_factory=list)
    invoice_flags: list[ValidationFlag] = Field(default_factory=list)

    def all_flags(self) -> list[ValidationFlag]:
        """Every flag — item-level then invoice-level — as one flat list."""
        flags: list[ValidationFlag] = []
        for item_validation in self.item_validations:
            flags.extend(item_validation.flags)
        flags.extend(self.invoice_flags)
        return flags

    def error_flags(self) -> list[ValidationFlag]:
        """Only the ``error``-severity flags (what forces the critique loop)."""
        return [flag for flag in self.all_flags() if flag.severity == "error"]

    @property
    def passes(self) -> bool:
        """True if validation found nothing to flag."""
        return len(self.all_flags()) == 0

    @property
    def has_errors(self) -> bool:
        """True if any ``error``-severity flag was raised."""
        return any(flag.severity == "error" for flag in self.all_flags())

    def to_dict(self) -> dict[str, Any]:
        """Fully JSON-serialisable view for the JSONL audit trail."""
        return {
            "invoice_id": self.invoice_id,
            "passes": self.passes,
            "has_errors": self.has_errors,
            "item_validations": [iv.model_dump() for iv in self.item_validations],
            "invoice_flags": [flag.model_dump() for flag in self.invoice_flags],
        }


# =============================================================================
# Approval contract (Phase 4 — Approval Agent)
# =============================================================================
class ApprovalDecision(str, Enum):
    """The three terminal outcomes of the approval agent.

    This is the structural reason the pipeline can express a *third* state beyond
    approve/reject: ``NEEDS_REVIEW`` lets the system escalate to a human when it is
    genuinely uncertain (the draft and revised passes disagreed, the deterministic
    gate flagged an unrecognised vendor, or the LLM was unreachable) rather than
    forcing a coin-flip. The graph's conditional edge routes on exactly this enum:
    ``APPROVED -> payment``, ``REJECTED -> log``, ``NEEDS_REVIEW -> escalation``.
    """

    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


class ReflectionTrace(BaseModel):
    """The audit trail of one draft -> critique -> revise reflection cycle.

    Captured verbatim so a human reviewer (or an auditor replacing the old VP email
    chain) can see exactly how the model reasoned, including where the revised pass
    diverged from the draft. Every field is optional because a fast-tracked or
    gate-escalated invoice never enters the loop and so has no transcript.
    """

    draft_decision: Optional[ApprovalDecision] = None
    draft_reasoning: Optional[str] = None
    critique: Optional[str] = None
    revised_decision: Optional[ApprovalDecision] = None
    revised_reasoning: Optional[str] = None

    @property
    def ran(self) -> bool:
        """True if the reflection loop actually executed (a draft was produced)."""
        return self.draft_decision is not None


class ApprovalResult(BaseModel):
    """The complete approval outcome for one invoice — the approval agent's output.

    ``decision`` is the committed routing verdict; ``reasoning`` is the
    human-readable justification; ``trace`` carries the full reflection transcript
    when the critique loop ran (empty otherwise). ``route`` records *which* gate
    path produced the decision (fast-approve, auto-review, or the full loop), so the
    audit log shows not just the verdict but why that verdict was reached cheaply or
    expensively.
    """

    invoice_id: Optional[str] = None
    decision: ApprovalDecision
    reasoning: str
    route: str
    trace: ReflectionTrace = Field(default_factory=ReflectionTrace)

    def to_dict(self) -> dict[str, Any]:
        """Fully JSON-serialisable view for the JSONL audit trail.

        Mirrors :meth:`ValidationResult.to_dict` so every agent's audit record has
        the same shape. ``json.dumps`` accepts this dict directly.
        """
        return {
            "invoice_id": self.invoice_id,
            "decision": self.decision.value,
            "reasoning": self.reasoning,
            "route": self.route,
            "trace": {
                "draft_decision": _decision_value(self.trace.draft_decision),
                "draft_reasoning": self.trace.draft_reasoning,
                "critique": self.trace.critique,
                "revised_decision": _decision_value(self.trace.revised_decision),
                "revised_reasoning": self.trace.revised_reasoning,
            },
        }


def _decision_value(decision: Optional[ApprovalDecision]) -> Optional[str]:
    """Render an optional decision enum as its string value (or None)."""
    return decision.value if decision is not None else None
