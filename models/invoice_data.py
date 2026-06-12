"""
InvoiceData — the shared extraction data contract for the invoice pipeline.

Every downstream agent (validation, approval, payment) consumes an ``InvoiceData``
object. It carries the four required invoice fields plus a per-field confidence
rating so downstream reasoning can weigh how much to trust each value. The
approval agent does not need to know *how* a vendor name was found — only whether
to trust it. The per-field confidence scores are the mechanism that carries that
signal forward.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Confidence(str, Enum):
    """Per-field trust level assigned during extraction."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Fields that participate in the overall-confidence majority rule.
CONFIDENCE_FIELDS: tuple[str, ...] = ("vendor", "amount", "items", "due_date")


class LineItem(BaseModel):
    """A single line on an invoice."""

    item: str
    quantity: int
    unit_price: float


class InvoiceData(BaseModel):
    """Structured, confidence-scored representation of one invoice.

    The four required fields (``vendor``, ``amount``, ``items``, ``due_date``) are
    the contract every downstream agent relies on. ``field_confidence`` maps each
    field name to a :class:`Confidence`; ``overall_confidence`` collapses those
    into a single rating via a simple majority rule.
    """

    vendor: str
    amount: float
    items: list[LineItem] = Field(default_factory=list)
    due_date: date
    invoice_id: Optional[str] = None
    field_confidence: dict[str, Confidence] = Field(default_factory=dict)

    @property
    def overall_confidence(self) -> Confidence:
        """Majority rule over the required fields.

        ``HIGH`` if 3+ required fields are ``HIGH``; ``LOW`` if 2+ are ``LOW``;
        otherwise ``MEDIUM``. Fields with no recorded confidence default to
        ``MEDIUM`` so a missing rating never inflates the score.
        """
        ratings = [
            self.field_confidence.get(name, Confidence.MEDIUM)
            for name in CONFIDENCE_FIELDS
        ]
        high = sum(1 for r in ratings if r is Confidence.HIGH)
        low = sum(1 for r in ratings if r is Confidence.LOW)
        if high >= 3:
            return Confidence.HIGH
        if low >= 2:
            return Confidence.LOW
        return Confidence.MEDIUM

    def model_dump_log(self) -> dict[str, Any]:
        """Return a fully JSON-serialisable dict for the JSONL audit trail."""
        return {
            "invoice_id": self.invoice_id,
            "vendor": self.vendor,
            "amount": self.amount,
            "due_date": self.due_date.isoformat(),
            "items": [item.model_dump() for item in self.items],
            "field_confidence": {
                name: rating.value for name, rating in self.field_confidence.items()
            },
            "overall_confidence": self.overall_confidence.value,
        }
