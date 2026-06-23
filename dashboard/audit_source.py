"""
Audit-log read model — turns ``logs/audit.jsonl`` into the view objects the web
dashboard renders.

This is the dashboard's *read* counterpart to :mod:`tools.audit` (which is the one
place that *writes* the trail). The split mirrors the rest of the codebase: a single
module owns the on-disk JSONL shape, and everything downstream consumes typed objects
instead of poking at raw dicts. The agents stay pure; the dashboard never imports the
graph — it only reads the durable artifact the graph already produced.

Nothing here invents data. Every KPI, row, and feed entry is computed from real
:class:`~models.results.AuditRecord` lines. The design prototype animated fake
invoices; the real dashboard shows what actually happened, polling the file for new
records as the pipeline appends them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from tools.audit import AUDIT_PATH

logger = logging.getLogger(__name__)

# The three terminal decisions, in the order the filter chips read (happy path first).
# Mirrors models.results.ApprovalDecision values without importing the enum — the
# dashboard reads strings off disk, not the live enum.
_DECISIONS = ("approved", "needs_review", "rejected")

# Human label per decision, matching the design's badge text.
_DECISION_LABEL = {
    "approved": "Approved",
    "needs_review": "Needs review",
    "rejected": "Rejected",
}


def _pretty_flag(code: str) -> str:
    """Render a flag code as a human label: ``UNKNOWN_ITEM`` -> ``Unknown item``.

    Matches the prototype's ``prettyFlag`` exactly (capitalise first letter, lower the
    rest, underscores to spaces) so a real ``STOCK_EXCEEDED`` reads as "Stock exceeded".
    """
    if not code:
        return ""
    return code[0] + code[1:].lower().replace("_", " ")


def _time_ago(timestamp: str, *, now: Optional[datetime] = None) -> str:
    """Compact "2m" / "3h" / "5d" relative age from an ISO timestamp.

    Fail-forward: an empty or unparseable timestamp yields "—" rather than raising,
    so one malformed line never breaks the whole table render.
    """
    if not timestamp:
        return "—"
    try:
        when = datetime.fromisoformat(timestamp)
    except ValueError:
        return "—"
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    reference = now or datetime.now(timezone.utc)
    seconds = max(0, int((reference - when).total_seconds()))
    if seconds < 60:
        return "just now" if seconds < 5 else f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    return f"{hours // 24}d"


def _payment_view(decision: str, payment_status: Optional[str]) -> str:
    """The text shown in the table's PAYMENT column for a record.

    Real records carry ``payment_status`` only on the approved/payment path (see
    :class:`~models.results.AuditRecord`); rejection and escalation leave it null. The
    design shows a word per outcome, so derive one: an approved invoice that was paid
    says so, an approved-but-not-yet-paid one is "Scheduled", a rejected one is
    "Voided", and anything in review is "On hold".
    """
    if decision == "approved":
        if payment_status and payment_status.lower() in {"success", "paid", "completed"}:
            return "Paid"
        if payment_status:
            return payment_status.capitalize()
        return "Scheduled"
    if decision == "rejected":
        return "Voided"
    return "On hold"


@dataclass(frozen=True)
class InvoiceRow:
    """One processed invoice, shaped for the table + expandable detail panel."""

    invoice_id: str
    id_short: str  # "#invoice_1042" — the display id with a leading hash
    vendor: str
    amount: float
    amount_fmt: str  # "$12,480"
    decision: str  # "approved" | "needs_review" | "rejected"
    badge_text: str  # "Approved" | "Needs review" | "Rejected"
    time_ago: str  # "2m"
    proc_ms: str  # "1.6s"
    payment_text: str  # "Scheduled" | "Paid" | "On hold" | "Voided"
    reasoning: str
    critique: str
    flags: list[str]  # prettified labels for the detail panel

    @property
    def no_flags(self) -> bool:
        return len(self.flags) == 0


@dataclass(frozen=True)
class FeedItem:
    """One entry in the live-processing feed (most-recent records, newest first)."""

    invoice_id: str
    id_short: str
    vendor: str
    amount_fmt: str
    decision: str
    label: str  # "Approved" / "Needs review" / "Rejected"


@dataclass(frozen=True)
class Kpis:
    """The four executive-summary headline numbers, all derived from real records."""

    processed: int
    processed_fmt: str
    auto_approval_rate: float  # 0.0–1.0
    rate_fmt: str  # "78.8%"
    rate_bar: str  # "78.8%" — CSS width for the progress bar
    avg_ms: int
    avg_fmt: str  # "1.8s"
    value_approved: float
    value_fmt: str  # "$4.84M" or "$12,480"


@dataclass(frozen=True)
class DashboardData:
    """Everything the index page and the polling endpoints render from."""

    kpis: Kpis
    rows: list[InvoiceRow]
    feed: list[FeedItem]
    counts: dict[str, int]  # per-decision counts incl. "all", for the filter chips
    total: int
    latest_timestamp: Optional[str]  # newest record's ISO timestamp, for poll diffing


def _money(amount: float) -> str:
    """``12480.0`` -> ``$12,480`` (thousands separator, no cents — matches the design)."""
    return f"${amount:,.0f}"


def _value_fmt(amount: float) -> str:
    """Headline money: compact ``$4.84M`` above a million, full ``$4,180`` below."""
    if amount >= 1_000_000:
        return f"${amount / 1_000_000:.2f}M"
    if amount >= 10_000:
        return f"${amount / 1_000:.1f}K"
    return _money(amount)


def _proc_fmt(processing_time_ms: Any) -> str:
    """Milliseconds as seconds to one decimal: ``1640`` -> ``1.6s``."""
    try:
        return f"{int(processing_time_ms) / 1000:.1f}s"
    except (TypeError, ValueError):
        return "—"


def _read_records(path: Path) -> list[dict[str, Any]]:
    """Parse every JSON line in the audit log; skip blank or corrupt lines.

    Fail-forward, like the writer: a single malformed line is logged and skipped rather
    than sinking the whole dashboard. Returns records in file order (oldest first).
    """
    if not path.exists():
        logger.warning("audit log not found at %s — dashboard will show an empty state", path)
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("audit log line %d is not valid JSON, skipping: %s", line_no, exc)
    return records


def _build_row(record: dict[str, Any], *, now: datetime) -> InvoiceRow:
    """Shape one audit record into a table row + detail view."""
    decision = record.get("decision") or "needs_review"
    amount = float(record.get("amount") or 0.0)
    invoice_id = record.get("invoice_id") or "—"
    vendor = record.get("vendor") or "Unknown vendor"
    flags = [_pretty_flag(code) for code in (record.get("flags") or [])]
    return InvoiceRow(
        invoice_id=invoice_id,
        id_short=f"#{invoice_id}",
        vendor=vendor,
        amount=amount,
        amount_fmt=_money(amount),
        decision=decision,
        badge_text=_DECISION_LABEL.get(decision, decision.replace("_", " ").title()),
        time_ago=_time_ago(record.get("timestamp", ""), now=now),
        proc_ms=_proc_fmt(record.get("processing_time_ms")),
        payment_text=_payment_view(decision, record.get("payment_status")),
        reasoning=record.get("reasoning") or "No reasoning recorded for this decision.",
        critique=record.get("critique") or "No critique recorded for this decision.",
        flags=flags,
    )


def _build_kpis(records: list[dict[str, Any]]) -> Kpis:
    """Compute the four headline numbers from the full record set."""
    total = len(records)
    approved = [r for r in records if r.get("decision") == "approved"]
    rate = (len(approved) / total) if total else 0.0
    times = [int(r["processing_time_ms"]) for r in records if isinstance(r.get("processing_time_ms"), (int, float))]
    avg_ms = round(sum(times) / len(times)) if times else 0
    value = sum(float(r.get("amount") or 0.0) for r in approved)
    return Kpis(
        processed=total,
        processed_fmt=f"{total:,}",
        auto_approval_rate=rate,
        rate_fmt=f"{rate * 100:.1f}%",
        rate_bar=f"{rate * 100:.1f}%",
        avg_ms=avg_ms,
        avg_fmt=f"{avg_ms / 1000:.1f}s",
        value_approved=value,
        value_fmt=_value_fmt(value),
    )


def load_dashboard_data(path: "str | Path | None" = None, *, feed_size: int = 8) -> DashboardData:
    """Read the audit log and build everything the dashboard renders.

    Pass ``path`` to point at a different log (the test suite uses a ``tmp_path``),
    mirroring :func:`tools.audit.write_audit_record`. ``feed_size`` caps how many of the
    most-recent records appear in the live-processing panel.
    """
    target = Path(path) if path is not None else AUDIT_PATH
    records = _read_records(target)
    now = datetime.now(timezone.utc)

    # Newest first for the table and feed; the file is appended oldest-to-newest.
    newest_first = list(reversed(records))
    rows = [_build_row(record, now=now) for record in newest_first]

    feed = [
        FeedItem(
            invoice_id=row.invoice_id,
            id_short=row.id_short,
            vendor=row.vendor,
            amount_fmt=row.amount_fmt,
            decision=row.decision,
            label=row.badge_text,
        )
        for row in rows[:feed_size]
    ]

    counts = {"all": len(records)}
    for decision in _DECISIONS:
        counts[decision] = sum(1 for r in records if r.get("decision") == decision)

    return DashboardData(
        kpis=_build_kpis(records),
        rows=rows,
        feed=feed,
        counts=counts,
        total=len(records),
        latest_timestamp=newest_first[0].get("timestamp") if newest_first else None,
    )
