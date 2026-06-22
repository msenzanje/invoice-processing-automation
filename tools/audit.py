"""
Audit-trail writer — the one place that appends to ``logs/audit.jsonl``.

Disk I/O lives here, not in an agent: the terminal nodes stay pure (state in,
state out) and the side effect of appending a record is isolated behind a small,
swappable helper, exactly as ``tools/db.py`` and ``tools/payment_api.py`` do.
Centralising the write means every terminal branch emits byte-identical JSONL and
the trail can later be redirected by changing one module.

Fail-forward: :func:`write_audit_record` never raises into the
graph. A failed audit write must not crash a payment that already succeeded — so an
I/O error is logged at ERROR and returned as a string for the caller to surface,
rather than propagated.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from models.results import AuditRecord

logger = logging.getLogger(__name__)

# logs/audit.jsonl sits at the project root (one level up from tools/), derived the
# same way tools/db.py derives DB_PATH so both anchor to the repo root identically.
AUDIT_PATH = Path(__file__).resolve().parent.parent / "logs" / "audit.jsonl"


def write_audit_record(record: AuditRecord, path: "str | Path | None" = None) -> Optional[str]:
    """Append one audit record as a single JSON line; never raise.

    Resolves the log path (default :data:`AUDIT_PATH`), ensures the parent
    directory exists, and appends ``json.dumps(record.to_dict())`` followed by a
    newline in append mode so the legally-significant trail is only ever extended,
    never truncated.

    Pass ``path`` to target a different file (the test suite points this at a
    ``tmp_path`` so it never touches the real log), mirroring how
    :func:`tools.db.get_connection` accepts a ``db_path`` override.

    Returns ``None`` on success, or the error message string on failure. The caller
    (a terminal node) surfaces that string into ``processing_log`` so a swallowed
    write is still visible to the operator; the write itself is never re-raised.
    """
    target = Path(path) if path is not None else AUDIT_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_dict())
        with target.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return None
    except OSError as exc:  # disk full, permissions, bad path — log and signal, don't crash
        message = f"{type(exc).__name__}: {exc}"
        logger.error("audit write failed for %s -> %s: %s", record.invoice_id, target, message)
        return message


def elapsed_ms(start_iso: str) -> int:
    """Whole milliseconds from an ISO start timestamp to now (UTC).

    Used by the terminal nodes to fill ``AuditRecord.processing_time_ms`` from the
    state's start anchor without widening the state contract.

    INVARIANT: ``start_iso`` must mark the start of *this* invoice's processing
    (``InvoiceState['timestamp']``, set per-invoice in ``build_initial_state``). The
    batch entrypoint must likewise set ``timestamp`` per-invoice; reusing one start
    across a batch would inflate every invoice after the first.

    Fail-forward: a missing or unparseable timestamp returns ``0`` rather than
    raising, so a malformed start anchor never crashes a terminal node.
    """
    if not start_iso:
        return 0
    try:
        start = datetime.fromisoformat(start_iso)
    except ValueError:
        return 0
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - start
    return max(0, int(delta.total_seconds() * 1000))
