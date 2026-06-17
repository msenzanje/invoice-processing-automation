"""
SQLite helpers for the inventory/vendor database.

The validation agent queries two tables here:

* ``inventory`` — item, stock, unit_price, category
* ``vendors``   — vendor_name, approved, added_date

Connection model: every call opens a fresh short-lived connection via
:func:`get_connection` (a context manager) and closes it on exit. SQLite
connections cannot be shared across threads by default, and the MVP processes
invoices sequentially, so per-call connections are both the simplest and the
thread-safe choice — no pooling, no shared mutable handle. Seeding lives in
``setup_db.py``; this module only owns the schema and read helpers.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, NamedTuple, Optional

logger = logging.getLogger(__name__)

# inventory.db sits at the project root (one level up from tools/).
DB_PATH = Path(__file__).resolve().parent.parent / "inventory.db"


class InventoryItem(NamedTuple):
    """A row from the ``inventory`` table."""

    item: str
    stock: int
    unit_price: Optional[float]
    category: Optional[str]


@contextmanager
def get_connection(db_path: "str | Path | None" = None) -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection and guarantee it is closed.

    Usage::

        with get_connection() as conn:
            row = conn.execute("SELECT ...").fetchone()

    Pass ``db_path`` to target a different database (the test suite points this
    at a temporary file so it never touches the real inventory.db).
    """
    path = str(db_path) if db_path is not None else str(DB_PATH)
    conn = sqlite3.connect(path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: "str | Path | None" = None) -> None:
    """Create both tables if absent and migrate a legacy 2-column inventory.

    Idempotent: safe to call on every run. An older inventory.db may have only
    ``(item, stock)`` (the Phase 1 scaffold schema); this adds the documented
    ``unit_price`` and ``category`` columns in place rather than dropping data.
    """
    with get_connection(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS inventory ("
            "  item TEXT PRIMARY KEY,"
            "  stock INTEGER,"
            "  unit_price REAL,"
            "  category TEXT"
            ")"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS vendors ("
            "  vendor_name TEXT PRIMARY KEY,"
            "  approved INTEGER,"
            "  added_date TEXT"
            ")"
        )
        _migrate_inventory_columns(conn)
        conn.commit()


def _migrate_inventory_columns(conn: sqlite3.Connection) -> None:
    """Add unit_price/category to a legacy inventory table that lacks them."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(inventory)")}
    if "unit_price" not in existing:
        logger.info("Migrating inventory: adding unit_price column")
        conn.execute("ALTER TABLE inventory ADD COLUMN unit_price REAL")
    if "category" not in existing:
        logger.info("Migrating inventory: adding category column")
        conn.execute("ALTER TABLE inventory ADD COLUMN category TEXT")


def get_inventory_item(
    conn: sqlite3.Connection, item_name: str
) -> Optional[InventoryItem]:
    """Return the inventory row for ``item_name``, or None if not stocked.

    The match is exact (item is the primary key); the ingestion agent is
    responsible for normalising item names before they reach here.
    """
    row = conn.execute(
        "SELECT item, stock, unit_price, category FROM inventory WHERE item = ?",
        (item_name,),
    ).fetchone()
    if row is None:
        return None
    return InventoryItem(item=row[0], stock=row[1], unit_price=row[2], category=row[3])


def is_vendor_approved(conn: sqlite3.Connection, vendor_name: str) -> bool:
    """True only if the vendor exists in ``vendors`` and ``approved = 1``.

    An unknown vendor and an explicitly flagged (``approved = 0``) vendor are
    treated the same by the caller: both trip VENDOR_UNRECOGNIZED.
    """
    if not vendor_name or not vendor_name.strip():
        return False
    row = conn.execute(
        "SELECT approved FROM vendors WHERE vendor_name = ?",
        (vendor_name.strip(),),
    ).fetchone()
    return row is not None and row[0] == 1
