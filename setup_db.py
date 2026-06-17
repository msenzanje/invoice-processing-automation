"""
One-shot database seeder for inventory.db.

Creates and seeds the two tables the validation agent queries: 
``inventory`` (item / stock / unit_price / category) and ``vendors``
(vendor_name / approved / added_date). Safe to re-run — every statement is
idempotent (``CREATE TABLE IF NOT EXISTS`` + ``INSERT OR IGNORE``), so an existing
database is never clobbered and seed rows are only added when missing.

Run before main.py:

    python setup_db.py
"""

from __future__ import annotations

from datetime import datetime, timezone

from tools.db import DB_PATH, get_connection, init_db

# Inventory the validation agent checks invoices against. Stock values match the
# levels the test invoices are designed to exercise (e.g. GadgetX=5 makes
# invoice_1005's request for 8 trip STOCK_EXCEEDED; FakeItem=0 trips ZERO_STOCK).
INVENTORY_SEED: tuple[tuple[str, int, float, str], ...] = (
    ("WidgetA", 15, 250.00, "widgets"),
    ("WidgetB", 10, 500.00, "widgets"),
    ("GadgetX", 5, 750.00, "gadgets"),
    ("FakeItem", 0, 0.00, "unknown"),
)

# Vendors known to the AP system. ``approved=1`` is a recognised, trusted vendor;
# anyone not in this table trips the VENDOR_UNRECOGNIZED warning. These names are
# the legitimate vendors the ingestion test fixtures extract cleanly.
VENDOR_SEED: tuple[str, ...] = (
    "Widgets Inc.",
    "Precision Parts Ltd.",
    "Acme Industrial Supplies",
    "MegaWidgets Corp",
    "Summit Manufacturing Co.",
    "Reliable Components Inc.",
)


def seed() -> None:
    """Create the schema (via init_db) and insert seed rows if not present."""
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        # Upsert inventory so re-running reconciles unit_price/category onto rows
        # carried over from the legacy (item, stock)-only scaffold schema.
        conn.executemany(
            "INSERT INTO inventory (item, stock, unit_price, category) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(item) DO UPDATE SET "
            "  stock=excluded.stock, unit_price=excluded.unit_price, "
            "  category=excluded.category",
            INVENTORY_SEED,
        )
        conn.executemany(
            "INSERT OR IGNORE INTO vendors (vendor_name, approved, added_date) "
            "VALUES (?, 1, ?)",
            [(name, now) for name in VENDOR_SEED],
        )
        conn.commit()


def _report() -> None:
    """Print the seeded contents so a manual run shows the resulting state."""
    with get_connection() as conn:
        inventory = conn.execute(
            "SELECT item, stock, unit_price, category FROM inventory ORDER BY item"
        ).fetchall()
        vendors = conn.execute(
            "SELECT vendor_name, approved FROM vendors ORDER BY vendor_name"
        ).fetchall()
    print(f"Database: {DB_PATH}")
    print("\ninventory:")
    for item, stock, unit_price, category in inventory:
        price = f"${unit_price:,.2f}" if unit_price is not None else "-"
        print(f"  {item:<12} stock={stock!s:<4} unit_price={price:<9} category={category}")
    print("\nvendors:")
    for vendor_name, approved in vendors:
        print(f"  {vendor_name:<28} approved={approved}")


if __name__ == "__main__":
    seed()
    _report()
