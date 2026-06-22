"""
CLI entrypoint

Builds the full LangGraph app — ingestion -> validation -> approval -> one of three
terminal nodes (payment / rejection log / human escalation) — invokes it on a single
invoice file, and renders the extracted fields, validation flags, and the approval
decision (with the reflection transcript when the critique loop ran).

Usage:
    python main.py --invoice_path=data/invoices/invoice_1001.txt
    python main.py --invoice_path=data/invoices/invoice_1001.txt --verbose
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from graph.workflow import build_graph
from models.state import InvoiceState
from tools.audit import AUDIT_PATH

console = Console()


def build_initial_state(invoice_path: str) -> InvoiceState:
    """Construct a fully-defaulted InvoiceState for a single invoice file."""
    path = Path(invoice_path)
    return {
        "invoice_id": path.stem,
        "filename": path.name,
        "file_format": path.suffix.lower().lstrip("."),
        "file_path": str(path),
        "raw_content": "",
        "parsed_data": {},
        "invoice_data": None,
        "processing_stage": "pending",
        "is_valid": None,
        "validation_errors": [],
        "validation_result": None,
        "approval_result": None,
        "inventory_updated": False,
        "db_record_id": None,
        "extraction_confidence": 0.0,
        "processing_log": [],
        "error_message": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _confidence_of(state_data, field: str) -> str:
    rating = state_data.field_confidence.get(field)
    return rating.value if rating is not None else "-"


def _render_result(state: InvoiceState) -> None:
    if state.get("error_message"):
        console.print(f"[bold red]Ingestion failed:[/] {state['error_message']}")
        return
    data = state.get("invoice_data")
    if data is None:
        console.print("[yellow]No invoice data was produced.[/]")
        return

    table = Table(title=f"Extracted Invoice — {data.invoice_id or state['filename']}")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value")
    table.add_column("Confidence", style="magenta")
    table.add_row("Vendor", data.vendor, _confidence_of(data, "vendor"))
    table.add_row("Amount", f"${data.amount:,.2f}", _confidence_of(data, "amount"))
    table.add_row("Due date", data.due_date.isoformat(), _confidence_of(data, "due_date"))
    table.add_row("Items", str(len(data.items)), _confidence_of(data, "items"))
    table.add_row("Overall", "", data.overall_confidence.value)
    console.print(table)
    for item in data.items:
        console.print(f"  • {item.item}  qty={item.quantity}  @ ${item.unit_price:,.2f}")
    _render_validation(state)
    _render_approval(state)


def _render_validation(state: InvoiceState) -> None:
    result = state.get("validation_result")
    if result is None:
        return
    if result.passes:
        console.print("[bold green]Validation: passed[/] (no flags)")
        return
    style = "red" if result.has_errors else "yellow"
    console.print(f"[bold {style}]Validation: {len(result.all_flags())} flag(s)[/]")
    for flag in result.all_flags():
        marker = "[red]✗[/]" if flag.severity == "error" else "[yellow]![/]"
        scope = f" ({flag.item})" if flag.item else ""
        console.print(f"  {marker} {flag.code.value}{scope}: {flag.message}")


_DECISION_STYLE = {
    "approved": "green",
    "rejected": "red",
    "needs_review": "yellow",
}


def _render_approval(state: InvoiceState) -> None:
    result = state.get("approval_result")
    if result is None:
        return
    decision = result.decision.value
    style = _DECISION_STYLE.get(decision, "white")
    console.print(f"[bold {style}]Approval: {decision.upper()}[/] (route={result.route})")
    console.print(f"  reason: {result.reasoning}")
    _render_reflection(result)


def _render_reflection(result) -> None:
    """Print the draft -> critique -> revised transcript when the loop actually ran."""
    trace = result.trace
    if not trace.ran:
        return
    console.print("  [dim]reflection transcript:[/]")
    console.print(f"    draft   ({_decision_text(trace.draft_decision)}): {trace.draft_reasoning}")
    console.print(f"    critique: {trace.critique}")
    console.print(f"    revised ({_decision_text(trace.revised_decision)}): {trace.revised_reasoning}")


def _decision_text(decision) -> str:
    return decision.value if decision is not None else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description="Galatiq invoice processing pipeline.")
    parser.add_argument("--invoice_path", required=True, help="Path to the invoice file.")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG-level logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    app = build_graph()
    result = app.invoke(build_initial_state(args.invoice_path))
    _render_result(result)
    # Printed unconditionally (not inside _render_result, which returns early on a
    # dead invoice) so every run tells the operator where the durable record landed.
    console.print(f"[dim]Audit record appended to {AUDIT_PATH}[/]")


if __name__ == "__main__":
    main()
