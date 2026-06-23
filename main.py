"""
CLI entrypoint

Builds the full LangGraph app — ingestion -> validation -> approval -> one of three
terminal nodes (payment / rejection log / human escalation) — invokes it on a single
invoice file, and renders the extracted fields, validation flags, and the approval
decision (with the reflection transcript when the critique loop ran).

Usage:
    python main.py --invoice_path=data/invoices/invoice_1001.txt
    python main.py --invoice_path=data/invoices/invoice_1001.txt --verbose
    python main.py --batch --invoice_dir=data/invoices/
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from rich.box import SIMPLE_HEAVY
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from graph.workflow import build_graph
from models.state import InvoiceState
from tools.audit import AUDIT_PATH, elapsed_ms

logger = logging.getLogger(__name__)
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


# Colour per confidence rating, shared by the field table and the overall chip.
_CONFIDENCE_STYLE = {"high": "green", "medium": "yellow", "low": "red"}


def _confidence_chip(value: str) -> str:
    """Render a confidence rating as a small colour-coded chip (●)."""
    style = _CONFIDENCE_STYLE.get(value, "dim")
    return f"[{style}] {value}[/]"


def _confidence_of(state_data, field: str) -> str:
    rating = state_data.field_confidence.get(field)
    return _confidence_chip(rating.value) if rating is not None else "[dim]—[/]"


def _render_result(state: InvoiceState) -> None:
    if state.get("error_message"):
        console.print(
            Panel(
                state["error_message"],
                title="[bold red] Ingestion failed[/]",
                title_align="left",
                border_style="red",
                expand=False,
            )
        )
        return
    data = state.get("invoice_data")
    if data is None:
        console.print(
            Panel(
                "No invoice data was produced.",
                title="[bold yellow] Empty result[/]",
                title_align="left",
                border_style="yellow",
                expand=False,
            )
        )
        return

    title = data.invoice_id or state["filename"]
    table = Table(
        title=f"[bold]Invoice[/] — {title}",
        box=SIMPLE_HEAVY,
        title_justify="left",
        header_style="bold",
        expand=False,
    )
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value")
    table.add_column("Confidence", justify="right")
    table.add_row("Vendor", data.vendor, _confidence_of(data, "vendor"))
    table.add_row("Amount", f"[bold]${data.amount:,.2f}[/]", _confidence_of(data, "amount"))
    table.add_row("Due date", data.due_date.isoformat(), _confidence_of(data, "due_date"))
    table.add_row("Items", str(len(data.items)), _confidence_of(data, "items"))
    table.add_section()
    table.add_row("[bold]Overall[/]", "", _confidence_chip(data.overall_confidence.value))
    console.print(table)

    if data.items:
        console.print(_items_table(data.items))
    _render_validation(state)
    _render_approval(state)


def _items_table(items) -> Table:
    """Line-item breakdown with a computed line total per row."""
    table = Table(box=SIMPLE_HEAVY, header_style="bold", expand=False, padding=(0, 2, 0, 0))
    table.add_column("Item", style="cyan")
    table.add_column("Qty", justify="right")
    table.add_column("Unit price", justify="right")
    table.add_column("Line total", justify="right")
    for item in items:
        table.add_row(
            item.item,
            str(item.quantity),
            f"${item.unit_price:,.2f}",
            f"${item.quantity * item.unit_price:,.2f}",
        )
    return table


def _render_validation(state: InvoiceState) -> None:
    result = state.get("validation_result")
    if result is None:
        return
    if result.passes:
        console.print(
            Panel(
                "[green] passed[/] — no flags",
                title="[bold]Validation[/]",
                title_align="left",
                border_style="green",
                expand=False,
            )
        )
        return
    style = "red" if result.has_errors else "yellow"
    lines = []
    for flag in result.all_flags():
        marker = "[red][/]" if flag.severity == "error" else "[yellow]![/]"
        scope = f" [dim]({flag.item})[/]" if flag.item else ""
        lines.append(f"{marker} [bold]{flag.code.value}[/]{scope}: {flag.message}")
    console.print(
        Panel(
            "\n".join(lines),
            title=f"[bold {style}]Validation — {len(result.all_flags())} flag(s)[/]",
            title_align="left",
            border_style=style,
            expand=False,
        )
    )


# Icon + colour + human label per terminal outcome — the single place the batch
# view styles a verdict. "approved" reads as "paid" because an approved invoice has
# already run through the payment node by the time it lands in the summary.
_OUTCOME_STYLE = {
    "approved": ("", "bold green", "paid"),
    "rejected": ("", "bold red", "rejected"),
    "needs_review": ("", "bold yellow", "review"),
    "failed": ("", "bold red3", "failed"),
}
_OUTCOME_FALLBACK = ("·", "dim", "unknown")

# Border colour per approval decision, reused by the single-invoice approval panel.
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
    icon, _, _ = _OUTCOME_STYLE.get(decision, _OUTCOME_FALLBACK)
    body = [f"[{style}]{icon} {decision.upper()}[/]  [dim](route={result.route})[/]", "", result.reasoning]
    reflection = _reflection_lines(result)
    if reflection:
        body.append("")
        body.extend(reflection)
    console.print(
        Panel(
            "\n".join(body),
            title="[bold]Approval[/]",
            title_align="left",
            border_style=style,
            expand=False,
        )
    )


def _reflection_lines(result) -> list[str]:
    """The draft -> critique -> revised transcript, as panel lines (empty if it didn't run)."""
    trace = result.trace
    if not trace.ran:
        return []
    return [
        "[dim]reflection transcript[/]",
        f"  [dim]draft[/]    ({_decision_text(trace.draft_decision)}): {trace.draft_reasoning}",
        f"  [dim]critique[/] {trace.critique}",
        f"  [dim]revised[/]  ({_decision_text(trace.revised_decision)}): {trace.revised_reasoning}",
    ]


def _decision_text(decision) -> str:
    return decision.value if decision is not None else "-"


# Extensions the ingestion agent knows how to read. Used to pick invoice files out
# of a batch directory and to skip everything else (e.g. data/invoices/*.py helpers).
_INVOICE_SUFFIXES = {".txt", ".json", ".csv", ".xml", ".pdf"}


def discover_invoices(invoice_dir: str) -> list[Path]:
    """Return the invoice files in ``invoice_dir``, sorted, one row per file.

    Filters to extensions the ingestion agent handles so non-invoice helpers in the
    directory (generators, READMEs) are skipped. PDF/TXT pairs for the same invoice
    are *both* returned — the directory is the unit of work the spec points at, and a
    failed row is fail-forward, not a reason to second-guess what to feed in.
    """
    root = Path(invoice_dir)
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _INVOICE_SUFFIXES)


def run_batch(app, invoice_dir: str) -> list[InvoiceState]:
    """Process every invoice in ``invoice_dir`` sequentially; return completed states.

    Each invoice gets a fresh initial state (so ``timestamp`` anchors per-invoice and
    ``elapsed_ms`` measures each row independently — see ``tools.audit.elapsed_ms``).
    Fail-forward: a node that crashes outright must not sink the rest of the batch, so
    the per-invoice ``invoke`` is guarded and a synthetic failed state is recorded in
    its place. The graph's own nodes never raise, so this only catches the unexpected.
    """
    invoices = discover_invoices(invoice_dir)
    if not invoices:
        console.print(f"[yellow]No invoice files found in {invoice_dir}[/]")
        return []

    console.print(f"[bold]Processing {len(invoices)} invoice(s)[/] from [cyan]{invoice_dir}[/]\n")
    results: list[InvoiceState] = []
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("· [dim]{task.fields[current]}[/]"),
        console=console,
        transient=True,  # clears the bar on completion so the summary owns the screen
    )
    with progress:
        task = progress.add_task("Invoices", total=len(invoices), current="")
        for path in invoices:
            progress.update(task, current=path.name)
            state = build_initial_state(str(path))
            try:
                results.append(app.invoke(state))
            except Exception as exc:  # graph nodes are fail-forward; this is the last net
                logger.exception("batch: invoke crashed for %s", path.name)
                results.append({
                    **state,
                    "processing_stage": "failed",
                    "is_valid": False,
                    "error_message": f"BATCH_CRASH: {type(exc).__name__}: {exc}",
                })
            progress.advance(task)
    return results


def _decision_of(state: InvoiceState) -> str:
    """One-word terminal outcome for a completed state, for the batch summary."""
    result = state.get("approval_result")
    if result is not None:
        return result.decision.value
    if state.get("error_message"):
        return "failed"
    return "-"


# Order the breakdown chips read in: the happy path first, problems last.
_OUTCOME_ORDER = ["approved", "rejected", "needs_review", "failed"]


def _render_batch_summary(results: list[InvoiceState]) -> None:
    """Render the batch result: a per-invoice table plus an at-a-glance breakdown.

    Two views, because an operator reads them differently: the table is the
    line-item record (what happened to each invoice), the panel is the headline
    (how the batch did overall and how long it took).
    """
    table = Table(
        title=f"[bold]Batch results[/] — {len(results)} invoice(s)",
        box=SIMPLE_HEAVY,
        title_justify="left",
        header_style="bold",
        row_styles=["", "on grey7"],
        expand=True,
    )
    table.add_column("Invoice", style="cyan", no_wrap=True)
    table.add_column("Outcome", no_wrap=True)
    table.add_column("Vendor", overflow="ellipsis", max_width=30)
    table.add_column("Amount", justify="right")
    table.add_column("Time", justify="right", style="dim")

    counts: dict[str, int] = {}
    total_amount = 0.0
    total_ms = 0
    for state in results:
        data = state.get("invoice_data")
        outcome = _decision_of(state)
        counts[outcome] = counts.get(outcome, 0) + 1
        icon, style, label = _OUTCOME_STYLE.get(outcome, _OUTCOME_FALLBACK)

        if data is not None:
            vendor = data.vendor
            amount = f"${data.amount:,.2f}"
            total_amount += data.amount
        else:
            vendor = "[dim]—[/]"
            amount = "[dim]—[/]"

        ms = elapsed_ms(state.get("timestamp", ""))
        total_ms += ms
        table.add_row(
            state.get("invoice_id", "-"),
            f"[{style}]{icon} {label}[/]",
            vendor,
            amount,
            f"{ms / 1000:.1f}s",
        )

    table.add_section()
    table.add_row(
        "[bold]Total[/]",
        "",
        "",
        f"[bold]${total_amount:,.2f}[/]",
        f"[bold]{total_ms / 1000:.1f}s[/]",
    )
    console.print(table)
    console.print(_breakdown_panel(counts, len(results), total_ms))
    console.print(f"[dim]Audit records appended to {AUDIT_PATH}[/]")


def _breakdown_panel(counts: dict[str, int], total: int, total_ms: int) -> Panel:
    """A compact 'X paid · Y review · Z failed' headline with timing, as a panel."""
    chips = []
    for outcome in _OUTCOME_ORDER:
        n = counts.get(outcome, 0)
        if n == 0:
            continue
        icon, style, label = _OUTCOME_STYLE[outcome]
        chips.append(f"[{style}]{icon} {n} {label}[/]")
    # Any outcome we don't have a chip order for (e.g. the "-" no-result sentinel).
    leftover = sum(n for key, n in counts.items() if key not in _OUTCOME_ORDER)
    if leftover:
        chips.append(f"[dim]· {leftover} unknown[/]")

    avg = total_ms / total / 1000 if total else 0.0
    body = (
        f"{'   '.join(chips)}\n"
        f"[dim]{total} invoice(s) · {total_ms / 1000:.1f}s total · {avg:.1f}s avg[/]"
    )
    return Panel(body, title="[bold]Summary[/]", title_align="left", border_style="cyan", expand=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Galatiq invoice processing pipeline.")
    parser.add_argument("--invoice_path", help="Path to a single invoice file.")
    parser.add_argument("--batch", action="store_true", help="Process a whole directory of invoices.")
    parser.add_argument(
        "--invoice_dir",
        default="data/invoices/",
        help="Directory of invoices to process in --batch mode (default: data/invoices/).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG-level logging.")
    args = parser.parse_args()

    if args.batch == bool(args.invoice_path):
        parser.error("provide exactly one of --invoice_path or --batch")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    app = build_graph()

    if args.batch:
        results = run_batch(app, args.invoice_dir)
        if results:
            _render_batch_summary(results)
        return

    result = app.invoke(build_initial_state(args.invoice_path))
    _render_result(result)
    # Printed unconditionally (not inside _render_result, which returns early on a
    # dead invoice) so every run tells the operator where the durable record landed.
    console.print(f"[dim]Audit record appended to {AUDIT_PATH}[/]")


if __name__ == "__main__":
    main()
