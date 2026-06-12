"""
LangGraph workflow definition.

Phase 2 wires a single node — ``ingest_invoice`` — which calls the ingestion
agent and writes the result into :class:`~models.state.InvoiceState`. Later phases
(validation, approval, payment) extend this graph by adding nodes and edges; the
node functions stay pure (state in, state out) and never raise into the graph.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from agents.ingestion import extract
from models.invoice_data import Confidence
from models.state import InvoiceState

logger = logging.getLogger(__name__)

# Map the extraction enum back onto the legacy 0.0–1.0 float in InvoiceState.
_CONFIDENCE_TO_FLOAT = {
    Confidence.HIGH: 0.9,
    Confidence.MEDIUM: 0.6,
    Confidence.LOW: 0.3,
}


def ingest_invoice(state: InvoiceState) -> InvoiceState:
    """Run the ingestion agent and merge its result into the state.

    Fail-forward (PROJECT_CONTEXT §3): an exhausted-extraction ``ValueError`` is
    captured into ``error_message`` rather than raised, so the orchestrator can
    route the invoice to human review instead of crashing the graph.
    """
    file_path = state["file_path"]
    log = list(state.get("processing_log") or [])
    try:
        data = extract(file_path)
    except ValueError as exc:
        logger.error("Extraction failed for %s: %s", file_path, exc)
        log.append("ingestion: EXTRACTION_FAILED")
        return {
            **state,
            "processing_stage": "failed",
            "is_valid": False,
            "error_message": f"EXTRACTION_FAILED: {exc}",
            "processing_log": log,
        }

    log.append(f"ingestion: extracted {data.invoice_id or file_path} ({data.overall_confidence.value})")
    return {
        **state,
        "invoice_data": data,
        "parsed_data": data.model_dump_log(),
        "extraction_confidence": _CONFIDENCE_TO_FLOAT[data.overall_confidence],
        "processing_stage": "extracted",
        "processing_log": log,
    }


def build_graph():
    """Compile the (currently single-node) ingestion workflow."""
    graph = StateGraph(InvoiceState)
    graph.add_node("ingest_invoice", ingest_invoice)
    graph.set_entry_point("ingest_invoice")
    graph.add_edge("ingest_invoice", END)
    return graph.compile()
