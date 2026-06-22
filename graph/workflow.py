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

from agents.approval import approve_invoice
from agents.ingestion import extract
from agents.payment import escalation_node, log_node, payment_node
from agents.validation import validate_invoice
from models.invoice_data import Confidence
from models.results import ApprovalDecision
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


# Routing target for each terminal branch off the approval decision.
_DECISION_ROUTE = {
    ApprovalDecision.APPROVED: "process_payment",
    ApprovalDecision.REJECTED: "log_rejection",
    ApprovalDecision.NEEDS_REVIEW: "escalate_review",
}


def route_after_approval(state: InvoiceState) -> str:
    """Conditional edge: pick the terminal node from ``approval_result.decision``.

    A missing ``approval_result`` means the invoice died upstream (failed ingestion,
    so approval no-opped). Rather than drop it, route to the rejection log so a dead
    invoice still terminates cleanly with an audit record (fail-forward).
    """
    result = state.get("approval_result")
    if result is None:
        logger.info("routing %s: no approval_result -> log_rejection", state.get("invoice_id"))
        return "log_rejection"
    # .get with an escalate default: a future ApprovalDecision member not yet mapped
    # here routes to human review rather than raising KeyError into the graph, keeping
    # the router bound to a "never raises" contract and failing
    # in the same safe direction approve_invoice already uses for LLM errors.
    return _DECISION_ROUTE.get(result.decision, "escalate_review")


def build_graph():
    """Compile the full ingestion -> validation -> approval -> terminal workflow.

    The pipeline is linear through approval, then fans out on the approval decision:
    approved -> payment, rejected -> rejection log, needs_review -> human escalation.
    Every node is fail-forward (state in, state out, never raises), so a failed
    extraction or an unreachable LLM still flows to a clean terminal node and END.
    """
    graph = StateGraph(InvoiceState)
    graph.add_node("ingest_invoice", ingest_invoice)
    graph.add_node("validate_invoice", validate_invoice)
    graph.add_node("approve_invoice", approve_invoice)
    graph.add_node("process_payment", payment_node)
    graph.add_node("log_rejection", log_node)
    graph.add_node("escalate_review", escalation_node)

    graph.set_entry_point("ingest_invoice")
    graph.add_edge("ingest_invoice", "validate_invoice")
    graph.add_edge("validate_invoice", "approve_invoice")
    graph.add_conditional_edges(
        "approve_invoice",
        route_after_approval,
        {
            "process_payment": "process_payment",
            "log_rejection": "log_rejection",
            "escalate_review": "escalate_review",
        },
    )
    graph.add_edge("process_payment", END)
    graph.add_edge("log_rejection", END)
    graph.add_edge("escalate_review", END)
    return graph.compile()
