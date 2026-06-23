"""
Invoice Processing Automation — FastAPI server.

A read-only stakeholder view over the durable audit trail. It never touches the graph
or the agents; it reads ``logs/audit.jsonl`` through :mod:`dashboard.audit_source` and
serves it as a single-page dashboard plus a polling endpoint that surfaces new records
as the pipeline appends them.

The page shell is static HTML/CSS/JS served from ``dashboard/static`` and
``dashboard/templates``; all data crosses the wire as JSON (the initial payload is
embedded at first paint so the page renders instantly without a flash of empty state,
then ``/api/summary`` is polled for updates). No server-side templating engine is
needed, which keeps the dependency list lean.

Run:
    uvicorn dashboard.app:app --reload
    # or: python -m dashboard
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from dashboard.audit_source import DashboardData, load_dashboard_data

logger = logging.getLogger(__name__)

_DASHBOARD_DIR = Path(__file__).resolve().parent
_TEMPLATE_PATH = _DASHBOARD_DIR / "templates" / "index.html"
_STATIC_DIR = _DASHBOARD_DIR / "static"

app = FastAPI(
    title="Invoice Processing Automation",
    description="Read-only stakeholder view over the invoice-processing audit trail.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


def _payload(data: DashboardData) -> dict[str, Any]:
    """The JSON shape the frontend renders from — KPIs, rows, feed, chip counts.

    ``asdict`` flattens the frozen dataclasses; the frontend reads these exact keys, so
    the dataclass field names are the wire contract. Rows and feed are already
    newest-first from the source.
    """
    return {
        "kpis": asdict(data.kpis),
        "rows": [asdict(row) for row in data.rows],
        "feed": [asdict(item) for item in data.feed],
        "counts": data.counts,
        "total": data.total,
        "latest_timestamp": data.latest_timestamp,
    }


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the dashboard shell with the initial data payload embedded.

    Embedding the first payload means the page paints fully populated — no empty-state
    flash, no extra round trip — and then the client polls ``/api/summary`` for changes.
    """
    data = load_dashboard_data()
    html = _TEMPLATE_PATH.read_text(encoding="utf-8")
    bootstrap = json.dumps(_payload(data))
    # The template carries a single placeholder the server replaces with the live
    # payload; everything else in the file is static and cache-friendly.
    html = html.replace("__BOOTSTRAP_DATA__", bootstrap)
    return HTMLResponse(content=html)


@app.get("/api/summary")
def summary() -> JSONResponse:
    """The full dashboard payload as JSON — what the client polls for live updates.

    One endpoint, not several: KPIs, rows, and the feed all recompute from the same
    file read, so splitting them would mean three reads of the same log for one refresh.
    The client diffs ``latest_timestamp`` to decide whether anything actually changed.
    """
    return JSONResponse(content=_payload(load_dashboard_data()))


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness probe — confirms the server is up and the audit log is reachable."""
    data = load_dashboard_data()
    return {"status": "ok", "records": str(data.total)}
