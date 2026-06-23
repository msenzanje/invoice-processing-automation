"""Run the dashboard with ``python -m dashboard``.

A thin launcher around uvicorn so the dashboard has a first-class entrypoint alongside
``python main.py`` for the CLI. Host/port/reload are overridable via env vars so a demo
can bind to ``0.0.0.0`` without editing code.
"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "dashboard.app:app",
        host=os.getenv("DASHBOARD_HOST", "127.0.0.1"),
        port=int(os.getenv("DASHBOARD_PORT", "8000")),
        reload=os.getenv("DASHBOARD_RELOAD", "").lower() in {"1", "true", "yes"},
    )


if __name__ == "__main__":
    main()
