"""
Mock payment API — stands in for the production banking integration.

The real system would POST to a banking API here; for the MVP this records intent
and returns a deterministic success envelope so the payment node has something
concrete to log. No real HTTP call is ever made (PROJECT_CONTEXT §7).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def mock_payment(vendor: str, amount: float) -> dict[str, Any]:
    """Pretend to pay ``vendor`` ``amount`` and return a success envelope.

    Returns a dict with a synthetic transaction id and an ISO timestamp, mirroring
    the shape a real banking API response would carry, so the audit record looks the
    same in the demo as it would in production.
    """
    transaction_id = f"PAY-{uuid.uuid4().hex[:12]}"
    logger.info("mock_payment: paid %s $%.2f (txn=%s)", vendor, amount, transaction_id)
    return {
        "status": "success",
        "transaction_id": transaction_id,
        "vendor": vendor,
        "amount": amount,
        "paid_at": datetime.now(timezone.utc).isoformat(),
    }
