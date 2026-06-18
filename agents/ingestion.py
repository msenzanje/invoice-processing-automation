"""
Ingestion Agent — turns a raw invoice file into a structured, confidence-scored
:class:`~models.invoice_data.InvoiceData` object.

Two-pass design:

* **Pass 1 (deterministic):** fast, zero-cost, format-specific extractors
  (``pdfplumber``/``PyMuPDF`` for PDF, compiled regexes for TXT, ``pandas`` for
  CSV, ``json`` for JSON). Clean matches earn ``HIGH`` confidence; fallback
  matches earn ``MEDIUM``.
* **Pass 2 (LLM):** invoked only when Pass 1 leaves a required field missing or
  low-confidence. A self-correcting retry loop re-prompts the model on validation
  failure (up to ``MAX_RETRIES``). LLM results are *merged* into the Pass 1
  results so a ``HIGH``-confidence deterministic field is never overwritten.

Public entry point: :func:`extract`.
"""

from __future__ import annotations

import io
import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from models.invoice_data import CONFIDENCE_FIELDS, Confidence, InvoiceData, LineItem
from tools.llm import LLMProtocol, get_llm_client, strip_json

# Backwards-compatible alias: the JSON stripper now lives in tools.llm (shared by
# the approval agent). Existing call sites in this module use the short name.
_strip_json = strip_json

logger = logging.getLogger(__name__)

# --- Tunables -----------------------------------------------------------------
MAX_RETRIES = 3
PDF_TEXT_MIN_CHARS = 50  # below this, a PDF text layer is treated as scanned
REQUIRED_FIELDS: tuple[str, ...] = ("vendor", "amount", "due_date")
DATE_FORMATS = (
    "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y",
    "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y", "%d-%b-%Y",
)

# --- TXT regexes (compiled once) ----------------------------------------------
# Vendor: match an explicit "Vendor/Supplier/Seller" label first (HIGH). A bare
# "From:" header is a weak signal (email senders, etc.) handled separately as a
# MEDIUM fallback. The value stops at a 2+ space column gap OR before a following
# "Label:" token, so a second field sharing the line (common in PDF layouts where
# whitespace collapses, e.g. "Vendor: X  Due: Y") is not swallowed into the value.
_VENDOR_STOP = r"(?:\s{2,}|\s+\w+\s*:|\s*$)"
_VENDOR_RE = re.compile(rf"^\s*(?:vendor|supplier|seller)\s*:?\s*(.+?){_VENDOR_STOP}", re.I | re.M)
_FROM_RE = re.compile(rf"^\s*from\s*:?\s*(.+?){_VENDOR_STOP}", re.I | re.M)
_TOTAL_RE = re.compile(
    r"(?<!sub)(?:total\s*amount|grand\s*total|amount\s*due|total)\s*:?\s*\$?\s*([\d,]+\.\d{2})",
    re.I,
)
_AMOUNT_FALLBACK_RE = re.compile(r"\$\s*([\d,]+\.\d{2})")
# Due date may appear mid-line (PDF column layouts); capture up to a column gap or
# line end. The character class excludes ':' so it stops before a following label.
_DUE_DATE_RE = re.compile(
    r"\bdue\s*(?:date)?\s*:?\s*([A-Za-z0-9][A-Za-z0-9,/\- ]*?)(?:\s{2,}|\s*$)", re.I | re.M
)
_INVOICE_ID_RE = re.compile(
    r"\binv(?:oice)?\.?\s*(?:number|num|no\.?|#|id)?\s*[:#-]?\s*((?:INV[- ]?)?\d[\w-]*)", re.I
)
# Optional leading bullet ("- ", "* ", "• ") so emailed item lists are captured.
_LINE_ITEM_RE = re.compile(
    r"^[ \t]*[-*•]?\s*([A-Za-z][\w ]*?)\s+(?:qty[:\s]*|x\s*)?(\d+)\b[^$\n]*\$\s*([\d,]+(?:\.\d{2})?)",
    re.I | re.M,
)

# --- JSON key aliases ---------------------------------------------------------
_VENDOR_KEYS = ("vendor", "supplier", "seller", "vendor_name", "from")
_AMOUNT_KEYS = ("total", "total_amount", "amount", "amount_due", "grand_total")
_DUE_KEYS = ("due_date", "due", "payment_due", "duedate")
_ITEMS_KEYS = ("line_items", "items", "lineitems")
_ID_KEYS = ("invoice_number", "invoice_id", "invoice_no", "number")


@dataclass
class ExtractedField:
    """One field recovered by Pass 1, with the confidence we place in it."""

    value: Any
    confidence: Confidence


Fields = dict[str, ExtractedField]


# =============================================================================
# Small parsing helpers
# =============================================================================
def _parse_amount(raw: str) -> Optional[float]:
    """Parse a currency-ish string ('$1,250.00') into a float, or None."""
    if raw is None:
        return None
    cleaned = str(raw).replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except (TypeError, ValueError):
        return None


def _parse_date(raw: str) -> Optional[date]:
    """Parse a date written in any of several common formats, or None."""
    if raw is None:
        return None
    text = str(raw).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


# =============================================================================
# Pass 1 — TXT
# =============================================================================
def _extract_from_txt(path: Path) -> tuple[str, Fields]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    return raw, _parse_text_fields(raw)


def _parse_text_fields(raw: str) -> Fields:
    fields: Fields = {}
    vendor = _extract_text_vendor(raw)
    if vendor is not None:
        fields["vendor"] = vendor
    amount = _extract_text_amount(raw)
    if amount is not None:
        fields["amount"] = amount
    due = _DUE_DATE_RE.search(raw)
    if due:
        parsed = _parse_date(due.group(1).strip())
        if parsed is not None:
            fields["due_date"] = ExtractedField(parsed, Confidence.HIGH)
    inv = _INVOICE_ID_RE.search(raw)
    if inv:
        fields["invoice_id"] = ExtractedField(inv.group(1).strip(), Confidence.HIGH)
    items = _extract_text_items(raw)
    if items:
        fields["items"] = ExtractedField(items, Confidence.HIGH)
    return fields


def _extract_text_vendor(raw: str) -> Optional[ExtractedField]:
    """Explicit vendor label wins (HIGH); a bare 'From:' header is a MEDIUM hint."""
    match = _VENDOR_RE.search(raw)
    if match and match.group(1).strip():
        return ExtractedField(match.group(1).strip(), Confidence.HIGH)
    fallback = _FROM_RE.search(raw)
    if fallback and fallback.group(1).strip():
        return ExtractedField(fallback.group(1).strip(), Confidence.MEDIUM)
    return None


def _extract_text_amount(raw: str) -> Optional[ExtractedField]:
    """Prefer an explicit 'Total' (HIGH); else fall back to the largest $ figure."""
    total = _TOTAL_RE.search(raw)
    if total:
        value = _parse_amount(total.group(1))
        if value is not None:
            return ExtractedField(value, Confidence.HIGH)
    candidates = [_parse_amount(m) for m in _AMOUNT_FALLBACK_RE.findall(raw)]
    values = [v for v in candidates if v is not None]
    if values:
        return ExtractedField(max(values), Confidence.MEDIUM)
    return None


def _extract_text_items(raw: str) -> list[LineItem]:
    items: list[LineItem] = []
    for match in _LINE_ITEM_RE.finditer(raw):
        price = _parse_amount(match.group(3))
        if price is None:
            continue
        items.append(
            LineItem(item=match.group(1).strip(), quantity=int(match.group(2)), unit_price=price)
        )
    return items


# =============================================================================
# Pass 1 — PDF
# =============================================================================
def _extract_from_pdf(path: Path) -> tuple[str, Fields]:
    raw = _read_pdf_text(path)
    return raw, _parse_text_fields(raw)


def _read_pdf_text(path: Path) -> str:
    """pdfplumber first; fall back to PyMuPDF when the text layer is sparse."""
    text = ""
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            text = "\n".join((page.extract_text() or "") for page in pdf.pages)
    except Exception as exc:  # pdfplumber raises a variety of parse errors
        logger.warning("pdfplumber failed on %s: %s", path.name, exc)
    if len(text.strip()) >= PDF_TEXT_MIN_CHARS:
        return text
    logger.info("PDF text layer sparse (%d chars); trying PyMuPDF", len(text.strip()))
    try:
        import fitz  # PyMuPDF

        with fitz.open(path) as doc:
            text = "\n".join(page.get_text() for page in doc)
    except Exception as exc:
        logger.warning("PyMuPDF failed on %s: %s", path.name, exc)
    return text


# =============================================================================
# Pass 1 — CSV
# =============================================================================
def _extract_from_csv(path: Path) -> tuple[str, Fields]:
    import pandas as pd

    raw = path.read_text(encoding="utf-8", errors="replace")
    frame = pd.read_csv(io.StringIO(raw), dtype=str, keep_default_na=False)
    header = [str(c).strip().lower() for c in frame.columns]
    if header == ["field", "value"]:
        return raw, _parse_vertical_csv(frame)
    return raw, _parse_columnar_csv(frame)


def _parse_vertical_csv(frame: Any) -> Fields:
    """Handle the key/value layout: a 'field,value' row per attribute."""
    pairs: dict[str, str] = {}
    items: list[dict[str, str]] = []
    pending: dict[str, str] = {}
    for _, row in frame.iterrows():
        key = str(row.iloc[0]).strip().lower()
        value = str(row.iloc[1]).strip()
        if key == "item":
            if pending:
                items.append(pending)
            pending = {"item": value}
        elif key in ("quantity", "qty") and pending:
            pending["quantity"] = value
        elif key in ("unit_price", "unit price", "price") and pending:
            pending["unit_price"] = value
        else:
            pairs[key] = value
    if pending:
        items.append(pending)
    return _csv_fields(pairs.get("vendor"), pairs.get("total") or pairs.get("amount"),
                       pairs.get("due_date"), pairs.get("invoice_number"), items)


def _parse_columnar_csv(frame: Any) -> Fields:
    """Handle the tabular layout: one row per line item plus trailing total rows."""
    frame.columns = [str(c).strip() for c in frame.columns]
    colmap = _csv_colmap(frame.columns)
    items: list[dict[str, str]] = []
    item_col = colmap.get("item")
    for _, row in frame.iterrows():
        name = str(row[item_col]).strip() if item_col else ""
        if not name:
            continue
        items.append({
            "item": name,
            "quantity": str(row[colmap["quantity"]]).strip() if "quantity" in colmap else "0",
            "unit_price": str(row[colmap["unit_price"]]).strip() if "unit_price" in colmap else "0",
        })
    vendor = _first_nonempty(frame, colmap.get("vendor"))
    due = _first_nonempty(frame, colmap.get("due_date"))
    invoice_id = _first_nonempty(frame, colmap.get("invoice_id"))
    total = _columnar_total(frame, colmap.get("line_total") or frame.columns[-1])
    return _csv_fields(vendor, total, due, invoice_id, items)


def _csv_colmap(columns: Any) -> dict[str, str]:
    """Map canonical field names to the actual column headers present."""
    aliases = {
        "item": ("item", "description", "product"),
        "quantity": ("qty", "quantity"),
        "unit_price": ("unit price", "unit_price", "rate", "price"),
        "line_total": ("line total", "line_total", "amount"),
        "vendor": ("vendor", "supplier", "seller"),
        "due_date": ("due date", "due_date", "due"),
        "invoice_id": ("invoice number", "invoice_number", "invoice", "inv #", "invoice #"),
    }
    lowered = {str(c).strip().lower(): str(c) for c in columns}
    mapping: dict[str, str] = {}
    for canonical, names in aliases.items():
        for name in names:
            if name in lowered:
                mapping[canonical] = lowered[name]
                break
    return mapping


def _first_nonempty(frame: Any, column: Optional[str]) -> Optional[str]:
    if not column or column not in frame.columns:
        return None
    for value in frame[column]:
        text = str(value).strip()
        if text:
            return text
    return None


def _columnar_total(frame: Any, total_col: str) -> Optional[str]:
    """Return the value on the 'Total:' row (excluding the 'Subtotal:' row)."""
    for _, row in frame.iterrows():
        joined = " ".join(str(v) for v in row.values).lower()
        if "total" in joined and "subtotal" not in joined:
            value = _parse_amount(str(row[total_col]))
            if value is not None:
                return str(value)
    return None


def _csv_fields(
    vendor: Optional[str],
    total: Optional[str],
    due: Optional[str],
    invoice_id: Optional[str],
    items: list[dict[str, str]],
) -> Fields:
    fields: Fields = {}
    if vendor and vendor.strip():
        fields["vendor"] = ExtractedField(vendor.strip(), Confidence.HIGH)
    amount = _parse_amount(total) if total is not None else None
    if amount is not None:
        fields["amount"] = ExtractedField(amount, Confidence.HIGH)
    parsed_due = _parse_date(due) if due else None
    if parsed_due is not None:
        fields["due_date"] = ExtractedField(parsed_due, Confidence.HIGH)
    if invoice_id and invoice_id.strip():
        fields["invoice_id"] = ExtractedField(invoice_id.strip(), Confidence.HIGH)
    line_items = _build_line_items(items)
    if line_items:
        fields["items"] = ExtractedField(line_items, Confidence.HIGH)
    return fields


def _build_line_items(rows: list[dict[str, str]]) -> list[LineItem]:
    items: list[LineItem] = []
    for row in rows:
        price = _parse_amount(row.get("unit_price", "0"))
        try:
            quantity = int(float(row.get("quantity", "0") or 0))
        except (TypeError, ValueError):
            quantity = 0
        items.append(LineItem(item=row.get("item", ""), quantity=quantity, unit_price=price or 0.0))
    return items


# =============================================================================
# Pass 1 — JSON
# =============================================================================
def _extract_from_json(path: Path) -> tuple[str, Fields]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Malformed JSON in %s: %s; routing to LLM", path.name, exc)
        return raw, {}
    if not isinstance(data, dict):
        return raw, {}
    return raw, _parse_json_fields(data)


def _parse_json_fields(data: dict[str, Any]) -> Fields:
    fields: Fields = {}
    vendor = _normalize_vendor(_lookup(data, _VENDOR_KEYS))
    if vendor:
        fields["vendor"] = ExtractedField(vendor, Confidence.HIGH)
    amount = _parse_amount(_lookup(data, _AMOUNT_KEYS))
    if amount is not None:
        fields["amount"] = ExtractedField(amount, Confidence.HIGH)
    due = _parse_date(_lookup(data, _DUE_KEYS))
    if due is not None:
        fields["due_date"] = ExtractedField(due, Confidence.HIGH)
    invoice_id = _lookup(data, _ID_KEYS)
    if invoice_id:
        fields["invoice_id"] = ExtractedField(str(invoice_id), Confidence.HIGH)
    items = _normalize_json_items(_lookup(data, _ITEMS_KEYS))
    if items:
        fields["items"] = ExtractedField(items, Confidence.HIGH)
    return fields


def _lookup(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    lowered = {str(k).lower(): v for k, v in data.items()}
    for key in keys:
        if key in lowered and lowered[key] not in (None, ""):
            return lowered[key]
    return None


def _normalize_vendor(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        value = value.get("name")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _normalize_json_items(value: Any) -> list[LineItem]:
    if not isinstance(value, list):
        return []
    items: list[LineItem] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        name = entry.get("item") or entry.get("name") or entry.get("description") or ""
        quantity = entry.get("quantity", entry.get("qty", 0))
        price = entry.get("unit_price", entry.get("price", 0))
        try:
            items.append(LineItem(item=str(name), quantity=int(quantity), unit_price=float(price)))
        except (TypeError, ValueError):
            continue
    return items


# =============================================================================
# Confidence gate (Pass 1 -> Pass 2 decision)
# =============================================================================
def _needs_llm(fields: Fields) -> bool:
    """True if any required field is absent, or any scored field is LOW."""
    for name in REQUIRED_FIELDS:
        field = fields.get(name)
        if field is None or field.value in (None, "", []):
            return True
    for name in CONFIDENCE_FIELDS:
        field = fields.get(name)
        if field is not None and field.confidence is Confidence.LOW:
            return True
    return False


def _build_invoice_data(fields: Fields) -> InvoiceData:
    """Promote a sufficient Pass 1 result directly to an InvoiceData object."""
    confidence = {
        name: fields[name].confidence if name in fields else Confidence.LOW
        for name in CONFIDENCE_FIELDS
    }
    items = fields["items"].value if "items" in fields else []
    return InvoiceData(
        vendor=fields["vendor"].value,
        amount=fields["amount"].value,
        items=items,
        due_date=fields["due_date"].value,
        invoice_id=fields["invoice_id"].value if "invoice_id" in fields else None,
        field_confidence=confidence,
    )


# =============================================================================
# Pass 2 — LLM extraction with self-correction
# =============================================================================
_EXTRACTION_INSTRUCTIONS = """You are an invoice data extraction engine. Extract the fields below \
from the invoice text and return ONLY a single JSON object — no markdown, no commentary.

{
  "vendor": "<company that issued the invoice>",
  "amount": <final total due as a plain number, no currency symbol or commas>,
  "items": [{"item": "<name>", "quantity": <integer>, "unit_price": <number>}],
  "due_date": "<YYYY-MM-DD>",
  "invoice_id": "<invoice number, or null>"
}

Rules:
- amount is the final total due (including tax) as a plain number.
- due_date MUST be ISO format YYYY-MM-DD; infer it if written in words.
- Make a best inference where data is ambiguous, but do not invent line items.

Invoice text:
---
{raw_text}
---"""


def _build_extraction_prompt(raw_text: str) -> str:
    return _EXTRACTION_INSTRUCTIONS.replace("{raw_text}", raw_text)


def _build_correction_prompt(raw_text: str, previous: str, error: str) -> str:
    return (
        _build_extraction_prompt(raw_text)
        + f"\n\nYour previous response was:\n{previous}\n\n"
        f"It failed validation with this error:\n{error}\n\n"
        "Return a corrected JSON object that resolves this error. Output ONLY the JSON."
    )


def _extract_with_llm(raw_text: str, fields: Fields, llm: LLMProtocol) -> InvoiceData:
    prompt = _build_extraction_prompt(raw_text)
    attempts: list[tuple[str, str]] = []
    for attempt in range(MAX_RETRIES):
        logger.info("Pass 2 LLM extraction attempt %d/%d", attempt + 1, MAX_RETRIES)
        response = llm.complete(prompt)
        try:
            llm_data = InvoiceData.model_validate_json(_strip_json(response))
            return _merge(fields, llm_data)
        except (ValidationError, ValueError) as exc:
            error = str(exc)
            attempts.append((response, error))
            logger.warning("Pass 2 attempt %d failed validation: %s", attempt + 1, error)
            prompt = _build_correction_prompt(raw_text, response, error)
    raise ValueError(_format_failure(raw_text, attempts))


def _merge(fields: Fields, llm_data: InvoiceData) -> InvoiceData:
    """Keep HIGH-confidence Pass 1 fields; fill the rest from the LLM (MEDIUM)."""
    merged: dict[str, Any] = {}
    confidence: dict[str, Confidence] = {}
    for name in CONFIDENCE_FIELDS:
        field = fields.get(name)
        if field is not None and field.confidence is Confidence.HIGH and field.value not in (None, "", []):
            merged[name] = field.value
            confidence[name] = Confidence.HIGH
        else:
            merged[name] = getattr(llm_data, name)
            confidence[name] = Confidence.MEDIUM
    invoice_id = fields["invoice_id"].value if "invoice_id" in fields else llm_data.invoice_id
    return InvoiceData(
        vendor=merged["vendor"],
        amount=merged["amount"],
        items=merged["items"],
        due_date=merged["due_date"],
        invoice_id=invoice_id,
        field_confidence=confidence,
    )


def _format_failure(raw_text: str, attempts: list[tuple[str, str]]) -> str:
    lines = [f"LLM extraction failed after {len(attempts)} attempts."]
    for index, (response, error) in enumerate(attempts, start=1):
        lines.append(f"--- Attempt {index} ---\nOutput: {response}\nError: {error}")
    lines.append(f"--- Raw text ---\n{raw_text}")
    return "\n".join(lines)


# =============================================================================
# Public entry point
# =============================================================================
def detect_format(path: Path) -> str:
    """Return the lowercase file extension without the dot (e.g. 'pdf')."""
    return path.suffix.lower().lstrip(".")


def _dispatch_pass1(path: Path, fmt: str) -> tuple[str, Fields]:
    match fmt:
        case "pdf":
            return _extract_from_pdf(path)
        case "txt":
            return _extract_from_txt(path)
        case "csv":
            return _extract_from_csv(path)
        case "json":
            return _extract_from_json(path)
        case _:
            logger.warning("Unsupported format %r; routing raw text to the LLM", fmt)
            return path.read_text(encoding="utf-8", errors="replace"), {}


def extract(file_path: "str | Path", llm_client: Optional[LLMProtocol] = None) -> InvoiceData:
    """Extract structured invoice data from ``file_path``.

    Runs deterministic Pass 1, consults the confidence gate, and only then falls
    back to the LLM (Pass 2) with self-correction. Raises ``ValueError`` if Pass 2
    exhausts its retries, so the orchestrator can route to human review.
    """
    path = Path(file_path)
    fmt = detect_format(path)
    logger.info("Ingesting %s (format=%s)", path.name, fmt)

    raw_text, fields = _dispatch_pass1(path, fmt)
    logger.info("Pass 1 recovered fields: %s", sorted(fields))

    if not _needs_llm(fields):
        data = _build_invoice_data(fields)
        logger.info("Pass 1 sufficient; LLM skipped (overall=%s)", data.overall_confidence.value)
        return data

    logger.info("Pass 1 incomplete; invoking Pass 2 LLM")
    llm = llm_client or get_llm_client()
    data = _extract_with_llm(raw_text, fields, llm)
    logger.info("Pass 2 complete (overall=%s)", data.overall_confidence.value)
    return data
