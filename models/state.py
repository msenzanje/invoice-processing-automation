from typing import TypedDict, Optional, Dict, Any, List

from models.invoice_data import InvoiceData
from models.results import ValidationResult


class InvoiceState(TypedDict):
    """
    Shared state contract for all agents in the invoice processing pipeline.
    Every agent reads from and writes to this object.
    """
    
    # File metadata
    invoice_id: str  # Unique identifier (e.g., "invoice_1001")
    filename: str  # Original filename
    file_format: str  # Format: "txt", "json", "csv", "xml"
    file_path: str  # Full path to the invoice file
    
    # Raw content
    raw_content: str  # Original file content as string
    
    # Parsed data (from raw content)
    parsed_data: Dict[str, Any]
    # Expected keys in parsed_data:
    # - invoice_number: str
    # - date: str
    # - vendor: str
    # - total_amount: float
    # - items: List[Dict[str, Any]]
    # - description: str (optional)

    # Structured extraction result (Phase 2 — Ingestion Agent).
    # Rich, typed, per-field-confidence object produced by agents.ingestion.extract().
    # parsed_data above is kept populated (from invoice_data.model_dump_log()) for
    # back-compat; downstream agents should prefer invoice_data.
    invoice_data: Optional[InvoiceData]
    
    # Processing state
    processing_stage: str  # "pending", "extracted", "validated", "stored"
    is_valid: Optional[bool]
    validation_errors: List[str]

    # Structured validation result (Phase 3 — Validation Agent).
    # Typed flags produced by agents.validation.validate_invoice(). The approval
    # agent routes on these (specifically flag severity). None until validation runs.
    validation_result: Optional[ValidationResult]
    
    # Database/Inventory state
    inventory_updated: bool 
    db_record_id: Optional[int]  # Database record ID if stored
    
    # Results and metadata
    extraction_confidence: float  # Confidence score of extraction (0.0-1.0)
    processing_log: List[str]  # Log of processing steps
    error_message: Optional[str]  # Error details if processing failed
    
    # Timestamp
    timestamp: str  # ISO format timestamp of when processing started
