# main.py - Core orchestration
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class DocumentType(Enum):
    SCANNED_PDF = "scanned_pdf"
    DIGITAL_PDF = "digital_pdf"
    IMAGE = "image"
    TABLE = "table"
    FORM = "form"

@dataclass
class SemanticChunk:
    id: str
    content: str
    type: str  # text, table, form, image, header, etc.
    bbox: tuple  # (x1, y1, x2, y2)
    page_num: int
    relationships: List[str]
    metadata: Dict[str, Any]
    confidence: float

@dataclass
class LayoutAnalysis:
    chunks: List[SemanticChunk]
    document_structure: Dict[str, Any]
    semantic_relationships: List[Dict[str, Any]]