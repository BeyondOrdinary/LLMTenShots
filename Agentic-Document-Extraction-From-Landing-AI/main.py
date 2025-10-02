# main.py - Core orchestration
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass,field
import numpy as np 
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
    type: str
    bbox: tuple
    page_num: int
    relationships: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary"""
        return {
            'id': self.id,
            'content': self.content,
            'type': self.type,
            'bbox': [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in self.bbox],
            'page_num': int(self.page_num),
            'relationships': self.relationships,
            'metadata': self._serialize_metadata(self.metadata),
            'confidence': float(self.confidence)
        }
    
    def _serialize_metadata(self, metadata: Dict) -> Dict:
        """Serialize metadata to be JSON compatible"""
        serialized = {}
        for key, value in metadata.items():
            if isinstance(value, (np.integer, np.floating)):
                serialized[key] = float(value)
            elif isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, dict):
                serialized[key] = self._serialize_metadata(value)
            elif isinstance(value, list):
                serialized[key] = [self._serialize_item(item) for item in value]
            else:
                serialized[key] = value
        return serialized
    
    def _serialize_item(self, item):
        """Serialize individual items"""
        if isinstance(item, (np.integer, np.floating)):
            return float(item)
        elif isinstance(item, np.ndarray):
            return item.tolist()
        elif isinstance(item, dict):
            return self._serialize_metadata(item)
        else:
            return item
@dataclass
class LayoutAnalysis:
    chunks: List[SemanticChunk]
    document_structure: Dict[str, Any]
    semantic_relationships: List[Dict[str, Any]]