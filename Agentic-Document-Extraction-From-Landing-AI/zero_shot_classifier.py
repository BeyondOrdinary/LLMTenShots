# models/zero_shot_classifier.py
from transformers import pipeline
import torch
from main import SemanticChunk
from typing import List, Dict, Any, Optional

class ZeroShotLayoutClassifier:
    def __init__(self):
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        self.layout_labels = [
            "header", "footer", "title", "subtitle", "body_text",
            "table", "form_field", "image_caption", "list_item",
            "section_header", "page_number", "footnote"
        ]
    
    def classify_chunk(self, content: str, context: str = "") -> Dict[str, float]:
        """Classify document chunk using zero-shot learning"""
        
        # Add context to improve classification
        if context:
            text_to_classify = f"{context}: {content}"
        else:
            text_to_classify = content
        
        result = self.classifier(text_to_classify, self.layout_labels)
        
        # Return top predictions with confidence scores
        return dict(zip(result['labels'], result['scores']))
    
    def batch_classify(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Batch classify multiple chunks"""
        classified_chunks = []
        
        for chunk in chunks:
            # Get context from nearby chunks
            context = self._get_context(chunk, chunks)
            classification = self.classify_chunk(chunk.content, context)
            
            # Update chunk with classification
            chunk.metadata['classification'] = classification
            chunk.type = max(classification.items(), key=lambda x: x[1])[0]
            
            classified_chunks.append(chunk)
        
        return classified_chunks
    
    def _get_context(self, chunk: SemanticChunk, all_chunks: List[SemanticChunk]) -> str:
        """Get context from nearby chunks"""
        # Simple context: nearby chunks on same page
        nearby_chunks = [
            c for c in all_chunks 
            if c.page_num == chunk.page_num and c.id != chunk.id
        ]
        
        # Sort by proximity
        nearby_chunks.sort(key=lambda x: self._distance(chunk.bbox, x.bbox))
        
        # Take top 3 nearest chunks as context
        context_parts = [c.content[:50] for c in nearby_chunks[:3]]
        return " ".join(context_parts)
    
    def _distance(self, bbox1: tuple, bbox2: tuple) -> float:
        """Calculate distance between two bounding boxes"""
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        return ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5