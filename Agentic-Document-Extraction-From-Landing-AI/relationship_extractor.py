# semantic/relationship_extractor.py
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Dict, Tuple,Any,Optional
from zero_shot_classifier import ZeroShotLayoutClassifier
from multi_modal_parser import MultiModalParser
from main import SemanticChunk
from main import Optional

class SemanticRelationshipExtractor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
    def extract_relationships(self, chunks: List[SemanticChunk]) -> List[Dict]:
        """Extract semantic relationships between document elements"""
        
        # 1. Spatial relationships
        spatial_relations = self._extract_spatial_relationships(chunks)
        
        # 2. Semantic similarity relationships
        semantic_relations = self._extract_semantic_relationships(chunks)
        
        # 3. Content-based relationships
        content_relations = self._extract_content_relationships(chunks)
        
        # Combine all relationships
        all_relations = spatial_relations + semantic_relations + content_relations
        
        return all_relations
    
    def _extract_spatial_relationships(self, chunks: List[SemanticChunk]) -> List[Dict]:
        """Extract spatial relationships (above, below, left, right, inside)"""
        relations = []
        
        for i, chunk1 in enumerate(chunks):
            for j, chunk2 in enumerate(chunks):
                if i != j:
                    rel_type = self._get_spatial_relationship(chunk1.bbox, chunk2.bbox)
                    if rel_type:
                        relations.append({
                            'source': chunk1.id,
                            'target': chunk2.id,
                            'type': 'spatial',
                            'relation': rel_type,
                            'confidence': 0.9
                        })
        
        return relations

    def _find_nearby_text(self, chunks: List[SemanticChunk], bbox: tuple, exclude_index: int) -> List[SemanticChunk]:
        """Find nearby text chunks that might be labels for form fields"""
        nearby_texts = []
        x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        for i, chunk in enumerate(chunks):
            if i == exclude_index or chunk.type not in ['text', 'body_text', 'title']:
                continue
                
            # Check if text is nearby (above, left, or close)
            chunk_x, chunk_y, chunk_w, chunk_h = chunk.bbox[0], chunk.bbox[1], chunk.bbox[2] - chunk.bbox[0], chunk.bbox[3] - chunk.bbox[1]
            
            # Calculate distance
            distance = self._distance(bbox, chunk.bbox)
            
            # Check if it's in a label-like position (above or to the left)
            is_above = chunk_y < y and abs(chunk_x - x) < w * 2
            is_left = chunk_x < x and abs(chunk_y - y) < h * 2
            
            if (is_above or is_left) and distance < 300:  # Threshold in pixels
                nearby_texts.append(chunk)
        
        return nearby_texts

    def _find_table_cells(self, chunks: List[SemanticChunk], table_bbox: tuple, exclude_index: int) -> List[SemanticChunk]:
        """Find chunks that are inside the table bounding box"""
        table_cells = []
        table_x1, table_y1, table_x2, table_y2 = table_bbox
        
        for i, chunk in enumerate(chunks):
            if i == exclude_index:
                continue
                
            # Check if chunk is inside table
            chunk_x1, chunk_y1, chunk_x2, chunk_y2 = chunk.bbox
            
            if (chunk_x1 >= table_x1 and chunk_y1 >= table_y1 and 
                chunk_x2 <= table_x2 and chunk_y2 <= table_y2):
                table_cells.append(chunk)
        
        return table_cells

    def _distance(self, bbox1: tuple, bbox2: tuple) -> float:
        """Calculate distance between two bounding boxes"""
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        return ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
    
    def _get_spatial_relationship(self, bbox1: tuple, bbox2: tuple) -> Optional[str]:
        """Determine spatial relationship between two bounding boxes"""
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        
        # Calculate centers
        center1 = ((x1 + x2) / 2, (y1 + y2) / 2)
        center2 = ((x3 + x4) / 2, (y3 + y4) / 2)
        
        # Calculate relative positions
        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]
        
        # Determine relationship based on relative positions and overlap
        if abs(dx) > abs(dy):
            if dx > 0:
                return "right_of"
            else:
                return "left_of"
        else:
            if dy > 0:
                return "below"
            else:
                return "above"
    
    def _extract_semantic_relationships(self, chunks: List[SemanticChunk]) -> List[Dict]:
        """Extract semantic relationships using embedding similarity"""
        # Get embeddings for all chunks
        embeddings = self._get_embeddings([chunk.content for chunk in chunks])
        
        relations = []
        threshold = 0.7  # Similarity threshold
        
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                if similarity > threshold:
                    relations.append({
                        'source': chunks[i].id,
                        'target': chunks[j].id,
                        'type': 'semantic_similarity',
                        'similarity': float(similarity),
                        'confidence': similarity
                    })
        
        return relations
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get sentence embeddings"""
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).numpy()
                embeddings.append(embedding[0])
        return np.array(embeddings)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _extract_content_relationships(self, chunks: List[SemanticChunk]) -> List[Dict]:
        """Extract relationships based on content patterns"""
        relations = []
        
        for i, chunk in enumerate(chunks):
            # Check for form field relationships
            if chunk.type == 'form_field':
                # Look for nearby text that might be a label
                nearby_text = self._find_nearby_text(chunks, chunk.bbox, i)
                for text_chunk in nearby_text:
                    relations.append({
                        'source': chunk.id,
                        'target': text_chunk.id,
                        'type': 'form_label',
                        'confidence': 0.8
                    })
            
            # Check for table relationships
            elif chunk.type == 'table':
                # Find table cells
                table_cells = self._find_table_cells(chunks, chunk.bbox, i)
                for cell in table_cells:
                    relations.append({
                        'source': chunk.id,
                        'target': cell.id,
                        'type': 'table_cell',
                        'confidence': 0.9
                    })
        
        return relations