# output/rag_formatter.py
from typing import List, Dict, Any, Optional
import json
from layout_extraction_pipeline import LayoutExtractionPipeline
from relationship_extractor import SemanticRelationshipExtractor
from zero_shot_classifier import ZeroShotLayoutClassifier
from multi_modal_parser import MultiModalParser
from main import SemanticChunk

class RAGFormatter:
    def __init__(self):
        pass
    
    def format_for_rag(self, extraction_result: Dict) -> List[Dict]:
        """Format extraction results for RAG applications"""
        
        formatted_chunks = []
        chunks = extraction_result['chunks']
        relationships = extraction_result['relationships']
        
        # Create relationship lookup
        rel_lookup = self._build_relationship_lookup(relationships)
        
        for chunk in chunks:
            # Enrich chunk with relationships
            chunk_rels = rel_lookup.get(chunk['id'], [])
            
            # Create RAG-friendly format
            rag_chunk = {
                'id': chunk['id'],
                'content': chunk['content'],
                'type': chunk['type'],
                'page': chunk['page_num'],
                'bbox': chunk['bbox'],
                'confidence': chunk['confidence'],
                'metadata': {
                    'classification': chunk.get('metadata', {}).get('classification', {}),
                    'relationships': chunk_rels
                },
                # Context-aware content for better RAG performance
                'contextual_content': self._build_contextual_content(chunk, chunk_rels, chunks)
            }
            
            formatted_chunks.append(rag_chunk)
        
        return formatted_chunks
    
    def _build_relationship_lookup(self, relationships: List[Dict]) -> Dict[str, List]:
        """Build efficient relationship lookup"""
        lookup = {}
        for rel in relationships:
            source = rel['source']
            if source not in lookup:
                lookup[source] = []
            lookup[source].append(rel)
        return lookup
    
    def _build_contextual_content(self, chunk: Dict, relationships: List[Dict], all_chunks: List[Dict]) -> str:
        """Build contextually rich content for better RAG performance"""
        
        content_parts = [chunk['content']]
        
        # Add related content
        for rel in relationships[:3]:  # Limit to top 3 relationships
            target_id = rel['target']
            target_chunk = next((c for c in all_chunks if c['id'] == target_id), None)
            if target_chunk:
                content_parts.append(f"[{rel['type']}: {target_chunk['content']}]")
        
        return " ".join(content_parts)
    
    def save_rag_format(self, formatted_chunks: List[Dict], output_path: str):
        """Save RAG-formatted chunks to file"""
        with open(output_path, 'w') as f:
            json.dump(formatted_chunks, f, indent=2)

# Complete usage example
async def process_document_for_rag(input_path: str, output_path: str):
    """Complete pipeline from document to RAG-ready format"""
    
    # Initialize pipeline
    pipeline = LayoutExtractionPipeline()
    formatter = RAGFormatter()
    
    # Process document
    print(f"Processing {input_path}...")
    extraction_result = await pipeline.process_document(input_path)
    
    # Format for RAG
    print("Formatting for RAG...")
    rag_chunks = formatter.format_for_rag(extraction_result)
    
    # Save results
    formatter.save_rag_format(rag_chunks, output_path)
    print(f"RAG-ready chunks saved to {output_path}")
    
    return rag_chunks

# Example usage
if __name__ == "__main__":
    import asyncio
    asyncio.run(process_document_for_rag("input_document.pdf", "rag_output.json"))