# pipeline/layout_extraction_pipeline.py
import uuid
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
from relationship_extractor import SemanticRelationshipExtractor
from zero_shot_classifier import ZeroShotLayoutClassifier
from multi_modal_parser import MultiModalParser
from main import SemanticChunk

class LayoutExtractionPipeline:
    def __init__(self):
        self.parser = MultiModalParser()
        self.relationship_extractor = SemanticRelationshipExtractor()
        self.classifier = ZeroShotLayoutClassifier()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def process_document(self, file_path: str) -> Dict[str, Any]:
        """Main pipeline for document processing"""
        
        # Step 1: Parse document into raw elements
        print("Parsing document...")
        raw_elements = await self.parser.parse_document(file_path)
        
        # Step 2: Convert to semantic chunks
        print("Converting to semantic chunks...")
        semantic_chunks = self._elements_to_chunks(raw_elements)
        
        # Step 3: Classify chunks
        print("Classifying chunks...")
        classified_chunks = self.classifier.batch_classify(semantic_chunks)
        
        # Step 4: Extract relationships
        print("Extracting relationships...")
        relationships = self.relationship_extractor.extract_relationships(classified_chunks)
        
        # Step 5: Build document structure
        print("Building document structure...")
        document_structure = self._build_document_structure(classified_chunks)
        
        # Step 6: Generate enriched output
        result = {
            'chunks': [chunk.__dict__ for chunk in classified_chunks],
            'relationships': relationships,
            'document_structure': document_structure,
            'metadata': {
                'total_chunks': len(classified_chunks),
                'processing_time': 'calculated_time',
                'document_path': file_path
            }
        }
        
        return result
    
    def _elements_to_chunks(self, elements: List[Dict]) -> List[SemanticChunk]:
        """Convert raw elements to semantic chunks"""
        chunks = []
        
        for element in elements:
            chunk = SemanticChunk(
                id=str(uuid.uuid4()),
                content=element.get('text', ''),
                type=element.get('type', 'unknown'),
                bbox=element.get('bbox', (0, 0, 0, 0)),
                page_num=element.get('page_num', 0),
                relationships=[],
                metadata=element.get('metadata', {}),
                confidence=element.get('confidence', 0.5)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _build_document_structure(self, chunks: List[SemanticChunk]) -> Dict[str, Any]:
        """Build hierarchical document structure"""
        structure = {
            'pages': {},
            'sections': {},
            'content_hierarchy': []
        }
        
        # Group by page
        for chunk in chunks:
            page_num = chunk.page_num
            if page_num not in structure['pages']:
                structure['pages'][page_num] = []
            structure['pages'][page_num].append(chunk.id)
        
        # Group by type
        type_groups = {}
        for chunk in chunks:
            chunk_type = chunk.type
            if chunk_type not in type_groups:
                type_groups[chunk_type] = []
            type_groups[chunk_type].append(chunk.id)
        
        structure['type_groups'] = type_groups
        
        return structure

# Usage example
async def main():
    pipeline = LayoutExtractionPipeline()
    
    # Process a document
    result = await pipeline.process_document("sample_document.pdf")
    
    # Save results
    import json
    with open("layout_extraction_result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print("Layout extraction completed!")

if __name__ == "__main__":
    asyncio.run(main())