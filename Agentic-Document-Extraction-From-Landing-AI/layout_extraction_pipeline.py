# pipeline/layout_extraction_pipeline.py
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
from relationship_extractor import SemanticRelationshipExtractor
from zero_shot_classifier import ZeroShotLayoutClassifier
from multi_modal_parser import MultiModalParser
from main import SemanticChunk
import pickle
import numpy as np

class LayoutExtractionPipeline:
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.parser = MultiModalParser()
        self.relationship_extractor = SemanticRelationshipExtractor()
        self.classifier = ZeroShotLayoutClassifier()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    async def process_document_with_checkpoints(self, file_path: str, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """Process document with checkpointing for error recovery"""
        
        doc_name = Path(file_path).stem
        checkpoint_base = self.checkpoint_dir / doc_name
        
        # Step 1: Parse document
        if resume_from is None or resume_from <= "parse":
            print("Parsing document...")
            raw_elements = await self.parser.parse_document(file_path)
            self._save_checkpoint(checkpoint_base, "parsed", raw_elements)
        else:
            raw_elements = self._load_checkpoint(checkpoint_base, "parsed")
            print("✓ Resumed from parsed checkpoint")
        
        # Step 2: Convert to semantic chunks
        if resume_from is None or resume_from <= "chunks":
            print("Converting to semantic chunks...")
            semantic_chunks = self._elements_to_chunks(raw_elements)
            self._save_checkpoint(checkpoint_base, "chunks", semantic_chunks)
        else:
            semantic_chunks = self._load_checkpoint(checkpoint_base, "chunks")
            print("✓ Resumed from chunks checkpoint")
        
        # Step 3: Classify chunks
        if resume_from is None or resume_from <= "classify":
            print("Classifying chunks...")
            classified_chunks = self.classifier.batch_classify(semantic_chunks)
            self._save_checkpoint(checkpoint_base, "classified", classified_chunks)
        else:
            classified_chunks = self._load_checkpoint(checkpoint_base, "classified")
            print("✓ Resumed from classified checkpoint")
        
        # Step 4: Extract relationships
        if resume_from is None or resume_from <= "relationships":
            print("Extracting relationships...")
            relationships = self.relationship_extractor.extract_relationships(classified_chunks)
            self._save_checkpoint(checkpoint_base, "relationships", relationships)
        else:
            relationships = self._load_checkpoint(checkpoint_base, "relationships")
            print("✓ Resumed from relationships checkpoint")
        
        # Step 5: Build document structure
        print("Building document structure...")
        document_structure = self._build_document_structure(classified_chunks)
        
        # Clean up checkpoints (optional)
        # self._cleanup_checkpoints(checkpoint_base)
        
        result = {
            'chunks': [chunk.to_dict() for chunk in classified_chunks],  # Use to_dict() instead of __dict__
            'relationships': self._serialize_relationships(relationships),
            'document_structure': self._serialize_document_structure(document_structure),
            'metadata': {
                'total_chunks': len(classified_chunks),
                'document_path': file_path,
                'processed_steps': ['parse', 'chunks', 'classify', 'relationships']
            }
        }
        
        return result

    def _serialize_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Serialize relationships to be JSON compatible"""
        serialized = []
        for rel in relationships:
            serialized_rel = {}
            for key, value in rel.items():
                if isinstance(value, (np.integer, np.floating)):
                    serialized_rel[key] = float(value)
                elif isinstance(value, np.ndarray):
                    serialized_rel[key] = value.tolist()
                else:
                    serialized_rel[key] = value
            serialized.append(serialized_rel)
        return serialized

    def _serialize_document_structure(self, structure: Dict) -> Dict:
        """Serialize document structure to be JSON compatible"""
        # Handle any numpy arrays in document structure
        def serialize_recursive(obj):
            if isinstance(obj, dict):
                return {k: serialize_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize_recursive(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        return serialize_recursive(structure)
    
    def _save_checkpoint(self, base_path: Path, step: str, data: Any):
        """Save checkpoint data"""
        checkpoint_file = base_path.with_suffix(f".{step}.pkl")
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"  → Checkpoint saved: {checkpoint_file}")
    
    def _load_checkpoint(self, base_path: Path, step: str) -> Any:
        """Load checkpoint data"""
        checkpoint_file = base_path.with_suffix(f".{step}.pkl")
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
        
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    
    def _cleanup_checkpoints(self, base_path: Path):
        """Clean up checkpoint files after successful processing"""
        for checkpoint_file in base_path.parent.glob(f"{base_path.stem}.*.pkl"):
            checkpoint_file.unlink()
    
    def list_checkpoints(self, file_path: str) -> List[str]:
        """List available checkpoints for a document"""
        # Handle both filename and full path
        doc_name = Path(file_path).stem
        
        # Debug: show what we're looking for
        print(f"Document name: {doc_name}")
        print(f"Checkpoint directory contents: {list(self.checkpoint_dir.iterdir())}")
        
        steps = []
        try:
            # Look for checkpoint files
            for file_path in self.checkpoint_dir.iterdir():
                if file_path.suffix == '.pkl' and file_path.name.startswith(doc_name + '.'):
                    # Extract step name
                    # From "input_document.chunks.pkl" get "chunks"
                    stem = file_path.stem  # "input_document.chunks"
                    if '.' in stem:
                        parts = stem.split('.')
                        step = parts[-1]  # Last part
                        steps.append(step)
                        print(f"Found checkpoint: {file_path.name} -> step: {step}")
        except Exception as e:
            print(f"Error listing checkpoints: {e}")
        
        return sorted(steps)
    
    def resume_processing(self, file_path: str, step: str = None) -> Dict[str, Any]:
        """Resume processing from a specific step or auto-detect"""
        if step is None:
            # Auto-detect last completed step
            completed_steps = self.list_checkpoints(file_path)
            if not completed_steps:
                step = None  # Start from beginning
            else:
                step = completed_steps[-1]  # Resume from last completed
                print(f"Auto-resuming from step: {step}")
        
        return asyncio.run(self.process_document_with_checkpoints(file_path, resume_from=step))
    
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


# Enhanced main function with resume capability
async def process_document_with_resume(input_path: str, resume: bool = False, step: str = None):
    """Enhanced document processing with resume capability"""
    
    pipeline = LayoutExtractionPipeline()
    
    if resume:
        print(f"Checking for checkpoints for {input_path}...")
        available_steps = pipeline.list_checkpoints(input_path)
        
        if available_steps:
            print(f"Available checkpoints: {available_steps}")
            result = pipeline.resume_processing(input_path, step)
        else:
            print("No checkpoints found. Starting from beginning...")
            result = await pipeline.process_document_with_checkpoints(input_path)
    else:
        result = await pipeline.process_document_with_checkpoints(input_path)
    
    return result

# Usage examples:
if __name__ == "__main__":
    import asyncio
    
    # Normal processing
    # asyncio.run(process_document_with_resume("input_document.pdf"))
    
    # Resume from last checkpoint
    # asyncio.run(process_document_with_resume("input_document.pdf", resume=True))
    
    # Resume from specific step
    asyncio.run(process_document_with_resume("input_document.pdf", resume=True, step="classify"))