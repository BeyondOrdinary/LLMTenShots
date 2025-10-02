# cli.py
import argparse
import asyncio
from pathlib import Path
from layout_extraction_pipeline import LayoutExtractionPipeline
import json

def main():
    parser = argparse.ArgumentParser(description="Layout Extraction Tool with Checkpointing")
    parser.add_argument("input_file", help="Input document to process")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--step", help="Resume from specific step (parse, chunks, classify, relationships)")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--list-checkpoints", action="store_true", help="List available checkpoints")
    
    args = parser.parse_args()
    
    pipeline = LayoutExtractionPipeline(checkpoint_dir=args.checkpoint_dir)
    
    if args.list_checkpoints:
        steps = pipeline.list_checkpoints(args.input_file)
        print(f"Available checkpoints for {args.input_file}: {steps}")
        return
    
    if args.resume or args.step:
        result = pipeline.resume_processing(args.input_file, args.step)
    else:
        result = asyncio.run(pipeline.process_document_with_checkpoints(args.input_file))
    
    # Save result
    output_file = Path(args.input_file).with_suffix(".layout.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()