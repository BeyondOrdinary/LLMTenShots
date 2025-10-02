# Agentic Document Extraction from Landing.AI

This 10-shot was generated from a few interations with the Qwen-480B-Cloud model hosted at Ollama and accessed using OpenWebUI from a local installation.

## The URL

https://landing.ai/agentic-document-extraction

## Qwen's Advice

```python
# Normal processing
python cli.py input_document.pdf

# List available checkpoints
python cli.py input_document.pdf --list-checkpoints

# Resume from last checkpoint
python cli.py input_document.pdf --resume

# Resume from specific step
python cli.py input_document.pdf --resume --step classify

# Resume with custom checkpoint directory
python cli.py input_document.pdf --resume --checkpoint-dir /tmp/my_checkpoints
```

This approach gives you:

    Automatic checkpointing after each major step
    Easy resume capability from any completed step
    Checkpoint cleanup after successful completion
    Flexible CLI interface for different workflows
    Error recovery without reprocessing completed steps
     
