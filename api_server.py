#!/usr/bin/env python3
"""
FastAPI server for Mamba model inference.
Compatible with OpenWebUI and standard REST clients.

Usage:
    python api_server.py --checkpoint ./checkpoints/reasoning_optimized/final --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import torch
from models.mamba_model import create_mamba_model
from models.hybrid_recursive_mamba import create_hybrid_recursive_mamba_model
from transformers import GPT2Tokenizer
import json
import argparse
import os
import uvicorn
from datetime import datetime


app = FastAPI(
    title="Mamba Inference API",
    description="REST API for Mamba language model inference",
    version="1.0.0"
)

# CORS middleware for OpenWebUI compatibility
# Use environment variable for allowed origins, defaulting to localhost
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model_instance = None
tokenizer_instance = None
config_instance = None


class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="Input prompt text")
    max_tokens: int = Field(200, description="Maximum tokens to generate", ge=1, le=2048)
    temperature: float = Field(0.7, description="Sampling temperature", ge=0.0, le=2.0)
    top_k: Optional[int] = Field(50, description="Top-k sampling parameter", ge=1)
    stream: bool = Field(False, description="Stream response (not yet implemented)")


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    prompt: str
    generated_text: str
    full_text: str
    tokens_generated: int
    model: str
    timestamp: str


class ChatMessage(BaseModel):
    """Chat message format."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """OpenWebUI-compatible chat request."""
    messages: List[ChatMessage] = Field(..., description="Chat message history")
    max_tokens: int = Field(200, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(50, ge=1)
    stream: bool = Field(False)


class ChatResponse(BaseModel):
    """OpenWebUI-compatible chat response."""
    message: ChatMessage
    model: str
    created: int
    done: bool


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    type: str
    parameters: int
    trainable_parameters: int
    architecture: Dict[str, Any]
    device: str


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model_instance, tokenizer_instance, config_instance

    checkpoint_dir = os.getenv('CHECKPOINT_DIR', './checkpoints/reasoning_optimized/final')
    device = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading model from: {checkpoint_dir}")
    print(f"Device: {device}")

    # Load configuration
    config_path = os.path.join(checkpoint_dir, 'config.json')
    with open(config_path, 'r') as f:
        config_instance = json.load(f)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer_instance = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer_instance.pad_token = tokenizer_instance.eos_token

    # Load model
    print("Loading model...")
    model_type = config_instance['model'].get('model_type', 'standard')

    if model_type == 'hybrid_recursive':
        model_instance = create_hybrid_recursive_mamba_model(config_instance['model'])
    else:
        model_instance = create_mamba_model(config_instance['model'])

    # Load weights
    model_path = os.path.join(checkpoint_dir, 'model.pt')
    model_instance.load_state_dict(torch.load(model_path, map_location=device))
    model_instance = model_instance.to(device)
    model_instance.eval()

    total_params = sum(p.numel() for p in model_instance.parameters())
    print(f"Model loaded successfully!")
    print(f"Type: {model_type}")
    print(f"Parameters: {total_params:,}")
    print(f"Ready for inference!\n")


def generate_text(
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.7,
    top_k: Optional[int] = 50
) -> tuple:
    """
    Generate text from prompt.

    Returns:
        (generated_text, tokens_generated)
    """
    device = next(model_instance.parameters()).device

    # Encode prompt
    input_ids = tokenizer_instance.encode(prompt, return_tensors='pt').to(device)
    input_length = len(input_ids[0])

    # Generate
    with torch.no_grad():
        output_ids = model_instance.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )

    # Decode
    full_text = tokenizer_instance.decode(output_ids[0], skip_special_tokens=True)
    generated_text = full_text[len(prompt):].strip()
    tokens_generated = len(output_ids[0]) - input_length

    return generated_text, tokens_generated, full_text


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Mamba Inference API",
        "version": "1.0.0",
        "endpoints": {
            "generate": "/generate",
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(next(model_instance.parameters()).device)
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text from prompt.

    Simple generation endpoint for testing.
    """
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        generated_text, tokens_generated, full_text = generate_text(
            request.prompt,
            request.max_tokens,
            request.temperature,
            request.top_k
        )

        model_type = config_instance['model'].get('model_type', 'standard')

        return GenerateResponse(
            prompt=request.prompt,
            generated_text=generated_text,
            full_text=full_text,
            tokens_generated=tokens_generated,
            model=model_type,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    """
    OpenWebUI-compatible chat completions endpoint.

    Compatible with OpenAI API format.
    """
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Build prompt from message history
        prompt_parts = []
        for msg in request.messages:
            role = msg.role.capitalize()
            prompt_parts.append(f"{role}: {msg.content}")

        prompt_parts.append("Assistant:")
        full_prompt = "\n".join(prompt_parts)

        # Generate response
        generated_text, _, _ = generate_text(
            full_prompt,
            request.max_tokens,
            request.temperature,
            request.top_k
        )

        # Clean up response (take first complete sentence/line)
        response_text = generated_text.split('\n')[0].strip()

        return ChatResponse(
            message=ChatMessage(role="assistant", content=response_text),
            model=config_instance['model'].get('model_type', 'mamba'),
            created=int(datetime.utcnow().timestamp()),
            done=True
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """
    List available models (OpenWebUI compatibility).
    """
    if model_instance is None:
        return {"data": []}

    model_type = config_instance['model'].get('model_type', 'standard')
    total_params = sum(p.numel() for p in model_instance.parameters())

    return {
        "object": "list",
        "data": [
            {
                "id": f"mamba-{model_type}",
                "object": "model",
                "created": int(datetime.utcnow().timestamp()),
                "owned_by": "local",
                "parameters": total_params,
                "architecture": model_type
            }
        ]
    }


@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    """Get detailed model information."""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model_type = config_instance['model'].get('model_type', 'standard')
    total_params = sum(p.numel() for p in model_instance.parameters())
    trainable_params = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
    device = str(next(model_instance.parameters()).device)

    return ModelInfo(
        name=f"mamba-{model_type}",
        type=model_type,
        parameters=total_params,
        trainable_parameters=trainable_params,
        architecture={
            "d_model": config_instance['model'].get('d_model'),
            "n_layers": config_instance['model'].get('n_layers'),
            "vocab_size": config_instance['model'].get('vocab_size'),
            "max_seq_len": config_instance['model'].get('max_seq_len')
        },
        device=device
    )


def main():
    parser = argparse.ArgumentParser(description="Mamba API Server")
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoints/reasoning_optimized/final',
                        help='Path to checkpoint directory')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to run server on')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run on')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of worker processes')

    args = parser.parse_args()

    # Set environment variables
    os.environ['CHECKPOINT_DIR'] = args.checkpoint
    os.environ['DEVICE'] = args.device

    print("="*70)
    print("Starting Mamba API Server")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Device: {args.device}")
    print(f"Workers: {args.workers}")
    print("="*70 + "\n")

    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )


if __name__ == '__main__':
    main()
