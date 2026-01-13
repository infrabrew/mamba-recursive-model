#!/usr/bin/env python3
"""
Text generation script using a trained Mamba model.
"""

import argparse
import torch
from transformers import AutoTokenizer
from models.mamba_model import create_mamba_model
from configs.model_configs import load_config


def main():
    parser = argparse.ArgumentParser(description="Generate text with trained Mamba model")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint directory')
    parser.add_argument('--prompt', type=str, default="Once upon a time",
                        help='Text prompt for generation')
    parser.add_argument('--max_tokens', type=int, default=100,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling parameter')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for generation')

    args = parser.parse_args()

    # Check device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # Load configuration
    print(f"Loading model from: {args.checkpoint}")
    config = load_config(f"{args.checkpoint}/config.json")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Create model
    print("Creating model...")
    model = create_mamba_model(config['model'])
    model.to(device)

    # Load weights
    print("Loading weights...")
    model.load_state_dict(torch.load(f"{args.checkpoint}/model.pt", map_location=device))
    model.eval()

    print(f"\nModel loaded successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Tokenize prompt
    print(f"\nPrompt: {args.prompt}")
    input_ids = tokenizer.encode(args.prompt, return_tensors='pt').to(device)

    print(f"Generating {args.max_tokens} tokens...\n")
    print("-" * 80)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )

    # Decode
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(generated_text)
    print("-" * 80)

    print(f"\nGeneration complete!")


if __name__ == '__main__':
    main()
