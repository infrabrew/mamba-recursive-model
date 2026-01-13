#!/usr/bin/env python3
"""
Interactive CLI inference for trained Mamba models.

Usage:
    python inference.py --checkpoint ./checkpoints/reasoning_optimized/final
"""

import torch
from models.mamba_model import create_mamba_model
from models.hybrid_recursive_mamba import create_hybrid_recursive_mamba_model
from transformers import GPT2Tokenizer
import json
import argparse
import os
from typing import Optional


class MambaInference:
    """Inference wrapper for Mamba models."""

    def __init__(self, checkpoint_dir: str, device: str = 'cuda'):
        """
        Initialize inference engine.

        Args:
            checkpoint_dir: Path to checkpoint directory
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        # Load configuration
        config_path = os.path.join(checkpoint_dir, 'config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        print("Loading model...")
        model_type = self.config['model'].get('model_type', 'standard')

        if model_type == 'hybrid_recursive':
            self.model = create_hybrid_recursive_mamba_model(self.config['model'])
        else:
            self.model = create_mamba_model(self.config['model'])

        # Load weights
        model_path = os.path.join(checkpoint_dir, 'model.pt')
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()

        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded: {model_type}")
        print(f"Parameters: {total_params:,}")
        print(f"Device: {device}\n")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_k: Optional[int] = 50,
        return_full_text: bool = False
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            top_k: Top-k sampling parameter
            return_full_text: If True, return prompt + generated text

        Returns:
            Generated text
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k
            )

        # Decode
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if not return_full_text:
            # Remove prompt from output
            output_text = output_text[len(prompt):].strip()

        return output_text

    def chat(self):
        """Interactive chat mode."""
        print("="*70)
        print("Mamba Interactive Inference")
        print("="*70)
        print("Type 'quit', 'exit', or 'q' to exit")
        print("Type 'clear' to clear conversation history")
        print("Type 'help' for options\n")

        conversation_history = []

        while True:
            try:
                user_input = input("\n📝 You: ").strip()

                if not user_input:
                    continue

                # Commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break

                if user_input.lower() == 'clear':
                    conversation_history = []
                    print("✅ Conversation history cleared")
                    continue

                if user_input.lower() == 'help':
                    self._print_help()
                    continue

                # Build prompt with history
                if conversation_history:
                    context = "\n".join(conversation_history[-3:])  # Last 3 exchanges
                    full_prompt = f"{context}\nUser: {user_input}\nAssistant:"
                else:
                    full_prompt = f"User: {user_input}\nAssistant:"

                # Generate response
                print("\n🤔 Thinking...")
                response = self.generate(
                    full_prompt,
                    max_tokens=300,
                    temperature=0.7,
                    return_full_text=False
                )

                # Clean up response
                response = response.split('\n')[0].strip()  # First line only

                print(f"\n🤖 Assistant: {response}")

                # Update history
                conversation_history.append(f"User: {user_input}")
                conversation_history.append(f"Assistant: {response}")

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

    def _print_help(self):
        """Print help information."""
        print("\n" + "="*70)
        print("Commands:")
        print("  quit/exit/q  - Exit the program")
        print("  clear        - Clear conversation history")
        print("  help         - Show this help message")
        print("\nTips:")
        print("  - Ask math questions: 'What is 15% of 240?'")
        print("  - Code questions: 'Write a function to sort a list'")
        print("  - Logic puzzles: 'If A > B and B > C, is A > C?'")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Mamba model inference")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint directory')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run on')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Single prompt to generate (non-interactive)')
    parser.add_argument('--max_tokens', type=int, default=200,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')

    args = parser.parse_args()

    # Initialize inference
    inference = MambaInference(args.checkpoint, args.device)

    if args.prompt:
        # Single generation
        response = inference.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"Response: {response}\n")
    else:
        # Interactive chat
        inference.chat()


if __name__ == '__main__':
    main()
