#!/usr/bin/env python3
"""
Batch testing for Mamba models.

Usage:
    python batch_test.py --checkpoint ./checkpoints/reasoning_optimized/final --questions test_questions.txt
"""

import torch
from models.mamba_model import create_mamba_model
from models.hybrid_recursive_mamba import create_hybrid_recursive_mamba_model
from transformers import GPT2Tokenizer
import json
import argparse
import os
from typing import List
from datetime import datetime


class BatchTester:
    """Batch testing for Mamba models."""

    def __init__(self, checkpoint_dir: str, device: str = 'cuda'):
        """Initialize batch tester."""
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

        model_path = os.path.join(checkpoint_dir, 'model.pt')
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded: {model_type} ({total_params:,} parameters)\n")

    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        """Generate text from prompt."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=50
            )

        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text

    def test_questions(self, questions: List[str], max_tokens: int = 200, temperature: float = 0.7):
        """Test a list of questions."""
        print("="*80)
        print(f"Batch Testing - {len(questions)} Questions")
        print("="*80 + "\n")

        results = []

        for i, question in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] Question: {question}")

            # Format prompt
            prompt = f"Question: {question}\nAnswer:"

            # Generate
            try:
                start_time = datetime.now()
                response = self.generate(prompt, max_tokens, temperature)
                elapsed = (datetime.now() - start_time).total_seconds()

                # Extract answer (remove prompt)
                answer = response[len(prompt):].strip()
                # Take first line or sentence
                answer = answer.split('\n')[0].strip()

                print(f"Answer: {answer}")
                print(f"Time: {elapsed:.2f}s\n")

                results.append({
                    'question': question,
                    'answer': answer,
                    'full_response': response,
                    'time_seconds': elapsed
                })

            except Exception as e:
                print(f"Error: {e}\n")
                results.append({
                    'question': question,
                    'answer': f"ERROR: {e}",
                    'full_response': "",
                    'time_seconds': 0
                })

            print("-"*80 + "\n")

        return results

    def save_results(self, results: List[dict], output_file: str):
        """Save results to JSON file."""
        output_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'model': self.config['model'].get('model_type', 'standard'),
            'checkpoint': self.checkpoint_dir,
            'total_questions': len(results),
            'results': results
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✅ Results saved to: {output_file}")


def load_questions_from_file(filepath: str) -> List[str]:
    """Load questions from text file (one per line)."""
    with open(filepath, 'r') as f:
        questions = [line.strip() for line in f if line.strip()]
    return questions


def get_default_questions() -> List[str]:
    """Get default test questions."""
    return [
        "What is 15% of 240?",
        "If x + 5 = 12, what is x?",
        "A rectangle has length 8 and width 5. What is its area?",
        "What is the square root of 144?",
        "If 3x - 7 = 20, solve for x.",
        "Calculate: (8 + 2) × 5 - 3",
        "What is 25% of 80?",
        "If y - 9 = 15, what is y?",
        "A circle has radius 4. What is its area? (Use π = 3.14)",
        "Simplify: 2(x + 3) - 4",
        "What is 2^5?",
        "If 4a = 28, what is a?",
        "What is the perimeter of a square with side length 6?",
        "Calculate: 100 ÷ 4 + 3 × 2",
        "If 2x + 5 = 17, what is x?",
        "What is 30% of 150?",
        "A triangle has base 10 and height 6. What is its area?",
        "What is the cube root of 27?",
        "If x/3 = 7, what is x?",
        "Calculate: (12 - 4) × (5 + 3)"
    ]


def main():
    parser = argparse.ArgumentParser(description="Batch test Mamba model")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint directory')
    parser.add_argument('--questions', type=str, default=None,
                        help='Path to questions file (one per line)')
    parser.add_argument('--output', type=str, default='batch_test_results.json',
                        help='Output JSON file for results')
    parser.add_argument('--max_tokens', type=int, default=200,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run on')

    args = parser.parse_args()

    # Load questions
    if args.questions:
        print(f"Loading questions from: {args.questions}")
        questions = load_questions_from_file(args.questions)
    else:
        print("Using default test questions")
        questions = get_default_questions()

    # Initialize tester
    tester = BatchTester(args.checkpoint, args.device)

    # Run tests
    results = tester.test_questions(questions, args.max_tokens, args.temperature)

    # Save results
    tester.save_results(results, args.output)

    # Summary
    total_time = sum(r['time_seconds'] for r in results)
    avg_time = total_time / len(results) if results else 0

    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Total questions: {len(results)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per question: {avg_time:.2f}s")
    print("="*80)


if __name__ == '__main__':
    main()
