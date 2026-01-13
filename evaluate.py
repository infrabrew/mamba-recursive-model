#!/usr/bin/env python3
"""
Benchmark evaluation for Mamba models on GSM8K and other datasets.

Usage:
    python evaluate.py --checkpoint ./checkpoints/reasoning_optimized/final --dataset gsm8k --num_examples 100
"""

import torch
from models.mamba_model import create_mamba_model
from models.hybrid_recursive_mamba import create_hybrid_recursive_mamba_model
from transformers import GPT2Tokenizer
from datasets import load_dataset
import json
import argparse
import os
import re
from typing import Optional, Dict, List
from datetime import datetime
from tqdm import tqdm


class BenchmarkEvaluator:
    """Evaluate Mamba models on benchmark datasets."""

    def __init__(self, checkpoint_dir: str, device: str = 'cuda'):
        """Initialize evaluator."""
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

    def generate(self, prompt: str, max_tokens: int = 300, temperature: float = 0.3) -> str:
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

    def evaluate_gsm8k(self, num_examples: Optional[int] = None, split: str = 'test') -> Dict:
        """
        Evaluate on GSM8K dataset.

        Args:
            num_examples: Number of examples to evaluate (None = all)
            split: Dataset split to use

        Returns:
            Dictionary with evaluation results
        """
        print(f"Loading GSM8K dataset ({split} split)...")
        dataset = load_dataset("openai/gsm8k", "main", split=split)

        if num_examples:
            dataset = dataset.select(range(min(num_examples, len(dataset))))

        print(f"Evaluating on {len(dataset)} examples...\n")

        correct = 0
        results = []

        for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
            question = example['question']
            # Extract numeric answer
            answer_text = example['answer']
            true_answer = answer_text.split("####")[-1].strip()

            # Format prompt
            prompt = f"Question: {question}\nAnswer: Let's solve this step by step.\n"

            # Generate response
            try:
                response = self.generate(prompt, max_tokens=300, temperature=0.3)

                # Extract predicted answer
                predicted = self._extract_answer(response)

                # Check if correct
                is_correct = self._compare_answers(predicted, true_answer)

                if is_correct:
                    correct += 1

                results.append({
                    'question': question,
                    'true_answer': true_answer,
                    'predicted_answer': predicted,
                    'correct': is_correct,
                    'full_response': response
                })

                # Print progress every 10 examples
                if (i + 1) % 10 == 0:
                    current_accuracy = (correct / (i + 1)) * 100
                    print(f"\nProgress: {i+1}/{len(dataset)} | Accuracy: {current_accuracy:.2f}%")

            except Exception as e:
                print(f"\nError on example {i+1}: {e}")
                results.append({
                    'question': question,
                    'true_answer': true_answer,
                    'predicted_answer': 'ERROR',
                    'correct': False,
                    'full_response': str(e)
                })

        accuracy = (correct / len(dataset)) * 100

        return {
            'dataset': 'GSM8K',
            'split': split,
            'total_examples': len(dataset),
            'correct': correct,
            'accuracy': accuracy,
            'results': results
        }

    def evaluate_math(self, num_examples: Optional[int] = None, split: str = 'test') -> Dict:
        """
        Evaluate on MATH dataset.

        Args:
            num_examples: Number of examples to evaluate
            split: Dataset split to use

        Returns:
            Dictionary with evaluation results
        """
        print(f"Loading MATH dataset ({split} split)...")
        dataset = load_dataset("hendrycks/competition_math", split=split)

        if num_examples:
            dataset = dataset.select(range(min(num_examples, len(dataset))))

        print(f"Evaluating on {len(dataset)} examples...\n")

        correct = 0
        results = []

        for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
            question = example['problem']
            true_answer = example['solution']

            # Format prompt
            prompt = f"Problem: {question}\nSolution:"

            # Generate response
            try:
                response = self.generate(prompt, max_tokens=400, temperature=0.3)

                # Extract predicted answer
                predicted = self._extract_answer(response)
                true_ans = self._extract_answer(true_answer)

                # Check if correct
                is_correct = self._compare_answers(predicted, true_ans)

                if is_correct:
                    correct += 1

                results.append({
                    'question': question,
                    'true_answer': true_ans,
                    'predicted_answer': predicted,
                    'correct': is_correct,
                    'level': example.get('level', 'unknown'),
                    'type': example.get('type', 'unknown')
                })

            except Exception as e:
                print(f"\nError on example {i+1}: {e}")
                results.append({
                    'question': question,
                    'true_answer': '',
                    'predicted_answer': 'ERROR',
                    'correct': False
                })

        accuracy = (correct / len(dataset)) * 100

        return {
            'dataset': 'MATH',
            'split': split,
            'total_examples': len(dataset),
            'correct': correct,
            'accuracy': accuracy,
            'results': results
        }

    def _extract_answer(self, text: str) -> str:
        """Extract numerical answer from text."""
        # Look for boxed answer (LaTeX format)
        boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
        if boxed:
            return boxed[-1].strip()

        # Look for "Answer: X" or "The answer is X"
        answer_patterns = [
            r'[Aa]nswer:\s*([+-]?\d+(?:\.\d+)?)',
            r'[Tt]he answer is\s*([+-]?\d+(?:\.\d+)?)',
            r'=\s*([+-]?\d+(?:\.\d+)?)\s*$',
            r'([+-]?\d+(?:\.\d+)?)\s*$'
        ]

        for pattern in answer_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[-1].strip()

        # Fallback: return last number in text
        numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', text)
        if numbers:
            return numbers[-1].strip()

        return ""

    def _compare_answers(self, predicted: str, true: str) -> bool:
        """Compare predicted and true answers."""
        if not predicted or not true:
            return False

        # Try exact match first
        if predicted == true:
            return True

        # Try numerical comparison
        try:
            pred_num = float(predicted.replace(',', ''))
            true_num = float(true.replace(',', ''))
            # Allow small floating point differences
            return abs(pred_num - true_num) < 0.01
        except:
            pass

        # Try string normalization
        pred_norm = predicted.lower().strip().replace(' ', '')
        true_norm = true.lower().strip().replace(' ', '')

        return pred_norm == true_norm

    def save_results(self, results: Dict, output_file: str):
        """Save evaluation results to JSON file."""
        output_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'model_checkpoint': self.checkpoint_dir,
            'model_type': self.config['model'].get('model_type', 'standard'),
            'evaluation': results
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✅ Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mamba model on benchmarks")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint directory')
    parser.add_argument('--dataset', type=str, default='gsm8k',
                        choices=['gsm8k', 'math'],
                        help='Benchmark dataset to evaluate on')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'test', 'validation'],
                        help='Dataset split to use')
    parser.add_argument('--num_examples', type=int, default=None,
                        help='Number of examples to evaluate (default: all)')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='Output JSON file for results')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run on')

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = BenchmarkEvaluator(args.checkpoint, args.device)

    # Run evaluation
    print("="*80)
    print(f"Evaluating on {args.dataset.upper()} ({args.split} split)")
    print("="*80 + "\n")

    if args.dataset == 'gsm8k':
        results = evaluator.evaluate_gsm8k(args.num_examples, args.split)
    elif args.dataset == 'math':
        results = evaluator.evaluate_math(args.num_examples, args.split)

    # Print summary
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    print(f"Dataset: {results['dataset']}")
    print(f"Split: {results['split']}")
    print(f"Total Examples: {results['total_examples']}")
    print(f"Correct: {results['correct']}")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print("="*80)

    # Save results
    evaluator.save_results(results, args.output)

    # Show some examples
    print("\nSample Results (first 5):")
    print("-"*80)
    for i, result in enumerate(results['results'][:5], 1):
        status = "✓" if result['correct'] else "✗"
        print(f"\n{i}. {status} Question: {result['question'][:100]}...")
        print(f"   True: {result['true_answer']}")
        print(f"   Predicted: {result['predicted_answer']}")


if __name__ == '__main__':
    main()
