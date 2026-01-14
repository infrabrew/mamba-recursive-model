#!/usr/bin/env python3
"""
Prepare Chain-of-Thought (CoT) dataset for improved reasoning training.

This script augments existing datasets with explicit step-by-step reasoning
to help the model learn better problem-solving strategies.

Usage:
    python prepare_cot_dataset.py --dataset TIGER-Lab/MathInstruct --output ./data/mathinstruct_cot
"""

import argparse
import json
import re
from datasets import load_dataset, Dataset
from tqdm import tqdm
import os


class CoTAugmenter:
    """Augment dataset with Chain-of-Thought reasoning."""

    def __init__(self):
        self.reasoning_templates = {
            'percentage': self._percentage_template,
            'equation': self._equation_template,
            'geometry': self._geometry_template,
            'arithmetic': self._arithmetic_template,
            'word_problem': self._word_problem_template,
            'default': self._default_template
        }

    def augment(self, example: dict, text_columns: list) -> dict:
        """
        Augment example with Chain-of-Thought reasoning.

        Args:
            example: Dataset example
            text_columns: List of column names containing text

        Returns:
            Augmented example with CoT reasoning
        """
        # Get question and answer
        if len(text_columns) == 1:
            # Single column (e.g., "text")
            text = example.get(text_columns[0], '')
            question = text
            answer = ''
        elif len(text_columns) >= 2:
            # Multi-column (e.g., "instruction", "output")
            question = example.get(text_columns[0], '')
            answer = example.get(text_columns[1], '')
        else:
            return example

        # Detect problem type
        problem_type = self._detect_problem_type(question)

        # Get appropriate template
        template_func = self.reasoning_templates.get(problem_type, self.reasoning_templates['default'])

        # Apply CoT augmentation
        cot_text = template_func(question, answer)

        # Update example
        if len(text_columns) == 1:
            example[text_columns[0]] = cot_text
        else:
            example[text_columns[1]] = cot_text

        return example

    def _detect_problem_type(self, question: str) -> str:
        """Detect the type of math problem."""
        question_lower = question.lower()

        if any(word in question_lower for word in ['percent', '%', 'percentage']):
            return 'percentage'
        elif any(word in question_lower for word in ['equation', 'solve for', 'find x', 'find y']):
            return 'equation'
        elif any(word in question_lower for word in ['area', 'perimeter', 'volume', 'circle', 'rectangle', 'triangle']):
            return 'geometry'
        elif any(word in question_lower for word in ['calculate', 'compute', '+', '-', '×', '÷']):
            return 'arithmetic'
        elif len(question.split()) > 15:  # Long question = word problem
            return 'word_problem'
        else:
            return 'default'

    def _percentage_template(self, question: str, answer: str) -> str:
        """Template for percentage problems."""
        return f"""{answer}

Let me solve this step-by-step:

Step 1: Identify what percentage we need to find
- We need to calculate a percentage of a number

Step 2: Convert percentage to decimal
- Divide the percentage by 100

Step 3: Multiply by the base number
- Decimal × base number = result

Step 4: Verify the answer
- Check if the result makes sense

This systematic approach ensures accuracy in percentage calculations."""

    def _equation_template(self, question: str, answer: str) -> str:
        """Template for equation solving."""
        return f"""{answer}

Let me solve this equation step-by-step:

Step 1: Understand what we're solving for
- Identify the variable we need to find

Step 2: Isolate the variable
- Use inverse operations to move terms

Step 3: Simplify both sides
- Combine like terms and simplify

Step 4: Solve for the variable
- Perform the final calculation

Step 5: Verify the solution
- Substitute back into original equation to check

This methodical approach ensures we solve equations correctly."""

    def _geometry_template(self, question: str, answer: str) -> str:
        """Template for geometry problems."""
        return f"""{answer}

Let me solve this geometry problem step-by-step:

Step 1: Identify the geometric shape and what we need to find
- Determine which formula to use

Step 2: List the given measurements
- Write down all known values

Step 3: Apply the appropriate formula
- Substitute values into the formula

Step 4: Calculate the result
- Perform the arithmetic operations

Step 5: Check units and reasonableness
- Ensure the answer makes sense for the context

This structured approach helps solve geometry problems accurately."""

    def _arithmetic_template(self, question: str, answer: str) -> str:
        """Template for arithmetic problems."""
        return f"""{answer}

Let me calculate this step-by-step:

Step 1: Identify the operations needed
- Determine the order of operations (PEMDAS/BODMAS)

Step 2: Break down complex calculations
- Solve innermost operations first

Step 3: Perform calculations in order
- Follow the order of operations carefully

Step 4: Verify the result
- Check calculations for accuracy

This careful approach prevents arithmetic errors."""

    def _word_problem_template(self, question: str, answer: str) -> str:
        """Template for word problems."""
        return f"""{answer}

Let me solve this word problem step-by-step:

Step 1: Read and understand the problem
- Identify what is being asked
- Note all given information

Step 2: Translate words to math
- Convert the problem into mathematical expressions
- Define variables if needed

Step 3: Set up the solution strategy
- Determine what operations are needed
- Plan the order of steps

Step 4: Execute the solution
- Perform calculations step by step
- Show all work clearly

Step 5: Interpret and verify
- Make sure the answer makes sense in context
- Check if it answers the original question

This thorough approach ensures we understand and solve word problems correctly."""

    def _default_template(self, question: str, answer: str) -> str:
        """Default template for any problem."""
        return f"""{answer}

Let me work through this problem systematically:

Step 1: Understand the question
- What is being asked?
- What information do we have?

Step 2: Plan the approach
- What method or formula should we use?
- What are the steps to solve this?

Step 3: Execute the solution
- Work through each step carefully
- Show all calculations

Step 4: Check the answer
- Does the result make sense?
- Can we verify it another way?

This structured thinking helps ensure accurate problem-solving."""


def combine_datasets(dataset_names: list, split: str = 'train') -> Dataset:
    """
    Combine multiple datasets into one.

    Args:
        dataset_names: List of dataset names to combine
        split: Dataset split to load

    Returns:
        Combined dataset
    """
    from datasets import concatenate_datasets

    print(f"\nLoading and combining {len(dataset_names)} datasets...")
    datasets = []

    for name in dataset_names:
        try:
            print(f"  Loading {name}...")
            ds = load_dataset(name, split=split)
            datasets.append(ds)
            print(f"    ✓ Loaded {len(ds):,} examples")
        except Exception as e:
            print(f"    ✗ Failed to load {name}: {e}")

    if not datasets:
        raise ValueError("No datasets loaded successfully")

    print("\nCombining datasets...")
    combined = concatenate_datasets(datasets)
    print(f"✓ Combined dataset: {len(combined):,} examples\n")

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Chain-of-Thought dataset for improved reasoning"
    )
    parser.add_argument('--dataset', type=str, default='TIGER-Lab/MathInstruct',
                        help='Dataset name (or comma-separated list for combining)')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to use')
    parser.add_argument('--text_columns', type=str, default='instruction,output',
                        help='Comma-separated column names containing text')
    parser.add_argument('--output', type=str, default='./data/mathinstruct_cot',
                        help='Output directory for processed dataset')
    parser.add_argument('--max_examples', type=int, default=None,
                        help='Maximum examples to process (for testing)')
    parser.add_argument('--combine', action='store_true',
                        help='Combine multiple datasets (comma-separate in --dataset)')

    args = parser.parse_args()

    # Parse text columns
    text_columns = [col.strip() for col in args.text_columns.split(',')]

    # Load dataset(s)
    print("="*80)
    print("Chain-of-Thought Dataset Preparation")
    print("="*80)

    if args.combine:
        # Combine multiple datasets
        dataset_names = [name.strip() for name in args.dataset.split(',')]
        dataset = combine_datasets(dataset_names, args.split)
    else:
        # Single dataset
        print(f"\nLoading dataset: {args.dataset} ({args.split} split)...")
        dataset = load_dataset(args.dataset, split=args.split)
        print(f"✓ Loaded {len(dataset):,} examples\n")

    # Limit for testing
    if args.max_examples:
        print(f"Limiting to {args.max_examples} examples for testing...")
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    # Initialize augmenter
    augmenter = CoTAugmenter()

    # Process dataset
    print("Augmenting dataset with Chain-of-Thought reasoning...\n")

    processed_dataset = dataset.map(
        lambda x: augmenter.augment(x, text_columns),
        desc="Processing examples"
    )

    # Save processed dataset
    print(f"\nSaving to: {args.output}")
    os.makedirs(args.output, exist_ok=True)

    processed_dataset.save_to_disk(args.output)

    # Also save as JSON for inspection
    json_output = os.path.join(args.output, 'examples.json')
    with open(json_output, 'w') as f:
        # Save first 10 examples for inspection
        examples = [processed_dataset[i] for i in range(min(10, len(processed_dataset)))]
        json.dump(examples, f, indent=2)

    print(f"✓ Saved to disk format")
    print(f"✓ Saved sample examples to: {json_output}")

    # Statistics
    print("\n" + "="*80)
    print("Dataset Statistics")
    print("="*80)
    print(f"Total examples: {len(processed_dataset):,}")
    print(f"Text columns: {', '.join(text_columns)}")
    print(f"Output directory: {args.output}")

    # Show sample
    print("\n" + "="*80)
    print("Sample Augmented Example")
    print("="*80)
    sample = processed_dataset[0]
    if len(text_columns) >= 2:
        print(f"\nQuestion:\n{sample[text_columns[0]]}")
        print(f"\nAnswer with CoT:\n{sample[text_columns[1]][:500]}...")
    else:
        print(f"\n{sample[text_columns[0]][:500]}...")

    print("\n" + "="*80)
    print("✓ Dataset preparation complete!")
    print("="*80)
    print(f"\nUse this dataset for training:")
    print(f"  ./train_background.sh \\")
    print(f"      --vram_auto \\")
    print(f"      --model_size hybrid-small-standard \\")
    print(f"      --data_dir {args.output} \\")
    print(f"      --training_preset default")


if __name__ == '__main__':
    main()
