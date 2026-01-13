"""
Reasoning model training utilities for Mamba.
Supports chain-of-thought (CoT), step-by-step reasoning, and structured thinking.
"""

from typing import List, Dict, Optional
import random


class ReasoningDataFormatter:
    """Format datasets for reasoning model training."""

    def __init__(self, use_thinking_tags: bool = True, cot_style: str = "openai"):
        """
        Initialize reasoning data formatter.

        Args:
            use_thinking_tags: Whether to wrap reasoning in <think></think> tags
            cot_style: Chain-of-thought style ('openai', 'deepseek', 'qwen', 'llama')
        """
        self.use_thinking_tags = use_thinking_tags
        self.cot_style = cot_style

    def format_reasoning_example(
        self,
        instruction: str,
        reasoning: str,
        answer: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Format a single reasoning example.

        Args:
            instruction: The problem/question
            reasoning: Step-by-step reasoning process
            answer: Final answer
            metadata: Optional metadata (difficulty, topic, etc.)

        Returns:
            Formatted training example
        """
        if self.cot_style == "openai":
            # OpenAI o1-style format
            if self.use_thinking_tags:
                formatted = f"""User: {instruction}

<think>
{reasoning}
</think>
