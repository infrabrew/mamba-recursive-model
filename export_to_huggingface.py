#!/usr/bin/env python3
"""
Export trained Mamba model to HuggingFace format for compatibility with vLLM and other tools.
"""

import argparse
import os
import json
import torch
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer
from models.mamba_model import MambaLanguageModel
from configs.model_configs import load_config


class MambaConfig(PretrainedConfig):
    """HuggingFace-compatible configuration for Mamba model."""

    model_type = "mamba"

    def __init__(
        self,
        vocab_size=50257,
        d_model=768,
        n_layers=12,
        d_state=16,
        expand_factor=2,
        dropout=0.1,
        max_seq_len=2048,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.expand_factor = expand_factor
        self.dropout = dropout
        self.max_seq_len = max_seq_len


class HuggingFaceMambaModel(PreTrainedModel):
    """HuggingFace-compatible wrapper for Mamba model."""

    config_class = MambaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = MambaLanguageModel(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            d_state=config.d_state,
            expand_factor=config.expand_factor,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len
        )

    def forward(self, input_ids, labels=None, **kwargs):
        return self.model(input_ids, labels)

    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=50, **kwargs):
        return self.model.generate(input_ids, max_new_tokens, temperature, top_k)


def export_to_huggingface(checkpoint_dir: str, output_dir: str):
    """
    Export Mamba checkpoint to HuggingFace format.

    Args:
        checkpoint_dir: Path to Mamba checkpoint directory
        output_dir: Path to save HuggingFace model
    """
    print(f"Loading checkpoint from: {checkpoint_dir}")

    # Load config
    config_path = os.path.join(checkpoint_dir, "config.json")
    mamba_config = load_config(config_path)

    # Create HuggingFace config
    hf_config = MambaConfig(
        vocab_size=mamba_config['model']['vocab_size'],
        d_model=mamba_config['model']['d_model'],
        n_layers=mamba_config['model']['n_layers'],
        d_state=mamba_config['model']['d_state'],
        expand_factor=mamba_config['model']['expand_factor'],
        dropout=mamba_config['model']['dropout'],
        max_seq_len=mamba_config['model']['max_seq_len']
    )

    # Create model
    print("Creating HuggingFace model...")
    hf_model = HuggingFaceMambaModel(hf_config)

    # Load weights
    print("Loading weights...")
    state_dict = torch.load(
        os.path.join(checkpoint_dir, "model.pt"),
        map_location='cpu'
    )

    # Transfer weights (add 'model.' prefix for wrapper)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict[f'model.{key}'] = value

    hf_model.load_state_dict(new_state_dict)

    # Save in HuggingFace format
    print(f"Saving to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    hf_model.save_pretrained(output_dir)
    hf_config.save_pretrained(output_dir)

    # Copy tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.save_pretrained(output_dir)

    # Create model card
    model_card = f"""---
language: en
license: mit
tags:
- mamba
- text-generation
---

# Mamba Language Model

This model was trained using the Mamba architecture (Selective State Space Models).

## Model Details

- Architecture: Mamba
- Parameters: ~{sum(p.numel() for p in hf_model.parameters()):,}
- Vocabulary Size: {hf_config.vocab_size}
- Model Dimension: {hf_config.d_model}
- Layers: {hf_config.n_layers}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{output_dir}")
tokenizer = AutoTokenizer.from_pretrained("{output_dir}")

# Generate text
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

## Training

Trained with the Mamba Trainer framework.
"""

    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(model_card)

    print("\n✓ Export complete!")
    print(f"\nModel saved to: {output_dir}")
    print("\nYou can now use this model with:")
    print("  - vLLM")
    print("  - HuggingFace Transformers")
    print("  - Any HuggingFace-compatible tool")


def main():
    parser = argparse.ArgumentParser(description="Export Mamba model to HuggingFace format")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to Mamba checkpoint directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save HuggingFace model')

    args = parser.parse_args()

    export_to_huggingface(args.checkpoint, args.output)


if __name__ == '__main__':
    main()
