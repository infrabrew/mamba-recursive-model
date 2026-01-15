#!/usr/bin/env python3
"""
Main training script for Mamba language models.
Supports various model sizes, VRAM configurations, and data sources.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, List
import math

# Enable unbuffered output for live logging
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

# Set PyTorch memory optimization flags BEFORE importing torch
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Local imports
from models.mamba_model import create_mamba_model
from models.hybrid_recursive_mamba import create_hybrid_recursive_mamba_model
from configs.model_configs import get_config, print_config, save_config, load_config
from utils.data_loader import DataLoader as FileDataLoader


class TextDataset(Dataset):
    """Dataset for text chunks with tokenization."""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 2048):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)

        # Create labels (shifted input_ids for language modeling)
        labels = input_ids.clone()

        return {
            'input_ids': input_ids,
            'labels': labels
        }


class Trainer:
    """Training manager for Mamba models."""

    def __init__(self, config: dict, output_dir: str = "checkpoints"):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Setup TensorBoard
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"runs/training_{timestamp}"
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logging to: {log_dir}")
        print(f"View with: tensorboard --logdir=runs")

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

            # Enable memory optimizations
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True

            # Set memory fraction if needed (prevent OOM)
            # torch.cuda.set_per_process_memory_fraction(0.95, 0)

        # Initialize tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Update vocab size in config
        self.config['model']['vocab_size'] = len(self.tokenizer)

        # Create model
        print("Creating model...")
        model_type = self.config['model'].get('model_type', 'standard')
        if model_type == 'hybrid_recursive':
            print("Using Hybrid Recursive Mamba architecture")
            self.model = create_hybrid_recursive_mamba_model(self.config['model'])
        else:
            self.model = create_mamba_model(self.config['model'])
        self.model.to(self.device)

        # Print model size
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            betas=(self.config['training']['adam_beta1'], self.config['training']['adam_beta2']),
            eps=self.config['training']['adam_epsilon'],
            weight_decay=self.config['training']['weight_decay']
        )

        # Setup mixed precision training
        self.use_fp16 = self.config['vram'].get('use_fp16', False)
        self.scaler = GradScaler() if self.use_fp16 else None

        # Setup gradient checkpointing
        if self.config['vram'].get('use_gradient_checkpointing', False):
            print("Enabling gradient checkpointing...")
            # Note: Implement gradient checkpointing in MambaBlock if needed

        # Training state
        self.global_step = 0
        self.epoch = 0

    def create_dataloader(self, texts: List[str], shuffle: bool = True) -> DataLoader:
        """Create a DataLoader from text chunks."""
        dataset = TextDataset(
            texts,
            self.tokenizer,
            self.config['model']['max_seq_len']
        )

        return DataLoader(
            dataset,
            batch_size=self.config['vram']['batch_size'],
            shuffle=shuffle,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=True if torch.cuda.is_available() else False
        )

    def get_lr_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler with warmup."""
        warmup_steps = self.config['training']['warmup_steps']

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps))
            )

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch) -> float:
        """Single training step."""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Forward pass
        if self.use_fp16:
            with autocast():
                logits, loss = self.model(input_ids, labels)
        else:
            logits, loss = self.model(input_ids, labels)

        return loss

    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None):
        """Main training loop."""
        print("\nStarting training...")
        print(f"Total training steps: {self.config['training']['max_steps']}")

        gradient_accumulation_steps = self.config['vram']['gradient_accumulation_steps']
        max_steps = self.config['training']['max_steps']
        eval_interval = self.config['training']['eval_interval']
        save_interval = self.config['training']['save_interval']
        logging_steps = self.config['training']['logging_steps']

        # Create scheduler
        scheduler = self.get_lr_scheduler(max_steps)

        # Training loop
        self.model.train()
        running_loss = 0.0
        optimizer_step_count = 0

        progress_bar = tqdm(total=max_steps, desc="Training")

        while self.global_step < max_steps:
            for batch in train_dataloader:
                # Training step
                loss = self.train_step(batch)

                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps

                # Backward pass
                if self.use_fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                running_loss += loss.item()

                # Update weights every gradient_accumulation_steps
                if (self.global_step + 1) % gradient_accumulation_steps == 0:
                    # Clip gradients
                    if self.use_fp16:
                        self.scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['max_grad_norm']
                    )

                    # Optimizer step
                    if self.use_fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    scheduler.step()
                    self.optimizer.zero_grad()
                    optimizer_step_count += 1

                # Logging
                if (self.global_step + 1) % logging_steps == 0:
                    avg_loss = running_loss / logging_steps
                    lr = scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{lr:.2e}'
                    })

                    # Log to TensorBoard
                    self.writer.add_scalar('train/loss', avg_loss, self.global_step + 1)
                    self.writer.add_scalar('train/learning_rate', lr, self.global_step + 1)
                    if torch.cuda.is_available():
                        mem_used = torch.cuda.memory_allocated() / 1e9
                        mem_reserved = torch.cuda.memory_reserved() / 1e9
                        self.writer.add_scalar('system/gpu_memory_used_gb', mem_used, self.global_step + 1)
                        self.writer.add_scalar('system/gpu_memory_reserved_gb', mem_reserved, self.global_step + 1)

                    running_loss = 0.0

                # Evaluation
                if eval_dataloader and (self.global_step + 1) % eval_interval == 0:
                    eval_loss = self.evaluate(eval_dataloader)
                    print(f"\nStep {self.global_step + 1} - Eval loss: {eval_loss:.4f}")
                    # Log eval loss to TensorBoard
                    self.writer.add_scalar('eval/loss', eval_loss, self.global_step + 1)
                    self.model.train()

                # Save checkpoint
                if (self.global_step + 1) % save_interval == 0:
                    self.save_checkpoint(f"step_{self.global_step + 1}")

                self.global_step += 1
                progress_bar.update(1)

                if self.global_step >= max_steps:
                    break

            if self.global_step >= max_steps:
                break

            self.epoch += 1

        progress_bar.close()

        # Final checkpoint
        self.save_checkpoint("final")
        print(f"\nTraining completed! Final checkpoint saved.")

        # Close TensorBoard writer
        self.writer.close()
        print(f"TensorBoard logs saved. View with: tensorboard --logdir=runs")

    @torch.no_grad()
    def evaluate(self, eval_dataloader: DataLoader) -> float:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        total_steps = 0

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            if self.use_fp16:
                with autocast():
                    logits, loss = self.model(input_ids, labels)
            else:
                logits, loss = self.model(input_ids, labels)

            total_loss += loss.item()
            total_steps += 1

        return total_loss / total_steps

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model
        torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))

        # Save optimizer
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))

        # Save config
        save_config(self.config, os.path.join(checkpoint_dir, "config.json"))

        # Save training state
        state = {
            'global_step': self.global_step,
            'epoch': self.epoch
        }
        torch.save(state, os.path.join(checkpoint_dir, "training_state.pt"))

        print(f"Checkpoint saved to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir: str):
        """Load model checkpoint."""
        # Load model
        self.model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.pt"), weights_only=True))

        # Load optimizer
        self.optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.pt"), weights_only=True))

        # Load training state
        state = torch.load(os.path.join(checkpoint_dir, "training_state.pt"), weights_only=True)
        self.global_step = state['global_step']
        self.epoch = state['epoch']

        print(f"Checkpoint loaded from {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Mamba language model")

    # Model and training configuration
    parser.add_argument('--model_size', type=str, default='medium',
                        choices=['small', 'medium-lite', 'medium-plus', 'medium-x', 'medium', 'medium-extra', 'large-lite', 'large-plus', 'large', 'xlarge', 'xxlarge', 'hybrid-small', 'hybrid-small-plus', 'hybrid-small-pro', 'hybrid-medium', 'hybrid-large'],
                        help='Model size preset')
    parser.add_argument('--vram', type=str, default='16gb',
                        help='VRAM configuration (e.g., "16gb", "24gb", or custom like "20.5" for 20.5GB)')
    parser.add_argument('--vram_auto', action='store_true',
                        help='Auto-detect and use all available GPU VRAM')
    parser.add_argument('--vram_safety_margin', type=int, default=2048,
                        help='Safety margin in MB to prevent OOM (default: 2048)')
    parser.add_argument('--training_preset', type=str, default='default',
                        choices=['default', 'fast', 'careful'],
                        help='Training hyperparameter preset')

    # Data sources
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing training files')
    parser.add_argument('--dataset', type=str, default=None,
                        help='HuggingFace dataset name (optional)')
    parser.add_argument('--dataset_split', type=str, default='train',
                        help='Dataset split to use')
    parser.add_argument('--text_column', type=str, default='text',
                        help='Column name for text in dataset')

    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints')

    # Optional overrides
    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom config JSON file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint directory')

    args = parser.parse_args()

    # Load or create configuration
    if args.config:
        config = load_config(args.config)
    else:
        # Check if using custom VRAM value or auto-detection
        from utils.vram_optimizer import create_vram_config, VRAMOptimizer

        if args.vram_auto:
            # Auto-detect GPU VRAM
            print("Auto-detecting GPU VRAM...")
            optimizer = VRAMOptimizer(safety_margin_mb=args.vram_safety_margin)
            vram_gb = optimizer.target_vram_gb
            print(f"Detected {vram_gb:.2f} GB total VRAM")

            # Get base config and optimize VRAM settings
            from configs.model_configs import MODEL_CONFIGS, TRAINING_CONFIGS
            config = {
                'model': MODEL_CONFIGS[args.model_size].copy(),
                'training': TRAINING_CONFIGS[args.training_preset].copy(),
            }
            config['vram'] = optimizer.optimize_config(config['model'], config['training'])

        elif args.vram.replace('.', '').replace('gb', '').isdigit():
            # Custom numeric VRAM value (e.g., "24", "20.5")
            vram_value = args.vram.replace('gb', '').strip()
            vram_gb = float(vram_value)

            print(f"Using custom VRAM target: {vram_gb:.2f} GB")

            # Get base config and optimize VRAM settings
            from configs.model_configs import MODEL_CONFIGS, TRAINING_CONFIGS
            config = {
                'model': MODEL_CONFIGS[args.model_size].copy(),
                'training': TRAINING_CONFIGS[args.training_preset].copy(),
            }

            optimizer = VRAMOptimizer(target_vram_gb=vram_gb, safety_margin_mb=args.vram_safety_margin)
            config['vram'] = optimizer.optimize_config(config['model'], config['training'])

        else:
            # Standard preset (8gb, 12gb, 16gb, etc.)
            config = get_config(args.model_size, args.vram, args.training_preset)

    # Print configuration
    print_config(config)

    # Load data
    print("Loading training data...")
    file_loader = FileDataLoader(args.data_dir, config['model']['max_seq_len'])

    texts = []

    # Load from files in data directory
    if os.path.exists(args.data_dir):
        documents = file_loader.load_directory()
        file_texts = file_loader.prepare_training_data(documents)
        texts.extend(file_texts)
        print(f"Loaded {len(file_texts)} chunks from local files")

    # Load from HuggingFace dataset if specified
    if args.dataset:
        dataset_docs = file_loader.load_huggingface_dataset(
            args.dataset,
            args.dataset_split,
            args.text_column
        )
        dataset_texts = file_loader.prepare_training_data(dataset_docs)
        texts.extend(dataset_texts)
        print(f"Loaded {len(dataset_texts)} chunks from dataset")

    if not texts:
        print("ERROR: No training data found!")
        print(f"Please add files to '{args.data_dir}' or specify a dataset with --dataset")
        return

    print(f"\nTotal training chunks: {len(texts)}")

    # Split into train/eval
    eval_size = min(1000, len(texts) // 10)
    eval_texts = texts[:eval_size]
    train_texts = texts[eval_size:]

    print(f"Training chunks: {len(train_texts)}")
    print(f"Evaluation chunks: {len(eval_texts)}")

    # Create trainer
    trainer = Trainer(config, args.output_dir)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Create dataloaders
    train_dataloader = trainer.create_dataloader(train_texts, shuffle=True)
    eval_dataloader = trainer.create_dataloader(eval_texts, shuffle=False)

    # Train
    trainer.train(train_dataloader, eval_dataloader)


if __name__ == '__main__':
    main()
