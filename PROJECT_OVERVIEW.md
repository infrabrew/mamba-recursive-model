# Mamba Trainer - Project Overview

## Summary

A complete training framework for Mamba language models with support for multiple model sizes, VRAM configurations, and diverse data sources.

## Key Features

✅ **Mamba Architecture Implementation**
- Selective State Space Models (SSM)
- Linear-time complexity O(n)
- Efficient selective scanning mechanism

✅ **Flexible Model Sizes**
- Small: ~50M parameters (512 dim, 8 layers)
- Medium: ~150M parameters (768 dim, 12 layers)
- Large: ~400M parameters (1024 dim, 24 layers)

✅ **VRAM-Optimized Configurations**
- 8GB: Batch=1, GradAcc=16, FP16, Checkpointing
- 12GB: Batch=2, GradAcc=8, FP16, Checkpointing
- 16GB: Batch=4, GradAcc=4, FP16, Checkpointing
- 24GB: Batch=8, GradAcc=2, FP16
- 32GB: Batch=12, GradAcc=2, FP16
- 48GB: Batch=16, GradAcc=1, FP32

✅ **Comprehensive Data Loading**
- PDF documents
- Text files (TXT, MD, RST)
- Source code (Python, JS, Java, C++, Go, Rust, etc.)
- Configuration files (JSON, YAML, TOML, XML)
- HuggingFace datasets
- Automatic chunking and preprocessing

✅ **Memory Optimizations**
- Mixed precision training (FP16)
- Gradient accumulation
- Gradient checkpointing
- Efficient data streaming

## Project Structure

```
mamba_trainer/
│
├── Core Training Scripts
│   ├── train.py                    # Main training script
│   ├── generate.py                 # Text generation
│   └── prepare_data.py             # Data inspection
│
├── Model Implementation
│   └── models/
│       ├── mamba_model.py          # Mamba architecture
│       │   ├── SelectiveSSM        # Core SSM component
│       │   ├── MambaBlock          # Single Mamba block
│       │   └── MambaLanguageModel  # Complete model
│       └── __init__.py
│
├── Configuration System
│   └── configs/
│       ├── model_configs.py        # Config presets
│       │   ├── MODEL_CONFIGS       # Small/Medium/Large
│       │   ├── VRAM_CONFIGS        # 8GB to 48GB
│       │   └── TRAINING_CONFIGS    # Default/Fast/Careful
│       ├── example_8gb.json        # 8GB example
│       ├── example_24gb.json       # 24GB example
│       └── __init__.py
│
├── Data Utilities
│   └── utils/
│       ├── data_loader.py          # Multi-format loading
│       │   ├── DataLoader          # Standard loader
│       │   └── StreamingDataLoader # Memory-efficient
│       └── __init__.py
│
├── Documentation
│   ├── README.md                   # Full documentation
│   ├── QUICKSTART.md               # Quick start guide
│   └── PROJECT_OVERVIEW.md         # This file
│
├── Setup & Configuration
│   ├── requirements.txt            # Python dependencies
│   └── setup.sh                    # Setup script
│
└── Data & Output
    ├── data/                       # Training data (you add)
    │   └── example.txt             # Example file
    └── checkpoints/                # Saved models
```

## Components Explained

### 1. Mamba Architecture (models/mamba_model.py)

**SelectiveSSM**: Core selective state space model
- Implements selective scanning mechanism
- Computes state transitions dynamically
- Uses learnable selection parameters (delta, B, C)

**MambaBlock**: Single transformer-style block
- Layer normalization
- Selective SSM application
- Residual connections
- SiLU activation

**MambaLanguageModel**: Complete model
- Token embeddings
- Positional embeddings
- Stack of Mamba blocks
- Language modeling head
- Generation capabilities

### 2. Configuration System (configs/)

**Three-level configuration**:
1. **Model Config**: Architecture parameters (d_model, n_layers, etc.)
2. **VRAM Config**: Memory optimization settings
3. **Training Config**: Hyperparameters (LR, steps, etc.)

**Presets**: Pre-configured combinations for common scenarios

**Custom Configs**: JSON files for full customization

### 3. Data Loading (utils/data_loader.py)

**DataLoader**:
- Scans directories for supported files
- Loads and parses multiple formats
- Chunks text for training
- Supports HuggingFace datasets

**StreamingDataLoader**:
- Memory-efficient streaming
- Processes one file at a time
- Ideal for very large datasets

### 4. Training Script (train.py)

**Trainer Class**:
- Model initialization
- Optimizer setup (AdamW)
- Mixed precision training
- Gradient accumulation
- Learning rate scheduling
- Checkpoint saving/loading
- Evaluation

**Training Loop**:
- Progress tracking
- Loss logging
- Periodic evaluation
- Checkpoint management

### 5. Generation (generate.py)

**Features**:
- Load trained models
- Autoregressive generation
- Temperature sampling
- Top-k filtering
- Configurable generation length

## Usage Workflow

### 1. Setup
```bash
pip install -r requirements.txt
# or run: bash setup.sh
```

### 2. Prepare Data
```bash
# Add files to data/
cp /path/to/files/* data/

# Inspect data
python prepare_data.py --show_stats
```

### 3. Choose Configuration

Based on your GPU:
- **8GB**: `--model_size small --vram 8gb`
- **16GB**: `--model_size medium --vram 16gb`
- **24GB+**: `--model_size large --vram 24gb`

### 4. Train
```bash
python train.py --model_size medium --vram 16gb
```

### 5. Generate
```bash
python generate.py --checkpoint checkpoints/final --prompt "Hello"
```

## Technical Details

### Mamba vs Transformers

| Feature | Transformer | Mamba |
|---------|------------|-------|
| Complexity | O(n²) | O(n) |
| Mechanism | Attention | Selective SSM |
| Memory | High | Lower |
| Speed | Slower | Faster |
| Context | Global | Selective |

### Memory Optimization Techniques

1. **Mixed Precision (FP16)**
   - Reduces memory by 50%
   - Faster computation on modern GPUs
   - Maintained with gradient scaling

2. **Gradient Accumulation**
   - Simulates larger batch sizes
   - N micro-batches = 1 effective batch
   - No accuracy loss

3. **Gradient Checkpointing**
   - Trades compute for memory
   - Recomputes activations during backward
   - Essential for 8-12GB VRAM

### Training Parameters

**Learning Rate Schedule**:
- Linear warmup over N steps
- Cosine decay to end of training
- Default LR: 3e-4 (can be adjusted)

**Optimization**:
- AdamW optimizer
- Weight decay: 0.1
- Gradient clipping: 1.0
- Beta1: 0.9, Beta2: 0.95

**Checkpointing**:
- Periodic saves (every 5K steps by default)
- Final checkpoint
- Includes: model, optimizer, config, training state

## Extension Points

### Adding Custom Data Formats

Extend `DataLoader` class:
```python
def load_custom_format(self, file_path):
    # Your logic here
    return text
```

### Modifying Model Architecture

Edit `models/mamba_model.py`:
- Change layer configurations
- Add new components
- Modify SSM implementation

### Custom Training Loops

Extend `Trainer` class in `train.py`:
- Add custom metrics
- Implement new optimization strategies
- Add logging/tracking

## Performance Tips

1. **Data on SSD**: Faster loading
2. **Large Batches**: Better GPU utilization
3. **Reduce Eval Frequency**: Faster training
4. **Use FP16**: Memory + speed gains
5. **Monitor GPU Utilization**: Check with `nvidia-smi`

## Common Modifications

### Change Vocabulary Size
Edit tokenizer or update `vocab_size` in config

### Longer Sequences
Increase `max_seq_len` (requires more memory)

### Different Tokenizer
Replace GPT-2 tokenizer in train.py

### Add Wandb Logging
Uncomment wandb in requirements, add to Trainer

### Distributed Training
Add DDP/FSDP wrappers (future enhancement)

## Limitations & Future Work

**Current Limitations**:
- No optimized CUDA kernels (uses PyTorch implementation)
- Single GPU training only
- Limited generation options

**Planned Enhancements**:
- Optimized selective scan kernels
- Multi-GPU support (DDP/FSDP)
- Advanced generation (beam search, nucleus sampling)
- Model evaluation metrics (perplexity, etc.)
- Experiment tracking integration
- Pre-trained model weights

## Dependencies

**Core**:
- PyTorch 2.0+
- Transformers
- Datasets

**Data Processing**:
- PyPDF2

**Utilities**:
- tqdm
- numpy

**Optional**:
- wandb (experiment tracking)
- triton (optimized kernels)
- einops (tensor operations)

## Citation

Based on the Mamba architecture from:
```
Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence
Modeling with Selective State Spaces. arXiv:2312.00752
```

## License

Educational and research use.

## Support

Check documentation:
- README.md: Full guide
- QUICKSTART.md: Quick reference
- This file: Technical overview

For issues, review troubleshooting sections.
