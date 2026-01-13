# Mamba Language Model Trainer

A comprehensive training framework for Mamba architecture language models with support for various model sizes, VRAM configurations, and multiple data sources.

## Features

- **Mamba Architecture**: Implementation of the efficient Mamba (Selective State Space Model) architecture
- **Multiple Model Sizes**: Pre-configured small (~50M), medium (~150M), and large (~400M) parameter models
- **VRAM-Optimized Configurations**: Optimized settings for 8GB, 12GB, 16GB, 24GB, 32GB, and 48GB VRAM
- **Flexible Data Loading**: Support for PDF, TXT, source code, Markdown, JSON, config files, and HuggingFace datasets
- **Memory Efficient**: Gradient accumulation, mixed precision (FP16), and gradient checkpointing
- **Easy Configuration**: Simple command-line interface with sensible defaults

## Supported File Formats

The trainer can process the following file types from the `data/` directory:

- **Documents**: PDF, TXT, Markdown (.md), reStructuredText (.rst)
- **Source Code**: Python, JavaScript, TypeScript, Java, C/C++, Go, Rust, Ruby, PHP, and more
- **Configuration**: JSON, YAML, TOML, INI, XML
- **HuggingFace Datasets**: Any text dataset from the HuggingFace Hub

## Installation

```bash
# Clone the repository
cd mamba_trainer

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Data

Add your training files to the `data/` directory:

```bash
mkdir -p data
# Add your PDF, TXT, code files, etc. to data/
cp /path/to/your/documents/* data/
```

### 2. Train a Model

Basic training with default settings (medium model, 16GB VRAM):

```bash
python train.py --data_dir data
```

### 3. Custom Configuration

Train with specific model size and VRAM configuration:

```bash
# Small model on 8GB VRAM
python train.py --model_size small --vram 8gb --data_dir data

# Large model on 24GB VRAM with fast training preset
python train.py --model_size large --vram 24gb --training_preset fast --data_dir data
```

## Configuration Options

### Model Sizes

| Size | Parameters | d_model | Layers | Best For |
|------|-----------|---------|--------|----------|
| **small** | ~50M | 512 | 8 | Quick experiments, limited VRAM |
| **medium** | ~150M | 768 | 12 | Balanced performance |
| **large** | ~400M | 1024 | 24 | Maximum quality, high VRAM |

### VRAM Configurations

| VRAM | Batch Size | Gradient Acc | FP16 | Recommended Models | Example GPUs |
|------|-----------|--------------|------|-------------------|--------------|
| **8GB** | 1 | 16 | ✓ | small | GTX 1080, RTX 2070 |
| **12GB** | 2 | 8 | ✓ | small, medium | RTX 3060, RTX 2080 Ti |
| **16GB** | 4 | 4 | ✓ | small, medium | RTX 4060 Ti, RTX 3080 |
| **24GB** | 8 | 2 | ✓ | all | RTX 3090, RTX 4090 |
| **32GB** | 12 | 2 | ✓ | all | V100, A5500 |
| **48GB** | 16 | 1 | - | all | A40, A6000 |

### Training Presets

- **default**: Balanced settings (LR: 3e-4, 100K steps)
- **fast**: Faster training with higher LR (LR: 5e-4, 50K steps)
- **careful**: Conservative training (LR: 1e-4, 200K steps)

## Command-Line Arguments

```bash
python train.py [OPTIONS]

Model and Training Configuration:
  --model_size {small,medium,large}    Model size preset (default: medium)
  --vram {8gb,12gb,16gb,24gb,32gb,48gb}  VRAM configuration (default: 16gb)
  --training_preset {default,fast,careful}  Training preset (default: default)

Data Sources:
  --data_dir PATH              Directory with training files (default: data)
  --dataset NAME               HuggingFace dataset name (optional)
  --dataset_split SPLIT        Dataset split (default: train)
  --text_column COLUMN         Text column name (default: text)

Output:
  --output_dir PATH            Checkpoint directory (default: checkpoints)

Advanced:
  --config PATH                Custom config JSON file
  --resume PATH                Resume from checkpoint
```

## Usage Examples

### Example 1: Train on Local Files

```bash
# Add your documents to data/
mkdir -p data
cp ~/my_documents/*.pdf data/
cp ~/my_code/*.py data/

# Train medium model on 16GB GPU
python train.py --model_size medium --vram 16gb
```

### Example 2: Train on HuggingFace Dataset

```bash
# Train on WikiText-2 dataset
python train.py --dataset wikitext --dataset_split train --text_column text --vram 24gb

# Train on OpenWebText
python train.py --dataset openwebtext --vram 32gb --model_size large
```

### Example 3: Combined Data Sources

```bash
# Train on both local files and HuggingFace dataset
python train.py \
  --data_dir data \
  --dataset wikitext \
  --model_size medium \
  --vram 16gb
```

### Example 4: Resume Training

```bash
# Resume from a checkpoint
python train.py --resume checkpoints/step_50000 --vram 24gb
```

### Example 5: Custom Configuration

```bash
# Use a custom config file
python train.py --config configs/example_24gb.json
```

## Project Structure

```
mamba_trainer/
├── train.py                 # Main training script
├── requirements.txt         # Python dependencies
├── README.md               # This file
│
├── configs/                # Configuration files
│   ├── __init__.py
│   ├── model_configs.py    # Model and VRAM presets
│   ├── example_8gb.json    # Example 8GB config
│   └── example_24gb.json   # Example 24GB config
│
├── models/                 # Model implementations
│   ├── __init__.py
│   └── mamba_model.py      # Mamba architecture
│
├── utils/                  # Utility modules
│   ├── __init__.py
│   └── data_loader.py      # Data loading utilities
│
├── data/                   # Training data (you create this)
│   ├── documents/
│   ├── code/
│   └── ...
│
└── checkpoints/            # Saved model checkpoints
    ├── step_5000/
    ├── step_10000/
    └── final/
```

## Understanding Checkpoints

Checkpoints are saved to the `checkpoints/` directory at regular intervals. Each checkpoint contains:

- `model.pt`: Model weights
- `optimizer.pt`: Optimizer state
- `config.json`: Full configuration
- `training_state.pt`: Training progress (step, epoch)

Resume training from any checkpoint:

```bash
python train.py --resume checkpoints/step_50000
```

## Memory Optimization Tips

### For Limited VRAM (8GB - 12GB):

1. Use `small` model size
2. Reduce `max_seq_len` in config
3. Enable gradient checkpointing (automatic)
4. Use FP16 (automatic)
5. Increase gradient accumulation steps

### For Medium VRAM (16GB - 24GB):

1. Use `small` or `medium` model
2. Standard settings work well
3. Can disable gradient checkpointing for speed

### For High VRAM (32GB+):

1. Use any model size including `large`
2. Increase batch size
3. Can use FP32 for maximum precision
4. Reduce gradient accumulation steps

## Custom Configuration

Create a custom JSON config file:

```json
{
  "model": {
    "d_model": 768,
    "n_layers": 12,
    "d_state": 16,
    "expand_factor": 2,
    "dropout": 0.1,
    "max_seq_len": 2048,
    "vocab_size": 50257
  },
  "vram": {
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "use_fp16": true,
    "use_gradient_checkpointing": true
  },
  "training": {
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "max_steps": 100000,
    "warmup_steps": 2000,
    "eval_interval": 1000,
    "save_interval": 5000
  }
}
```

Use it with:

```bash
python train.py --config my_config.json
```

## Performance Tips

1. **Use SSD for data**: Store training data on SSD for faster loading
2. **Mixed Precision**: FP16 is enabled automatically for most configs
3. **Gradient Accumulation**: Allows larger effective batch sizes with limited VRAM
4. **Eval Interval**: Reduce for faster training, increase for more frequent evaluation
5. **Save Interval**: Adjust based on training time and disk space

## Troubleshooting

### Out of Memory Error

1. Reduce `--model_size` to a smaller model
2. Use a smaller VRAM configuration: `--vram 8gb`
3. Reduce sequence length in config file
4. Ensure no other processes are using GPU

### Slow Training

1. Verify GPU is being used (check training output)
2. Reduce `eval_interval` and `save_interval`
3. Use `--training_preset fast`
4. Enable FP16 training (automatic for most configs)

### No Data Loaded

1. Check that files exist in `--data_dir`
2. Verify file formats are supported
3. Check file permissions
4. Look for errors in data loading output

## Advanced Features

### Streaming Data Loading

For very large datasets that don't fit in RAM, the `StreamingDataLoader` class provides memory-efficient loading:

```python
from utils.data_loader import StreamingDataLoader

loader = StreamingDataLoader('data', chunk_size=2048)
for chunk in loader.stream_files():
    # Process chunk
    pass
```

### Custom File Processing

Extend the `DataLoader` class to support additional file formats:

```python
from utils.data_loader import DataLoader

class CustomDataLoader(DataLoader):
    def load_custom_format(self, file_path):
        # Your custom loading logic
        pass
```

## Citation

If you use this code, please cite the original Mamba paper:

```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Areas for improvement:

- Optimized CUDA kernels for selective scan
- Additional data preprocessing options
- Distributed training support
- Model evaluation metrics
- Generation examples and utilities

## Support

For issues or questions:

1. Check the troubleshooting section
2. Review configuration examples
3. Ensure all dependencies are installed correctly
