# Getting Started with Mamba Trainer

Welcome! This guide will help you get started training Mamba language models.

## What is This?

This is a complete training framework for **Mamba** - a new language model architecture that's faster and more memory-efficient than transformers. It uses Selective State Space Models (SSMs) to achieve linear-time complexity.

## What Can You Do?

✅ Train language models on your own documents
✅ Train on code, PDFs, text files, and more
✅ Train on HuggingFace datasets
✅ Choose model sizes that fit your GPU (8GB to 48GB)
✅ Generate text with your trained models

## Quick Start (3 Steps)

### Step 1: Install

```bash
# Install dependencies
pip install -r requirements.txt

# Or use the setup script
bash setup.sh
```

### Step 2: Verify Setup

```bash
# Run verification tests
python test_setup.py
```

This checks:
- All packages are installed
- Model can be created
- Data loader works
- GPU is detected (if available)

### Step 3: Train!

```bash
# The example data is already in data/example.txt
# Just run the trainer!

# For 8GB GPU
python train.py --model_size small --vram 8gb

# For 16GB GPU
python train.py --model_size medium --vram 16gb

# For 24GB+ GPU
python train.py --model_size large --vram 24gb
```

## Training on Your Own Data

### Supported File Types

The trainer automatically loads:

**Documents**:
- PDF files (.pdf)
- Text files (.txt)
- Markdown (.md, .markdown)
- reStructuredText (.rst)

**Code Files**:
- Python (.py)
- JavaScript/TypeScript (.js, .jsx, .ts, .tsx)
- Java (.java)
- C/C++ (.c, .cpp, .h)
- Go (.go)
- Rust (.rs)
- And many more!

**Configuration**:
- JSON (.json)
- YAML (.yaml, .yml)
- TOML (.toml)
- XML (.xml)

### Add Your Files

```bash
# Simply copy your files to the data directory
cp /path/to/your/documents/*.pdf data/
cp /path/to/your/code/*.py data/

# Check what will be loaded
python prepare_data.py --show_stats
```

### Train on Your Data

```bash
python train.py --data_dir data --model_size medium --vram 16gb
```

## Using HuggingFace Datasets

Train on any text dataset from HuggingFace:

```bash
# WikiText-2
python train.py --dataset wikitext --vram 16gb

# OpenWebText
python train.py --dataset openwebtext --vram 24gb

# TinyStories
python train.py --dataset roneneldan/TinyStories --vram 12gb
```

## Choosing the Right Configuration

### By GPU Memory

| Your GPU | VRAM | Command |
|----------|------|---------|
| GTX 1080, RTX 2070 | 8GB | `--model_size small --vram 8gb` |
| RTX 3060, 2080 Ti | 12GB | `--model_size medium --vram 12gb` |
| RTX 3080, 4060 Ti | 16GB | `--model_size medium --vram 16gb` |
| RTX 3090, 4090 | 24GB | `--model_size large --vram 24gb` |
| V100, A5500 | 32GB | `--model_size large --vram 32gb` |
| A40, A6000 | 48GB | `--model_size large --vram 48gb` |

### By Model Size

| Size | Parameters | Training Speed | Quality | Best For |
|------|-----------|----------------|---------|----------|
| **Small** | ~50M | Fast | Good | Experiments, limited GPU |
| **Medium** | ~150M | Medium | Better | General use |
| **Large** | ~400M | Slower | Best | High-quality models |

## Understanding Training Output

When training starts, you'll see:

```
Using device: cuda
GPU: NVIDIA RTX 3090
VRAM: 24.00 GB

Total parameters: 150,253,312
Trainable parameters: 150,253,312

Starting training...
Total training steps: 100000

Training: 45%|████████ | 45000/100000 [2:30:00<3:00:00, loss=2.34, lr=2.1e-4]
```

**What this means**:
- `loss`: Lower is better (starts high, decreases over time)
- `lr`: Learning rate (varies during training)
- Progress bar shows completion percentage

## Checkpoints

Models are saved automatically:

```
checkpoints/
├── step_5000/      # Saved at 5K steps
├── step_10000/     # Saved at 10K steps
├── step_15000/     # Saved at 15K steps
└── final/          # Final model
```

Each checkpoint contains:
- Model weights
- Optimizer state
- Configuration
- Training progress

## Generating Text

After training, generate text with your model:

```bash
python generate.py \
  --checkpoint checkpoints/final \
  --prompt "Once upon a time" \
  --max_tokens 100
```

Options:
- `--temperature`: Higher = more random (default: 1.0)
- `--top_k`: Limits vocab to top K tokens (default: 50)
- `--max_tokens`: How many tokens to generate

## Common Tasks

### Resume Training

```bash
python train.py --resume checkpoints/step_50000
```

### Train Faster

```bash
python train.py --training_preset fast --vram 16gb
```

### Train Longer (Better Quality)

```bash
python train.py --training_preset careful --vram 24gb
```

### Use Custom Config

```bash
# Copy and modify an example
cp configs/example_24gb.json my_config.json
# Edit my_config.json with your settings

python train.py --config my_config.json
```

## Troubleshooting

### "Out of Memory" Error

**Solution 1**: Use smaller model
```bash
python train.py --model_size small --vram 8gb
```

**Solution 2**: Use smaller VRAM config
```bash
python train.py --vram 8gb  # Even with medium model
```

### Training is Slow

This is normal! Language model training takes time. Tips:

- Use `--training_preset fast` for quicker experiments
- Reduce `eval_interval` and `save_interval` in config
- Ensure you're using GPU (check output for "Using device: cuda")

### No Data Found

```bash
# Check that files exist
ls data/

# Verify data can be loaded
python prepare_data.py --show_stats
```

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or use the setup script
bash setup.sh
```

## File Descriptions

| File | Purpose |
|------|---------|
| `train.py` | Main training script |
| `generate.py` | Generate text with trained model |
| `prepare_data.py` | Inspect training data |
| `test_setup.py` | Verify installation |
| `setup.sh` | Automated setup |
| `README.md` | Full documentation |
| `QUICKSTART.md` | Quick reference |
| `PROJECT_OVERVIEW.md` | Technical details |

## Next Steps

1. ✅ Run `python test_setup.py` to verify everything works
2. ✅ Add your data to `data/` directory
3. ✅ Run `python prepare_data.py --show_stats` to inspect data
4. ✅ Start training with appropriate settings for your GPU
5. ✅ Monitor training progress
6. ✅ Generate text with your trained model

## Learning More

- **README.md**: Complete documentation with all options
- **QUICKSTART.md**: Quick command reference
- **PROJECT_OVERVIEW.md**: Technical architecture details
- **configs/**: Example configurations

## Tips for Success

1. **Start Small**: Begin with small model and short training
2. **Monitor GPU**: Use `nvidia-smi` to check GPU usage
3. **Save Checkpoints**: Don't skip checkpointing
4. **Experiment**: Try different data sources and configurations
5. **Be Patient**: Training takes time, but results are worth it!

## Getting Help

1. Run `python test_setup.py` to diagnose issues
2. Check the troubleshooting sections
3. Review example configurations
4. Read the full README.md

## Example Training Session

```bash
# 1. Verify setup
python test_setup.py

# 2. Add your data
cp ~/my_documents/*.pdf data/
cp ~/my_code/*.py data/

# 3. Check data
python prepare_data.py --show_stats --show_samples 2

# 4. Train (adjust for your GPU)
python train.py --model_size medium --vram 16gb

# 5. Generate text
python generate.py \
  --checkpoint checkpoints/final \
  --prompt "Your prompt here" \
  --max_tokens 200

# 6. If you want to continue training
python train.py --resume checkpoints/final
```

## You're Ready!

You now have everything you need to train Mamba language models. Start with the test script, add your data, and begin training!

Happy training! 🚀
