# Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Training Data

```bash
# Add your files to the data directory
cp /path/to/your/files/* data/
```

Supported formats: PDF, TXT, Python, JavaScript, Markdown, JSON, and more!

### 3. Train Your Model

```bash
# For 8GB GPU (e.g., GTX 1080)
python train.py --model_size small --vram 8gb

# For 16GB GPU (e.g., RTX 3080)
python train.py --model_size medium --vram 16gb

# For 24GB+ GPU (e.g., RTX 4090)
python train.py --model_size large --vram 24gb
```

### 4. Generate Text

```bash
python generate.py --checkpoint checkpoints/final --prompt "Your prompt here"
```

## Common Commands

### Inspect Your Data

```bash
python prepare_data.py --show_stats --show_samples 3
```

### Train on HuggingFace Dataset

```bash
python train.py --dataset wikitext --vram 16gb
```

### Resume Training

```bash
python train.py --resume checkpoints/step_50000
```

### Use Custom Config

```bash
python train.py --config configs/example_8gb.json
```

## Model Size Recommendations

| GPU Memory | Recommended Model | Command |
|-----------|------------------|---------|
| 8GB | Small | `--model_size small --vram 8gb` |
| 12GB | Small/Medium | `--model_size medium --vram 12gb` |
| 16GB | Medium | `--model_size medium --vram 16gb` |
| 24GB | Large | `--model_size large --vram 24gb` |
| 32GB+ | Large | `--model_size large --vram 32gb` |

## Troubleshooting

**Out of Memory?**
- Use smaller model: `--model_size small`
- Use smaller VRAM config: `--vram 8gb`

**Slow Training?**
- Use fast preset: `--training_preset fast`
- Reduce eval interval in config

**No Data Found?**
- Check files are in `data/` directory
- Run `python prepare_data.py` to verify

## Full Documentation

See [README.md](README.md) for complete documentation.
