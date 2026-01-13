# Mamba Trainer - Complete Index

## 🚀 Start Here

**New to the project?** Start with these in order:

1. **[GETTING_STARTED.md](GETTING_STARTED.md)** ← **START HERE!**
   - Complete beginner's guide
   - Installation instructions
   - Your first training run
   - Troubleshooting

2. **[USAGE_GUIDE.md](USAGE_GUIDE.md)** ← **DETAILED EXAMPLES!**
   - Complete syntax reference for all scripts
   - Step-by-step examples for every feature
   - Real-world use cases
   - Configuration examples

3. **[QUICKSTART.md](QUICKSTART.md)**
   - Quick command reference
   - Common use cases
   - Model size recommendations

4. **[README.md](README.md)**
   - Full documentation
   - All features and options
   - Advanced usage

5. **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)**
   - Technical architecture
   - Implementation details
   - Extension guide

## 📋 Quick Reference

### Essential Commands

```bash
# Verify setup
python test_setup.py

# Inspect data
python prepare_data.py --show_stats

# Train (choose based on your GPU)
python train.py --model_size small --vram 8gb    # 8GB GPU
python train.py --model_size medium --vram 16gb  # 16GB GPU
python train.py --model_size large --vram 24gb   # 24GB+ GPU

# Generate text
python generate.py --checkpoint checkpoints/final --prompt "Hello"
```

### Files by Purpose

#### Want to...

**Train a model?**
→ `train.py` - Main training script
→ See: GETTING_STARTED.md, README.md

**Generate text?**
→ `generate.py` - Text generation
→ See: GETTING_STARTED.md

**Check your data?**
→ `prepare_data.py` - Data inspection
→ See: README.md section "Preparing Your Data"

**Verify installation?**
→ `test_setup.py` - Setup verification
→ See: GETTING_STARTED.md

**Configure training?**
→ `configs/model_configs.py` - All presets
→ `configs/example_8gb.json` - 8GB example
→ `configs/example_24gb.json` - 24GB example
→ See: README.md section "Configuration Options"

**Understand the model?**
→ `models/mamba_model.py` - Model implementation
→ See: PROJECT_OVERVIEW.md

**Load custom data?**
→ `utils/data_loader.py` - Data loading
→ See: README.md section "Custom Data"

## 📁 File Reference

### Scripts (Python)

| File | Lines | Description |
|------|-------|-------------|
| `train.py` | 389 | Main training script with memory optimization |
| `generate.py` | 78 | Text generation with trained models |
| `prepare_data.py` | 125 | Data inspection and statistics |
| `test_setup.py` | 188 | Setup verification tests |
| `models/mamba_model.py` | 281 | Mamba architecture implementation |
| `configs/model_configs.py` | 188 | Configuration presets |
| `utils/data_loader.py` | 207 | Multi-format data loading |

### Documentation (Markdown)

| File | Pages | Purpose | Best For |
|------|-------|---------|----------|
| `GETTING_STARTED.md` | ~10 | Complete beginner guide | First-time users |
| `USAGE_GUIDE.md` | ~25 | Detailed syntax & examples | Learning all features |
| `README.md` | ~15 | Full documentation | Reference |
| `QUICKSTART.md` | ~3 | Quick reference | Frequent users |
| `PROJECT_OVERVIEW.md` | ~12 | Technical details | Developers |
| `INDEX.md` | ~3 | This file | Navigation |

### Configuration Files

| File | Purpose |
|------|---------|
| `configs/example_8gb.json` | Example config for 8GB VRAM |
| `configs/example_24gb.json` | Example config for 24GB VRAM |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Git ignore rules |

### Setup

| File | Purpose |
|------|---------|
| `setup.sh` | Automated setup script |

## 🎯 Use Case Guide

### I want to...

#### Train on my own documents
1. Read: GETTING_STARTED.md → "Training on Your Own Data"
2. Add files to `data/` directory
3. Run: `python prepare_data.py --show_stats`
4. Run: `python train.py --model_size medium --vram 16gb`

#### Train on a HuggingFace dataset
1. Read: README.md → "Using HuggingFace Datasets"
2. Run: `python train.py --dataset wikitext --vram 16gb`

#### Train with limited GPU memory (8GB)
1. Read: QUICKSTART.md → "Model Size Recommendations"
2. Read: README.md → "Memory Optimization Tips"
3. Run: `python train.py --model_size small --vram 8gb`

#### Generate better quality text
1. Train with larger model: `--model_size large`
2. Train longer: `--training_preset careful`
3. Use more training data
4. See: README.md → "Training Presets"

#### Resume interrupted training
1. Run: `python train.py --resume checkpoints/step_50000`
2. See: README.md → "Resuming Training"

#### Use custom configuration
1. Copy: `configs/example_24gb.json` to `my_config.json`
2. Edit: `my_config.json` with your settings
3. Run: `python train.py --config my_config.json`
4. See: README.md → "Custom Configuration"

#### Understand how Mamba works
1. Read: PROJECT_OVERVIEW.md → "Mamba Architecture"
2. Read: `models/mamba_model.py` source code
3. See: Original paper (cited in README.md)

#### Extend the code
1. Read: PROJECT_OVERVIEW.md → "Extension Points"
2. See: `models/mamba_model.py` for model changes
3. See: `utils/data_loader.py` for data format changes
4. See: `train.py` for training loop changes

## 🔍 Feature Index

### Model Sizes
- **Small** (~50M params): `--model_size small` → README.md, model_configs.py:8
- **Medium** (~150M params): `--model_size medium` → README.md, model_configs.py:18
- **Large** (~400M params): `--model_size large` → README.md, model_configs.py:28

### VRAM Configurations
- **8GB**: `--vram 8gb` → README.md, model_configs.py:42
- **12GB**: `--vram 12gb` → README.md, model_configs.py:52
- **16GB**: `--vram 16gb` → README.md, model_configs.py:62
- **24GB**: `--vram 24gb` → README.md, model_configs.py:72
- **32GB**: `--vram 32gb` → README.md, model_configs.py:82
- **48GB**: `--vram 48gb` → README.md, model_configs.py:92

### Supported File Formats
- **PDF**: data_loader.py:47 (load_pdf)
- **Text**: data_loader.py:60 (load_text_file)
- **JSON**: data_loader.py:67 (load_json)
- **Code**: data_loader.py:15 (SUPPORTED_CODE_EXTENSIONS)
- **Config**: data_loader.py:27 (SUPPORTED_CONFIG_EXTENSIONS)
- **Datasets**: data_loader.py:109 (load_huggingface_dataset)

### Memory Optimizations
- **FP16 Training**: train.py:117 (use_fp16)
- **Gradient Accumulation**: train.py:152 (gradient_accumulation_steps)
- **Gradient Checkpointing**: train.py:120 (use_gradient_checkpointing)
- **Streaming Loading**: data_loader.py:162 (StreamingDataLoader)

### Generation Options
- **Temperature Sampling**: generate.py:47, mamba_model.py:209
- **Top-k Filtering**: generate.py:48, mamba_model.py:219
- **Max Length Control**: generate.py:46

## 📊 Statistics

- **Total Python Code**: ~1,613 lines
- **Total Documentation**: ~1,145 lines
- **Total Files**: 20
- **Model Implementations**: 3 sizes (small, medium, large)
- **VRAM Configurations**: 6 presets (8GB to 48GB)
- **Supported File Types**: 20+ extensions
- **Documentation Files**: 5 comprehensive guides

## 🆘 Troubleshooting Index

| Problem | Solution | See |
|---------|----------|-----|
| Out of memory | Use smaller model/VRAM config | GETTING_STARTED.md → Troubleshooting |
| Slow training | Use fast preset, check GPU | GETTING_STARTED.md → Troubleshooting |
| No data found | Check data/ directory | GETTING_STARTED.md → Troubleshooting |
| Import errors | Reinstall requirements | GETTING_STARTED.md → Troubleshooting |
| Can't find GPU | Check CUDA installation | test_setup.py |

## 🔗 Quick Links

- **First Time Setup**: GETTING_STARTED.md
- **Detailed Examples & Syntax**: USAGE_GUIDE.md
- **Command Reference**: QUICKSTART.md
- **Full Documentation**: README.md
- **Technical Details**: PROJECT_OVERVIEW.md
- **Model Code**: models/mamba_model.py
- **Config System**: configs/model_configs.py
- **Data Loading**: utils/data_loader.py

## 📝 Version Info

- **Architecture**: Mamba (Selective State Space Models)
- **Based on**: "Mamba: Linear-Time Sequence Modeling" (Gu & Dao, 2023)
- **Framework**: PyTorch 2.0+
- **Python**: 3.8+

## 🎓 Learning Path

1. **Beginner** → GETTING_STARTED.md (setup & first run)
2. **Learning** → USAGE_GUIDE.md (detailed examples for all features)
3. **User** → QUICKSTART.md (quick reference), README.md (full docs)
4. **Developer** → PROJECT_OVERVIEW.md, source code
5. **Researcher** → mamba_model.py, original paper

---

**Need help?** Start with test_setup.py to verify installation, then read GETTING_STARTED.md!
