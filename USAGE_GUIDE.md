# Complete Usage Guide - Mamba Trainer

This guide provides detailed examples and syntax for every feature of the Mamba Trainer.

## Table of Contents

1. [Script Syntax Reference](#script-syntax-reference)
2. [Training on Your Documents](#training-on-your-documents)
3. [Training on Datasets](#training-on-datasets)
4. [GPU Optimization](#gpu-optimization)
5. [Text Generation](#text-generation)
6. [Custom Configuration](#custom-configuration)
7. [Advanced Usage](#advanced-usage)
8. [Complete Examples](#complete-examples)

---

## Script Syntax Reference

### train.py - Main Training Script

```bash
python train.py [OPTIONS]

Required: None (uses defaults)
Optional: All arguments below

Model Configuration:
  --model_size {small,medium,large}
      Model size preset
      Default: medium
      Example: --model_size large

  --vram {8gb,12gb,16gb,24gb,32gb,48gb}
      VRAM optimization configuration
      Default: 16gb
      Example: --vram 24gb

  --training_preset {default,fast,careful}
      Training hyperparameter preset
      Default: default
      Example: --training_preset fast

Data Sources:
  --data_dir PATH
      Directory containing training files
      Default: data
      Example: --data_dir /path/to/my/documents

  --dataset NAME
      HuggingFace dataset name
      Default: None (no dataset)
      Example: --dataset wikitext

  --dataset_split SPLIT
      Dataset split to use
      Default: train
      Example: --dataset_split validation

  --text_column COLUMN
      Column name for text in dataset
      Default: text
      Example: --text_column content

Output:
  --output_dir PATH
      Directory for saving checkpoints
      Default: checkpoints
      Example: --output_dir /path/to/output

Advanced:
  --config PATH
      Path to custom JSON configuration
      Default: None (use presets)
      Example: --config my_config.json

  --resume PATH
      Resume training from checkpoint
      Default: None (start fresh)
      Example: --resume checkpoints/step_50000
```

### generate.py - Text Generation

```bash
python generate.py [OPTIONS]

Required:
  --checkpoint PATH
      Path to checkpoint directory
      Example: --checkpoint checkpoints/final

Optional:
  --prompt TEXT
      Text prompt for generation
      Default: "Once upon a time"
      Example: --prompt "The future of AI"

  --max_tokens INT
      Maximum tokens to generate
      Default: 100
      Range: 1-2048
      Example: --max_tokens 500

  --temperature FLOAT
      Sampling temperature (creativity)
      Default: 1.0
      Range: 0.1-2.0 (lower=focused, higher=creative)
      Example: --temperature 0.7

  --top_k INT
      Top-k sampling parameter
      Default: 50
      Range: 1-1000
      Example: --top_k 100

  --device {cuda,cpu}
      Device to use for generation
      Default: cuda
      Example: --device cpu
```

### prepare_data.py - Data Inspection

```bash
python prepare_data.py [OPTIONS]

Optional:
  --data_dir PATH
      Directory containing training files
      Default: data
      Example: --data_dir /path/to/documents

  --max_length INT
      Maximum sequence length
      Default: 2048
      Example: --max_length 1024

  --show_stats
      Show detailed statistics
      Default: False (off)
      Example: --show_stats

  --show_samples INT
      Number of sample chunks to display
      Default: 0 (none)
      Example: --show_samples 5
```

### test_setup.py - Setup Verification

```bash
python test_setup.py

No arguments required.
Runs comprehensive tests:
- Package imports
- Project structure
- Model creation
- Configuration system
- Data loading
- CUDA availability
```

---

## Training on Your Documents

### Supported File Formats

| Category | Extensions | Details |
|----------|-----------|---------|
| **Documents** | `.pdf`, `.txt`, `.md`, `.markdown`, `.rst`, `.tex`, `.log` | Automatically extracted |
| **Python** | `.py` | Source code training |
| **JavaScript** | `.js`, `.jsx`, `.ts`, `.tsx` | All JS variants |
| **Web** | `.html`, `.css`, `.vue`, `.svelte` | Web technologies |
| **Systems** | `.c`, `.cpp`, `.h`, `.go`, `.rs`, `.swift` | C, C++, Go, Rust, Swift |
| **JVM** | `.java`, `.kt`, `.scala` | Java, Kotlin, Scala |
| **Other Code** | `.rb`, `.php`, `.sh`, `.bash`, `.r`, `.sql` | Ruby, PHP, Shell, R, SQL |
| **Config** | `.json`, `.yaml`, `.yml`, `.toml`, `.ini`, `.cfg`, `.xml` | Configuration files |

### Step-by-Step: Training on Your Documents

#### Step 1: Organize Your Files

```bash
# Create organized structure (optional but recommended)
mkdir -p data/documents
mkdir -p data/code
mkdir -p data/configs

# Add your files
cp ~/my_pdfs/*.pdf data/documents/
cp ~/my_projects/**/*.py data/code/
cp ~/my_configs/*.json data/configs/
```

**Pro Tip**: You can just dump everything into `data/` - the loader will find all supported files recursively.

#### Step 2: Inspect Your Data

```bash
# Basic inspection
python prepare_data.py

# Detailed statistics
python prepare_data.py --show_stats

# See sample chunks
python prepare_data.py --show_stats --show_samples 3
```

**Example Output**:
```
Loading data from: data

Loaded: data/documents/manual.pdf (45234 chars)
Loaded: data/code/main.py (3421 chars)
Loaded: data/configs/config.json (892 chars)

Found 3 documents

Prepared 45 training chunks from 3 documents

DETAILED STATISTICS
============================================================

Files by type:
  .pdf      :    1 files
  .py       :    1 files
  .json     :    1 files

Document statistics:
  Total documents: 3
  Total characters: 49,547
  Avg chars/document: 16,516

Chunk statistics:
  Total chunks: 45
  Total characters: 49,547
  Avg chars/chunk: 1,101
  Estimated tokens: 12,387
```

#### Step 3: Train on Your Files

```bash
# Basic training (16GB GPU)
python train.py --data_dir data --model_size medium --vram 16gb

# For smaller GPU (8GB)
python train.py --data_dir data --model_size small --vram 8gb

# For powerful GPU (24GB+)
python train.py --data_dir data --model_size large --vram 24gb

# Fast training for experiments
python train.py --data_dir data --training_preset fast --vram 16gb

# Careful training for best quality
python train.py --data_dir data --training_preset careful --vram 24gb
```

### Real-World Examples

#### Example 1: Train on Python Codebase

```bash
# Copy your Python project
cp -r ~/my_project/src/*.py data/

# Inspect
python prepare_data.py --show_stats

# Train
python train.py \
  --data_dir data \
  --model_size medium \
  --vram 16gb \
  --output_dir checkpoints/python_model
```

#### Example 2: Train on PDF Documentation

```bash
# Add PDFs
cp ~/manuals/*.pdf data/

# Check if PyPDF2 is installed
python -c "import PyPDF2; print('PDF support ready!')"

# If not installed:
pip install PyPDF2

# Train
python train.py \
  --data_dir data \
  --model_size small \
  --vram 12gb
```

#### Example 3: Train on Mixed Content

```bash
# Create organized structure
mkdir -p data/books
mkdir -p data/articles
mkdir -p data/code
mkdir -p data/docs

# Add diverse content
cp ~/books/*.txt data/books/
cp ~/articles/*.md data/articles/
cp ~/projects/**/*.py data/code/
cp ~/documentation/*.pdf data/docs/

# Verify all files are found
python prepare_data.py --show_stats

# Train with larger model for diverse content
python train.py \
  --data_dir data \
  --model_size large \
  --vram 24gb \
  --training_preset careful
```

#### Example 4: Train on Specific Subdirectory

```bash
# Train only on code subdirectory
python train.py \
  --data_dir data/code \
  --model_size medium \
  --vram 16gb \
  --output_dir checkpoints/code_model

# Train only on documents
python train.py \
  --data_dir data/documents \
  --model_size medium \
  --vram 16gb \
  --output_dir checkpoints/doc_model
```

---

## Training on Datasets

### Using HuggingFace Datasets

The trainer can use any text dataset from [HuggingFace Hub](https://huggingface.co/datasets).

#### Syntax

```bash
python train.py \
  --dataset DATASET_NAME \
  [--dataset_split SPLIT] \
  [--text_column COLUMN] \
  [other options...]
```

### Popular Datasets

#### Example 1: WikiText-2

```bash
# Small dataset, good for testing
python train.py \
  --dataset wikitext \
  --dataset_split train \
  --text_column text \
  --model_size medium \
  --vram 16gb
```

**Dataset Info**:
- Size: ~100MB
- Type: Wikipedia articles
- Best for: General language modeling

#### Example 2: OpenWebText

```bash
# Large dataset, high quality
python train.py \
  --dataset openwebtext \
  --model_size large \
  --vram 24gb \
  --training_preset careful
```

**Dataset Info**:
- Size: ~40GB
- Type: Web content
- Best for: High-quality general models

#### Example 3: TinyStories

```bash
# Small, story-focused dataset
python train.py \
  --dataset roneneldan/TinyStories \
  --model_size small \
  --vram 8gb
```

**Dataset Info**:
- Size: ~2GB
- Type: Short stories
- Best for: Creative writing models

#### Example 4: Code Datasets

```bash
# Python code dataset
python train.py \
  --dataset codeparrot/github-code \
  --text_column code \
  --model_size medium \
  --vram 16gb
```

#### Example 5: Books Dataset

```bash
# BookCorpus
python train.py \
  --dataset bookcorpus \
  --model_size large \
  --vram 32gb
```

### Dataset Splits

Most datasets have multiple splits:

```bash
# Train on training split (default)
python train.py --dataset wikitext --dataset_split train

# Use validation split for testing
python train.py --dataset wikitext --dataset_split validation

# Use test split
python train.py --dataset wikitext --dataset_split test

# Some datasets use different names
python train.py --dataset dataset_name --dataset_split train[:10%]  # First 10%
```

### Custom Text Column

If the dataset uses a different column name for text:

```bash
# Default column is 'text'
python train.py --dataset wikitext --text_column text

# Some datasets use 'content'
python train.py --dataset my_dataset --text_column content

# Or 'article', 'story', etc.
python train.py --dataset news --text_column article
```

### Combining Local Files and Datasets

```bash
# Train on both local files AND HuggingFace dataset
python train.py \
  --data_dir data \
  --dataset wikitext \
  --model_size medium \
  --vram 16gb
```

**How it works**:
1. Loads all files from `data/`
2. Loads dataset from HuggingFace
3. Combines both into training set
4. Shuffles and trains on everything

#### Example: Code + Documentation

```bash
# Local code files + online documentation
cp ~/my_code/*.py data/

python train.py \
  --data_dir data \
  --dataset wikipedia \
  --dataset_split train \
  --model_size large \
  --vram 24gb
```

---

## GPU Optimization

### VRAM Configuration Guide

Each VRAM preset optimizes batch size, gradient accumulation, and precision for your GPU.

#### 8GB VRAM Configuration

**Target GPUs**: GTX 1080, RTX 2060, RTX 2070, RTX 3050

```bash
python train.py \
  --model_size small \
  --vram 8gb \
  --data_dir data
```

**Optimizations Applied**:
- Batch Size: 1
- Gradient Accumulation: 16 steps
- FP16: Enabled
- Gradient Checkpointing: Enabled
- Max Sequence Length: 512
- Effective Batch Size: 16

**Training Speed**: Slower due to gradient accumulation
**Recommended Models**: Small only

#### 12GB VRAM Configuration

**Target GPUs**: RTX 3060, RTX 2080 Ti, RTX 3060 Ti

```bash
python train.py \
  --model_size medium \
  --vram 12gb \
  --data_dir data
```

**Optimizations Applied**:
- Batch Size: 2
- Gradient Accumulation: 8 steps
- FP16: Enabled
- Gradient Checkpointing: Enabled
- Max Sequence Length: 1024
- Effective Batch Size: 16

**Training Speed**: Moderate
**Recommended Models**: Small, Medium

#### 16GB VRAM Configuration

**Target GPUs**: RTX 4060 Ti, RTX 3080, RTX 4070

```bash
python train.py \
  --model_size medium \
  --vram 16gb \
  --data_dir data
```

**Optimizations Applied**:
- Batch Size: 4
- Gradient Accumulation: 4 steps
- FP16: Enabled
- Gradient Checkpointing: Enabled
- Max Sequence Length: 1024
- Effective Batch Size: 16

**Training Speed**: Good
**Recommended Models**: Small, Medium

#### 24GB VRAM Configuration

**Target GPUs**: RTX 3090, RTX 4090, RTX A5000

```bash
python train.py \
  --model_size large \
  --vram 24gb \
  --data_dir data
```

**Optimizations Applied**:
- Batch Size: 8
- Gradient Accumulation: 2 steps
- FP16: Enabled
- Gradient Checkpointing: Disabled (not needed)
- Max Sequence Length: 2048
- Effective Batch Size: 16

**Training Speed**: Fast
**Recommended Models**: All (Small, Medium, Large)

#### 32GB VRAM Configuration

**Target GPUs**: V100, RTX A5500, A6000 Ada

```bash
python train.py \
  --model_size large \
  --vram 32gb \
  --data_dir data
```

**Optimizations Applied**:
- Batch Size: 12
- Gradient Accumulation: 2 steps
- FP16: Enabled
- Gradient Checkpointing: Disabled
- Max Sequence Length: 2048
- Effective Batch Size: 24

**Training Speed**: Very Fast
**Recommended Models**: All

#### 48GB VRAM Configuration

**Target GPUs**: A40, A6000, RTX 6000 Ada

```bash
python train.py \
  --model_size large \
  --vram 48gb \
  --data_dir data
```

**Optimizations Applied**:
- Batch Size: 16
- Gradient Accumulation: 1 step
- FP16: Disabled (can use FP32 for max precision)
- Gradient Checkpointing: Disabled
- Max Sequence Length: 2048
- Effective Batch Size: 16

**Training Speed**: Maximum
**Recommended Models**: All

### Model Size Guide

#### Small Model (~50M parameters)

**Configuration**:
- d_model: 512
- n_layers: 8
- d_state: 16
- Parameters: ~50M

**When to Use**:
- 8GB VRAM GPUs
- Quick experiments
- Limited training data
- Fast iteration

**Example**:
```bash
python train.py \
  --model_size small \
  --vram 8gb \
  --training_preset fast
```

#### Medium Model (~150M parameters)

**Configuration**:
- d_model: 768
- n_layers: 12
- d_state: 16
- Parameters: ~150M

**When to Use**:
- 12GB-24GB VRAM
- General-purpose models
- Balanced quality/speed
- Standard use case

**Example**:
```bash
python train.py \
  --model_size medium \
  --vram 16gb \
  --training_preset default
```

#### Large Model (~400M parameters)

**Configuration**:
- d_model: 1024
- n_layers: 24
- d_state: 16
- Parameters: ~400M

**When to Use**:
- 24GB+ VRAM
- Best quality needed
- Large training datasets
- Production models

**Example**:
```bash
python train.py \
  --model_size large \
  --vram 24gb \
  --training_preset careful
```

### Optimization Examples

#### Maximum Speed

```bash
python train.py \
  --model_size small \
  --vram 16gb \
  --training_preset fast
```

#### Maximum Quality

```bash
python train.py \
  --model_size large \
  --vram 48gb \
  --training_preset careful
```

#### Balanced

```bash
python train.py \
  --model_size medium \
  --vram 24gb \
  --training_preset default
```

---

## Text Generation

### Basic Generation

```bash
python generate.py \
  --checkpoint checkpoints/final \
  --prompt "Your prompt here"
```

### Generation Parameters

#### Temperature (Creativity Control)

Temperature controls randomness in generation:

**Low Temperature (0.1-0.5)**: Focused, deterministic
```bash
python generate.py \
  --checkpoint checkpoints/final \
  --prompt "The capital of France is" \
  --temperature 0.3
```
**Output**: "The capital of France is Paris."

**Medium Temperature (0.7-1.0)**: Balanced
```bash
python generate.py \
  --checkpoint checkpoints/final \
  --prompt "Once upon a time" \
  --temperature 0.8
```
**Output**: Creative but coherent

**High Temperature (1.5-2.0)**: Very creative, random
```bash
python generate.py \
  --checkpoint checkpoints/final \
  --prompt "The future of AI" \
  --temperature 1.8
```
**Output**: Highly creative, potentially chaotic

#### Top-K Sampling

Limits vocabulary to top K most likely tokens:

**Low Top-K (10-30)**: Very focused
```bash
python generate.py \
  --checkpoint checkpoints/final \
  --prompt "Python programming" \
  --top_k 10
```

**Medium Top-K (50-100)**: Balanced
```bash
python generate.py \
  --checkpoint checkpoints/final \
  --prompt "Write a story" \
  --top_k 50
```

**High Top-K (200+)**: More variety
```bash
python generate.py \
  --checkpoint checkpoints/final \
  --prompt "Create something unique" \
  --top_k 200
```

#### Max Tokens

Control generation length:

**Short (10-50 tokens)**:
```bash
python generate.py \
  --checkpoint checkpoints/final \
  --prompt "Complete this sentence:" \
  --max_tokens 20
```

**Medium (100-200 tokens)**:
```bash
python generate.py \
  --checkpoint checkpoints/final \
  --prompt "Write a paragraph about" \
  --max_tokens 150
```

**Long (500+ tokens)**:
```bash
python generate.py \
  --checkpoint checkpoints/final \
  --prompt "Write a detailed story" \
  --max_tokens 500
```

### Generation Examples by Use Case

#### Code Completion

```bash
python generate.py \
  --checkpoint checkpoints/code_model \
  --prompt "def calculate_fibonacci(n):" \
  --temperature 0.2 \
  --top_k 20 \
  --max_tokens 100
```

#### Creative Writing

```bash
python generate.py \
  --checkpoint checkpoints/final \
  --prompt "In a world where dreams become reality," \
  --temperature 1.2 \
  --top_k 100 \
  --max_tokens 300
```

#### Technical Documentation

```bash
python generate.py \
  --checkpoint checkpoints/doc_model \
  --prompt "## Installation Guide\n\nTo install this software," \
  --temperature 0.4 \
  --top_k 30 \
  --max_tokens 200
```

#### Dialogue Generation

```bash
python generate.py \
  --checkpoint checkpoints/final \
  --prompt "Alice: How are you today?\nBob:" \
  --temperature 0.9 \
  --top_k 60 \
  --max_tokens 150
```

### Using Different Checkpoints

```bash
# Use latest checkpoint
python generate.py --checkpoint checkpoints/final

# Use specific step
python generate.py --checkpoint checkpoints/step_50000

# Use earlier checkpoint for comparison
python generate.py --checkpoint checkpoints/step_10000
```

### CPU Generation

If no GPU available:

```bash
python generate.py \
  --checkpoint checkpoints/final \
  --prompt "Hello world" \
  --device cpu
```

---

## Custom Configuration

### Configuration File Structure

JSON configuration files have three sections:

```json
{
  "model": { /* Model architecture */ },
  "vram": { /* Memory optimization */ },
  "training": { /* Training hyperparameters */ }
}
```

### Creating Custom Configurations

#### Method 1: Modify Example

```bash
# Copy example
cp configs/example_24gb.json my_config.json

# Edit my_config.json
nano my_config.json  # or your favorite editor

# Use it
python train.py --config my_config.json
```

#### Method 2: Start from Scratch

Create `my_config.json`:

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
    "max_seq_len": 2048,
    "use_fp16": true,
    "use_gradient_checkpointing": true,
    "use_cpu_offload": false
  },
  "training": {
    "learning_rate": 0.0003,
    "weight_decay": 0.1,
    "max_steps": 100000,
    "warmup_steps": 2000,
    "eval_interval": 1000,
    "save_interval": 5000,
    "logging_steps": 10,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "adam_epsilon": 1e-08,
    "max_grad_norm": 1.0,
    "label_smoothing": 0.0
  }
}
```

### Configuration Parameters Explained

#### Model Section

```json
"model": {
  "d_model": 768,           // Model dimension (512, 768, 1024)
  "n_layers": 12,           // Number of Mamba blocks (8, 12, 24)
  "d_state": 16,            // SSM state dimension (usually 16)
  "expand_factor": 2,       // Inner dimension multiplier (usually 2)
  "dropout": 0.1,           // Dropout rate (0.0-0.2)
  "max_seq_len": 2048,      // Maximum sequence length
  "vocab_size": 50257       // Vocabulary size (GPT-2 default)
}
```

#### VRAM Section

```json
"vram": {
  "batch_size": 4,                      // Batch size per step
  "gradient_accumulation_steps": 4,     // Steps before optimizer update
  "max_seq_len": 2048,                  // Override model max_seq_len if needed
  "use_fp16": true,                     // Enable mixed precision
  "use_gradient_checkpointing": true,   // Trade compute for memory
  "use_cpu_offload": false              // Offload to CPU (slow)
}
```

#### Training Section

```json
"training": {
  "learning_rate": 0.0003,    // Learning rate (1e-5 to 5e-4)
  "weight_decay": 0.1,        // Weight decay for regularization
  "max_steps": 100000,        // Total training steps
  "warmup_steps": 2000,       // Warmup period
  "eval_interval": 1000,      // Steps between evaluations
  "save_interval": 5000,      // Steps between checkpoints
  "logging_steps": 10,        // Steps between log outputs
  "adam_beta1": 0.9,          // Adam beta1 parameter
  "adam_beta2": 0.95,         // Adam beta2 parameter
  "adam_epsilon": 1e-08,      // Adam epsilon
  "max_grad_norm": 1.0,       // Gradient clipping threshold
  "label_smoothing": 0.0      // Label smoothing (0.0-0.1)
}
```

### Custom Configuration Examples

#### Example 1: Fast Experimentation

```json
{
  "model": {
    "d_model": 512,
    "n_layers": 6,
    "d_state": 16,
    "expand_factor": 2,
    "dropout": 0.1,
    "max_seq_len": 512,
    "vocab_size": 50257
  },
  "vram": {
    "batch_size": 8,
    "gradient_accumulation_steps": 2,
    "use_fp16": true,
    "use_gradient_checkpointing": false
  },
  "training": {
    "learning_rate": 0.0005,
    "max_steps": 10000,
    "warmup_steps": 500,
    "eval_interval": 500,
    "save_interval": 2000,
    "logging_steps": 10
  }
}
```

**Use**:
```bash
python train.py --config fast_experiment.json
```

#### Example 2: High-Quality Production Model

```json
{
  "model": {
    "d_model": 1024,
    "n_layers": 24,
    "d_state": 16,
    "expand_factor": 2,
    "dropout": 0.0,
    "max_seq_len": 2048,
    "vocab_size": 50257
  },
  "vram": {
    "batch_size": 16,
    "gradient_accumulation_steps": 1,
    "use_fp16": false,
    "use_gradient_checkpointing": false
  },
  "training": {
    "learning_rate": 0.0001,
    "weight_decay": 0.1,
    "max_steps": 500000,
    "warmup_steps": 10000,
    "eval_interval": 5000,
    "save_interval": 10000,
    "logging_steps": 100
  }
}
```

**Use**:
```bash
python train.py --config production.json --data_dir large_dataset
```

#### Example 3: Code-Specific Model

```json
{
  "model": {
    "d_model": 768,
    "n_layers": 16,
    "d_state": 16,
    "expand_factor": 2,
    "dropout": 0.05,
    "max_seq_len": 1024,
    "vocab_size": 50257
  },
  "vram": {
    "batch_size": 6,
    "gradient_accumulation_steps": 3,
    "use_fp16": true,
    "use_gradient_checkpointing": true
  },
  "training": {
    "learning_rate": 0.0002,
    "weight_decay": 0.05,
    "max_steps": 200000,
    "warmup_steps": 4000,
    "eval_interval": 2000,
    "save_interval": 10000,
    "logging_steps": 50
  }
}
```

**Use**:
```bash
python train.py --config code_model.json --data_dir code_data
```

### Saving Current Configuration

After running with presets, save the configuration:

```python
from configs.model_configs import get_config, save_config

config = get_config('medium', '16gb', 'default')
save_config(config, 'my_saved_config.json')
```

---

## Advanced Usage

### Resuming Training

Resume from any checkpoint:

```bash
# Resume from specific step
python train.py --resume checkpoints/step_50000

# Resume with different data
python train.py \
  --resume checkpoints/step_50000 \
  --data_dir new_data

# Resume with modified config (not recommended)
python train.py \
  --resume checkpoints/step_50000 \
  --config modified_config.json
```

### Training Multiple Models

Train different models in parallel:

```bash
# Terminal 1: Small model
python train.py \
  --model_size small \
  --vram 8gb \
  --output_dir checkpoints/small_model

# Terminal 2: Medium model
python train.py \
  --model_size medium \
  --vram 16gb \
  --output_dir checkpoints/medium_model
```

### Monitoring Training

Watch training progress:

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor checkpoints
watch -n 10 ls -lh checkpoints/

# Follow training logs (if logging to file)
tail -f training.log
```

### Data Preprocessing

Pre-inspect and validate data:

```bash
# Check file counts
python prepare_data.py --show_stats | grep "files"

# Verify all formats are supported
python prepare_data.py --show_stats

# Sample first chunks
python prepare_data.py --show_samples 10 > samples.txt
```

### Batch Processing

Train on multiple datasets sequentially:

```bash
#!/bin/bash

# Script: train_all.sh

datasets=("wikitext" "openwebtext" "bookcorpus")

for dataset in "${datasets[@]}"; do
    echo "Training on $dataset"
    python train.py \
        --dataset "$dataset" \
        --model_size medium \
        --vram 24gb \
        --output_dir "checkpoints/${dataset}_model"
done
```

---

## Complete Examples

### Example 1: Train on Your Code Repository

```bash
# Step 1: Clone or copy your repository
git clone https://github.com/yourusername/your-repo.git
cp -r your-repo/**/*.py data/code/
cp -r your-repo/**/*.js data/code/

# Step 2: Add documentation
cp -r your-repo/docs/*.md data/docs/
cp your-repo/README.md data/docs/

# Step 3: Inspect
python prepare_data.py --show_stats --show_samples 2

# Step 4: Train
python train.py \
  --data_dir data \
  --model_size medium \
  --vram 16gb \
  --training_preset default \
  --output_dir checkpoints/code_model

# Step 5: Generate code
python generate.py \
  --checkpoint checkpoints/code_model/final \
  --prompt "def calculate_" \
  --temperature 0.2 \
  --max_tokens 100
```

### Example 2: Train on Research Papers (PDFs)

```bash
# Step 1: Collect PDFs
mkdir -p data/papers
cp ~/Downloads/papers/*.pdf data/papers/

# Step 2: Install PDF support
pip install PyPDF2

# Step 3: Verify PDF loading
python prepare_data.py --show_stats

# Step 4: Train with larger model (papers are complex)
python train.py \
  --data_dir data \
  --model_size large \
  --vram 24gb \
  --training_preset careful \
  --output_dir checkpoints/research_model

# Step 5: Generate technical content
python generate.py \
  --checkpoint checkpoints/research_model/final \
  --prompt "Abstract: This paper presents" \
  --temperature 0.7 \
  --max_tokens 200
```

### Example 3: Train on Mixed Sources

```bash
# Step 1: Organize data
mkdir -p data/{local,remote}

# Local files
cp ~/documents/*.txt data/local/
cp ~/code/**/*.py data/local/

# Step 2: Train on local + HuggingFace
python train.py \
  --data_dir data/local \
  --dataset wikitext \
  --model_size medium \
  --vram 16gb \
  --output_dir checkpoints/mixed_model

# Step 3: Generate
python generate.py \
  --checkpoint checkpoints/mixed_model/final \
  --prompt "In conclusion," \
  --temperature 0.8
```

### Example 4: Full Production Pipeline

```bash
#!/bin/bash
# production_train.sh

# Setup
echo "Setting up environment..."
python test_setup.py || exit 1

# Prepare data
echo "Preparing data..."
mkdir -p data/production
cp -r ~/training_data/* data/production/

# Validate
echo "Validating data..."
python prepare_data.py \
  --data_dir data/production \
  --show_stats > data_report.txt

# Train
echo "Starting training..."
python train.py \
  --data_dir data/production \
  --model_size large \
  --vram 48gb \
  --training_preset careful \
  --output_dir checkpoints/production_v1 \
  2>&1 | tee training.log

# Test generation
echo "Testing generation..."
python generate.py \
  --checkpoint checkpoints/production_v1/final \
  --prompt "Test prompt" \
  --max_tokens 50 > test_output.txt

echo "Training complete!"
```

### Example 5: Experiment with Different Configs

```bash
# Create experiment directory
mkdir -p experiments

# Experiment 1: High learning rate
cat > experiments/high_lr.json << 'EOF'
{
  "model": {"d_model": 512, "n_layers": 8, "d_state": 16, "expand_factor": 2, "dropout": 0.1, "max_seq_len": 1024, "vocab_size": 50257},
  "vram": {"batch_size": 4, "gradient_accumulation_steps": 4, "use_fp16": true, "use_gradient_checkpointing": true},
  "training": {"learning_rate": 0.001, "max_steps": 50000, "warmup_steps": 1000, "eval_interval": 1000, "save_interval": 5000, "logging_steps": 10}
}
EOF

# Experiment 2: Low learning rate
cat > experiments/low_lr.json << 'EOF'
{
  "model": {"d_model": 512, "n_layers": 8, "d_state": 16, "expand_factor": 2, "dropout": 0.1, "max_seq_len": 1024, "vocab_size": 50257},
  "vram": {"batch_size": 4, "gradient_accumulation_steps": 4, "use_fp16": true, "use_gradient_checkpointing": true},
  "training": {"learning_rate": 0.00005, "max_steps": 50000, "warmup_steps": 1000, "eval_interval": 1000, "save_interval": 5000, "logging_steps": 10}
}
EOF

# Run experiments
python train.py --config experiments/high_lr.json --output_dir checkpoints/exp_high_lr
python train.py --config experiments/low_lr.json --output_dir checkpoints/exp_low_lr

# Compare results
python generate.py --checkpoint checkpoints/exp_high_lr/final --prompt "Test"
python generate.py --checkpoint checkpoints/exp_low_lr/final --prompt "Test"
```

---

## Quick Command Reference

### Most Common Commands

```bash
# Setup and test
python test_setup.py

# Inspect data
python prepare_data.py --show_stats

# Train (8GB GPU)
python train.py --model_size small --vram 8gb

# Train (16GB GPU)
python train.py --model_size medium --vram 16gb

# Train (24GB+ GPU)
python train.py --model_size large --vram 24gb

# Train on dataset
python train.py --dataset wikitext --vram 16gb

# Resume training
python train.py --resume checkpoints/step_50000

# Generate text
python generate.py --checkpoint checkpoints/final --prompt "Hello"

# Generate creative text
python generate.py --checkpoint checkpoints/final --prompt "Story:" --temperature 1.2 --max_tokens 300
```

---

## Tips and Best Practices

1. **Always verify setup first**: Run `python test_setup.py`
2. **Inspect data before training**: Use `prepare_data.py --show_stats`
3. **Start small**: Use `small` model and `fast` preset for experiments
4. **Match VRAM to GPU**: Check your GPU memory with `nvidia-smi`
5. **Save configurations**: Keep successful configs in JSON files
6. **Monitor GPU usage**: Use `watch -n 1 nvidia-smi` during training
7. **Checkpoint regularly**: Default saves every 5000 steps
8. **Experiment with generation**: Try different temperature values
9. **Use appropriate data**: Match model to your use case
10. **Be patient**: Training takes time, but quality improves

---

For more information, see:
- **GETTING_STARTED.md** - Beginner guide
- **README.md** - Full documentation
- **PROJECT_OVERVIEW.md** - Technical details
