# Custom VRAM Configuration Guide

Automatically optimize training configuration based on your exact GPU memory with safety margins to prevent OOM errors.

## ⚠️ Important: Actual vs Advertised GPU Memory

**Your GPU shows less memory than advertised!**

Example: A "16GB" GPU typically shows only **15.57GB available** in `nvidia-smi`:
```
1743MiB / 16376MiB
```

This is because:
- ~430MB is reserved by the system/CUDA
- The "16GB" is marketing (16,000MB), but actual is 16,384MB
- After system reserves: 15,976MB (~15.57GB) is what you actually see

**Therefore, we use a 1024MB (1GB) safety margin by default** to account for:
- System/CUDA reserves: ~400-500MB
- PyTorch overhead: ~200-300MB
- Memory fragmentation: ~200-300MB

## Overview

The VRAM optimizer:
- ✅ Auto-detects single or multi-GPU setups
- ✅ Accepts custom VRAM values (e.g., 24GB, 20.5GB, 15.6GB)
- ✅ Applies configurable safety margin (default: 1024MB (1GB))
- ✅ Calculates optimal batch size and sequence length
- ✅ Estimates memory usage for your model
- ✅ Prevents out-of-memory errors

## Quick Examples

### Auto-Detect GPU Memory

```bash
# Automatically detect and use all available VRAM
python train.py --model_size medium --vram_auto --data_dir synthetic_data
```

**What it does**:
- Detects total VRAM (e.g., 16GB)
- Subtracts 1024MB (1GB) safety margin → 15.0GB usable
- Optimizes batch size, sequence length, etc.

### Custom VRAM Value

```bash
# Use exactly 24GB
python train.py --model_size medium --vram 24 --data_dir synthetic_data

# Use 20.5GB (if you have 24GB but want to reserve some)
python train.py --model_size medium --vram 20.5 --data_dir synthetic_data

# Use 15.6GB (your 16GB minus safety margin)
python train.py --model_size medium --vram 15.6 --data_dir synthetic_data
```

### Multi-GPU Cluster

```bash
# 2x 24GB GPUs = 48GB total
python train.py --model_size large --vram 48 --data_dir synthetic_data

# Or auto-detect
python train.py --model_size large --vram_auto --data_dir synthetic_data
```

### Custom Safety Margin

```bash
# Use 800MB safety margin instead of default 1024MB
python train.py --model_size medium --vram 24 --vram_safety_margin 500 --data_dir data

# Smaller margin (300MB) for more VRAM usage
python train.py --model_size medium --vram 24 --vram_safety_margin 300 --data_dir data

# Large margin (1GB = 1024MB) for very safe training
python train.py --model_size medium --vram 24 --vram_safety_margin 1024 --data_dir data
```

## How It Works

### Step 1: Detect or Use Custom VRAM

```
Your GPU: 24GB total
Safety margin: -1024MB (1GB)
Usable VRAM: 23.0GB (24,576MB - 1,024MB = 23,552MB)
```

### Step 2: Estimate Model Memory

```
Medium model:
  - Model weights (FP16): ~300MB
  - Optimizer states: ~1024MB
  - Total model: ~900MB
```

### Step 3: Calculate Batch Memory Budget

```
Available for batches: 23.6GB - 0.9GB = 22.7GB
```

### Step 4: Optimize Batch Configuration

Based on available memory, calculates:
- Optimal batch size
- Safe sequence length
- Gradient accumulation steps
- Whether to use gradient checkpointing

## Command Reference

### Basic Arguments

```bash
--vram VALUE                # VRAM to use (e.g., "24", "20.5", "16gb")
--vram_auto                 # Auto-detect GPU VRAM
--vram_safety_margin MB     # Safety margin in MB (default: 600)
```

### Examples by GPU Size

#### 8GB GPU (RTX 2060, GTX 1080)

```bash
# Auto-detect
python train.py --model_size small --vram_auto --data_dir data

# Manual (7.4GB after 1024MB margin)
python train.py --model_size small --vram 7.6 --data_dir data

# Conservative (leave 1GB free)
python train.py --model_size small --vram 7 --data_dir data
```

#### 12GB GPU (RTX 3060, RTX 2080 Ti)

```bash
# Auto-detect
python train.py --model_size medium --vram_auto --data_dir data

# Manual (11.6GB after margin)
python train.py --model_size medium --vram 11.6 --data_dir data
```

#### 16GB GPU (RTX 4060 Ti, RTX 3080)

```bash
# Auto-detect
python train.py --model_size medium --vram_auto --data_dir data

# Manual (15.6GB after margin)
python train.py --model_size medium --vram 15.6 --data_dir data

# Your exact case (from nvidia-smi: 16376MB = 16GB)
python train.py --model_size medium --vram 15.976 --data_dir data
```

#### 24GB GPU (RTX 3090, RTX 4090)

```bash
# Auto-detect
python train.py --model_size large --vram_auto --data_dir data

# Manual (23.6GB after margin)
python train.py --model_size large --vram 23.6 --data_dir data

# Aggressive (only 200MB margin)
python train.py --model_size large --vram 23.8 --vram_safety_margin 200 --data_dir data
```

#### 48GB Multi-GPU or A6000

```bash
# Auto-detect
python train.py --model_size large --vram_auto --data_dir data

# Manual
python train.py --model_size large --vram 47.6 --data_dir data
```

## Understanding the Output

When you run with custom VRAM, you'll see:

```
============================================================
VRAM Optimizer
============================================================

Detected GPUs: 1
  GPU 0: NVIDIA GeForce RTX 3080 (16.00 GB)

Target VRAM: 16.00 GB
Safety Margin: 1024 MB (1GB)
Usable VRAM: 15.00 GB (15360 MB)

Estimated Model Memory: 900 MB (0.88 GB)
Available for Batches: 14460 MB (14.12 GB)

============================================================
Optimized Configuration
============================================================
  batch_size                    : 2
  gradient_accumulation_steps   : 8
  max_seq_len                   : 512
  use_fp16                      : True
  use_gradient_checkpointing    : True
  use_cpu_offload               : False
  effective_batch_size          : 16
  description                   : Auto-optimized for 16.0GB VRAM
============================================================
```

## Multi-GPU Support

The optimizer detects all GPUs and sums their memory:

```bash
# You have 2x RTX 3090 (24GB each) = 48GB total
python train.py --model_size large --vram_auto --data_dir data
```

Output:
```
Detected GPUs: 2
  GPU 0: NVIDIA RTX 3090 (24.00 GB)
  GPU 1: NVIDIA RTX 3090 (24.00 GB)

Total VRAM: 48.00 GB
Target VRAM: 48.00 GB
Usable VRAM: 47.00 GB (48,128MB after 1024MB safety margin)
```

## Safety Margin Explained

The safety margin prevents OOM by reserving memory:

```
Total VRAM:     24,576 MB (24GB)
Safety Margin:    -1,024 MB (1GB)
Usable:         23,552 MB (23.0 GB)
```

**Why 1024MB (1GB) default?**
- System/CUDA reserves: ~400-500 MB (already taken before PyTorch even loads)
- PyTorch overhead: ~200-300 MB
- Memory fragmentation: ~200-300 MB
- **Total needed: ~1000MB minimum**

**Real-world example:**
- Your "16GB" GPU shows only **15.57GB available** in nvidia-smi
- That's 430MB already reserved by system/CUDA
- Plus PyTorch needs 200-300MB overhead
- Plus fragmentation needs 200-300MB buffer
- **= 1024MB (1GB) total safety margin**

This ensures training won't OOM across all GPU models and workloads.

**Adjust if needed**:
```bash
# More conservative (1GB margin)
--vram_safety_margin 1024

# Less conservative (200MB margin)
--vram_safety_margin 200

# Aggressive (100MB margin) - risky!
--vram_safety_margin 100
```

## Advanced Usage

### Test VRAM Optimizer

```bash
# Test the optimizer directly
cd utils
python vram_optimizer.py
```

### Verify GPU Detection

```bash
# In Python
python3 << 'EOF'
import torch
print(f"GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name}, {props.total_memory/1024**3:.2f} GB")
EOF
```

### Calculate Exact VRAM from nvidia-smi

Your nvidia-smi shows: `1743MiB / 16376MiB`

This means you have a **16GB GPU** (16,384MB total), but only **15,976MB (15.6GB) is available** after system reserves.

```bash
# Option 1: Use auto-detect (recommended)
python train.py --model_size medium --vram_auto --data_dir data

# Option 2: Specify 16GB (system will use 15GB after 1GB margin)
python train.py --model_size medium --vram 16 --data_dir data

# Option 3: Conservative - use less to be extra safe
python train.py --model_size medium --vram 14 --data_dir data
```

### Compare Configurations

```bash
# Preset
python train.py --model_size medium --vram 16gb --data_dir data

# Custom (same VRAM, auto-optimized)
python train.py --model_size medium --vram 16 --data_dir data

# Custom with your exact free memory (14.6GB free from nvidia-smi)
python train.py --model_size medium --vram 14.6 --data_dir data
```

## Optimization Strategy

The optimizer uses these rules:

| VRAM (GB) | Batch Size | Seq Length | Gradient Acc | Checkpointing |
|-----------|-----------|-----------|--------------|---------------|
| ≤ 8       | 1         | 256       | 16           | Yes           |
| 9-12      | 2         | 512       | 8            | Yes           |
| 13-16     | 2         | 512       | 8            | Yes           |
| 17-24     | 4         | 1024      | 4            | No            |
| 25-32     | 8         | 1024      | 2            | No            |
| 33+       | 12        | 2048      | 2            | No            |

**Then adjusts based on**:
- Actual available memory
- Model size (small/medium/large)
- Estimated memory requirements

## Troubleshooting

### Still Getting OOM

**Try larger safety margin**:
```bash
python train.py --model_size medium --vram 16 --vram_safety_margin 800 --data_dir data
```

**Or reduce VRAM target**:
```bash
# Use 14GB instead of 16GB
python train.py --model_size medium --vram 14 --data_dir data
```

### Not Using Enough Memory

**Reduce safety margin**:
```bash
python train.py --model_size medium --vram 16 --vram_safety_margin 200 --data_dir data
```

**Or increase VRAM target**:
```bash
python train.py --model_size medium --vram 16.5 --data_dir data
```

### Multi-GPU Not Detected

```bash
# Check CUDA visibility
echo $CUDA_VISIBLE_DEVICES

# Should be empty or "0,1"
# If it's "0", only one GPU is visible

# Fix:
export CUDA_VISIBLE_DEVICES=0,1
python train.py --vram_auto ...
```

### Wrong VRAM Detected

```bash
# Manually specify
python train.py --vram 24 --data_dir data

# Don't use --vram_auto if detection is wrong
```

## Real-World Examples

### Your 16GB GPU Case

From your nvidia-smi: `1743MiB / 16376MiB` (only 1.7GB used)

```bash
# Option 1: Auto-detect (safest)
python train.py --model_size medium --vram_auto --data_dir synthetic_data

# Option 2: Use exact total (16GB)
python train.py --model_size medium --vram 16 --data_dir synthetic_data

# Option 3: Use after standard margin (15.6GB)
python train.py --model_size medium --vram 15.6 --data_dir synthetic_data

# Option 4: Use available (14.6GB free from nvidia-smi)
python train.py --model_size medium --vram 14.6 --data_dir synthetic_data
```

### Mixed GPU Cluster (2x different GPUs)

```bash
# 1x RTX 3090 (24GB) + 1x RTX 3080 (10GB) = 34GB total
python train.py --vram 34 --model_size large --data_dir data
```

### Conservative for Stability

```bash
# Use 20GB of your 24GB GPU (4GB reserved)
python train.py --vram 20 --model_size large --data_dir data

# Equivalent with safety margin
python train.py --vram 24 --vram_safety_margin 4096 --model_size large --data_dir data
```

## Integration with Background Training

```bash
# Background training with custom VRAM
./train_background.sh --model_size medium --vram 20.5 --data_dir data

# Background with auto-detect
./train_background.sh --model_size medium --vram_auto --data_dir data

# Background with custom margin
./train_background.sh --vram 24 --vram_safety_margin 1024 --data_dir data
```

## Summary

**Auto-detect (easiest)**:
```bash
python train.py --model_size medium --vram_auto --data_dir data
```

**Custom VRAM (precise control)**:
```bash
python train.py --model_size medium --vram 24 --data_dir data
```

**With safety margin**:
```bash
python train.py --vram 24 --vram_safety_margin 600 --data_dir data
```

**For your 16GB GPU**:
```bash
python train.py --model_size medium --vram 16 --data_dir synthetic_data
```

The system automatically:
- ✅ Detects single/multi-GPU
- ✅ Applies safety margin
- ✅ Optimizes batch size and sequence length
- ✅ Prevents OOM errors

No more manual tuning needed!
