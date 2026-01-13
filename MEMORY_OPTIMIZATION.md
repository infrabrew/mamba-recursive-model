# Memory Optimization Guide

This guide helps you resolve CUDA out of memory errors when training Mamba models.

## Quick Fixes

If you get `torch.OutOfMemoryError`, try these in order:

### 1. Use Smaller Batch Size and Sequence Length

```bash
# Most conservative (should work on 16GB)
python train.py \
  --model_size small \
  --vram 8gb \
  --data_dir your_data
```

### 2. Set Environment Variable

```bash
# Before training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py --model_size medium --vram 16gb --data_dir your_data
```

### 3. Use Custom Config with Smaller Settings

Create `config_16gb_safe.json`:
```json
{
  "model": {
    "d_model": 512,
    "n_layers": 8,
    "d_state": 16,
    "expand_factor": 2,
    "dropout": 0.1,
    "max_seq_len": 256,
    "vocab_size": 50257
  },
  "vram": {
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "max_seq_len": 256,
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

Then train:
```bash
python train.py --config config_16gb_safe.json --data_dir your_data
```

## Understanding the Error

The error message shows:
```
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 2.00 MiB. GPU 0 has a total capacity of 15.57 GiB
of which 2.44 MiB is free.
```

This means your GPU memory is full. The Mamba model's selective scan operation processes sequences step-by-step, which can use significant memory.

## Memory Optimizations Applied

I've updated the code with these optimizations:

### 1. ✅ Chunked Processing in Selective Scan

The `selective_scan` method now processes sequences in chunks of 64 timesteps instead of all at once, with cache clearing between chunks.

### 2. ✅ Pre-allocated Tensors

Output tensors are pre-allocated instead of using list appending, reducing memory fragmentation.

### 3. ✅ Environment Variables

The training script now sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` automatically.

### 4. ✅ Conservative 16GB Config

The 16GB VRAM configuration is now more conservative:
- Batch size: 2 (was 4)
- Sequence length: 512 (was 1024)
- Gradient accumulation: 8 (was 4)

## Recommended Settings by GPU

| GPU Memory | Model Size | Batch Size | Seq Length | Command |
|-----------|------------|-----------|-----------|---------|
| 8GB | Small | 1 | 256 | `--model_size small --vram 8gb` |
| 12GB | Small | 2 | 512 | `--model_size small --vram 12gb` |
| 16GB | Small | 2 | 512 | `--model_size small --vram 16gb` |
| 16GB | Medium | 1 | 256 | Custom config (see above) |
| 24GB+ | Medium | 4+ | 1024+ | `--model_size medium --vram 24gb` |

## Step-by-Step Troubleshooting

### Step 1: Check Your GPU

```bash
nvidia-smi
```

Make sure:
- GPU is visible
- Memory is not being used by other processes
- CUDA is available

### Step 2: Start with Smallest Configuration

```bash
python train.py \
  --model_size small \
  --vram 8gb \
  --data_dir synthetic_data
```

If this works, gradually increase:
1. Try `--vram 12gb`
2. Try `--model_size medium` with `--vram 8gb`
3. Increase batch size in custom config

### Step 3: Monitor Memory Usage

In another terminal:
```bash
watch -n 1 nvidia-smi
```

This shows memory usage in real-time.

### Step 4: Reduce Sequence Length

If still OOM, reduce `max_seq_len`:

```json
{
  "model": {
    "max_seq_len": 128  // Very small
  },
  "vram": {
    "max_seq_len": 128,
    "batch_size": 1
  }
}
```

## Advanced Optimizations

### 1. Gradient Checkpointing

Enable in config (already enabled for 8-16GB):
```json
"vram": {
  "use_gradient_checkpointing": true
}
```

This trades compute for memory - slower but uses less RAM.

### 2. Mixed Precision (FP16)

Enable in config (already enabled):
```json
"vram": {
  "use_fp16": true
}
```

This reduces memory by ~50% with minimal accuracy loss.

### 3. Reduce Model Size

Use fewer layers or smaller dimension:
```json
"model": {
  "d_model": 256,  // Very small
  "n_layers": 4
}
```

### 4. CPU Offloading (Last Resort)

```json
"vram": {
  "use_cpu_offload": true
}
```

**Warning**: This is VERY slow but prevents OOM.

## Configuration Examples

### Ultra-Safe 16GB Config

```bash
python train.py \
  --model_size small \
  --vram 8gb \
  --data_dir your_data
```

Settings:
- Batch size: 1
- Sequence length: 256
- Gradient accumulation: 16
- FP16: Yes
- Gradient checkpointing: Yes

### Balanced 16GB Config

Create `config_balanced_16gb.json`:
```json
{
  "model": {
    "d_model": 512,
    "n_layers": 8,
    "d_state": 16,
    "expand_factor": 2,
    "dropout": 0.1,
    "max_seq_len": 512,
    "vocab_size": 50257
  },
  "vram": {
    "batch_size": 2,
    "gradient_accumulation_steps": 8,
    "max_seq_len": 512,
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
    "logging_steps": 10
  }
}
```

## Common Causes and Solutions

### Cause: Sequence Too Long

**Solution**: Reduce `max_seq_len` in config
```json
"model": {"max_seq_len": 256}
```

### Cause: Batch Size Too Large

**Solution**: Reduce batch size, increase gradient accumulation
```json
"vram": {
  "batch_size": 1,
  "gradient_accumulation_steps": 16
}
```

### Cause: Model Too Large

**Solution**: Use smaller model
```bash
--model_size small
```

### Cause: Memory Fragmentation

**Solution**: Set environment variable (now automatic)
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Cause: Other Processes Using GPU

**Solution**: Kill other GPU processes
```bash
nvidia-smi  # Find process ID
kill -9 <PID>
```

## Testing Your Configuration

Before long training, test with small data:

```bash
# Test configuration
python train.py \
  --config your_config.json \
  --data_dir data \
  --training_preset fast
```

Watch for OOM errors in first few steps.

## Memory Usage Estimation

Approximate memory usage:

**Model Weights**:
- Small: ~200 MB
- Medium: ~600 MB
- Large: ~1.6 GB

**Activations** (per batch, FP16):
- Batch size 1, seq 256: ~100 MB
- Batch size 2, seq 512: ~400 MB
- Batch size 4, seq 1024: ~1.6 GB

**Optimizer States** (AdamW):
- 2x model weights

**Example: Medium model, batch 2, seq 512**:
- Model: 600 MB
- Activations: 400 MB
- Optimizer: 1200 MB
- Total: ~2.2 GB
- With overhead: ~3-4 GB

This should fit in 16GB easily, but the selective scan's intermediate states use additional memory.

## If Nothing Works

1. **Use CPU training** (very slow):
```bash
# Force CPU
CUDA_VISIBLE_DEVICES= python train.py ...
```

2. **Train on cloud GPU**:
- Google Colab (free 16GB)
- AWS/GCP/Azure
- Lambda Labs
- RunPod

3. **Use smaller chunks of data**:
Train on subsets sequentially

## Getting Help

If you're still having issues:

1. Run `nvidia-smi` and share output
2. Share your config file
3. Share the exact error message
4. Mention your GPU model

## Summary

**Quick fix for 16GB OOM**:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train.py --model_size small --vram 8gb --data_dir your_data
```

**If that doesn't work**:
Use the custom config with `batch_size: 1` and `max_seq_len: 256`

The optimizations in the code should handle most cases now!
