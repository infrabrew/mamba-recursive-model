# Final Fix for 16GB GPU OOM Error

## Problem Summary

Even with 2048MB (2GB) safety margin, the **medium model still causes OOM** on the 16GB GPU.

**Root Cause:**
- The 16GB GPU only shows **15.57GB actually available**
- Medium model + batch_size=2 + seq_len=512 uses **15.25GB** 
- This leaves only ~320MB free, which is insufficient for training operations

## Solution: Use SMALL Model Instead

The medium model (~150M parameters) is **too large** for a 16GB GPU during training.

### Recommended Command

```bash
# Use SMALL model instead of MEDIUM
python train.py --vram 16 --model_size small --data_dir ./data

# Or with auto-detect
python train.py --vram_auto --model_size small --data_dir ./data

# Background training
./train_background.sh --vram_auto --model_size small --data_dir ./data
```

### Why Small Model Works

**Medium model:**
- ~150M parameters
- Model + optimizer: ~900MB
- But activations/gradients use 15.25GB total → **TOO LARGE**

**Small model:**
- ~50M parameters  
- Model + optimizer: ~300MB
- Activations/gradients: ~5-6GB total → **FITS COMFORTABLY**

## Alternative: Keep Medium But Reduce Further

If you MUST use medium model:

```bash
# Ultra-conservative: batch_size=1, seq_len=256, 3GB safety margin
python train.py --vram 13 --model_size medium --data_dir ./data
```

But this will be **VERY SLOW** and may still fail.

## Comparison Table

| Model  | Parameters | GPU Memory | Recommended for 16GB? |
|--------|------------|------------|----------------------|
| Small  | ~50M       | ~6-8GB     | ✅ **YES** |
| Medium | ~150M      | ~15-16GB   | ❌ **NO** (OOM risk) |
| Large  | ~400M      | ~30-35GB   | ❌ **NO** (won't fit) |

## Final Recommendation

**For your 16GB GPU, use the SMALL model:**

```bash
cd /home/coreai/llm-projects/mamba_trainer
source venv/bin/activate
python train.py --vram_auto --model_size small --data_dir ./data
```

This will:
- ✅ Fit comfortably in 15.57GB available
- ✅ Leave room for training operations
- ✅ Train successfully without OOM
- ✅ Still provide good results (50M parameters is respectable)

## If You Want Medium Model

You'll need:
- **24GB GPU** (RTX 3090, RTX 4090, A5000, etc.)
- Or use **gradient checkpointing + CPU offloading** (much slower)
