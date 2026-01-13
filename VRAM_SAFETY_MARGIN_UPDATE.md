# VRAM Safety Margin Update

## Summary

Updated default safety margin from **400MB to 600MB** across the entire codebase.

## Rationale

Based on real-world GPU behavior:
- A 16GB GPU shows only **15.57GB available** in nvidia-smi
- This means ~430MB is already reserved by the system
- 600MB safety margin provides adequate buffer across all GPU models

## Calculations with 600MB Margin

### 16GB GPU
```
Total VRAM:     16,384 MB (16.0 GB)
Safety Margin:    -600 MB
Usable VRAM:    15,784 MB (15.41 GB)
```

### 24GB GPU
```
Total VRAM:     24,576 MB (24.0 GB)
Safety Margin:    -600 MB
Usable VRAM:    23,976 MB (23.41 GB)
```

### 48GB Multi-GPU (2×24GB)
```
Total VRAM:     49,152 MB (48.0 GB)
Safety Margin:    -600 MB
Usable VRAM:    48,552 MB (47.41 GB)
```

## Files Updated

1. **utils/vram_optimizer.py**
   - `VRAMOptimizer.__init__()`: default `safety_margin_mb=600`
   - `create_vram_config()`: default `safety_margin_mb=600`

2. **train.py**
   - `--vram_safety_margin`: default changed to `600`

3. **CUSTOM_VRAM.md**
   - All examples updated to reflect 600MB margin
   - Explanation section updated with real-world GPU context

## Usage

### Default (600MB margin)
```bash
python train.py --vram_auto --model_size medium --data_dir data
python train.py --vram 24 --model_size large --data_dir data
```

### Custom margin
```bash
# More conservative (1GB margin)
python train.py --vram 24 --vram_safety_margin 1024 --data_dir data

# Less conservative (300MB margin)
python train.py --vram 24 --vram_safety_margin 300 --data_dir data
```

## Verification

Tested with:
- 16GB: 15.41 GB usable ✓
- 24GB: 23.41 GB usable ✓
- 48GB: 47.41 GB usable ✓

All calculations verified correct.
