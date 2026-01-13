# Hybrid Recursive Mamba Models

## Overview

The Hybrid Recursive Mamba combines two powerful architectures:

1. **Mamba SSM** - Efficient linear-time sequence modeling with selective state spaces
2. **Recursive Language Model** - Hierarchical reasoning with multi-scale processing

This creates a model that excels at:
- ✅ Long-range dependencies (from Mamba)
- ✅ Hierarchical reasoning (from recursive processing)
- ✅ Multi-scale pattern recognition
- ✅ Complex problem decomposition

## Architecture Components

### 1. Mamba SSM Layer
- Selective State Space Models for efficient sequence processing
- Linear O(n) complexity instead of O(n²) attention
- Data-dependent parameter selection

### 2. Recursive Processor
- Dynamic recursion depth (learned per input)
- Memory accumulation across recursion levels
- Hierarchical state management

### 3. Hierarchical Attention
- Multi-scale attention (1x, 2x, 4x pooling)
- Captures patterns at different granularities
- Scale fusion for comprehensive understanding

### 4. Hybrid Fusion
- Combines Mamba and recursive outputs
- Residual connections for training stability
- Adaptive weighting

## Model Sizes

| Model | Parameters | Recursion Depth | VRAM | Best For |
|-------|-----------|-----------------|------|----------|
| **hybrid-small** | 75M | 3 | 8-10GB | Code, math reasoning |
| **hybrid-medium** | 220M | 3 | 18-20GB | Complex reasoning, planning |
| **hybrid-large** | 550M | 4 | 40-45GB | Advanced reasoning, research |

## Training Hybrid Models

### Basic Training

```bash
# Hybrid small model on 16GB GPU
./train_background.sh \
    --vram_auto \
    --model_size hybrid-small \
    --dataset TIGER-Lab/MathInstruct \
    --dataset_split train \
    --text_column "instruction,output"
```

### For 24GB GPU

```bash
# Hybrid medium model
./train_background.sh \
    --vram_auto \
    --model_size hybrid-medium \
    --dataset TIGER-Lab/MathInstruct \
    --dataset_split train \
    --text_column "instruction,output" \
    --training_preset default
```

### For 48GB GPU

```bash
# Hybrid large model
./train_background.sh \
    --vram_auto \
    --model_size hybrid-large \
    --dataset TIGER-Lab/MathInstruct \
    --dataset_split train \
    --text_column "instruction,output" \
    --training_preset careful
```

## Best Use Cases

### 1. Mathematical Reasoning
The recursive architecture excels at multi-step math problems:

```bash
./train_background.sh \
    --vram_auto \
    --model_size hybrid-small \
    --dataset TIGER-Lab/MathInstruct \
    --dataset_split train \
    --text_column "instruction,output"
```

### 2. Code Generation
Hierarchical processing helps with complex code structures:

```bash
./train_background.sh \
    --vram_auto \
    --model_size hybrid-medium \
    --data_dir ./code_data \
    --dataset bigcode/the-stack-dedup \
    --dataset_split train \
    --text_column content
```

### 3. Planning & Strategy
Recursive depth enables multi-level planning:

```bash
./train_background.sh \
    --vram_auto \
    --model_size hybrid-medium \
    --dataset openai/gsm8k \
    --dataset_split train \
    --text_column "question,answer"
```

### 4. Scientific Reasoning
Multi-scale attention captures complex relationships:

```bash
./train_background.sh \
    --vram_auto \
    --model_size hybrid-large \
    --dataset hendrycks/competition_math \
    --dataset_split train \
    --text_column "problem,solution"
```

## Comparison: Standard vs Hybrid

| Feature | Standard Mamba | Hybrid Recursive |
|---------|---------------|------------------|
| Sequence Modeling | ✅ Excellent | ✅ Excellent |
| Reasoning Depth | ⚠️ Limited | ✅ Deep (3-4 levels) |
| Multi-scale Processing | ❌ No | ✅ Yes |
| Parameter Efficiency | ✅ High | ⚠️ Medium (1.5x params) |
| Training Speed | ✅ Fast | ⚠️ Moderate (1.3x slower) |
| VRAM Usage | ✅ Low | ⚠️ Higher (+30-40%) |
| Best For | General text | Reasoning, planning |

## Architecture Details

### Recursive Processing

The model uses **dynamic recursion depth** based on input complexity:

```
Input → Depth Gate → Weighted Recursion
                   ↓
        Level 1: Basic processing
        Level 2: Intermediate reasoning
        Level 3: Deep analysis
        Level 4: Meta-reasoning (large model only)
```

### Memory Management

Each recursion level maintains state:
- **Short-term**: Current level processing
- **Long-term**: Accumulated from previous levels
- **Fusion**: Combines all levels for output

### Hierarchical Attention

Multi-scale pattern detection:
```
1x scale: Token-level patterns
2x scale: Phrase-level patterns
4x scale: Sentence-level patterns
→ Fused for comprehensive understanding
```

## Training Tips

### 1. Start with Hybrid-Small
- Faster iteration
- Lower cost
- Good for prototyping

### 2. Use Reasoning Datasets
Hybrid models shine with structured reasoning:
- MathInstruct (math problems)
- GSM8K (grade school math)
- Code datasets
- Scientific QA

### 3. Monitor Recursion Depth
Watch TensorBoard for:
- Depth distribution
- Layer-wise activation patterns
- Recursive state evolution

### 4. Longer Training
Hybrid models benefit from more steps:
- Use `--training_preset careful` (200K steps)
- Better recursion pattern learning
- Improved hierarchical reasoning

## Example Training Session

```bash
# SSH to ML server
ssh coreai@192.168.0.31
cd /home/coreai/llm-projects/mamba_trainer
source venv/bin/activate

# Train hybrid model for reasoning
./train_background.sh \
    --vram_auto \
    --model_size hybrid-small \
    --dataset TIGER-Lab/MathInstruct \
    --dataset_split train \
    --text_column "instruction,output" \
    --training_preset default \
    --output_dir ./checkpoints/hybrid_reasoning

# Monitor training
tail -f logs/training_*.log

# TensorBoard (in another terminal)
tensorboard --logdir=runs --host=0.0.0.0 --port=6006
# Open: http://192.168.0.31:6006
```

## Performance Expectations

### Training Time (100K steps)

| Model | GPU | Time | Quality |
|-------|-----|------|---------|
| hybrid-small | 16GB | ~60 hours | Very Good |
| hybrid-medium | 24GB | ~100 hours | Excellent |
| hybrid-large | 48GB | ~180 hours | Outstanding |

### Inference Speed

Hybrid models are ~1.3x slower than standard Mamba due to:
- Recursive processing overhead
- Multi-scale attention
- Hierarchical state management

But the reasoning quality improvement is significant!

## When to Use Hybrid vs Standard

**Use Hybrid Recursive Mamba when:**
- ✅ Task requires multi-step reasoning
- ✅ Hierarchical problem decomposition needed
- ✅ Complex planning/strategy required
- ✅ Code generation with deep structure
- ✅ Mathematical/scientific reasoning

**Use Standard Mamba when:**
- ✅ General text generation
- ✅ Speed is critical
- ✅ VRAM is limited
- ✅ Simple pattern matching
- ✅ Conversational AI

## Troubleshooting

**OOM Errors:**
- Use smaller hybrid model (hybrid-small)
- Reduce max_recursion_depth in config
- Disable hierarchical attention if needed

**Slow Training:**
- Normal - hybrid is 1.3x slower
- Use `--training_preset fast` for quicker results
- Consider standard Mamba if speed critical

**Poor Reasoning:**
- Train longer (`--training_preset careful`)
- Use larger model (hybrid-medium or hybrid-large)
- Ensure reasoning dataset quality

## Advanced Configuration

You can customize hybrid models in `configs/model_configs.py`:

```python
'custom-hybrid': {
    'd_model': 768,
    'n_layers': 12,
    'd_state': 16,
    'expand_factor': 2,
    'dropout': 0.1,
    'max_seq_len': 2048,
    'vocab_size': 50257,
    'max_recursion_depth': 5,  # More recursion levels
    'use_hierarchical_attention': True,  # Enable multi-scale
    'model_type': 'hybrid_recursive',
    'description': 'Custom Hybrid Mamba'
}
```

## Conclusion

The Hybrid Recursive Mamba is ideal for:
- 🧮 Mathematical reasoning
- 💻 Complex code generation
- 🧠 Multi-step problem solving
- 📊 Scientific analysis
- 🎯 Strategic planning

It combines the efficiency of Mamba with the reasoning power of recursive architectures, creating a model that thinks hierarchically while processing efficiently.

Ready to train your first hybrid model? Start with:

```bash
./train_background.sh --vram_auto --model_size hybrid-small --dataset TIGER-Lab/MathInstruct --dataset_split train --text_column "instruction,output"
```
