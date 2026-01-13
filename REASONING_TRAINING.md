# Training Mamba Reasoning Models

This guide explains how to train Mamba models for reasoning tasks using chain-of-thought (CoT) and step-by-step problem solving.

## What is a Reasoning Model?

Reasoning models are trained to:
- Show step-by-step thinking
- Use chain-of-thought (CoT) prompting
- Break down complex problems
- Provide detailed explanations before answers

## Best Reasoning Datasets

### 1. MathInstruct (262K examples - Recommended)
```bash
./train_background.sh \
    --vram_auto \
    --model_size medium-x \
    --dataset TIGER-Lab/MathInstruct \
    --dataset_split train \
    --text_column "instruction,output"
```

### 2. GSM8K (Grade School Math)
```bash
./train_background.sh \
    --vram_auto \
    --model_size medium-x \
    --dataset gsm8k \
    --dataset_split train \
    --text_column "question,answer"
```

### 3. MATH (Competition Math)
```bash
./train_background.sh \
    --vram_auto \
    --model_size medium-x \
    --dataset hendrycks/competition_math \
    --dataset_split train \
    --text_column "problem,solution"
```

### 4. OpenMathInstruct (1.8M examples)
```bash
./train_background.sh \
    --vram_auto \
    --model_size medium-x \
    --dataset nvidia/OpenMathInstruct-1 \
    --dataset_split train \
    --text_column "instruction,response"
```

### 5. MetaMath (395K examples)
```bash
./train_background.sh \
    --vram_auto \
    --model_size medium-x \
    --dataset meta-math/MetaMathQA \
    --dataset_split train \
    --text_column "query,response"
```

## Training Configuration for Reasoning

### Recommended Settings

**For 16GB GPU:**
```bash
./train_background.sh \
    --vram_auto \
    --model_size medium-lite \
    --dataset TIGER-Lab/MathInstruct \
    --dataset_split train \
    --text_column "instruction,output" \
    --training_preset careful
```

**For 24GB GPU:**
```bash
./train_background.sh \
    --vram_auto \
    --model_size medium-extra \
    --dataset TIGER-Lab/MathInstruct \
    --dataset_split train \
    --text_column "instruction,output" \
    --training_preset default
```

**For 48GB+ GPU:**
```bash
./train_background.sh \
    --vram_auto \
    --model_size large-lite \
    --dataset TIGER-Lab/MathInstruct \
    --dataset_split train \
    --text_column "instruction,output" \
    --training_preset fast
```

## Combining Multiple Reasoning Datasets

You can train on local reasoning examples + HuggingFace dataset:

```bash
# Create local reasoning examples
mkdir -p data/reasoning
cat > data/reasoning/custom_math.txt << 'EXAMPLE'
Problem: What is 15% of 80?
Solution: 
Step 1: Convert 15% to decimal: 15/100 = 0.15
Step 2: Multiply: 0.15 × 80 = 12
Answer: 12
EXAMPLE

# Train on both
./train_background.sh \
    --vram_auto \
    --model_size medium-x \
    --data_dir ./data \
    --dataset TIGER-Lab/MathInstruct \
    --dataset_split train \
    --text_column "instruction,output"
```

## Training Tips for Reasoning Models

### 1. Longer Sequence Lengths
Reasoning requires more tokens for step-by-step explanations:
- Small models: 1024 tokens
- Medium models: 1536-2048 tokens  
- Large models: 2048-4096 tokens

### 2. More Training Steps
Reasoning models benefit from longer training:
- Use `--training_preset careful` for more steps (200K)
- Or `--training_preset default` for balanced (100K)

### 3. Start with Smaller Models
- Train medium-lite or medium-plus first
- Evaluate reasoning quality
- Scale up if needed

### 4. Monitor Eval Loss
Watch TensorBoard for:
- Steady decrease in eval loss
- No overfitting (train loss << eval loss)

```bash
tensorboard --logdir=runs --host=0.0.0.0 --port=6006
```

## Example: Complete Reasoning Model Training

```bash
# SSH to your ML server
ssh coreai@192.168.0.31
cd /home/coreai/llm-projects/mamba_trainer
source venv/bin/activate

# Train a 135M parameter reasoning model
./train_background.sh \
    --vram_auto \
    --model_size medium-x \
    --dataset TIGER-Lab/MathInstruct \
    --dataset_split train \
    --text_column "instruction,output" \
    --training_preset default \
    --output_dir ./checkpoints/reasoning

# Monitor training
tail -f logs/training_*.log

# Start TensorBoard (in another terminal)
tensorboard --logdir=runs --host=0.0.0.0 --port=6006
# Then open: http://192.168.0.31:6006
```

## Reasoning Model Performance by Size

| Model Size | Parameters | VRAM | Training Time* | Quality |
|------------|-----------|------|----------------|---------|
| medium-lite | 90M | 10-12GB | ~48 hours | Good |
| medium-plus | 120M | 13-14GB | ~60 hours | Better |
| medium-x | 135M | 14-15GB | ~70 hours | Very Good |
| medium | 150M | 15-16GB | ~80 hours | Excellent |
| large-lite | 250M | 18-20GB | ~120 hours | Outstanding |

*Estimated for 100K steps on MathInstruct

## Post-Training

After training completes, your reasoning model will be in:
```
checkpoints/reasoning/final/
```

The model will have learned to:
- Break down complex problems
- Show step-by-step reasoning
- Provide detailed explanations
- Arrive at correct answers through logical steps

## Troubleshooting

**OOM Errors:**
- Use smaller model size
- Reduce max_seq_len in config
- Increase safety margin: `--vram_safety_margin 2048`

**Poor Reasoning Quality:**
- Train longer (use `--training_preset careful`)
- Use larger model
- Check dataset quality
- Monitor eval loss in TensorBoard

**Slow Training:**
- Use `--training_preset fast` for fewer steps
- Smaller model size
- Check GPU utilization: `watch -n 1 nvidia-smi`
