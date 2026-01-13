# Deployment Guide - Full Precision (No Quality Loss)

This guide focuses on deploying your Mamba model **without quantization** to preserve maximum accuracy.

## Overview

Quantization (Q4_K_M, Q5_K_M, etc.) reduces model size but **decreases accuracy**. For production use where quality matters, use full precision formats.

## Recommended Approaches

### ✅ Best: HuggingFace Format (FP16/FP32)

**Advantages**:
- No accuracy loss
- Works with vLLM (fastest inference)
- Works with HuggingFace Transformers
- Native PyTorch precision

**Use with**: vLLM, HuggingFace Transformers, custom servers

---

### ✅ Good: GGUF FP16 Format

**Advantages**:
- Minimal accuracy loss (16-bit float)
- Smaller than FP32 but maintains quality
- Works with Ollama and LM Studio

**Use with**: Ollama, LM Studio (if you need these tools)

---

## Method 1: Use HuggingFace Format (Recommended)

This preserves full model accuracy with no quantization.

### Step 1: Export to HuggingFace

```bash
python export_to_huggingface.py \
  --checkpoint checkpoints/final \
  --output models/mamba-hf
```

### Step 2: Deploy with vLLM (Production-Ready)

vLLM provides the fastest inference for HuggingFace models:

```bash
# Install vLLM
pip install vllm

# Serve with full precision
vllm serve models/mamba-hf \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16
```

**Use the API**:
```python
from vllm import LLM, SamplingParams

# Load model in FP16 (no quantization)
llm = LLM(
    model="models/mamba-hf",
    dtype="float16"  # Full precision
)

sampling_params = SamplingParams(
    temperature=0.8,
    top_k=50,
    max_tokens=200
)

outputs = llm.generate(["Your prompt here"], sampling_params)
print(outputs[0].outputs[0].text)
```

### Step 3: OpenAI-Compatible API

```bash
# Start server
vllm serve models/mamba-hf --dtype float16
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.completions.create(
    model="models/mamba-hf",
    prompt="Once upon a time",
    max_tokens=200,
    temperature=0.8
)

print(response.choices[0].text)
```

---

## Method 2: HuggingFace Transformers (Direct)

Use the model directly with HuggingFace Transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load in FP16 for efficiency (still full precision)
model = AutoModelForCausalLM.from_pretrained(
    "models/mamba-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("models/mamba-hf")

# Generate
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.8,
    do_sample=True,
    top_k=50
)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
```

---

## Method 3: GGUF FP16 (For Ollama/LM Studio)

If you must use Ollama or LM Studio, use **FP16 instead of quantized versions**.

### Step 1: Export to HuggingFace

```bash
python export_to_huggingface.py \
  --checkpoint checkpoints/final \
  --output models/mamba-hf
```

### Step 2: Convert to GGUF FP16 (No Quantization)

```bash
# Install prerequisites
pip install gguf
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && make && cd ..

# Convert to FP16 GGUF (NO quantization)
python llama.cpp/convert.py models/mamba-hf \
  --outfile models/mamba-fp16.gguf \
  --outtype f16
```

**Important**: Use `--outtype f16` for full precision, **NOT** Q4_K_M or other quantization.

### Step 3: Use with Ollama (FP16)

Create `Modelfile-fp16`:
```
FROM ./models/mamba-fp16.gguf

PARAMETER temperature 0.8
PARAMETER top_k 50
PARAMETER top_p 0.9

TEMPLATE """{{ .Prompt }}"""
```

Import and run:
```bash
ollama create mamba-fp16 -f Modelfile-fp16
ollama run mamba-fp16 "Your prompt here"
```

### Step 4: Use with LM Studio (FP16)

1. Open LM Studio
2. Import `models/mamba-fp16.gguf`
3. Load and use normally

**Note**: FP16 GGUF files are much larger than quantized versions, but preserve accuracy.

---

## Method 4: Custom FastAPI Server (Full Control)

Deploy your own API server with full precision control:

```python
# serve_fullprecision.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# Load model in FP16 (full precision, no quantization)
model = AutoModelForCausalLM.from_pretrained(
    "models/mamba-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("models/mamba-hf")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9

@app.post("/generate")
def generate(request: GenerateRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=True,
            top_k=request.top_k,
            top_p=request.top_p
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"text": text}

@app.get("/health")
def health():
    return {"status": "ok", "precision": "float16"}

# Run with: uvicorn serve_fullprecision:app --host 0.0.0.0 --port 8000
```

Start the server:
```bash
pip install fastapi uvicorn
uvicorn serve_fullprecision:app --host 0.0.0.0 --port 8000
```

Use the API:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_tokens": 200,
    "temperature": 0.8
  }'
```

---

## Precision Comparison

| Format | Accuracy | Size (Medium Model) | Speed | Memory |
|--------|----------|---------------------|-------|--------|
| **FP32** | 100% (reference) | ~600 MB | Slower | High |
| **FP16** | 99.9% | ~300 MB | Fast | Medium |
| **Q8_0** | ~98% | ~150 MB | Fast | Low |
| **Q5_K_M** | ~95% | ~100 MB | Faster | Low |
| **Q4_K_M** | ~90-93% | ~75 MB | Fastest | Lowest |

### Recommendations by Use Case

**Production (Quality Critical)**:
- Use **HuggingFace FP16** with vLLM
- Fastest inference + full accuracy

**Research/Development**:
- Use **HuggingFace FP16** with Transformers
- Direct PyTorch control

**Local GUI (Quality Critical)**:
- Use **GGUF FP16** with LM Studio
- No quantization loss

**Local CLI (Quality Critical)**:
- Use **GGUF FP16** with Ollama
- No quantization loss

**Resource-Constrained (Accept 5-10% Quality Loss)**:
- Only then consider Q5_K_M or Q8_0
- Never use Q4_K_M for production

---

## File Size Examples (Medium Model ~150M params)

```
Original PyTorch (.pt)         : ~600 MB (FP32)
HuggingFace FP16              : ~300 MB
HuggingFace FP32              : ~600 MB
GGUF FP16                     : ~300 MB
GGUF Q8_0                     : ~150 MB (2% accuracy loss)
GGUF Q5_K_M                   : ~100 MB (5% accuracy loss)
GGUF Q4_K_M                   : ~75 MB (7-10% accuracy loss)
```

---

## Performance Optimization (Without Quantization)

### 1. Use FP16 Instead of FP32

FP16 gives you:
- 50% smaller files
- 2x faster inference
- <0.1% accuracy loss (negligible)

```python
# FP16 is the sweet spot
model = AutoModelForCausalLM.from_pretrained(
    "models/mamba-hf",
    torch_dtype=torch.float16  # Use FP16
)
```

### 2. Enable Flash Attention (vLLM)

```bash
vllm serve models/mamba-hf \
  --dtype float16 \
  --enable-prefix-caching \
  --max-model-len 2048
```

### 3. Batch Processing

```python
# Process multiple prompts at once
prompts = [
    "Prompt 1",
    "Prompt 2",
    "Prompt 3"
]

outputs = llm.generate(prompts, sampling_params)
# Much faster than sequential generation
```

### 4. GPU Optimization

```python
# Use GPU if available
model = AutoModelForCausalLM.from_pretrained(
    "models/mamba-hf",
    torch_dtype=torch.float16,
    device_map="auto"  # Automatically use GPU
)
```

---

## Direct PyTorch (No Conversion)

For maximum control and no conversion overhead:

```python
import torch
from transformers import AutoTokenizer
from models.mamba_model import create_mamba_model
from configs.model_configs import load_config

# Load config
config = load_config("checkpoints/final/config.json")

# Create model
model = create_mamba_model(config['model'])

# Load weights in FP16
model.load_state_dict(
    torch.load("checkpoints/final/model.pt", map_location="cuda")
)
model = model.half()  # Convert to FP16
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Generate
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')

with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=200,
        temperature=0.8,
        top_k=50
    )

text = tokenizer.decode(output_ids[0])
print(text)
```

---

## Summary

### For Maximum Accuracy:

1. **Best Choice**: Export to HuggingFace → Use with vLLM (FP16)
   ```bash
   python export_to_huggingface.py --checkpoint checkpoints/final --output models/mamba-hf
   vllm serve models/mamba-hf --dtype float16
   ```

2. **Alternative**: GGUF FP16 for Ollama/LM Studio
   ```bash
   python llama.cpp/convert.py models/mamba-hf --outfile models/mamba-fp16.gguf --outtype f16
   ollama create mamba-fp16 -f Modelfile-fp16
   ```

3. **Development**: Direct PyTorch with FP16
   ```python
   model = model.half()  # FP16
   ```

### Avoid:
- ❌ Q4_K_M (7-10% accuracy loss)
- ❌ Q5_K_M (5% accuracy loss)
- ⚠️ Q8_0 (2% accuracy loss) - only if desperate for space

### Use FP16 Everywhere:
- ✅ Negligible accuracy loss (<0.1%)
- ✅ 50% size reduction
- ✅ 2x faster inference
- ✅ Supported by all tools

---

For complete deployment documentation, see **DEPLOYMENT_GUIDE.md**.
