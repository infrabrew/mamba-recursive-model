# Deployment Guide - Using Your Trained Mamba Model

This guide shows how to use your trained Mamba model with popular inference tools like Ollama, vLLM, and LM Studio.

## Overview

The Mamba trainer saves models in **PyTorch format** (`.pt` files). To use them with inference tools, you need to convert them:

```
Mamba Checkpoint (.pt)
    ↓
HuggingFace Format
    ↓
GGUF Format (for Ollama/LM Studio)
```

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Export to HuggingFace Format (vLLM)](#export-to-huggingface-format)
3. [Export to GGUF Format (Ollama/LM Studio)](#export-to-gguf-format)
4. [Using with vLLM](#using-with-vllm)
5. [Using with Ollama](#using-with-ollama)
6. [Using with LM Studio](#using-with-lm-studio)
7. [Direct Inference (Python)](#direct-inference-python)

---

## Quick Reference

```bash
# Step 1: Export to HuggingFace format
python export_to_huggingface.py \
  --checkpoint checkpoints/final \
  --output models/mamba-hf

# Step 2a: Use with vLLM (HuggingFace format)
vllm serve models/mamba-hf

# Step 2b: Export to GGUF for Ollama/LM Studio
python export_to_gguf.py \
  --hf_model models/mamba-hf \
  --output models/mamba.gguf \
  --quantization Q4_K_M
```

---

## Export to HuggingFace Format

### Step 1: Export Your Model

```bash
python export_to_huggingface.py \
  --checkpoint checkpoints/final \
  --output models/mamba-hf
```

**What this does**:
- Converts PyTorch checkpoint to HuggingFace format
- Adds tokenizer (GPT-2 by default)
- Creates model card and README
- Makes model compatible with Transformers library

**Output**:
```
models/mamba-hf/
├── config.json           # Model configuration
├── pytorch_model.bin     # Model weights
├── tokenizer.json        # Tokenizer
├── tokenizer_config.json # Tokenizer config
└── README.md            # Model card
```

### Step 2: Test the Export

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("models/mamba-hf")
tokenizer = AutoTokenizer.from_pretrained("models/mamba-hf")

# Generate text
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### Step 3: (Optional) Upload to HuggingFace Hub

```python
from huggingface_hub import HfApi

api = HfApi()

# Login first
# huggingface-cli login

# Upload
model.push_to_hub("your-username/mamba-model")
tokenizer.push_to_hub("your-username/mamba-model")
```

---

## Export to GGUF Format

GGUF format is required for Ollama and LM Studio.

### Prerequisites

```bash
# Install gguf package
pip install gguf

# Clone llama.cpp (required for conversion)
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make
cd ..
```

### Step 1: Export to HuggingFace First

```bash
# Must export to HuggingFace format first
python export_to_huggingface.py \
  --checkpoint checkpoints/final \
  --output models/mamba-hf
```

### Step 2: Convert to GGUF

```bash
# Convert HuggingFace → GGUF (FP16)
python llama.cpp/convert.py models/mamba-hf \
  --outfile models/mamba-f16.gguf \
  --outtype f16
```

### Step 3: Quantize (Optional but Recommended)

Quantization reduces model size significantly:

```bash
# Quantize to Q4_K_M (recommended, ~4GB for medium model)
./llama.cpp/quantize models/mamba-f16.gguf \
  models/mamba-Q4_K_M.gguf \
  Q4_K_M
```

**Quantization Options**:
- `Q4_K_M` - 4-bit, medium quality (recommended, smallest)
- `Q5_K_M` - 5-bit, high quality (balanced)
- `Q8_0` - 8-bit, highest quality (larger)
- `f16` - 16-bit float (no quantization, largest)

### Using the Helper Script

```bash
# All-in-one conversion
python export_to_gguf.py \
  --hf_model models/mamba-hf \
  --output models/mamba.gguf \
  --quantization Q4_K_M
```

---

## Using with vLLM

vLLM is optimized for fast inference and high throughput.

### Installation

```bash
pip install vllm
```

### Method 1: Serve via API

```bash
# Serve the model
vllm serve models/mamba-hf \
  --host 0.0.0.0 \
  --port 8000
```

**Use the API**:
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "models/mamba-hf",
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "temperature": 0.8
  }'
```

### Method 2: Python API

```python
from vllm import LLM, SamplingParams

# Load model
llm = LLM(model="models/mamba-hf")

# Configure sampling
sampling_params = SamplingParams(
    temperature=0.8,
    top_k=50,
    max_tokens=100
)

# Generate
prompts = ["Once upon a time", "The future of AI"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

### Method 3: OpenAI-Compatible API

```bash
# Start server
vllm serve models/mamba-hf \
  --host 0.0.0.0 \
  --port 8000
```

```python
# Use with OpenAI client
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.completions.create(
    model="models/mamba-hf",
    prompt="Once upon a time",
    max_tokens=100
)

print(response.choices[0].text)
```

---

## Using with Ollama

Ollama provides a simple interface for running models locally.

### Installation

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

### Step 1: Create Modelfile

Create `Modelfile`:
```
FROM ./models/mamba-Q4_K_M.gguf

PARAMETER temperature 0.8
PARAMETER top_k 50
PARAMETER top_p 0.9

TEMPLATE """{{ .Prompt }}"""

SYSTEM """You are a helpful AI assistant."""
```

### Step 2: Import Model

```bash
# Create the model in Ollama
ollama create mamba -f Modelfile
```

### Step 3: Run the Model

```bash
# Interactive chat
ollama run mamba

# Or one-shot generation
ollama run mamba "Once upon a time"
```

### Step 4: Use via API

```bash
# Start Ollama server
ollama serve
```

```python
# Python client
import requests

response = requests.post('http://localhost:11434/api/generate', json={
    'model': 'mamba',
    'prompt': 'Once upon a time',
    'stream': False
})

print(response.json()['response'])
```

---

## Using with LM Studio

LM Studio provides a GUI for running GGUF models.

### Step 1: Install LM Studio

Download from [lmstudio.ai](https://lmstudio.ai)

### Step 2: Import Model

1. Open LM Studio
2. Click **"Import Model"**
3. Select your GGUF file: `models/mamba-Q4_K_M.gguf`
4. Wait for import to complete

### Step 3: Load and Chat

1. Go to **"Chat"** tab
2. Select your model from dropdown
3. Configure parameters:
   - Temperature: 0.8
   - Top-K: 50
   - Max Length: 2048
4. Start chatting!

### Step 4: Use Local Server

LM Studio can run a local API server:

1. Go to **"Server"** tab
2. Click **"Start Server"**
3. Use the OpenAI-compatible API:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="mamba",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

---

## Direct Inference (Python)

Use the model directly without conversion.

### Option 1: Using generate.py

```bash
# Use the included generation script
python generate.py \
  --checkpoint checkpoints/final \
  --prompt "Once upon a time" \
  --temperature 0.8 \
  --max_tokens 200
```

### Option 2: Python Script

```python
import torch
from transformers import AutoTokenizer
from models.mamba_model import create_mamba_model
from configs.model_configs import load_config

# Load model
config = load_config("checkpoints/final/config.json")
model = create_mamba_model(config['model'])
model.load_state_dict(torch.load("checkpoints/final/model.pt"))
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Generate
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=100,
        temperature=0.8,
        top_k=50
    )

text = tokenizer.decode(output_ids[0])
print(text)
```

### Option 3: REST API with FastAPI

Create `serve.py`:
```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
from models.mamba_model import create_mamba_model
from configs.model_configs import load_config

app = FastAPI()

# Load model at startup
config = load_config("checkpoints/final/config.json")
model = create_mamba_model(config['model'])
model.load_state_dict(torch.load("checkpoints/final/model.pt"))
model.eval()
tokenizer = AutoTokenizer.from_pretrained('gpt2')

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 50

@app.post("/generate")
def generate(request: GenerateRequest):
    input_ids = tokenizer.encode(request.prompt, return_tensors='pt')

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k
        )

    text = tokenizer.decode(output_ids[0])
    return {"text": text}

# Run with: uvicorn serve:app --host 0.0.0.0 --port 8000
```

---

## Comparison of Deployment Options

| Tool | Format | Speed | Memory | Use Case |
|------|--------|-------|--------|----------|
| **vLLM** | HuggingFace | Very Fast | High | Production API, high throughput |
| **Ollama** | GGUF | Fast | Low | Local use, easy setup |
| **LM Studio** | GGUF | Medium | Low | GUI, experimentation |
| **Direct (Python)** | PyTorch | Medium | Medium | Development, custom logic |

## Performance Tips

1. **Use Quantization**: Q4_K_M reduces size by ~75% with minimal quality loss
2. **GPU Acceleration**: All tools support CUDA for faster inference
3. **Batch Processing**: vLLM excels at batched requests
4. **Model Size**: Start with quantized models for faster loading

## Troubleshooting

### Export Fails

**Problem**: `export_to_huggingface.py` errors
**Solution**: Check that checkpoint has all required files:
```bash
ls checkpoints/final/
# Should have: model.pt, config.json, optimizer.pt, training_state.pt
```

### GGUF Conversion Fails

**Problem**: llama.cpp conversion errors
**Solution**:
- Update llama.cpp: `git pull` in llama.cpp directory
- Ensure HuggingFace export completed successfully
- Try without quantization first (f16)

### Model Doesn't Load in Ollama

**Problem**: Ollama can't load GGUF
**Solution**:
- Verify GGUF file is valid: `ls -lh models/mamba-Q4_K_M.gguf`
- Check Modelfile syntax
- Try re-creating: `ollama create mamba -f Modelfile`

### Slow Inference

**Problem**: Generation is slow
**Solution**:
- Use GPU if available
- Use quantized model (Q4_K_M)
- For vLLM, increase `--tensor-parallel-size`
- Reduce `max_tokens`

---

## Next Steps

1. Export your trained model to HuggingFace format
2. Choose your deployment tool (vLLM for production, Ollama for local)
3. Convert to GGUF if using Ollama/LM Studio
4. Test generation with sample prompts
5. Deploy to your application

For more information, see:
- **USAGE_GUIDE.md** - Training and generation examples
- **README.md** - Full documentation
- **generate.py** - Direct inference script
