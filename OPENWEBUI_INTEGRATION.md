# OpenWebUI Integration Guide

Complete guide to integrate your trained Mamba model with OpenWebUI.

---

## What is OpenWebUI?

**OpenWebUI** (formerly Ollama WebUI) is a self-hosted, feature-rich web interface for LLMs. It provides:
- ChatGPT-like interface
- Multi-model support
- Chat history
- Model switching
- RAG (Retrieval Augmented Generation)
- User management

**Official site**: https://docs.openwebui.com/

---

## Prerequisites

1. ✅ Trained Mamba model checkpoint
2. ✅ Python 3.8+ with PyTorch
3. ✅ OpenWebUI installed
4. ✅ Network access between OpenWebUI and API server

---

## Step 1: Start Mamba API Server

The Mamba API server implements OpenAI-compatible endpoints that OpenWebUI can connect to.

### Start the Server

```bash
cd /home/coreai/llm-projects/mamba_trainer
source venv/bin/activate

python api_server.py \
    --checkpoint ./checkpoints/reasoning_optimized/final \
    --port 8000 \
    --host 0.0.0.0 \
    --device cuda
```

### Verify Server is Running

```bash
# Test health endpoint
curl http://192.168.0.31:8000/health

# Expected response:
# {"status":"healthy","model_loaded":true,"device":"cuda"}

# List available models
curl http://192.168.0.31:8000/v1/models

# Test generation
curl -X POST http://192.168.0.31:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is 2+2?", "max_tokens": 50}'
```

### Run as Background Service (Recommended)

Create a systemd service file:

```bash
sudo nano /etc/systemd/system/mamba-api.service
```

Add this content:

```ini
[Unit]
Description=Mamba API Server
After=network.target

[Service]
Type=simple
User=coreai
WorkingDirectory=/home/coreai/llm-projects/mamba_trainer
Environment="PATH=/home/coreai/llm-projects/mamba_trainer/venv/bin:/usr/bin"
ExecStart=/home/coreai/llm-projects/mamba_trainer/venv/bin/python api_server.py \
    --checkpoint ./checkpoints/reasoning_optimized/final \
    --port 8000 \
    --host 0.0.0.0 \
    --device cuda
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable mamba-api
sudo systemctl start mamba-api

# Check status
sudo systemctl status mamba-api

# View logs
sudo journalctl -u mamba-api -f
```

---

## Step 2: Install OpenWebUI

### Option A: Docker (Recommended)

```bash
docker run -d \
  --name openwebui \
  -p 3000:8080 \
  -v open-webui:/app/backend/data \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

Access at: http://localhost:3000

### Option B: Python Installation

```bash
pip install open-webui
open-webui serve
```

Access at: http://localhost:8080

### Option C: Build from Source

```bash
git clone https://github.com/open-webui/open-webui.git
cd open-webui
npm install
npm run build
```

---

## Step 3: Configure OpenWebUI to Use Mamba API

### Method 1: Using OpenWebUI Admin Panel

1. **Open OpenWebUI** in your browser (http://localhost:3000 or http://localhost:8080)

2. **Create Admin Account** (first time)
   - Sign up with email and password
   - First user becomes admin

3. **Go to Settings** (click gear icon ⚙️)

4. **Navigate to "Connections"** or "External Connections"

5. **Add OpenAI API Connection**:
   - **API Base URL**: `http://192.168.0.31:8000/v1`
   - **API Key**: `dummy` (not used, but required field)
   - **Model**: Leave blank (will auto-discover)

6. **Save and Test Connection**

7. **Select Model**:
   - Go to chat interface
   - Click model selector dropdown
   - Choose `mamba-hybrid_recursive` (or your model name)

8. **Start Chatting!** 🎉

### Method 2: Using Environment Variables

Set these before starting OpenWebUI:

```bash
export OPENAI_API_BASE_URL=http://192.168.0.31:8000/v1
export OPENAI_API_KEY=dummy

# Start OpenWebUI
docker run -d \
  --name openwebui \
  -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://192.168.0.31:8000/v1 \
  -e OPENAI_API_KEY=dummy \
  -v open-webui:/app/backend/data \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

### Method 3: Configuration File

Create or edit `~/.open-webui/config.json`:

```json
{
  "openai": {
    "api_base": "http://192.168.0.31:8000/v1",
    "api_key": "dummy"
  },
  "models": [
    {
      "id": "mamba-hybrid_recursive",
      "name": "Mamba Reasoning Model",
      "provider": "openai"
    }
  ]
}
```

---

## Step 4: Test the Integration

### In OpenWebUI Chat Interface

1. **Select your Mamba model** from the dropdown

2. **Ask a math question**:
   ```
   User: What is 15% of 240?

   Mamba: Let me solve this step by step.
   Step 1: Convert 15% to decimal: 0.15
   Step 2: Multiply: 0.15 × 240 = 36
   Answer: 36
   ```

3. **Try a coding question**:
   ```
   User: Write a Python function to check if a number is prime

   Mamba: Here's a function to check if a number is prime:

   def is_prime(n):
       if n <= 1:
           return False
       for i in range(2, int(n**0.5) + 1):
           if n % i == 0:
               return False
       return True
   ```

4. **Test reasoning**:
   ```
   User: If A > B and B > C, what's the relationship between A and C?

   Mamba: Since A > B and B > C, we can conclude that A > C.
   This is the transitive property of inequality.
   ```

---

## Advanced Configuration

### Custom Model Parameters in OpenWebUI

You can adjust these in the chat interface:

1. **Temperature** (0.0 - 2.0)
   - 0.1-0.3: Deterministic, best for math/logic
   - 0.7: Balanced (default)
   - 1.0+: More creative

2. **Max Tokens** (1 - 2048)
   - Controls response length
   - Default: 200

3. **Top-k Sampling** (1 - 100)
   - Default: 50
   - Lower = more focused
   - Higher = more diverse

### Multiple Models

You can run multiple Mamba models and switch between them:

```bash
# Model 1: Math reasoning (port 8000)
python api_server.py \
    --checkpoint ./checkpoints/math_model/final \
    --port 8000 \
    --device cuda

# Model 2: Code generation (port 8001)
python api_server.py \
    --checkpoint ./checkpoints/code_model/final \
    --port 8001 \
    --device cuda
```

Add both to OpenWebUI:
- Math Model: `http://192.168.0.31:8000/v1`
- Code Model: `http://192.168.0.31:8001/v1`

### Load Balancing with Nginx

For production, use Nginx to load balance multiple API servers:

```nginx
upstream mamba_api {
    server 192.168.0.31:8000;
    server 192.168.0.31:8001;
    server 192.168.0.31:8002;
}

server {
    listen 80;
    server_name mamba-api.local;

    location / {
        proxy_pass http://mamba_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## API Endpoints Reference

The Mamba API server provides these OpenAI-compatible endpoints:

### 1. Chat Completions (OpenWebUI uses this)

```bash
POST /v1/chat/completions
```

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "What is 2+2?"}
  ],
  "max_tokens": 100,
  "temperature": 0.7,
  "top_k": 50
}
```

**Response:**
```json
{
  "message": {
    "role": "assistant",
    "content": "2+2 equals 4."
  },
  "model": "mamba-hybrid_recursive",
  "created": 1704844800,
  "done": true
}
```

### 2. List Models

```bash
GET /v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "mamba-hybrid_recursive",
      "object": "model",
      "created": 1704844800,
      "owned_by": "local",
      "parameters": 112809496,
      "architecture": "hybrid_recursive"
    }
  ]
}
```

### 3. Simple Generation

```bash
POST /generate
```

**Request:**
```json
{
  "prompt": "What is the capital of France?",
  "max_tokens": 50,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "prompt": "What is the capital of France?",
  "generated_text": "The capital of France is Paris.",
  "full_text": "What is the capital of France? The capital of France is Paris.",
  "tokens_generated": 8,
  "model": "hybrid_recursive",
  "timestamp": "2024-01-10T12:00:00Z"
}
```

### 4. Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### 5. Model Info

```bash
GET /model/info
```

**Response:**
```json
{
  "name": "mamba-hybrid_recursive",
  "type": "hybrid_recursive",
  "parameters": 112809496,
  "trainable_parameters": 112809496,
  "architecture": {
    "d_model": 608,
    "n_layers": 10,
    "vocab_size": 50257,
    "max_seq_len": 1408
  },
  "device": "cuda"
}
```

---

## Troubleshooting

### Issue: OpenWebUI can't connect to Mamba API

**Check 1: Is the API server running?**
```bash
curl http://192.168.0.31:8000/health
```

**Check 2: Firewall blocking port 8000?**
```bash
sudo ufw allow 8000
# or
sudo firewall-cmd --add-port=8000/tcp --permanent
sudo firewall-cmd --reload
```

**Check 3: Correct IP address?**
```bash
# Find server IP
hostname -I
```

**Check 4: Check API server logs**
```bash
sudo journalctl -u mamba-api -f
```

---

### Issue: Model loads but gives errors

**Check 1: Model compatibility**
```bash
# Verify model loaded correctly
curl http://192.168.0.31:8000/model/info
```

**Check 2: GPU memory**
```bash
nvidia-smi
# Make sure model fits in VRAM
```

**Check 3: Test generation directly**
```bash
curl -X POST http://192.168.0.31:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test", "max_tokens": 10}'
```

---

### Issue: Slow responses

**Solution 1: Check GPU utilization**
```bash
watch -n 1 nvidia-smi
# GPU should be near 100% during generation
```

**Solution 2: Reduce max_tokens**
- In OpenWebUI settings, set max tokens to 150-200

**Solution 3: Use smaller model**
- Train or use `hybrid-small` instead of `hybrid-small-standard`

**Solution 4: Check CPU bottleneck**
```bash
htop
# If CPU is maxed, you may need better hardware
```

---

### Issue: OpenWebUI shows "Model not found"

**Solution 1: Restart OpenWebUI**
```bash
docker restart openwebui
# or
sudo systemctl restart open-webui
```

**Solution 2: Clear OpenWebUI cache**
```bash
docker exec openwebui rm -rf /app/backend/data/cache
docker restart openwebui
```

**Solution 3: Manually add model**
- Go to OpenWebUI settings
- Add model ID: `mamba-hybrid_recursive`
- Set base URL: `http://192.168.0.31:8000/v1`

---

## Security Considerations

### 1. API Key Authentication (Optional)

Modify `api_server.py` to require API keys:

```python
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != "your-secret-key":
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

@app.post("/generate", dependencies=[Depends(verify_api_key)])
async def generate(request: GenerateRequest):
    # ... existing code
```

### 2. Rate Limiting

Install and configure rate limiting:

```bash
pip install slowapi

# In api_server.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/generate")
@limiter.limit("10/minute")  # 10 requests per minute
async def generate(request: Request, ...):
    # ... existing code
```

### 3. HTTPS/TLS

Use Nginx or Caddy as reverse proxy with SSL:

```bash
# Install Caddy
sudo apt install caddy

# Configure /etc/caddy/Caddyfile
mamba-api.yourdomain.com {
    reverse_proxy localhost:8000
}

# Restart
sudo systemctl restart caddy
```

---

## Performance Optimization

### 1. Use FP16 Inference

Modify model loading to use half precision:

```python
model = model.half()  # Convert to FP16
# 2x faster, 2x less VRAM
```

### 2. Batch Multiple Requests

If using multiple workers, enable batching in `api_server.py`.

### 3. Model Quantization

Quantize model to INT8:

```python
import torch.quantization as quantization

model = quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
# 4x smaller, slightly faster
```

### 4. Use Multiple GPUs

Run multiple API servers, one per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python api_server.py --port 8000 &
CUDA_VISIBLE_DEVICES=1 python api_server.py --port 8001 &
```

---

## Complete Setup Script

Here's a complete script to set everything up:

```bash
#!/bin/bash

# Mamba + OpenWebUI Complete Setup

# 1. Start Mamba API Server
cd /home/coreai/llm-projects/mamba_trainer
source venv/bin/activate

echo "Starting Mamba API server..."
python api_server.py \
    --checkpoint ./checkpoints/reasoning_optimized/final \
    --port 8000 \
    --host 0.0.0.0 \
    --device cuda &

MAMBA_PID=$!
echo "Mamba API PID: $MAMBA_PID"

# Wait for server to start
sleep 10

# 2. Test API
echo "Testing API..."
curl -s http://localhost:8000/health | jq .

# 3. Start OpenWebUI with Docker
echo "Starting OpenWebUI..."
docker run -d \
  --name openwebui \
  -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://192.168.0.31:8000/v1 \
  -e OPENAI_API_KEY=dummy \
  -v open-webui:/app/backend/data \
  --restart always \
  ghcr.io/open-webui/open-webui:main

# 4. Wait for OpenWebUI to start
echo "Waiting for OpenWebUI to start..."
sleep 15

# 5. Show status
echo ""
echo "==============================================="
echo "Setup Complete!"
echo "==============================================="
echo "Mamba API: http://192.168.0.31:8000"
echo "OpenWebUI: http://192.168.0.31:3000"
echo ""
echo "1. Open http://192.168.0.31:3000 in browser"
echo "2. Create an account (first user = admin)"
echo "3. Select 'mamba-hybrid_recursive' model"
echo "4. Start chatting!"
echo "==============================================="
```

Save as `start_openwebui.sh` and run:

```bash
chmod +x start_openwebui.sh
./start_openwebui.sh
```

---

## Summary

✅ **Mamba API Server**: Runs your trained model with OpenAI-compatible endpoints
✅ **OpenWebUI**: Provides ChatGPT-like interface
✅ **Integration**: Connect OpenWebUI to Mamba API
✅ **Features**: Chat, history, model switching, RAG
✅ **Production Ready**: Systemd service, monitoring, security

Your Mamba model is now accessible through a beautiful web interface! 🎉
