# Using Your Trained Mamba Model - Quick Reference

After training completes, here's everything you can do with your model.

---

## Prerequisites

Your training has finished and you have a checkpoint at:
```
./checkpoints/reasoning_optimized/final/
├── model.pt          # Model weights
├── config.json       # Model configuration
└── optimizer.pt      # Optimizer state (for resuming)
```

---

## Feature 1: Interactive CLI Chat

**What it does**: Chat with your model in the terminal

**Command**:
```bash
cd /home/coreai/llm-projects/mamba_trainer
source venv/bin/activate
python inference.py --checkpoint ./checkpoints/reasoning_optimized/final
```

**Example Session**:
```
📝 You: What is 15% of 240?

🤔 Thinking...

🤖 Assistant: To find 15% of 240, multiply 240 by 0.15 = 36

📝 You: If x + 5 = 12, what is x?

🤔 Thinking...

🤖 Assistant: x = 7 (because 7 + 5 = 12)

📝 You: quit
👋 Goodbye!
```

**Commands**:
- `quit` / `exit` / `q` - Exit
- `clear` - Clear conversation history
- `help` - Show help

**Single prompt (non-interactive)**:
```bash
python inference.py \
    --checkpoint ./checkpoints/reasoning_optimized/final \
    --prompt "What is the square root of 144?" \
    --max_tokens 100 \
    --temperature 0.7
```

---

## Feature 2: API Server (REST API)

**What it does**: Expose your model as a REST API for applications

**Start Server**:
```bash
cd /home/coreai/llm-projects/mamba_trainer
source venv/bin/activate

# Install dependencies (first time only)
pip install fastapi uvicorn

# Start server
python api_server.py \
    --checkpoint ./checkpoints/reasoning_optimized/final \
    --port 8000 \
    --host 0.0.0.0 \
    --device cuda
```

**Server Output**:
```
Starting Mamba API Server
═══════════════════════════════════════
Checkpoint: ./checkpoints/reasoning_optimized/final
Host: 0.0.0.0
Port: 8000
Device: cuda
═══════════════════════════════════════

Loading model...
Model loaded: hybrid_recursive (112,809,496 parameters)
Ready for inference!

INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Test the API**:

```bash
# Health check
curl http://192.168.0.31:8000/health

# Simple generation
curl -X POST http://192.168.0.31:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is 2+2?",
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Chat completion (OpenAI-compatible)
curl -X POST http://192.168.0.31:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is 15% of 240?"}
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }'

# List models
curl http://192.168.0.31:8000/v1/models

# Model info
curl http://192.168.0.31:8000/model/info
```

**Run as background service**:

```bash
# Using nohup
nohup python api_server.py \
    --checkpoint ./checkpoints/reasoning_optimized/final \
    --port 8000 \
    --host 0.0.0.0 \
    --device cuda > api_server.log 2>&1 &

# Or use systemd (see OPENWEBUI_INTEGRATION.md for full setup)
```

**API Endpoints**:
- `GET /` - API info
- `GET /health` - Health check
- `POST /generate` - Simple text generation
- `POST /v1/chat/completions` - OpenAI-compatible chat
- `GET /v1/models` - List models
- `GET /model/info` - Model details

---

## Feature 3: Batch Testing

**What it does**: Test your model on multiple questions and save results

**Default questions**:
```bash
cd /home/coreai/llm-projects/mamba_trainer
source venv/bin/activate

python batch_test.py \
    --checkpoint ./checkpoints/reasoning_optimized/final \
    --output results.json
```

**Custom questions from file**:
```bash
# Create questions file
cat > my_questions.txt << EOF
What is 15% of 240?
If x + 5 = 12, what is x?
What is the area of a circle with radius 4?
Calculate: (8 + 2) × 5 - 3
What is 2^5?
EOF

# Run batch test
python batch_test.py \
    --checkpoint ./checkpoints/reasoning_optimized/final \
    --questions my_questions.txt \
    --output results.json \
    --max_tokens 200 \
    --temperature 0.7
```

**Example Output**:
```
════════════════════════════════════════════════════════════════════════════════
Batch Testing - 5 Questions
════════════════════════════════════════════════════════════════════════════════

[1/5] Question: What is 15% of 240?
Answer: 36
Time: 2.34s

────────────────────────────────────────────────────────────────────────────────

[2/5] Question: If x + 5 = 12, what is x?
Answer: x = 7
Time: 1.98s

────────────────────────────────────────────────────────────────────────────────

...

✅ Results saved to: results.json

════════════════════════════════════════════════════════════════════════════════
Summary
════════════════════════════════════════════════════════════════════════════════
Total questions: 5
Total time: 10.45s
Average time per question: 2.09s
════════════════════════════════════════════════════════════════════════════════
```

**Results JSON format**:
```json
{
  "timestamp": "2026-01-13T04:00:00Z",
  "model": "hybrid_recursive",
  "checkpoint": "./checkpoints/reasoning_optimized/final",
  "total_questions": 5,
  "results": [
    {
      "question": "What is 15% of 240?",
      "answer": "36",
      "full_response": "Question: What is 15% of 240?\nAnswer: 36",
      "time_seconds": 2.34
    }
  ]
}
```

---

## Feature 4: Benchmark Evaluation

**What it does**: Measure accuracy on standard datasets (GSM8K, MATH)

**Evaluate on GSM8K** (grade school math):
```bash
cd /home/coreai/llm-projects/mamba_trainer
source venv/bin/activate

python evaluate.py \
    --checkpoint ./checkpoints/reasoning_optimized/final \
    --dataset gsm8k \
    --split test \
    --num_examples 100 \
    --output gsm8k_results.json
```

**Evaluate on MATH** (competition math):
```bash
python evaluate.py \
    --checkpoint ./checkpoints/reasoning_optimized/final \
    --dataset math \
    --split test \
    --num_examples 50 \
    --output math_results.json
```

**Example Output**:
```
Loading GSM8K dataset (test split)...
Evaluating on 100 examples...

Evaluating: 100%|██████████| 100/100 [08:23<00:00,  5.03s/it]

Progress: 100/100 | Accuracy: 72.00%

════════════════════════════════════════════════════════════════════════════════
Evaluation Results
════════════════════════════════════════════════════════════════════════════════
Dataset: GSM8K
Split: test
Total Examples: 100
Correct: 72
Accuracy: 72.00%
════════════════════════════════════════════════════════════════════════════════

✅ Results saved to: gsm8k_results.json

Sample Results (first 5):

1. ✓ Question: Natalia sold clips to 48 of her friends in April...
   True: 192
   Predicted: 192

2. ✓ Question: Weng earns $12 an hour for babysitting...
   True: 21
   Predicted: 21

3. ✗ Question: Betty is saving money for a new wallet...
   True: 45
   Predicted: 42
```

**Full evaluation (all test examples)**:
```bash
# This will take several hours
python evaluate.py \
    --checkpoint ./checkpoints/reasoning_optimized/final \
    --dataset gsm8k \
    --split test \
    --output full_gsm8k_results.json
```

---

## Feature 5: Export for Deployment

**What it does**: Package your model for production deployment

**Export model**:
```bash
cd /home/coreai/llm-projects/mamba_trainer
source venv/bin/activate

python export_model.py \
    --checkpoint ./checkpoints/reasoning_optimized/final \
    --output ./deployment
```

**Example Output**:
```
════════════════════════════════════════════════════════════════════════════════
Mamba Model Exporter
════════════════════════════════════════════════════════════════════════════════
Checkpoint: ./checkpoints/reasoning_optimized/final
Output: ./deployment
════════════════════════════════════════════════════════════════════════════════

Loading model...
Model loaded: hybrid_recursive

Exporting model to: ./deployment

1. Exporting PyTorch weights...
   ✓ Saved to: ./deployment/model.pt

2. Exporting TorchScript model...
   ✓ Saved to: ./deployment/model_scripted.pt

3. Exporting configuration...
   ✓ Saved to: ./deployment/config.json

4. Generating model info...
   ✓ Saved to: ./deployment/model_info.json

5. Copying inference scripts...
   ✓ Copied: inference.py
   ✓ Copied: api_server.py
   ✓ Copied: batch_test.py
   ✓ Copied: evaluate.py

6. Copying model source code...
   ✓ Copied models directory

7. Creating deployment README...
   ✓ Saved to: ./deployment/README.md

8. Creating requirements.txt...
   ✓ Saved to: ./deployment/requirements.txt

════════════════════════════════════════════════════════════════════════════════
Export Complete!
════════════════════════════════════════════════════════════════════════════════
Model Type: hybrid_recursive
Parameters: 112,809,496
Model Size: 428.65 MB
Min VRAM: 13.4 GB

All files saved to: ./deployment
════════════════════════════════════════════════════════════════════════════════
```

**Deployment directory structure**:
```
deployment/
├── model.pt                    # PyTorch weights
├── model_scripted.pt          # TorchScript (production)
├── config.json                # Configuration
├── model_info.json            # Metadata
├── requirements.txt           # Dependencies
├── README.md                  # Deployment guide
├── scripts/
│   ├── inference.py
│   ├── api_server.py
│   ├── batch_test.py
│   └── evaluate.py
└── models/
    ├── mamba_model.py
    └── hybrid_recursive_mamba.py
```

**Deploy to another server**:
```bash
# Package and transfer
cd /home/coreai/llm-projects/mamba_trainer
tar -czf deployment.tar.gz deployment/
scp deployment.tar.gz user@production-server:/opt/models/

# On production server
cd /opt/models
tar -xzf deployment.tar.gz
cd deployment
pip install -r requirements.txt
python scripts/api_server.py --checkpoint .
```

---

## Feature 6: OpenWebUI Integration

**What it does**: Connect your model to OpenWebUI for a ChatGPT-like interface

See **OPENWEBUI_INTEGRATION.md** for complete setup guide.

**Quick Start**:

1. **Start API server**:
```bash
python api_server.py \
    --checkpoint ./checkpoints/reasoning_optimized/final \
    --port 8000 \
    --host 0.0.0.0 \
    --device cuda
```

2. **Start OpenWebUI** (Docker):
```bash
docker run -d \
  --name openwebui \
  -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://192.168.0.31:8000/v1 \
  -e OPENAI_API_KEY=dummy \
  -v open-webui:/app/backend/data \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

3. **Access OpenWebUI**:
   - Open browser: http://192.168.0.31:3000
   - Create account (first user = admin)
   - Select your Mamba model
   - Start chatting!

---

## Common Workflows

### Workflow 1: Quick Testing
```bash
# Interactive testing
python inference.py --checkpoint ./checkpoints/reasoning_optimized/final

# Ask a few questions
# Type 'quit' when done
```

### Workflow 2: Systematic Evaluation
```bash
# 1. Create test questions
cat > test_questions.txt << EOF
What is 15% of 240?
If x + 5 = 12, solve for x
What is the area of a circle with radius 4?
EOF

# 2. Run batch test
python batch_test.py \
    --checkpoint ./checkpoints/reasoning_optimized/final \
    --questions test_questions.txt \
    --output results.json

# 3. Evaluate on benchmark
python evaluate.py \
    --checkpoint ./checkpoints/reasoning_optimized/final \
    --dataset gsm8k \
    --num_examples 100 \
    --output gsm8k_results.json

# 4. Review results
cat results.json | jq '.results[] | {question, answer}'
cat gsm8k_results.json | jq '.evaluation | {accuracy, correct, total_examples}'
```

### Workflow 3: Production Deployment
```bash
# 1. Export model
python export_model.py \
    --checkpoint ./checkpoints/reasoning_optimized/final \
    --output ./deployment

# 2. Start API server as service
python api_server.py \
    --checkpoint ./checkpoints/reasoning_optimized/final \
    --port 8000 \
    --host 0.0.0.0 \
    --workers 4

# 3. Set up OpenWebUI
docker run -d \
  --name openwebui \
  -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://192.168.0.31:8000/v1 \
  ghcr.io/open-webui/open-webui:main

# 4. Test integration
curl http://192.168.0.31:8000/health
curl http://192.168.0.31:3000
```

### Workflow 4: Sharing with Team
```bash
# 1. Export model
python export_model.py \
    --checkpoint ./checkpoints/reasoning_optimized/final \
    --output ./deployment

# 2. Package
tar -czf mamba_model.tar.gz deployment/

# 3. Share
# Upload to cloud storage or share directly
# Team members can extract and run:
#   tar -xzf mamba_model.tar.gz
#   cd deployment
#   pip install -r requirements.txt
#   python scripts/api_server.py --checkpoint .
```

---

## Monitoring & Debugging

### Check Model Performance
```bash
# Quick test
python inference.py \
    --checkpoint ./checkpoints/reasoning_optimized/final \
    --prompt "What is 2+2?" \
    --max_tokens 50

# Batch accuracy test
python batch_test.py \
    --checkpoint ./checkpoints/reasoning_optimized/final \
    --output test_results.json

# Full benchmark
python evaluate.py \
    --checkpoint ./checkpoints/reasoning_optimized/final \
    --dataset gsm8k \
    --num_examples 100
```

### Monitor GPU Usage
```bash
# While running inference
watch -n 1 nvidia-smi

# Or in another terminal
nvidia-smi dmon -s puct -c 100
```

### Debug API Server
```bash
# Run with verbose logging
python api_server.py \
    --checkpoint ./checkpoints/reasoning_optimized/final \
    --port 8000 \
    --device cuda 2>&1 | tee api_debug.log

# Check logs
tail -f api_debug.log

# Test endpoints
curl -v http://localhost:8000/health
curl -v http://localhost:8000/model/info
```

---

## Tips & Best Practices

### For Inference
- Use **temperature 0.1-0.3** for math/logic (more deterministic)
- Use **temperature 0.7-1.0** for creative tasks
- Set **max_tokens 200-300** for most questions
- Use **top_k 50** for balanced diversity

### For API Server
- Use **CUDA** (`--device cuda`) for GPU acceleration
- Run as **background service** for production
- Enable **HTTPS** with reverse proxy (Nginx/Caddy)
- Add **rate limiting** for public APIs
- Monitor with **Prometheus + Grafana** (optional)

### For Batch Testing
- Start with **small batches** (10-20 questions) to verify
- Use **lower temperature** (0.3) for consistent results
- Save results to **JSON** for analysis
- Use **multiple seeds** for ensemble testing

### For Benchmarks
- **GSM8K**: Best for grade school math (easier)
- **MATH**: Best for competition math (harder)
- Start with **100 examples** for quick evaluation
- Run **full dataset** for official metrics
- Compare with **baseline models** for context

---

## Summary

You now have 6 powerful features:

1. ✅ **Interactive CLI** - Quick testing and debugging
2. ✅ **API Server** - REST API for applications
3. ✅ **Batch Testing** - Systematic evaluation
4. ✅ **Benchmark Evaluation** - Standard metrics (GSM8K, MATH)
5. ✅ **Export for Deployment** - Production-ready packaging
6. ✅ **OpenWebUI Integration** - ChatGPT-like interface

**Next Steps**:
1. Test your model with interactive CLI
2. Run batch tests to verify quality
3. Evaluate on GSM8K benchmark
4. Set up API server + OpenWebUI for daily use
5. Export and share with your team

Enjoy your trained Mamba model! 🎉
