#!/usr/bin/env python3
"""
Export trained Mamba models for production deployment.

Usage:
    python export_model.py --checkpoint ./checkpoints/reasoning_optimized/final --output ./deployment
"""

import torch
from models.mamba_model import create_mamba_model
from models.hybrid_recursive_mamba import create_hybrid_recursive_mamba_model
from transformers import GPT2Tokenizer
import json
import argparse
import os
import shutil
from datetime import datetime


class ModelExporter:
    """Export Mamba models for deployment."""

    def __init__(self, checkpoint_dir: str, device: str = 'cpu'):
        """Initialize exporter."""
        self.checkpoint_dir = checkpoint_dir
        self.device = device

        # Load configuration
        config_path = os.path.join(checkpoint_dir, 'config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Load model
        print("Loading model...")
        model_type = self.config['model'].get('model_type', 'standard')

        if model_type == 'hybrid_recursive':
            self.model = create_hybrid_recursive_mamba_model(self.config['model'])
        else:
            self.model = create_mamba_model(self.config['model'])

        model_path = os.path.join(checkpoint_dir, 'model.pt')
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()

        print(f"Model loaded: {model_type}\n")

    def export(self, output_dir: str):
        """
        Export model for production deployment.

        Args:
            output_dir: Directory to save exported model
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"Exporting model to: {output_dir}\n")

        # 1. Export PyTorch model weights
        print("1. Exporting PyTorch weights...")
        model_output = os.path.join(output_dir, 'model.pt')
        torch.save(self.model.state_dict(), model_output)
        print(f"   ✓ Saved to: {model_output}")

        # 2. Export TorchScript model (for production)
        print("\n2. Exporting TorchScript model...")
        try:
            scripted_model = torch.jit.script(self.model)
            script_output = os.path.join(output_dir, 'model_scripted.pt')
            torch.jit.save(scripted_model, script_output)
            print(f"   ✓ Saved to: {script_output}")
        except Exception as e:
            print(f"   ⚠ TorchScript export failed: {e}")
            print("   ℹ Standard PyTorch model still available")

        # 3. Export configuration
        print("\n3. Exporting configuration...")
        config_output = os.path.join(output_dir, 'config.json')
        with open(config_output, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"   ✓ Saved to: {config_output}")

        # 4. Export model info
        print("\n4. Generating model info...")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        model_info = {
            'export_timestamp': datetime.utcnow().isoformat(),
            'checkpoint_source': self.checkpoint_dir,
            'model_type': self.config['model'].get('model_type', 'standard'),
            'architecture': {
                'd_model': self.config['model'].get('d_model'),
                'n_layers': self.config['model'].get('n_layers'),
                'vocab_size': self.config['model'].get('vocab_size'),
                'max_seq_len': self.config['model'].get('max_seq_len'),
                'd_state': self.config['model'].get('d_state'),
                'expand_factor': self.config['model'].get('expand_factor')
            },
            'parameters': {
                'total': total_params,
                'trainable': trainable_params
            },
            'size_mb': os.path.getsize(model_output) / (1024 * 1024),
            'deployment': {
                'pytorch_model': 'model.pt',
                'torchscript_model': 'model_scripted.pt',
                'config': 'config.json',
                'recommended_device': 'cuda',
                'min_vram_gb': self._estimate_vram_requirement()
            }
        }

        info_output = os.path.join(output_dir, 'model_info.json')
        with open(info_output, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"   ✓ Saved to: {info_output}")

        # 5. Copy inference scripts
        print("\n5. Copying inference scripts...")
        scripts_to_copy = [
            'inference.py',
            'api_server.py',
            'batch_test.py',
            'evaluate.py'
        ]

        scripts_dir = os.path.join(output_dir, 'scripts')
        os.makedirs(scripts_dir, exist_ok=True)

        for script in scripts_to_copy:
            if os.path.exists(script):
                shutil.copy(script, os.path.join(scripts_dir, script))
                print(f"   ✓ Copied: {script}")

        # 6. Copy model source code
        print("\n6. Copying model source code...")
        models_dir = os.path.join(output_dir, 'models')
        if os.path.exists('models'):
            shutil.copytree('models', models_dir, dirs_exist_ok=True)
            print(f"   ✓ Copied models directory")

        # 7. Create README
        print("\n7. Creating deployment README...")
        readme_content = self._generate_readme(model_info)
        readme_output = os.path.join(output_dir, 'README.md')
        with open(readme_output, 'w') as f:
            f.write(readme_content)
        print(f"   ✓ Saved to: {readme_output}")

        # 8. Create requirements.txt
        print("\n8. Creating requirements.txt...")
        requirements = [
            'torch>=2.0.0',
            'transformers>=4.30.0',
            'fastapi>=0.100.0',
            'uvicorn>=0.23.0',
            'datasets>=2.14.0',
            'numpy>=1.24.0',
            'tqdm>=4.65.0',
            'pydantic>=2.0.0'
        ]
        requirements_output = os.path.join(output_dir, 'requirements.txt')
        with open(requirements_output, 'w') as f:
            f.write('\n'.join(requirements))
        print(f"   ✓ Saved to: {requirements_output}")

        # Summary
        print("\n" + "="*80)
        print("Export Complete!")
        print("="*80)
        print(f"Model Type: {model_info['model_type']}")
        print(f"Parameters: {model_info['parameters']['total']:,}")
        print(f"Model Size: {model_info['size_mb']:.2f} MB")
        print(f"Min VRAM: {model_info['deployment']['min_vram_gb']:.1f} GB")
        print(f"\nAll files saved to: {output_dir}")
        print("="*80)

    def _estimate_vram_requirement(self) -> float:
        """Estimate VRAM requirement in GB."""
        total_params = sum(p.numel() for p in self.model.parameters())

        # Rough estimate: params * 4 bytes (fp32) * 1.5 (overhead)
        bytes_required = total_params * 4 * 1.5
        gb_required = bytes_required / (1024**3)

        return round(gb_required, 1)

    def _generate_readme(self, model_info: dict) -> str:
        """Generate deployment README."""
        return f"""# Mamba Model Deployment

## Model Information

- **Model Type**: {model_info['model_type']}
- **Total Parameters**: {model_info['parameters']['total']:,}
- **Model Size**: {model_info['size_mb']:.2f} MB
- **Exported**: {model_info['export_timestamp']}

## Architecture

- **d_model**: {model_info['architecture']['d_model']}
- **Layers**: {model_info['architecture']['n_layers']}
- **Vocabulary**: {model_info['architecture']['vocab_size']}
- **Max Sequence Length**: {model_info['architecture']['max_seq_len']}

## System Requirements

- **Minimum VRAM**: {model_info['deployment']['min_vram_gb']:.1f} GB
- **Recommended Device**: {model_info['deployment']['recommended_device']}
- **Python**: 3.8+
- **PyTorch**: 2.0+

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {{torch.__version__}}')"
```

## Usage

### 1. Interactive CLI

```bash
python scripts/inference.py --checkpoint .
```

### 2. API Server

```bash
# Start server
python scripts/api_server.py --checkpoint . --port 8000

# Test endpoint
curl -X POST http://localhost:8000/generate \\
  -H "Content-Type: application/json" \\
  -d '{{"prompt": "What is 2+2?", "max_tokens": 100}}'
```

### 3. Python Integration

```python
import torch
from models.mamba_model import create_mamba_model
from transformers import GPT2Tokenizer
import json

# Load config
with open('config.json') as f:
    config = json.load(f)

# Load model
model = create_mamba_model(config['model'])
model.load_state_dict(torch.load('model.pt'))
model.eval()

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Generate
input_ids = tokenizer.encode("Question: What is 2+2?", return_tensors='pt')
output_ids = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(output_ids[0]))
```

### 4. Batch Testing

```bash
# Create test questions file
echo "What is 15% of 240?" > questions.txt
echo "If x + 5 = 12, what is x?" >> questions.txt

# Run batch test
python scripts/batch_test.py --checkpoint . --questions questions.txt
```

### 5. Benchmark Evaluation

```bash
# Evaluate on GSM8K
python scripts/evaluate.py \\
  --checkpoint . \\
  --dataset gsm8k \\
  --num_examples 100
```

## OpenWebUI Integration

This model is compatible with OpenWebUI using the API server:

1. Start the API server:
```bash
python scripts/api_server.py --checkpoint . --port 8000 --host 0.0.0.0
```

2. In OpenWebUI settings:
   - Add API endpoint: `http://your-server-ip:8000`
   - Select model: `mamba-{model_info['model_type']}`
   - Start chatting!

The API implements OpenAI-compatible endpoints:
- `POST /v1/chat/completions` - Chat completions
- `GET /v1/models` - List models
- `GET /health` - Health check

## Files Included

- `model.pt` - PyTorch model weights
- `model_scripted.pt` - TorchScript model (production-ready)
- `config.json` - Model configuration
- `model_info.json` - Detailed model information
- `requirements.txt` - Python dependencies
- `scripts/` - Inference and evaluation scripts
- `models/` - Model source code

## Performance Tips

1. **GPU Acceleration**: Always use CUDA when available
   ```bash
   python scripts/api_server.py --checkpoint . --device cuda
   ```

2. **Batch Processing**: Use batch_test.py for multiple inputs

3. **Temperature Tuning**:
   - Low (0.1-0.3): More deterministic, better for math
   - Medium (0.5-0.7): Balanced creativity
   - High (0.8-1.0): More creative, less predictable

4. **Memory Optimization**: Use FP16 for inference if VRAM is limited

## Troubleshooting

### Out of Memory
- Reduce `max_tokens` parameter
- Use smaller batch sizes
- Enable FP16 inference

### Slow Inference
- Ensure CUDA is being used (`--device cuda`)
- Check GPU utilization with `nvidia-smi`
- Consider using TorchScript model for better performance

### Import Errors
- Verify all dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)

## Support

For issues or questions, refer to the main repository documentation.

---

**Generated**: {model_info['export_timestamp']}
**Source**: {model_info['checkpoint_source']}
"""


def main():
    parser = argparse.ArgumentParser(description="Export Mamba model for deployment")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint directory')
    parser.add_argument('--output', type=str, default='./deployment',
                        help='Output directory for exported model')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cuda', 'cpu'],
                        help='Device to load model on (use cpu for export)')

    args = parser.parse_args()

    print("="*80)
    print("Mamba Model Exporter")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print("="*80 + "\n")

    # Export model
    exporter = ModelExporter(args.checkpoint, args.device)
    exporter.export(args.output)


if __name__ == '__main__':
    main()
