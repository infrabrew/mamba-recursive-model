#!/bin/bash
# Setup script for Mamba Trainer

echo "=========================================="
echo "Mamba Language Model Trainer Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment (optional)
read -p "Create virtual environment? (y/n): " create_venv
if [ "$create_venv" = "y" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Virtual environment created and activated"
fi

# Install requirements
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check for CUDA
echo ""
echo "Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data
mkdir -p checkpoints
mkdir -p logs

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Add your training data to the 'data/' directory"
echo "2. Run: python prepare_data.py --show_stats"
echo "3. Run: python train.py --model_size small --vram 8gb"
echo ""
echo "For more information, see README.md"
