#!/usr/bin/env python3
"""
Test script to verify the setup is correct.
Runs basic checks without requiring GPU or training.
"""

import sys
import os


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    required = ['torch', 'transformers', 'datasets', 'tqdm', 'numpy']
    optional = ['PyPDF2']

    failed = []
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - REQUIRED")
            failed.append(package)

    for package in optional:
        try:
            __import__(package)
            print(f"  ✓ {package} (optional)")
        except ImportError:
            print(f"  ⚠ {package} - optional (install for PDF support)")

    return len(failed) == 0


def test_project_structure():
    """Test that all required files exist."""
    print("\nTesting project structure...")
    required_files = [
        'train.py',
        'generate.py',
        'prepare_data.py',
        'requirements.txt',
        'models/mamba_model.py',
        'configs/model_configs.py',
        'utils/data_loader.py'
    ]

    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - MISSING")
            all_exist = False

    return all_exist


def test_model_creation():
    """Test that model can be created."""
    print("\nTesting model creation...")
    try:
        from models.mamba_model import create_mamba_model

        config = {
            'd_model': 128,
            'n_layers': 2,
            'd_state': 8,
            'expand_factor': 2,
            'dropout': 0.1,
            'max_seq_len': 256,
            'vocab_size': 1000
        }

        model = create_mamba_model(config)
        total_params = sum(p.numel() for p in model.parameters())

        print(f"  ✓ Model created successfully")
        print(f"  ✓ Total parameters: {total_params:,}")
        return True
    except Exception as e:
        print(f"  ✗ Error creating model: {e}")
        return False


def test_config_system():
    """Test configuration system."""
    print("\nTesting configuration system...")
    try:
        from configs.model_configs import get_config, MODEL_CONFIGS, VRAM_CONFIGS

        # Test getting a config
        config = get_config('small', '8gb', 'default')

        print(f"  ✓ Configuration system working")
        print(f"  ✓ Available model sizes: {list(MODEL_CONFIGS.keys())}")
        print(f"  ✓ Available VRAM configs: {list(VRAM_CONFIGS.keys())}")
        return True
    except Exception as e:
        print(f"  ✗ Error with config system: {e}")
        return False


def test_data_loader():
    """Test data loading utilities."""
    print("\nTesting data loader...")
    try:
        from utils.data_loader import DataLoader

        # Create loader
        loader = DataLoader('data', max_length=256)

        # Test loading example file
        if os.path.exists('data/example.txt'):
            documents = loader.load_directory('data')
            print(f"  ✓ Data loader working")
            print(f"  ✓ Found {len(documents)} example document(s)")

            if documents:
                chunks = loader.prepare_training_data(documents)
                print(f"  ✓ Created {len(chunks)} training chunks")
        else:
            print(f"  ⚠ No example data found (this is OK)")

        return True
    except Exception as e:
        print(f"  ✗ Error with data loader: {e}")
        return False


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ CUDA version: {torch.version.cuda}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ VRAM: {vram:.2f} GB")
        else:
            print(f"  ⚠ CUDA not available (CPU-only mode)")
            print(f"    Training will be slower but still works")

        return True
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False


def main():
    print("="*60)
    print("Mamba Trainer Setup Verification")
    print("="*60)
    print()

    tests = [
        ("Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("Model Creation", test_model_creation),
        ("Configuration System", test_config_system),
        ("Data Loader", test_data_loader),
        ("CUDA", test_cuda)
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ Unexpected error in {name}: {e}")
            results.append((name, False))

    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} - {name}")

    all_passed = all(r for _, r in results)

    print()
    if all_passed:
        print("✓ All tests passed! You're ready to train.")
        print()
        print("Next steps:")
        print("1. Add training data to the 'data/' directory")
        print("2. Run: python prepare_data.py --show_stats")
        print("3. Run: python train.py --model_size small --vram 8gb")
    else:
        print("⚠ Some tests failed. Please install missing dependencies:")
        print("   pip install -r requirements.txt")

    print()
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
