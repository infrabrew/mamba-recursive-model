"""
Model configuration presets for different model sizes and VRAM constraints.
"""

# Model size configurations
MODEL_CONFIGS = {
    'small': {
        'd_model': 512,
        'n_layers': 8,
        'd_state': 16,
        'expand_factor': 2,
        'dropout': 0.1,
        'max_seq_len': 1024,
        'vocab_size': 50257,
        'description': 'Small Mamba model (~50M parameters)',
        'estimated_params': '~50M'
    },
    'medium-lite': {
        'd_model': 640,
        'n_layers': 10,
        'd_state': 16,
        'expand_factor': 2,
        'dropout': 0.1,
        'max_seq_len': 1024,
        'vocab_size': 50257,
        'description': 'Medium-Lite Mamba model (~90M parameters) - Memory: 10-12GB',
        'estimated_params': '~90M'
    },
    'medium-plus': {
        'd_model': 704,
        'n_layers': 11,
        'd_state': 16,
        'expand_factor': 2,
        'dropout': 0.1,
        'max_seq_len': 1536,
        'vocab_size': 50257,
        'description': 'Medium-Plus Mamba model (~120M parameters) - Memory: 13-14GB',
        'estimated_params': '~120M'
    },
    'medium-x': {
        'd_model': 736,
        'n_layers': 12,
        'd_state': 16,
        'expand_factor': 2,
        'dropout': 0.1,
        'max_seq_len': 1792,
        'vocab_size': 50257,
        'description': 'Medium-X Mamba model (~135M parameters) - Memory: 14-15GB',
        'estimated_params': '~135M'
    },
    'medium': {
        'd_model': 768,
        'n_layers': 12,
        'd_state': 16,
        'expand_factor': 2,
        'dropout': 0.1,
        'max_seq_len': 2048,
        'vocab_size': 50257,
        'description': 'Medium Mamba model (~150M parameters) - Memory: 15-16GB',
        'estimated_params': '~150M'
    },
    'medium-extra': {
        'd_model': 832,
        'n_layers': 14,
        'd_state': 16,
        'expand_factor': 2,
        'dropout': 0.1,
        'max_seq_len': 2048,
        'vocab_size': 50257,
        'description': 'Medium-Extra Mamba model (~180M parameters) - Memory: 16-18GB',
        'estimated_params': '~180M'
    },
    'large-lite': {
        'd_model': 896,
        'n_layers': 18,
        'd_state': 16,
        'expand_factor': 2,
        'dropout': 0.1,
        'max_seq_len': 2048,
        'vocab_size': 50257,
        'description': 'Large-Lite Mamba model (~250M parameters) - Memory: 18-20GB',
        'estimated_params': '~250M'
    },
    'large-plus': {
        'd_model': 960,
        'n_layers': 21,
        'd_state': 16,
        'expand_factor': 2,
        'dropout': 0.1,
        'max_seq_len': 2048,
        'vocab_size': 50257,
        'description': 'Large-Plus Mamba model (~320M parameters) - Memory: 20-22GB',
        'estimated_params': '~320M'
    },
    'large': {
        'd_model': 1024,
        'n_layers': 24,
        'd_state': 16,
        'expand_factor': 2,
        'dropout': 0.1,
        'max_seq_len': 2048,
        'vocab_size': 50257,
        'description': 'Large Mamba model (~400M parameters) - Memory: 30-35GB',
        'estimated_params': '~400M'
    },
    'xlarge': {
        'd_model': 1280,
        'n_layers': 32,
        'd_state': 16,
        'expand_factor': 2,
        'dropout': 0.1,
        'max_seq_len': 2048,
        'vocab_size': 50257,
        'description': 'XLarge Mamba model (~650M parameters) - Memory: 45-50GB',
        'estimated_params': '~650M'
    },
    'xxlarge': {
        'd_model': 1536,
        'n_layers': 40,
        'd_state': 16,
        'expand_factor': 2,
        'dropout': 0.1,
        'max_seq_len': 2048,
        'vocab_size': 50257,
        'description': 'XXLarge Mamba model (~1B parameters) - Memory: 70-80GB',
        'estimated_params': '~1B'
    },
    # Hybrid Recursive Mamba models
    'hybrid-small': {
        'd_model': 512,
        'n_layers': 8,
        'd_state': 16,
        'expand_factor': 2,
        'dropout': 0.1,
        'max_seq_len': 1024,
        'vocab_size': 50257,
        'max_recursion_depth': 3,
        'use_hierarchical_attention': True,
        'model_type': 'hybrid_recursive',
        'description': 'Hybrid Recursive Mamba (~75M parameters) - Memory: 8-10GB',
        'estimated_params': '~75M'
    },
    'hybrid-small-mini': {
        'd_model': 576,
        'n_layers': 9,
        'd_state': 16,
        'expand_factor': 2,
        'dropout': 0.1,
        'max_seq_len': 1280,
        'vocab_size': 50257,
        'max_recursion_depth': 3,
        'use_hierarchical_attention': True,
        'model_type': 'hybrid_recursive',
        'description': 'Hybrid Recursive Mamba (~100M parameters) - Memory: 11-13GB',
        'estimated_params': '~100M'
    },
    'hybrid-small-standard': {
        'd_model': 608,
        'n_layers': 10,
        'd_state': 16,
        'expand_factor': 2,
        'dropout': 0.1,
        'max_seq_len': 1408,
        'vocab_size': 50257,
        'max_recursion_depth': 3,
        'use_hierarchical_attention': True,
        'model_type': 'hybrid_recursive',
        'description': 'Hybrid Recursive Mamba (~112M parameters) - Memory: 13-14.5GB',
        'estimated_params': '~112M'
    },
    'hybrid-small-plus': {
        'd_model': 640,
        'n_layers': 10,
        'd_state': 16,
        'expand_factor': 2,
        'dropout': 0.1,
        'max_seq_len': 1536,
        'vocab_size': 50257,
        'max_recursion_depth': 3,
        'use_hierarchical_attention': True,
        'model_type': 'hybrid_recursive',
        'description': 'Hybrid Recursive Mamba (~125M parameters) - Memory: 13-15GB',
        'estimated_params': '~125M'
    },
    'hybrid-small-pro': {
        'd_model': 704,
        'n_layers': 11,
        'd_state': 16,
        'expand_factor': 2,
        'dropout': 0.1,
        'max_seq_len': 1792,
        'vocab_size': 50257,
        'max_recursion_depth': 3,
        'use_hierarchical_attention': True,
        'model_type': 'hybrid_recursive',
        'description': 'Hybrid Recursive Mamba (~165M parameters) - Memory: 15-16GB',
        'estimated_params': '~165M'
    },
    'hybrid-medium': {
        'd_model': 768,
        'n_layers': 12,
        'd_state': 16,
        'expand_factor': 2,
        'dropout': 0.1,
        'max_seq_len': 2048,
        'vocab_size': 50257,
        'max_recursion_depth': 3,
        'use_hierarchical_attention': True,
        'model_type': 'hybrid_recursive',
        'description': 'Hybrid Recursive Mamba (~220M parameters) - Memory: 18-20GB',
        'estimated_params': '~220M'
    },
    'hybrid-large': {
        'd_model': 1024,
        'n_layers': 16,
        'd_state': 16,
        'expand_factor': 2,
        'dropout': 0.1,
        'max_seq_len': 2048,
        'vocab_size': 50257,
        'max_recursion_depth': 4,
        'use_hierarchical_attention': True,
        'model_type': 'hybrid_recursive',
        'description': 'Hybrid Recursive Mamba (~550M parameters) - Memory: 40-45GB',
        'estimated_params': '~550M'
    }
}

# VRAM-optimized training configurations
VRAM_CONFIGS = {
    '8gb': {
        'batch_size': 1,
        'gradient_accumulation_steps': 16,
        'max_seq_len': 512,
        'use_fp16': True,
        'use_gradient_checkpointing': True,
        'use_cpu_offload': False,
        'recommended_models': ['small'],
        'description': '8GB VRAM configuration (GTX 1080, RTX 2070, etc.)',
        'effective_batch_size': 16
    },
    '12gb': {
        'batch_size': 2,
        'gradient_accumulation_steps': 8,
        'max_seq_len': 1024,
        'use_fp16': True,
        'use_gradient_checkpointing': True,
        'use_cpu_offload': False,
        'recommended_models': ['small', 'medium'],
        'description': '12GB VRAM configuration (RTX 3060, RTX 2080 Ti, etc.)',
        'effective_batch_size': 16
    },
    '16gb': {
        'batch_size': 2,
        'gradient_accumulation_steps': 8,
        'max_seq_len': 512,
        'use_fp16': True,
        'use_gradient_checkpointing': True,
        'use_cpu_offload': False,
        'recommended_models': ['small'],
        'description': '16GB VRAM configuration (RTX 4060 Ti, RTX 3080, etc.) - Conservative for small model',
        'effective_batch_size': 16
    },
    '16gb_medium': {
        'batch_size': 1,
        'gradient_accumulation_steps': 16,
        'max_seq_len': 256,
        'use_fp16': True,
        'use_gradient_checkpointing': True,
        'use_cpu_offload': True,
        'recommended_models': ['medium'],
        'description': '16GB VRAM configuration for medium model - Ultra conservative with CPU offload',
        'effective_batch_size': 16
    },
    '24gb': {
        'batch_size': 8,
        'gradient_accumulation_steps': 2,
        'max_seq_len': 2048,
        'use_fp16': True,
        'use_gradient_checkpointing': False,
        'use_cpu_offload': False,
        'recommended_models': ['small', 'medium', 'large'],
        'description': '24GB VRAM configuration (RTX 3090, RTX 4090, A5000, etc.)',
        'effective_batch_size': 16
    },
    '32gb': {
        'batch_size': 12,
        'gradient_accumulation_steps': 2,
        'max_seq_len': 2048,
        'use_fp16': True,
        'use_gradient_checkpointing': False,
        'use_cpu_offload': False,
        'recommended_models': ['small', 'medium', 'large'],
        'description': '32GB VRAM configuration (V100, A5500, etc.)',
        'effective_batch_size': 24
    },
    '48gb': {
        'batch_size': 16,
        'gradient_accumulation_steps': 1,
        'max_seq_len': 2048,
        'use_fp16': False,
        'use_gradient_checkpointing': False,
        'use_cpu_offload': False,
        'recommended_models': ['small', 'medium', 'large'],
        'description': '48GB VRAM configuration (A40, A6000, RTX 6000 Ada, etc.)',
        'effective_batch_size': 16
    }
}

# Training hyperparameters
TRAINING_CONFIGS = {
    'default': {
        'learning_rate': 3e-4,
        'weight_decay': 0.1,
        'max_steps': 100000,
        'warmup_steps': 2000,
        'eval_interval': 1000,
        'save_interval': 5000,
        'logging_steps': 10,
        'adam_beta1': 0.9,
        'adam_beta2': 0.95,
        'adam_epsilon': 1e-8,
        'max_grad_norm': 1.0,
        'label_smoothing': 0.0
    },
    'fast': {
        'learning_rate': 5e-4,
        'weight_decay': 0.1,
        'max_steps': 50000,
        'warmup_steps': 1000,
        'eval_interval': 500,
        'save_interval': 2500,
        'logging_steps': 10,
        'adam_beta1': 0.9,
        'adam_beta2': 0.95,
        'adam_epsilon': 1e-8,
        'max_grad_norm': 1.0,
        'label_smoothing': 0.0
    },
    'careful': {
        'learning_rate': 1e-4,
        'weight_decay': 0.1,
        'max_steps': 200000,
        'warmup_steps': 5000,
        'eval_interval': 2000,
        'save_interval': 10000,
        'logging_steps': 10,
        'adam_beta1': 0.9,
        'adam_beta2': 0.95,
        'adam_epsilon': 1e-8,
        'max_grad_norm': 1.0,
        'label_smoothing': 0.0
    },
    # Muon optimizer presets (faster convergence, 30-40% fewer steps)
    'muon_fast': {
        'learning_rate': 4e-3,
        'weight_decay': 0.1,
        'max_steps': 30000,
        'warmup_steps': 500,
        'eval_interval': 750,
        'save_interval': 2500,
        'logging_steps': 10,
        'adam_beta1': 0.9,
        'adam_beta2': 0.95,
        'adam_epsilon': 1e-8,
        'max_grad_norm': 1.0,
        'label_smoothing': 0.0
    },
    'muon_default': {
        'learning_rate': 3e-3,
        'weight_decay': 0.1,
        'max_steps': 60000,
        'warmup_steps': 1000,
        'eval_interval': 1500,
        'save_interval': 5000,
        'logging_steps': 10,
        'adam_beta1': 0.9,
        'adam_beta2': 0.95,
        'adam_epsilon': 1e-8,
        'max_grad_norm': 1.0,
        'label_smoothing': 0.0
    },
    'muon_careful': {
        'learning_rate': 2e-3,
        'weight_decay': 0.1,
        'max_steps': 120000,
        'warmup_steps': 3000,
        'eval_interval': 3000,
        'save_interval': 10000,
        'logging_steps': 10,
        'adam_beta1': 0.9,
        'adam_beta2': 0.95,
        'adam_epsilon': 1e-8,
        'max_grad_norm': 1.0,
        'label_smoothing': 0.0
    }
}


def get_config(model_size: str = 'medium', vram: str = '16gb',
               training_preset: str = 'default') -> dict:
    """
    Get a complete configuration by combining model, VRAM, and training configs.

    Args:
        model_size: Model size ('small', 'medium', 'large')
        vram: VRAM configuration ('8gb', '12gb', '16gb', '24gb', '32gb', '48gb')
        training_preset: Training preset ('default', 'fast', 'careful')

    Returns:
        Dictionary containing complete configuration
    """
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(MODEL_CONFIGS.keys())}")

    if vram not in VRAM_CONFIGS:
        raise ValueError(f"Unknown VRAM config: {vram}. Choose from {list(VRAM_CONFIGS.keys())}")

    if training_preset not in TRAINING_CONFIGS:
        raise ValueError(f"Unknown training preset: {training_preset}. Choose from {list(TRAINING_CONFIGS.keys())}")

    # Check if model size is recommended for this VRAM
    vram_config = VRAM_CONFIGS[vram]
    if model_size not in vram_config['recommended_models']:
        print(f"WARNING: {model_size} model is not recommended for {vram} VRAM.")
        print(f"Recommended models: {vram_config['recommended_models']}")

    # Merge configurations
    config = {
        'model': MODEL_CONFIGS[model_size].copy(),
        'vram': VRAM_CONFIGS[vram].copy(),
        'training': TRAINING_CONFIGS[training_preset].copy()
    }

    # Override model max_seq_len with VRAM constraint if needed
    if vram_config['max_seq_len'] < config['model']['max_seq_len']:
        print(f"Reducing max_seq_len from {config['model']['max_seq_len']} to {vram_config['max_seq_len']} for VRAM constraints")
        config['model']['max_seq_len'] = vram_config['max_seq_len']

    return config


def print_config(config: dict):
    """Print configuration in a readable format."""
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)

    print("\nMODEL CONFIG:")
    print("-" * 60)
    for key, value in config['model'].items():
        print(f"  {key:20s}: {value}")

    print("\nVRAM OPTIMIZATION CONFIG:")
    print("-" * 60)
    for key, value in config['vram'].items():
        print(f"  {key:30s}: {value}")

    print("\nTRAINING HYPERPARAMETERS:")
    print("-" * 60)
    for key, value in config['training'].items():
        print(f"  {key:20s}: {value}")

    print("="*60 + "\n")


def save_config(config: dict, path: str):
    """Save configuration to JSON file."""
    import json
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {path}")


def load_config(path: str) -> dict:
    """Load configuration from JSON file."""
    import json
    with open(path, 'r') as f:
        config = json.load(f)
    print(f"Configuration loaded from {path}")
    return config
