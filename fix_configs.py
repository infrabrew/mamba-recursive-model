#!/usr/bin/env python3
"""Fix model_configs.py to add Muon presets"""

import re

with open("configs/model_configs.py", "r") as f:
    content = f.read()

# Find the end of 'careful' preset in TRAINING_CONFIGS
# and add Muon presets before the closing brace

muon_presets = """    ,
    # Muon optimizer presets (faster convergence)
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
    }"""

# Pattern: Find the closing of 'careful' dict and the closing brace of TRAINING_CONFIGS
pattern = r"('careful': \{[^}]+\})\n\}\n\n\ndef get_config"

replacement = r"\1" + muon_presets + r"\n}\n\n\ndef get_config"

content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

with open("configs/model_configs.py", "w") as f:
    f.write(content)

print("✓ Added Muon presets to TRAINING_CONFIGS")
