#!/usr/bin/env python3
"""
Script to add Muon optimizer support to train.py safely.
"""

import re

def add_muon_support():
    """Add Muon optimizer support to train.py"""

    with open('train.py', 'r') as f:
        content = f.read()

    # 1. Add import after torch import
    if 'from optimizers import create_muon_optimizer' not in content:
        content = content.replace(
            'import torch\n',
            'import torch\nfrom optimizers import create_muon_optimizer\n'
        )
        print("✓ Added Muon import")

    # 2. Add optimizer argument to argparse
    if '--optimizer' not in content:
        # Find the seed argument and add optimizer after it
        pattern = r"(parser\.add_argument\('--seed'[^\)]+\))"
        replacement = r"\1\n    parser.add_argument('--optimizer', type=str, default='adamw',\n                        choices=['adamw', 'muon'],\n                        help='Optimizer to use (adamw or muon)')"
        content = re.sub(pattern, replacement, content)
        print("✓ Added --optimizer argument")

    # 3. Replace optimizer setup
    old_pattern = r"# Setup optimizer\s+self\.optimizer = torch\.optim\.AdamW\(\s+self\.model\.parameters\(\),\s+lr=self\.config\['training'\]\['learning_rate'\],\s+betas=\(self\.config\['training'\]\['adam_beta1'\], self\.config\['training'\]\['adam_beta2'\]\),\s+eps=self\.config\['training'\]\['adam_epsilon'\],\s+weight_decay=self\.config\['training'\]\['weight_decay'\]\s+\)"

    new_optimizer_code = """# Setup optimizer
        optimizer_type = self.config.get('optimizer', 'adamw')

        if optimizer_type == 'muon':
            print("Using Muon optimizer...")
            self.optimizer = create_muon_optimizer(
                self.model,
                lr=self.config['training']['learning_rate'],
                momentum=0.95,
                weight_decay=self.config['training']['weight_decay'],
                nesterov=True,
                backend='newtonschulz5',
                decoupled_wd=True
            )
        else:
            print("Using AdamW optimizer...")
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                betas=(self.config['training']['adam_beta1'], self.config['training']['adam_beta2']),
                eps=self.config['training']['adam_epsilon'],
                weight_decay=self.config['training']['weight_decay']
            )"""

    content = re.sub(old_pattern, new_optimizer_code, content, flags=re.MULTILINE | re.DOTALL)
    print("✓ Updated optimizer setup")

    # 4. Add optimizer to config dict
    pattern = r"('training': TRAINING_CONFIGS\[args\.training_preset\]\.copy\(\),)"
    replacement = r"\1\n                'optimizer': args.optimizer,"
    if "'optimizer': args.optimizer" not in content:
        content = re.sub(pattern, replacement, content)
        print("✓ Added optimizer to config")

    # Write back
    with open('train.py', 'w') as f:
        f.write(content)

    print("\n✓ Successfully added Muon optimizer support!")
    print("\nYou can now use:")
    print("  --optimizer adamw  (default)")
    print("  --optimizer muon   (faster convergence)")

if __name__ == '__main__':
    add_muon_support()
