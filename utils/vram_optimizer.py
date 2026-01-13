"""
Dynamic VRAM optimization based on available GPU memory.
Automatically configures batch size, sequence length, and other parameters.
"""

import torch
import math
from typing import Dict, List, Tuple, Optional


class VRAMOptimizer:
    """Optimize training configuration based on available VRAM."""

    def __init__(self, target_vram_gb: Optional[float] = None, safety_margin_mb: int = 2048):
        """
        Initialize VRAM optimizer.

        Args:
            target_vram_gb: Target VRAM to use (in GB). If None, auto-detect.
            safety_margin_mb: Safety margin to prevent OOM (in MB, default: 2048)
        """
        self.safety_margin_mb = safety_margin_mb
        self.gpu_info = self._detect_gpus()
        self.target_vram_gb = target_vram_gb or self._calculate_total_vram()

    def _detect_gpus(self) -> List[Dict]:
        """Detect all available GPUs and their memory."""
        if not torch.cuda.is_available():
            return []

        gpu_info = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info = {
                'id': i,
                'name': props.name,
                'total_memory_gb': props.total_memory / (1024**3),
                'total_memory_mb': props.total_memory / (1024**2),
            }
            gpu_info.append(info)

        return gpu_info

    def _calculate_total_vram(self) -> float:
        """Calculate total VRAM across all GPUs."""
        if not self.gpu_info:
            return 0.0
        return sum(gpu['total_memory_gb'] for gpu in self.gpu_info)

    def get_usable_vram_mb(self) -> float:
        """Get usable VRAM after applying safety margin."""
        vram_mb = self.target_vram_gb * 1024
        usable_vram_mb = vram_mb - self.safety_margin_mb
        return usable_vram_mb

    def get_usable_vram_gb(self) -> float:
        """Get usable VRAM in GB after applying safety margin."""
        return self.get_usable_vram_mb() / 1024

    def is_multi_gpu(self) -> bool:
        """Check if multiple GPUs are available."""
        return len(self.gpu_info) > 1

    def estimate_model_memory(self, model_config: Dict) -> float:
        """
        Estimate model memory usage in MB.

        Args:
            model_config: Model configuration dict

        Returns:
            Estimated memory in MB
        """
        d_model = model_config.get('d_model', 768)
        n_layers = model_config.get('n_layers', 12)
        vocab_size = model_config.get('vocab_size', 50257)
        d_state = model_config.get('d_state', 16)
        expand_factor = model_config.get('expand_factor', 2)

        # Estimate parameter count
        # Embedding: vocab_size * d_model
        embedding_params = vocab_size * d_model

        # Per layer: roughly d_model * d_model * 4 (projections) + d_model * d_state * 2
        per_layer_params = d_model * d_model * 4 + d_model * d_state * 2 + d_model * expand_factor * d_model * 2

        total_params = embedding_params + (per_layer_params * n_layers)

        # FP16: 2 bytes per param, FP32: 4 bytes per param
        bytes_per_param = 2  # Assuming FP16
        model_size_mb = (total_params * bytes_per_param) / (1024**2)

        # Optimizer states (AdamW): 2x model size
        optimizer_size_mb = model_size_mb * 2

        # Total model + optimizer
        total_mb = model_size_mb + optimizer_size_mb

        return total_mb

    def estimate_batch_memory(self, model_config: Dict, batch_size: int, seq_len: int, use_fp16: bool = True) -> float:
        """
        Estimate memory for a single batch.

        Args:
            model_config: Model configuration
            batch_size: Batch size
            seq_len: Sequence length
            use_fp16: Whether using FP16

        Returns:
            Estimated memory in MB
        """
        d_model = model_config.get('d_model', 768)
        n_layers = model_config.get('n_layers', 12)
        d_state = model_config.get('d_state', 16)

        bytes_per_element = 2 if use_fp16 else 4

        # Input embeddings
        input_memory = batch_size * seq_len * d_model * bytes_per_element

        # Activations per layer (simplified)
        per_layer_activations = batch_size * seq_len * d_model * 4  # Rough estimate
        per_layer_states = batch_size * d_model * d_state * bytes_per_element

        total_activations = (per_layer_activations + per_layer_states) * n_layers

        # Gradients (roughly same as activations)
        gradients = total_activations

        # Total batch memory
        batch_memory_bytes = input_memory + total_activations + gradients
        batch_memory_mb = batch_memory_bytes / (1024**2)

        return batch_memory_mb

    def optimize_config(self, model_config: Dict, training_config: Dict) -> Dict:
        """
        Optimize VRAM configuration based on available memory.

        Args:
            model_config: Model configuration
            training_config: Training configuration

        Returns:
            Optimized VRAM configuration
        """
        usable_vram_mb = self.get_usable_vram_mb()
        usable_vram_gb = self.get_usable_vram_gb()

        print(f"\n{'='*60}")
        print("VRAM Optimizer")
        print(f"{'='*60}")

        # Print GPU info
        if self.gpu_info:
            print(f"\nDetected GPUs: {len(self.gpu_info)}")
            for gpu in self.gpu_info:
                print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['total_memory_gb']:.2f} GB)")

        print(f"\nTarget VRAM: {self.target_vram_gb:.2f} GB")
        print(f"Safety Margin: {self.safety_margin_mb} MB")
        print(f"Usable VRAM: {usable_vram_gb:.2f} GB ({usable_vram_mb:.0f} MB)")

        # Estimate model memory
        model_memory_mb = self.estimate_model_memory(model_config)
        print(f"\nEstimated Model Memory: {model_memory_mb:.0f} MB ({model_memory_mb/1024:.2f} GB)")

        # Calculate available memory for batches
        available_for_batches_mb = usable_vram_mb - model_memory_mb
        print(f"Available for Batches: {available_for_batches_mb:.0f} MB ({available_for_batches_mb/1024:.2f} GB)")

        if available_for_batches_mb < 500:
            print("\n⚠️  WARNING: Very little memory available for batches!")
            print("   Consider using a smaller model or more VRAM.")

        # Determine optimal batch size and sequence length
        optimal_config = self._calculate_optimal_batch_config(
            model_config,
            available_for_batches_mb,
            usable_vram_gb
        )

        print(f"\n{'='*60}")
        print("Optimized Configuration")
        print(f"{'='*60}")
        for key, value in optimal_config.items():
            if not key.startswith('_'):
                print(f"  {key:30s}: {value}")
        print(f"{'='*60}\n")

        return optimal_config

    def _calculate_optimal_batch_config(self, model_config: Dict, available_mb: float, total_vram_gb: float) -> Dict:
        """Calculate optimal batch size and sequence length."""
        use_fp16 = True  # Always use FP16 for efficiency

        # Start with conservative estimates based on VRAM
        if total_vram_gb <= 8:
            base_batch_size = 1
            base_seq_len = 256
            gradient_accumulation = 16
        elif total_vram_gb <= 12:
            base_batch_size = 2
            base_seq_len = 512
            gradient_accumulation = 8
        elif total_vram_gb <= 16:
            base_batch_size = 2
            base_seq_len = 512
            gradient_accumulation = 8
        elif total_vram_gb <= 24:
            base_batch_size = 4
            base_seq_len = 1024
            gradient_accumulation = 4
        elif total_vram_gb <= 32:
            base_batch_size = 8
            base_seq_len = 1024
            gradient_accumulation = 2
        else:  # 48GB+
            base_batch_size = 12
            base_seq_len = 2048
            gradient_accumulation = 2

        # Verify this fits in available memory
        batch_memory = self.estimate_batch_memory(model_config, base_batch_size, base_seq_len, use_fp16)

        # If doesn't fit, reduce batch size or sequence length
        while batch_memory > available_mb * 0.8 and base_batch_size > 1:
            base_batch_size = max(1, base_batch_size // 2)
            gradient_accumulation *= 2
            batch_memory = self.estimate_batch_memory(model_config, base_batch_size, base_seq_len, use_fp16)

        while batch_memory > available_mb * 0.8 and base_seq_len > 128:
            base_seq_len = max(128, base_seq_len // 2)
            batch_memory = self.estimate_batch_memory(model_config, base_batch_size, base_seq_len, use_fp16)

        # Determine gradient checkpointing
        use_gradient_checkpointing = total_vram_gb < 24

        effective_batch_size = base_batch_size * gradient_accumulation

        return {
            'batch_size': base_batch_size,
            'gradient_accumulation_steps': gradient_accumulation,
            'max_seq_len': base_seq_len,
            'use_fp16': use_fp16,
            'use_gradient_checkpointing': use_gradient_checkpointing,
            'use_cpu_offload': False,
            'effective_batch_size': effective_batch_size,
            'description': f'Auto-optimized for {total_vram_gb:.1f}GB VRAM',
            '_estimated_batch_memory_mb': batch_memory,
            '_usable_vram_gb': total_vram_gb,
            '_safety_margin_mb': self.safety_margin_mb,
        }

    def print_summary(self):
        """Print summary of GPU configuration."""
        print(f"\n{'='*60}")
        print("GPU Summary")
        print(f"{'='*60}")

        if not self.gpu_info:
            print("No CUDA GPUs detected")
            return

        print(f"\nTotal GPUs: {len(self.gpu_info)}")
        print(f"Multi-GPU: {'Yes' if self.is_multi_gpu() else 'No'}")
        print(f"\nGPU Details:")

        for gpu in self.gpu_info:
            print(f"  GPU {gpu['id']}: {gpu['name']}")
            print(f"    Total Memory: {gpu['total_memory_gb']:.2f} GB")

        print(f"\nTotal VRAM: {self._calculate_total_vram():.2f} GB")
        print(f"Target VRAM: {self.target_vram_gb:.2f} GB")
        print(f"Usable VRAM: {self.get_usable_vram_gb():.2f} GB (after {self.safety_margin_mb}MB safety margin)")
        print(f"{'='*60}\n")


def create_vram_config(target_vram_gb: Optional[float] = None,
                      model_config: Optional[Dict] = None,
                      training_config: Optional[Dict] = None,
                      safety_margin_mb: int = 2048) -> Dict:
    """
    Create optimized VRAM configuration.

    Args:
        target_vram_gb: Target VRAM in GB (e.g., 24.0). If None, auto-detect.
        model_config: Model configuration dict
        training_config: Training configuration dict
        safety_margin_mb: Safety margin in MB (default: 2048)

    Returns:
        Optimized VRAM configuration dict
    """
    optimizer = VRAMOptimizer(target_vram_gb, safety_margin_mb)
    optimizer.print_summary()

    if model_config:
        return optimizer.optimize_config(model_config, training_config or {})
    else:
        # Return basic info
        return {
            'usable_vram_gb': optimizer.get_usable_vram_gb(),
            'usable_vram_mb': optimizer.get_usable_vram_mb(),
            'total_vram_gb': optimizer.target_vram_gb,
            'num_gpus': len(optimizer.gpu_info),
            'is_multi_gpu': optimizer.is_multi_gpu(),
            'gpu_info': optimizer.gpu_info,
        }


if __name__ == '__main__':
    # Test the optimizer
    print("Testing VRAM Optimizer\n")

    # Example model config
    model_config = {
        'd_model': 768,
        'n_layers': 12,
        'd_state': 16,
        'expand_factor': 2,
        'vocab_size': 50257,
    }

    # Test with different VRAM targets
    for vram_gb in [8, 12, 16, 24, 32, 48]:
        print(f"\n{'#'*60}")
        print(f"Testing with {vram_gb}GB VRAM target")
        print(f"{'#'*60}")

        config = create_vram_config(
            target_vram_gb=vram_gb,
            model_config=model_config,
            safety_margin_mb=1024
        )
