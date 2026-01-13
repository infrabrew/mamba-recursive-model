"""
Hybrid Recursive Mamba Model.
Combines Mamba SSM with recursive language modeling for hierarchical reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

from .mamba_model import SelectiveSSM, MambaBlock


class RecursiveProcessor(nn.Module):
    """Recursive processing layer for hierarchical reasoning."""

    def __init__(self, d_model: int, max_depth: int = 3, dropout: float = 0.1):
        """
        Initialize recursive processor.

        Args:
            d_model: Model dimension
            max_depth: Maximum recursion depth
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.max_depth = max_depth

        # Gating mechanism to decide recursion depth
        self.depth_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, max_depth),
            nn.Softmax(dim=-1)
        )

        # Recursive transformation layers (one per depth level)
        self.recursive_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model)
            )
            for _ in range(max_depth)
        ])

        # Memory accumulator for recursive states
        self.memory_mixer = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor, prev_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply recursive processing.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            prev_state: Previous recursive state (optional)

        Returns:
            output: Processed output
            state: Updated recursive state
        """
        batch, seq_len, _ = x.shape

        # Determine recursion depth dynamically
        depth_weights = self.depth_gate(x.mean(dim=1))  # (batch, max_depth)

        # Initialize state - ensure 3D shape (batch, seq_len, d_model)
        if prev_state is None:
            state = torch.zeros_like(x)  # (batch, seq_len, d_model)
        else:
            state = prev_state

        output = x

        # Apply recursive layers weighted by depth gate
        for depth in range(self.max_depth):
            # Get weight for this depth
            weight = depth_weights[:, depth].view(batch, 1, 1)  # (batch, 1, 1)

            # Recursive transformation - maintains (batch, seq_len, d_model)
            transformed = self.recursive_layers[depth](output)

            # Combine with previous state (memory) - concatenate along last dimension
            # transformed: (batch, seq_len, d_model)
            # state: (batch, seq_len, d_model)
            # combined: (batch, seq_len, d_model * 2)
            combined = torch.cat([transformed, state], dim=-1)

            # memory_mixer projects back to d_model, maintaining 3D shape
            # new_state: (batch, seq_len, d_model)
            new_state = self.memory_mixer(combined)

            # Weighted accumulation - all tensors are (batch, seq_len, d_model)
            output = output + weight * transformed
            state = state + weight * new_state

        return output, state


class HierarchicalAttention(nn.Module):
    """Multi-scale hierarchical attention for recursive processing."""

    def __init__(self, d_model: int, n_heads: int = 8, scales: List[int] = [1, 2, 4]):
        """
        Initialize hierarchical attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            scales: Different scales for hierarchical processing
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.scales = scales
        self.head_dim = d_model // n_heads

        # Multi-scale projections
        self.scale_projections = nn.ModuleList([
            nn.Linear(d_model, d_model)
            for _ in scales
        ])

        # Attention layers for each scale
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in scales
        ])

        # Scale fusion
        self.scale_fusion = nn.Linear(d_model * len(scales), d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply hierarchical multi-scale attention.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Attended output
        """
        batch, seq_len, d_model = x.shape
        scale_outputs = []

        for scale_idx, scale in enumerate(self.scales):
            # Pool to different scales
            if scale > 1:
                # Average pooling to reduce sequence length
                pooled = F.avg_pool1d(
                    x.transpose(1, 2),
                    kernel_size=scale,
                    stride=scale
                ).transpose(1, 2)
            else:
                pooled = x

            # Project and attend
            projected = self.scale_projections[scale_idx](pooled)
            attended, _ = self.scale_attentions[scale_idx](projected, projected, projected)

            # Upsample back if needed
            if scale > 1:
                attended = F.interpolate(
                    attended.transpose(1, 2),
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)

            scale_outputs.append(attended)

        # Fuse all scales
        fused = torch.cat(scale_outputs, dim=-1)
        output = self.scale_fusion(fused)

        return output


class RecursiveMambaBlock(nn.Module):
    """Hybrid block combining Mamba SSM with recursive processing."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand_factor: int = 2,
        dropout: float = 0.1,
        max_recursion_depth: int = 3,
        use_hierarchical_attention: bool = True
    ):
        """
        Initialize recursive Mamba block.

        Args:
            d_model: Model dimension
            d_state: State dimension for SSM
            expand_factor: Expansion factor
            dropout: Dropout rate
            max_recursion_depth: Maximum recursion depth
            use_hierarchical_attention: Whether to use hierarchical attention
        """
        super().__init__()
        self.d_model = d_model

        # Standard Mamba block
        self.mamba_block = MambaBlock(d_model, d_state, expand_factor, dropout)

        # Recursive processor
        self.recursive_processor = RecursiveProcessor(
            d_model,
            max_depth=max_recursion_depth,
            dropout=dropout
        )

        # Optional hierarchical attention
        self.use_hierarchical_attention = use_hierarchical_attention
        if use_hierarchical_attention:
            self.hierarchical_attention = HierarchicalAttention(d_model)

        # Fusion layer - projects concatenated outputs back to d_model
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, recursive_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through hybrid block.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            recursive_state: Previous recursive state

        Returns:
            output: Processed output
            new_recursive_state: Updated recursive state
        """
        # Mamba processing (efficient sequential modeling)
        mamba_out = self.mamba_block(x)

        # Recursive processing (hierarchical reasoning)
        recursive_out, new_recursive_state = self.recursive_processor(mamba_out, recursive_state)

        # Optional hierarchical attention
        if self.use_hierarchical_attention:
            recursive_out = recursive_out + self.hierarchical_attention(recursive_out)

        # Fuse Mamba and recursive outputs
        combined = torch.cat([mamba_out, recursive_out], dim=-1)
        output = self.fusion(combined)

        # Residual connection
        output = output + x

        return output, new_recursive_state


class HybridRecursiveMambaModel(nn.Module):
    """Hybrid model combining Mamba SSM with recursive language modeling."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        d_state: int = 16,
        expand_factor: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        max_recursion_depth: int = 3,
        use_hierarchical_attention: bool = True
    ):
        """
        Initialize hybrid recursive Mamba model.

        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_layers: Number of layers
            d_state: State dimension for SSM
            expand_factor: Expansion factor
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            max_recursion_depth: Maximum recursion depth
            use_hierarchical_attention: Whether to use hierarchical attention
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_recursion_depth = max_recursion_depth

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        # Hybrid recursive Mamba blocks
        self.blocks = nn.ModuleList([
            RecursiveMambaBlock(
                d_model,
                d_state,
                expand_factor,
                dropout,
                max_recursion_depth,
                use_hierarchical_attention
            )
            for _ in range(n_layers)
        ])

        # Final normalization
        self.norm_f = nn.LayerNorm(d_model)

        # Output head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        recursive_states: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            labels: Target token IDs (batch, seq_len) for loss computation
            recursive_states: List of recursive states for each layer (not used during training)

        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
            loss: Cross-entropy loss if labels provided, else None
        """
        batch, seq_len = input_ids.shape

        # Token + positional embeddings
        x = self.embedding(input_ids)
        x = x + self.pos_embedding[:, :seq_len, :]

        # Initialize recursive states if not provided
        if recursive_states is None:
            recursive_states = [None] * self.n_layers

        new_recursive_states = []

        # Apply hybrid blocks
        for i, block in enumerate(self.blocks):
            x, new_state = block(x, recursive_states[i])
            new_recursive_states.append(new_state)

        # Final normalization and projection
        x = self.norm_f(x)
        logits = self.lm_head(x)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )

        return logits, loss


def create_hybrid_recursive_mamba_model(config: dict) -> HybridRecursiveMambaModel:
    """
    Create a hybrid recursive Mamba model from config.

    Args:
        config: Model configuration dictionary

    Returns:
        Initialized hybrid model
    """
    model = HybridRecursiveMambaModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        d_state=config.get('d_state', 16),
        expand_factor=config.get('expand_factor', 2),
        dropout=config.get('dropout', 0.1),
        max_seq_len=config.get('max_seq_len', 2048),
        max_recursion_depth=config.get('max_recursion_depth', 3),
        use_hierarchical_attention=config.get('use_hierarchical_attention', True)
    )

    return model
