"""
Mamba architecture implementation for sequence modeling.
Based on the Mamba: Linear-Time Sequence Modeling with Selective State Spaces paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SelectiveSSM(nn.Module):
    """Selective State Space Model - core component of Mamba."""

    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        # Projections for selective scan
        self.x_proj = nn.Linear(d_model, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        # State space parameters
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape

        # Compute selective parameters
        x_proj_out = self.x_proj(x)  # (batch, seq_len, dt_rank + 2*d_state)

        delta = self.dt_proj(x_proj_out[..., :self.dt_rank])  # (batch, seq_len, d_model)
        B = x_proj_out[..., self.dt_rank:self.dt_rank + self.d_state]  # (batch, seq_len, d_state)
        C = x_proj_out[..., self.dt_rank + self.d_state:]  # (batch, seq_len, d_state)

        # Discretize continuous parameters
        delta = F.softplus(delta)
        A = -torch.exp(self.A_log)  # (d_model, d_state)

        # Simplified selective scan (for educational purposes)
        # In production, this should use optimized CUDA kernels
        y = self.selective_scan(x, delta, A, B, C)

        # Add skip connection
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)

        return y

    def selective_scan(self, x: torch.Tensor, delta: torch.Tensor,
                       A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient selective scan implementation.
        For production, use optimized causal-conv1d and selective-scan CUDA kernels.
        """
        batch, seq_len, d_model = x.shape

        # Pre-allocate output tensor to avoid list appending
        outputs = torch.zeros(batch, seq_len, d_model, device=x.device, dtype=x.dtype)

        # Initialize state - use smaller dtype in training if needed
        h = torch.zeros(batch, d_model, self.d_state, device=x.device, dtype=x.dtype)

        # Process in chunks to reduce memory
        chunk_size = min(64, seq_len)  # Process 64 timesteps at a time

        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)

            # Sequential scan within chunk
            for t in range(chunk_start, chunk_end):
                # Discretize (reuse tensors to avoid allocations)
                dt = delta[:, t, :].unsqueeze(-1)  # (batch, d_model, 1)
                dA = torch.exp(dt * A.unsqueeze(0))  # (batch, d_model, d_state)
                dB = dt * B[:, t, :].unsqueeze(1)  # (batch, 1, d_state)

                # State update (in-place where possible)
                h = h * dA + x[:, t, :].unsqueeze(-1) * dB  # (batch, d_model, d_state)

                # Output - write directly to pre-allocated tensor
                outputs[:, t, :] = torch.einsum('bds,bs->bd', h, C[:, t, :])

            # Clear CUDA cache between chunks if needed
            if chunk_end < seq_len and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return outputs


class MambaBlock(nn.Module):
    """Single Mamba block with normalization and projections."""

    def __init__(self, d_model: int, d_state: int = 16, expand_factor: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand_factor

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Selective SSM
        self.ssm = SelectiveSSM(self.d_inner, d_state)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)

        # Split projection into two paths
        x_proj = self.in_proj(x)
        x_1, x_2 = x_proj.chunk(2, dim=-1)

        # Apply activation to one path
        x_1 = F.silu(x_1)

        # Apply SSM to the other path
        x_2 = self.ssm(x_2)

        # Combine paths
        x = x_1 * x_2

        # Output projection
        x = self.out_proj(x)
        x = self.dropout(x)

        return x + residual


class MambaLanguageModel(nn.Module):
    """Mamba-based language model for text generation."""

    def __init__(self, vocab_size: int, d_model: int = 768, n_layers: int = 12,
                 d_state: int = 16, expand_factor: int = 2, dropout: float = 0.1,
                 max_seq_len: int = 2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, expand_factor, dropout)
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
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            labels: Target token IDs of shape (batch, seq_len) for loss computation

        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size)
            loss: Cross-entropy loss if labels provided, else None
        """
        batch, seq_len = input_ids.shape

        # Embed tokens
        x = self.embedding(input_ids)

        # Add positional embedding
        x = x + self.pos_embedding[:, :seq_len, :]

        # Apply Mamba blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.norm_f(x)

        # Project to vocabulary
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

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Initial token IDs of shape (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter

        Returns:
            Generated token IDs of shape (batch, seq_len + max_new_tokens)
        """
        self.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                logits, _ = self.forward(input_ids)

                # Get logits for last token
                logits = logits[:, -1, :] / temperature

                # Optional top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


def create_mamba_model(config: dict) -> MambaLanguageModel:
    """
    Create a Mamba model from configuration.

    Args:
        config: Dictionary containing model configuration

    Returns:
        MambaLanguageModel instance
    """
    return MambaLanguageModel(
        vocab_size=config.get('vocab_size', 50257),
        d_model=config.get('d_model', 768),
        n_layers=config.get('n_layers', 12),
        d_state=config.get('d_state', 16),
        expand_factor=config.get('expand_factor', 2),
        dropout=config.get('dropout', 0.1),
        max_seq_len=config.get('max_seq_len', 2048)
    )
