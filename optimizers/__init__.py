"""
Optimizers package for Mamba training.

Available optimizers:
- Muon: Momentum-orthogonalized optimizer (fast convergence)
- MuonW: Muon with decoupled weight decay (recommended)
"""

from .muon import Muon, MuonW, create_muon_optimizer

__all__ = ['Muon', 'MuonW', 'create_muon_optimizer']
