"""
FedALARC: Federated Learning with Adaptive Local Aggregation and Robust Clipping

This module combines two techniques for robust personalized federated learning:

1. FedALA (Adaptive Local Aggregation) - From AAAI 2023
   - Learns element-wise weights to selectively aggregate global and local models
   - Captures desired information from global model while preserving local knowledge
   - Applied on client side during local initialization

2. ARC (Adaptive Robust Clipping) - From ICLR 2025
   - Adaptive pre-aggregation gradient clipping for Byzantine robustness
   - Clips largest k gradients where k = floor(2*(f/n)*(n-f))
   - Clipping threshold is the (k+1)-th largest gradient norm
   - Applied on server side before aggregation

Usage:
    from openfgl.flcore.fedalarc import FedALARCClient, FedALARCServer
    
    # For Byzantine robustness evaluation
    from openfgl.flcore.fedalarc import ByzantineAttack, create_byzantine_clients

Configuration (in args):
    ALA parameters:
        - eta: Learning rate for ALA weights (default: 1.0)
        - rand_percent: Percentage of data for ALA (default: 80)
        - layer_idx: Number of layers to adapt (default: 1, 0=all)
        - threshold: Convergence threshold (default: 0.1)
        - num_pre_loss: Losses for convergence check (default: 10)
    
    ARC parameters:
        - use_arc: Enable ARC clipping (default: False)
        - max_byzantine: Number of Byzantine clients to tolerate (f)
    
    Byzantine simulation:
        - byzantine_ids: List of Byzantine client IDs
        - attack_type: Attack type ('sign_flip', 'gaussian_noise', etc.)
        - attack_params: Attack-specific parameters (dict)
"""

from .client import FedALARCClient
from .server import FedALARCServer
from .ala_module import FedALARC_ALA
from .byzantine_attacks import ByzantineAttack, create_byzantine_clients

__all__ = [
    'FedALARCClient',
    'FedALARCServer', 
    'FedALARC_ALA',
    'ByzantineAttack',
    'create_byzantine_clients'
]

__version__ = '1.0.0'
