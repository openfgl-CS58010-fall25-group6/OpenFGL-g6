"""
Byzantine Attack Strategies for Robust Federated Learning Evaluation - Corrected Version

Implements various attack types from Byzantine ML literature:
- sign_flip: Flips the sign of gradients (most effective, from ARC paper)
- gaussian_noise: Adds large Gaussian noise
- zero: Sends zero gradients (non-participation)
- random: Sends random gradients
- label_flip: Implicit attack via label manipulation (not implemented here)

Key corrections:
1. Robust handling of Parameter vs Tensor types
2. Better documentation of attack mathematics
3. Added FOE (Fall of Empires) attack variant
"""

import torch
import copy
from typing import List, Optional, Dict, Any, Union


class ByzantineAttack:
    """
    Simulates various Byzantine (adversarial) attack strategies.
    
    These attacks are used to evaluate the robustness of federated learning
    algorithms like FedALARC against malicious clients.
    
    Usage:
        attack = ByzantineAttack('sign_flip')
        attacked_weights = attack.attack_weights(local_weights, global_weights)
    """
    
    SUPPORTED_ATTACKS = ['sign_flip', 'gaussian_noise', 'zero', 'random', 'foe', 'alie']
    
    def __init__(self, attack_type: str = 'sign_flip', attack_params: Optional[Dict[str, Any]] = None):
        """
        Initialize Byzantine attack.
        
        Args:
            attack_type (str): Type of attack to execute. Supported types:
                - 'sign_flip': Flip gradient sign (most effective)
                - 'gaussian_noise': Add Gaussian noise
                - 'zero': Send zero gradients
                - 'random': Send random weights
                - 'foe': Fall of Empires attack
                - 'alie': A Little Is Enough attack
            attack_params (dict): Additional parameters for specific attacks:
                - noise_scale: Scale for gaussian_noise (default: 10.0)
                - random_scale: Scale for random attack (default: 1.0)
                - foe_epsilon: Scaling factor for FOE attack (default: 1.0)
        """
        if attack_type not in self.SUPPORTED_ATTACKS:
            print(f"Warning: Unknown attack type '{attack_type}'. "
                  f"Supported: {self.SUPPORTED_ATTACKS}")
        
        self.attack_type = attack_type
        self.attack_params = attack_params or {}
    
    def _get_data(self, param: Union[torch.nn.Parameter, torch.Tensor]) -> torch.Tensor:
        """
        Safely extract tensor data from Parameter or Tensor.
        
        Args:
            param: Either a Parameter or Tensor
            
        Returns:
            Tensor data
        """
        if hasattr(param, 'data'):
            return param.data
        return param
        
    def attack_weights(
        self, 
        local_weights: List[Union[torch.nn.Parameter, torch.Tensor]], 
        global_weights: Optional[List[Union[torch.nn.Parameter, torch.Tensor]]] = None
    ) -> List[torch.Tensor]:
        """
        Apply Byzantine attack to model weights.
        
        Args:
            local_weights: List of local model parameters
            global_weights: List of global model parameters (for gradient-based attacks)
            
        Returns:
            Attacked weights (list of tensors)
        """
        if self.attack_type == 'sign_flip':
            return self._sign_flip_attack(local_weights, global_weights)
        elif self.attack_type == 'gaussian_noise':
            return self._gaussian_noise_attack(local_weights)
        elif self.attack_type == 'zero':
            return self._zero_attack(local_weights)
        elif self.attack_type == 'random':
            return self._random_attack(local_weights)
        elif self.attack_type == 'foe':
            return self._foe_attack(local_weights, global_weights)
        elif self.attack_type == 'alie':
            return self._alie_attack(local_weights, global_weights)
        else:
            # Unknown attack type, return original
            print(f"Warning: Unknown attack type '{self.attack_type}', returning original weights")
            return [self._get_data(w).clone() for w in local_weights]
    
    def _sign_flip_attack(
        self, 
        local_weights: List, 
        global_weights: Optional[List]
    ) -> List[torch.Tensor]:
        """
        Sign-flipping attack: Flip the sign of the gradient.
        
        This is one of the most effective Byzantine attacks because:
        1. It pushes the model in the opposite direction of convergence
        2. It has maximum impact on aggregation
        3. It's hard to detect without robust aggregation
        
        Mathematics:
            gradient = w_local - w_global
            flipped_gradient = -gradient
            w_attacked = w_global + flipped_gradient
                       = w_global - (w_local - w_global)
                       = 2 * w_global - w_local
        
        Args:
            local_weights: Local model parameters after training
            global_weights: Global model parameters (before local training)
            
        Returns:
            Attacked weights with flipped gradients
        """
        if global_weights is None:
            # If no global weights, just flip the local weights directly
            # This is less effective but still adversarial
            return [(-1.0 * self._get_data(param).clone()) for param in local_weights]
        
        # Compute gradient and flip it
        attacked = []
        for local_param, global_param in zip(local_weights, global_weights):
            local_data = self._get_data(local_param)
            global_data = self._get_data(global_param)
            
            # gradient = local - global (the update)
            gradient = local_data - global_data
            
            # Flip the gradient direction
            flipped_gradient = -gradient
            
            # w_attacked = w_global + flipped_gradient = 2*w_global - w_local
            attacked_weight = global_data + flipped_gradient
            attacked.append(attacked_weight.clone())
        
        return attacked
    
    def _foe_attack(
        self, 
        local_weights: List, 
        global_weights: Optional[List]
    ) -> List[torch.Tensor]:
        """
        Fall of Empires (FOE) attack.
        
        A sophisticated attack that scales the negative gradient by a factor.
        More aggressive than simple sign-flip.
        
        Mathematics:
            gradient = w_local - w_global
            w_attacked = w_global - epsilon * gradient
        
        Reference: Xie et al., "Fall of Empires: Breaking Byzantine-tolerant SGD"
        
        Args:
            local_weights: Local model parameters after training
            global_weights: Global model parameters
            
        Returns:
            Attacked weights
        """
        epsilon = self.attack_params.get('foe_epsilon', 1.0)
        
        if global_weights is None:
            return [(-epsilon * self._get_data(param).clone()) for param in local_weights]
        
        attacked = []
        for local_param, global_param in zip(local_weights, global_weights):
            local_data = self._get_data(local_param)
            global_data = self._get_data(global_param)
            
            gradient = local_data - global_data
            attacked_weight = global_data - epsilon * gradient
            attacked.append(attacked_weight.clone())
        
        return attacked
    
    def _alie_attack(
        self, 
        local_weights: List, 
        global_weights: Optional[List]
    ) -> List[torch.Tensor]:
        """
        A Little Is Enough (ALIE) attack.
        
        A stealthy attack that adds a small perturbation in the negative
        gradient direction. Designed to be harder to detect.
        
        Reference: Baruch et al., "A Little Is Enough: Circumventing Defenses"
        
        Args:
            local_weights: Local model parameters after training
            global_weights: Global model parameters
            
        Returns:
            Attacked weights with small perturbation
        """
        # ALIE uses a small scaling factor to stay under the radar
        z_max = self.attack_params.get('alie_z', 1.0)
        
        if global_weights is None:
            return [self._get_data(param).clone() for param in local_weights]
        
        attacked = []
        for local_param, global_param in zip(local_weights, global_weights):
            local_data = self._get_data(local_param)
            global_data = self._get_data(global_param)
            
            gradient = local_data - global_data
            
            # Small perturbation in negative direction
            # ALIE tries to find the maximum perturbation that won't be clipped
            std = gradient.std() if gradient.numel() > 1 else torch.tensor(1.0)
            perturbation = -z_max * std * torch.sign(gradient)
            
            attacked_weight = local_data + perturbation
            attacked.append(attacked_weight.clone())
        
        return attacked
    
    def _gaussian_noise_attack(self, local_weights: List) -> List[torch.Tensor]:
        """
        Gaussian noise attack: Add large Gaussian noise to weights.
        
        This attack aims to disrupt aggregation by sending very noisy gradients.
        Less effective than sign-flip but can still degrade performance.
        
        Args:
            local_weights: Local model parameters
            
        Returns:
            Weights with added Gaussian noise
        """
        noise_scale = self.attack_params.get('noise_scale', 10.0)
        
        attacked = []
        for param in local_weights:
            param_data = self._get_data(param)
            noise = torch.randn_like(param_data) * noise_scale
            attacked.append((param_data + noise).clone())
        
        return attacked
    
    def _zero_attack(self, local_weights: List) -> List[torch.Tensor]:
        """
        Zero attack: Send zero gradients (equivalent to not participating).
        
        This is a "lazy" attack that doesn't contribute to learning.
        Easy to detect but can slow down convergence.
        
        Args:
            local_weights: Local model parameters (used only for shape)
            
        Returns:
            Zero weights matching the shape of local_weights
        """
        return [torch.zeros_like(self._get_data(param)) for param in local_weights]
    
    def _random_attack(self, local_weights: List) -> List[torch.Tensor]:
        """
        Random attack: Send random weights.
        
        This attack sends completely random values, disrupting aggregation.
        Not as effective as targeted attacks but can cause instability.
        
        Args:
            local_weights: Local model parameters (used only for shape)
            
        Returns:
            Random weights matching the shape of local_weights
        """
        random_scale = self.attack_params.get('random_scale', 1.0)
        
        attacked = []
        for param in local_weights:
            param_data = self._get_data(param)
            random_weight = torch.randn_like(param_data) * random_scale
            attacked.append(random_weight.clone())
        
        return attacked


def create_byzantine_clients(
    num_clients: int, 
    num_byzantine: int, 
    attack_type: str = 'sign_flip', 
    attack_params: Optional[Dict[str, Any]] = None,
    random_selection: bool = False
) -> Dict[str, Any]:
    """
    Helper function to create Byzantine client configuration.
    
    Args:
        num_clients (int): Total number of clients
        num_byzantine (int): Number of Byzantine clients
        attack_type (str): Type of attack
        attack_params (dict): Attack parameters
        random_selection (bool): If True, randomly select Byzantine clients
                                 If False, use first num_byzantine clients
        
    Returns:
        dict: Configuration dict with:
            - byzantine_ids: List of Byzantine client IDs
            - attacks: Dict mapping client_id to ByzantineAttack object
            - attack_type: The attack type used
            - num_byzantine: Number of Byzantine clients
            - num_clients: Total number of clients
        
    Raises:
        ValueError: If num_byzantine >= num_clients // 2 (violates Byzantine assumption)
    """
    if num_byzantine >= num_clients // 2:
        raise ValueError(
            f"Number of Byzantine clients ({num_byzantine}) must be less than "
            f"half of total clients ({num_clients}). Byzantine assumption requires f < n/2."
        )
    
    if num_byzantine < 0:
        raise ValueError(f"Number of Byzantine clients cannot be negative: {num_byzantine}")
    
    # Select Byzantine clients
    if random_selection:
        import random
        byzantine_ids = random.sample(range(num_clients), num_byzantine)
        byzantine_ids.sort()
    else:
        # Select first num_byzantine clients as Byzantine
        byzantine_ids = list(range(num_byzantine))
    
    # Create attack objects
    attacks = {
        client_id: ByzantineAttack(attack_type, attack_params)
        for client_id in byzantine_ids
    }
    
    return {
        'byzantine_ids': byzantine_ids,
        'attacks': attacks,
        'attack_type': attack_type,
        'num_byzantine': num_byzantine,
        'num_clients': num_clients
    }
