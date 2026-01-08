"""
Byzantine attack strategies for robust federated learning evaluation.

Implements various attack types:
- sign_flip: Flips the sign of gradients (most effective)
- gaussian_noise: Adds large Gaussian noise
- zero: Sends zero gradients (non-participation)
- random: Sends random gradients
"""

import torch
import copy


class ByzantineAttack:
    """
    Simulates various Byzantine (adversarial) attack strategies.
    
    Usage:
        attack = ByzantineAttack('sign_flip')
        attacked_weights = attack.attack_weights(local_weights, global_weights)
    """
    
    def __init__(self, attack_type='sign_flip', attack_params=None):
        """
        Args:
            attack_type (str): Type of attack to execute
            attack_params (dict): Additional parameters for specific attacks
                - noise_scale: Scale for gaussian_noise (default: 10.0)
                - random_scale: Scale for random attack (default: 1.0)
        """
        self.attack_type = attack_type
        self.attack_params = attack_params or {}
        
    def attack_weights(self, local_weights, global_weights=None):
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
        else:
            # Unknown attack type, return original
            print(f"Warning: Unknown attack type '{self.attack_type}', returning original weights")
            return local_weights
    
    def _sign_flip_attack(self, local_weights, global_weights):
        """
        Sign-flipping attack: Flip the sign of the gradient.
        
        This is one of the most effective Byzantine attacks because:
        1. It pushes the model in the opposite direction of convergence
        2. It has maximum impact on aggregation
        3. It's hard to detect without robust aggregation
        
        Attack: w_attacked = w_global - (w_local - w_global) = 2*w_global - w_local
        """
        if global_weights is None:
            # If no global weights, just flip the local weights
            return [(-1.0 * param.data.clone()) for param in local_weights]
        
        # Compute gradient and flip it
        attacked = []
        for local_param, global_param in zip(local_weights, global_weights):
            gradient = local_param.data - global_param.data
            flipped_gradient = -gradient
            attacked_weight = global_param.data + flipped_gradient
            attacked.append(attacked_weight.clone())
        
        return attacked
    
    def _gaussian_noise_attack(self, local_weights):
        """
        Gaussian noise attack: Add large Gaussian noise to weights.
        
        This attack aims to disrupt aggregation by sending very noisy gradients.
        """
        noise_scale = self.attack_params.get('noise_scale', 10.0)
        
        attacked = []
        for param in local_weights:
            noise = torch.randn_like(param.data) * noise_scale
            attacked.append((param.data + noise).clone())
        
        return attacked
    
    def _zero_attack(self, local_weights):
        """
        Zero attack: Send zero gradients (equivalent to not participating).
        
        This is a "lazy" attack that doesn't contribute to learning.
        """
        return [torch.zeros_like(param.data) for param in local_weights]
    
    def _random_attack(self, local_weights):
        """
        Random attack: Send random weights.
        
        This attack sends completely random values, disrupting aggregation.
        """
        random_scale = self.attack_params.get('random_scale', 1.0)
        
        attacked = []
        for param in local_weights:
            random_weight = torch.randn_like(param.data) * random_scale
            attacked.append(random_weight.clone())
        
        return attacked


def create_byzantine_clients(num_clients, num_byzantine, attack_type='sign_flip', attack_params=None):
    """
    Helper function to create Byzantine client configuration.
    
    Args:
        num_clients (int): Total number of clients
        num_byzantine (int): Number of Byzantine clients
        attack_type (str): Type of attack
        attack_params (dict): Attack parameters
        
    Returns:
        dict: Configuration dict with byzantine_ids and attack objects
        
    Raises:
        ValueError: If num_byzantine >= num_clients // 2 (violates Byzantine assumption)
    """
    if num_byzantine >= num_clients // 2:
        raise ValueError(
            f"Number of Byzantine clients ({num_byzantine}) must be less than "
            f"half of total clients ({num_clients}). Byzantine assumption requires f < n/2."
        )
    
    # Select first num_byzantine clients as Byzantine (can be randomized)
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