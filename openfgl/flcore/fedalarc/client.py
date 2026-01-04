"""
FedALARC Client - Corrected Version

Implements FedALA client with Byzantine attack simulation support.
Key corrections:
1. More robust Byzantine attack parameter parsing
2. Better handling of Parameter vs Tensor types
3. Cleaner separation of concerns
"""

import torch
import copy
import json
from openfgl.flcore.base import BaseClient
from openfgl.flcore.fedalarc.ala_module import FedALARC_ALA
from openfgl.flcore.fedalarc.byzantine_attacks import ByzantineAttack


class FedALARCClient(BaseClient):
    """
    FedALARC Client: FedALA with Adaptive Robust Clipping support.
    
    Implements:
    1. ALA module for adaptive local aggregation (client-side personalization)
    2. Byzantine attack simulation (optional, for robustness evaluation)
    3. Standard local training
    
    The ALA module learns element-wise weights to selectively aggregate
    global and local model parameters, capturing desired information
    from the global model while preserving local knowledge.
    """
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedALARCClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        
        # ========================================
        # Initialize ALA Module
        # ========================================
        self._setup_ala(args)
        
        # ========================================
        # Byzantine Attack Configuration
        # ========================================
        self._setup_byzantine_attack(args, client_id)
    
    def _setup_ala(self, args):
        """
        Setup ALA module with parameters from args.
        
        ALA Parameters (from FedALA paper):
        - eta: Learning rate for weight learning (default: 1.0)
        - rand_percent: Percentage of local data to use (s% in paper, default: 80)
        - layer_idx: Number of higher layers to adapt (p in paper, default: 1)
                     0 = adapt all layers, >0 = adapt only last `layer_idx` layers
        - threshold: Convergence threshold for weight learning (default: 0.1)
        - num_pre_loss: Number of losses to check for convergence (default: 10)
        """
        self.eta = getattr(args, 'eta', 1.0)
        self.rand_percent = getattr(args, 'rand_percent', 80)
        self.layer_idx = getattr(args, 'layer_idx', 1)
        self.threshold = getattr(args, 'threshold', 0.1)    
        self.num_pre_loss = getattr(args, 'num_pre_loss', 10) 
        
        self.ala = FedALARC_ALA(
            client_id=self.client_id,
            task=self.task,
            device=self.device,
            batch_size=args.batch_size,
            rand_percent=self.rand_percent,
            layer_idx=self.layer_idx,
            eta=self.eta,
            threshold=self.threshold, 
            num_pre_loss=self.num_pre_loss
        )
        
    def _setup_byzantine_attack(self, args, client_id):
        """
        Setup Byzantine attack if this client is adversarial.
        
        Handles parameter parsing robustly for various input formats:
        - byzantine_ids: Can be list, string ("0,1,2"), or empty
        - attack_type: String specifying attack type
        - attack_params: Dict or JSON string with attack-specific parameters
        
        Args:
            args: Command-line arguments
            client_id: This client's ID
        """
        # ========================================
        # Parse byzantine_ids
        # ========================================
        byzantine_ids = getattr(args, 'byzantine_ids', [])
        
        # Handle different input formats
        if isinstance(byzantine_ids, str):
            # Parse comma-separated string: "0,1,2" -> [0, 1, 2]
            if byzantine_ids.strip():
                try:
                    byzantine_ids = [int(x.strip()) for x in byzantine_ids.split(',')]
                except ValueError as e:
                    print(f"Warning: Could not parse byzantine_ids '{byzantine_ids}': {e}")
                    byzantine_ids = []
            else:
                byzantine_ids = []
        elif isinstance(byzantine_ids, (int, float)):
            # Single value: convert to list
            byzantine_ids = [int(byzantine_ids)]
        elif not isinstance(byzantine_ids, list):
            # Unknown type
            print(f"Warning: Unknown byzantine_ids type {type(byzantine_ids)}, using empty list")
            byzantine_ids = []
        
        # Ensure all IDs are integers
        byzantine_ids = [int(x) for x in byzantine_ids]
        
        # ========================================
        # Check if this client is Byzantine
        # ========================================
        self.is_byzantine = client_id in byzantine_ids
        self.attack = None
        
        if self.is_byzantine:
            # Get attack type
            attack_type = str(getattr(args, 'attack_type', 'sign_flip'))
            
            # Parse attack_params
            attack_params = getattr(args, 'attack_params', {})
            if isinstance(attack_params, str):
                # Parse JSON string
                try:
                    attack_params = json.loads(attack_params) if attack_params.strip() else {}
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse attack_params '{attack_params}': {e}")
                    attack_params = {}
            elif not isinstance(attack_params, dict):
                attack_params = {}
            
            # Create attack object
            self.attack = ByzantineAttack(attack_type, attack_params)
            print(f"✗ Client {client_id} initialized as BYZANTINE")
            print(f"  Attack type: '{attack_type}'")
            if attack_params:
                print(f"  Attack params: {attack_params}")
        else:
            print(f"✓ Client {client_id} initialized as HONEST")
    
    def execute(self):
        """
        Client execution for one round.
        
        Steps:
        1. Download global model weights from server (via message pool)
        2. Run ALA (Adaptive Local Aggregation) for adaptive initialization
           - Learns element-wise weights to aggregate global and local models
           - Captures desired information from global while preserving local knowledge
        3. Run standard local training on the initialized model
        
        Note: Byzantine attacks are applied in send_message(), not here.
        This ensures honest local training even for Byzantine clients,
        with attacks only affecting what gets sent to the server.
        """
        # Get global weights from message pool
        global_weights = self.message_pool["server"]["weight"]
        
        # Perform ALA if global model exists (skip on first round)
        if global_weights is not None:
            self.ala.adaptive_local_aggregation(global_weights)
        
        # Proceed with standard local training
        self.task.train()

    def send_message(self):
        """
        Send model weights to server.
        
        If this client is Byzantine, apply the configured attack before sending.
        This simulates adversarial behavior where a compromised client sends
        malicious updates to try to degrade the global model.
        
        Attack types include:
        - sign_flip: Flips gradient direction (most effective attack)
        - gaussian_noise: Adds large noise to weights
        - zero: Sends zero gradients (non-participation)
        - random: Sends random weights
        """
        # Get local model weights as a list of Parameters
        local_weights = list(self.task.model.parameters())
        
        # ========================================
        # Apply Byzantine attack if configured
        # ========================================
        if self.is_byzantine and self.attack is not None:
            # Get global weights for gradient-based attacks (like sign-flip)
            global_weights = self.message_pool.get("server", {}).get("weight", None)
            
            # Apply attack - this modifies the weights before sending
            attacked_weights = self.attack.attack_weights(local_weights, global_weights)
            
            # Ensure all weights are Parameters (attacks may return Tensors)
            local_weights = self._ensure_parameters(attacked_weights)
        
        # ========================================
        # Send to server via message pool
        # ========================================
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": local_weights
        }
    
    def _ensure_parameters(self, weights):
        """
        Ensure all weights are torch.nn.Parameter objects.
        
        Some operations (like Byzantine attacks) may return plain Tensors.
        This method converts them back to Parameters for consistency.
        
        Args:
            weights: List of weights (Parameters or Tensors)
            
        Returns:
            List of Parameters
        """
        result = []
        for w in weights:
            if isinstance(w, torch.nn.Parameter):
                result.append(w)
            elif isinstance(w, torch.Tensor):
                # Clone to ensure we don't share memory with original
                result.append(torch.nn.Parameter(w.clone().detach()))
            else:
                # Try to convert to tensor first
                result.append(torch.nn.Parameter(torch.tensor(w)))
        return result
