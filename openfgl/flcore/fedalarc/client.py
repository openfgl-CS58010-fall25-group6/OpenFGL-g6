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
    1. ALA module for adaptive local aggregation
    2. Byzantine attack simulation (optional)
    3. Standard local training
    """
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedALARCClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        
        # ========================================
        # Initialize ALA Module
        # ========================================
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
        
        # ========================================
        # Byzantine Attack Configuration
        # ========================================
        self._setup_byzantine_attack(args, client_id)
        
    def _setup_byzantine_attack(self, args, client_id):
        """
        Setup Byzantine attack if this client is adversarial.
        Handles parameter parsing robustly.
        
        Args:
            args: Command-line arguments
            client_id: This client's ID
        """
        # Get byzantine_ids and ensure it's a list
        byzantine_ids = getattr(args, 'byzantine_ids', [])
        
        # Parse byzantine_ids if it's a string (e.g., from YAML: "0,1,2")
        if isinstance(byzantine_ids, str):
            if byzantine_ids.strip():
                byzantine_ids = [int(x.strip()) for x in byzantine_ids.split(',')]
            else:
                byzantine_ids = []
        elif not isinstance(byzantine_ids, list):
            byzantine_ids = []
        
        # Check if this client is Byzantine
        self.is_byzantine = client_id in byzantine_ids
        self.attack = None
        
        if self.is_byzantine:
            # Get attack type
            attack_type = getattr(args, 'attack_type', 'sign_flip')
            
            # Parse attack_params (handle both string and dict)
            attack_params = getattr(args, 'attack_params', {})
            if isinstance(attack_params, str):
                try:
                    attack_params = json.loads(attack_params)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse attack_params '{attack_params}', using empty dict")
                    attack_params = {}
            elif not isinstance(attack_params, dict):
                attack_params = {}
            
            # Create attack object
            self.attack = ByzantineAttack(attack_type, attack_params)
            print(f"✗ Client {client_id} initialized as BYZANTINE with '{attack_type}' attack")
        else:
            print(f"✓ Client {client_id} initialized as HONEST")
    
    def execute(self):
        """
        Client execution for one round:
        1. Download Global Model from server
        2. Run ALA (Adaptive Local Aggregation) for initialization
        3. Run standard local training
        """
        # Get global weights from message pool
        global_weights = self.message_pool["server"]["weight"]
        
        # Perform ALA if global model exists (not first round)
        if global_weights is not None:
            self.ala.adaptive_local_aggregation(global_weights)
        
        # Proceed with standard local training
        self.task.train()

    def send_message(self):
        """
        Send model weights to server.
        If Byzantine, apply attack before sending.
        """
        # Get local model weights
        local_weights = list(self.task.model.parameters())
        
        # If Byzantine, attack the weights before sending
        if self.is_byzantine and self.attack is not None:
            # Get global weights for gradient-based attacks (like sign-flip)
            global_weights = self.message_pool.get("server", {}).get("weight", None)
            
            # Apply attack
            local_weights = self.attack.attack_weights(local_weights, global_weights)
            
            # Ensure weights are Parameters (not just tensors)
            local_weights = [
                w if isinstance(w, torch.nn.Parameter) else torch.nn.Parameter(w)
                for w in local_weights
            ]
        
        # Send to server
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": local_weights
        }