import torch
import copy
from openfgl.flcore.base import BaseClient
from openfgl.flcore.fedala_plus.ala_module import ALA

class FedALAPlusClient(BaseClient):
    """
    FedALAPlusClient implements the client-side logic for FedALA+.
    Enhanced with disagreement-based sampling, caching, and fallback mechanisms.
    """
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedALAPlusClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        
        # Original FedALA parameters
        self.eta = getattr(args, 'eta', 1.0)
        self.rand_percent = getattr(args, 'rand_percent', 40)  # Default changed to 40%
        self.layer_idx = getattr(args, 'layer_idx', 1)
        self.threshold = getattr(args, 'threshold', 0.1)    
        self.num_pre_loss = getattr(args, 'num_pre_loss', 10)
        
        # NEW: Disagreement sampling parameters
        self.use_disagreement = getattr(args, 'use_disagreement', True)
        
        # NEW: Caching parameters (Solution 1)
        self.selection_frequency = getattr(args, 'selection_frequency', 1)  # Default: recompute every round
        
        # NEW: Fallback parameters (Solution 2)
        self.min_disagreement_samples = getattr(args, 'min_disagreement_samples', None)  # Default: 50% of rand_percent

        if hasattr(self.task, 'num_samples'):
            train_samples = self.task.num_samples
        else:
            train_samples = "Unknown"

        #debug prints
        """if client_id == 0:  # Only print for first client to avoid spam
            print(f"\n{'='*60}")
            print(f"FedALA+ Client {client_id} Initialization")
            print(f"{'='*60}")
            print(f"rand_percent: {self.rand_percent}")
            print(f"use_disagreement: {self.use_disagreement}")
            print(f"Training samples: {train_samples}")
            if isinstance(train_samples, int):
                print(f"Samples for ALA: {int(train_samples * self.rand_percent / 100)}")
            print(f"selection_frequency: {self.selection_frequency}")
            print(f"min_disagreement_samples: {self.min_disagreement_samples}")
            print(f"eta: {self.eta}")
            print(f"threshold: {self.threshold}")
            print(f"num_pre_loss: {self.num_pre_loss}")
            print(f"{'='*60}\n")"""
        
        self.ala = ALA(
            client_id=self.client_id,
            task=self.task,
            device=self.device,
            batch_size=args.batch_size,
            rand_percent=self.rand_percent,
            layer_idx=self.layer_idx,
            eta=self.eta,
            threshold=self.threshold, 
            num_pre_loss=self.num_pre_loss,
            selection_frequency=self.selection_frequency,  # NEW
            min_disagreement_samples=self.min_disagreement_samples  # NEW
        )
        
    def execute(self):
        """
        1. Download Global Model.
        2. Run ALA (Adaptive Initialization with disagreement sampling).
        3. Run Standard Local Training.
        """

        #debug prints
        """if self.client_id == 0 and self.message_pool.get("round", 0) == 0:
            print(f"\nClient {self.client_id}: execute() called")
            print(f"  use_disagreement = {self.use_disagreement}")
            print(f"  rand_percent = {self.rand_percent}\n")"""
        # Get global weights from message pool
        global_weights = self.message_pool["server"]["weight"]
        
        if global_weights is not None:
            # Perform Adaptive Local Aggregation with disagreement sampling
            self.ala.adaptive_local_aggregation(
                global_weights, 
                use_disagreement=self.use_disagreement
            )
        
        # Proceed with standard local training
        self.task.train()

    def send_message(self):
        """
        Standard FedAvg upload: num_samples and weights.
        Also includes ALA metrics for analysis.
        """
        # Get ALA metrics
        ala_metrics = self.ala.get_metrics_summary()
        
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
            "ala_metrics": ala_metrics  # NEW: Include metrics for server-side analysis
        }
    
    def get_ala_metrics(self):
        """
        Public method to retrieve ALA metrics for logging/analysis.
        """
        return self.ala.get_metrics_summary()