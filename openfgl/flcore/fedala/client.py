import torch
import copy
from openfgl.flcore.base import BaseClient
from openfgl.flcore.fedala.ala_module import ALA

class FedALAClient(BaseClient):
    """
    FedALAClient implements the client-side logic for FedALA.
    It uses the ALA module to initialize the local model adaptively
    using the global model downloaded from the server.
    """
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedALAClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        
        # Initialize ALA Module
        # args should contain hyperparameters like eta, rand_percent, layer_idx
        self.eta = getattr(args, 'eta', 1.0)
        self.rand_percent = getattr(args, 'rand_percent', 80)
        self.layer_idx = getattr(args, 'layer_idx', 0) # 0 means adapt all layers
        
        self.threshold = getattr(args, 'threshold', 0.1)    
        self.num_pre_loss = getattr(args, 'num_pre_loss', 10) 
        
        self.ala = ALA(
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
        
    def execute(self):
        """
        1. Download Global Model.
        2. Run ALA (Adaptive Initialization).
        3. Run Standard Local Training.
        """
        # Get global weights from message pool
        global_weights = self.message_pool["server"]["weight"]
        
        # Check if it's the very first round (global weights might be None or init)
        # We perform ALA only if we have a valid global model to adapt to.
        if global_weights is not None:
             # Convert list of tensors to list of parameters/tensors structure matching model
             # The message pool stores them as a list of tensors.
             
             # Perform Adaptive Local Aggregation
             # detailed in [cite: 8]
             self.ala.adaptive_local_aggregation(global_weights)
        
        # Proceed with standard local training
        self.task.train()

    def send_message(self):
        """
        Standard FedAvg upload: num_samples and weights.
        """
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "weight": list(self.task.model.parameters())
            }