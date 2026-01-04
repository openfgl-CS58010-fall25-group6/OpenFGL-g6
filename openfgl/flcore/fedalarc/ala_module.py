"""
FedALARC ALA Module - Corrected Version

Implements Adaptive Local Aggregation (ALA) from FedALA paper.
Key correction: Properly implements layer_idx for selective layer adaptation.
"""

import torch
import torch.nn as nn
import copy
import numpy as np


class FedALARC_ALA:
    def __init__(self,
                 client_id,
                 task,
                 device,
                 batch_size=32,
                 rand_percent=80,
                 layer_idx=0,
                 eta=1.0,
                 threshold=0.1,
                 num_pre_loss=10):
        """
        FedALA Module for OpenFGL.
        
        Args:
            client_id: Client identifier
            task: Task object containing model and data
            device: Torch device (cuda/cpu)
            batch_size: Batch size for ALA weight learning
            rand_percent: Percentage of local data to use for ALA (s% in paper)
            layer_idx: Number of higher layers to apply ALA to (p in paper)
                       0 means apply to ALL layers
                       >0 means apply only to the last `layer_idx` layers
            eta: Learning rate for ALA weight learning
            threshold: Convergence threshold for weight learning
            num_pre_loss: Number of losses to check for convergence
        """
        self.cid = client_id
        self.task = task 
        self.device = device
        self.batch_size = batch_size
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        
        self.weights = None 
        self.start_phase = True

    def _get_adaptive_layer_mask(self, num_params):
        """
        Determine which layers should be adapted vs. overwritten.
        
        According to FedALA paper (Section "Adaptive Local Aggregation"):
        - Lower layers learn more generic information (desired by all clients)
        - Higher layers learn more task-specific information (need adaptation)
        - We apply ALA only on p higher layers and overwrite lower layers
        
        Args:
            num_params: Total number of parameter groups in model
            
        Returns:
            start_idx: Index from which to start ALA adaptation
                       Layers [0, start_idx) are overwritten (weight=1)
                       Layers [start_idx, num_params) are adapted
        """
        if self.layer_idx <= 0:
            # Apply ALA to ALL layers
            return 0
        else:
            # Apply ALA only to last `layer_idx` layers
            return max(0, num_params - self.layer_idx)

    def adaptive_local_aggregation(self, global_model_params):
        """
        Perform Adaptive Local Aggregation.
        
        Aggregates global and local model parameters using learned element-wise weights:
            param_initialized = param_local + (param_global - param_local) * weight
        
        where weight is in [0, 1]:
            - weight = 0: Keep local parameter entirely
            - weight = 1: Take global parameter entirely (overwrite)
        
        Args:
            global_model_params: Global model parameters (dict, list, or nn.Module)
        """
        # Parse global model parameters
        if isinstance(global_model_params, dict):
            params_g = list(global_model_params.values())
        elif isinstance(global_model_params, list):
            params_g = global_model_params
        else:
            params_g = list(global_model_params.parameters())
        
        # Get local model parameters
        params_l = list(self.task.model.parameters())
        num_params = len(params_l)

        # Initialize weights if first round
        if self.weights is None:
            self.weights = [torch.ones_like(p.data).to(self.device) for p in params_l]

        # ========================================
        # Apply layer_idx: Determine which layers to adapt
        # ========================================
        start_idx = self._get_adaptive_layer_mask(num_params)
        
        # For layers before start_idx, keep weights at 1 (take global, i.e., overwrite)
        # This implements Equation (4) from the FedALA paper:
        # Θ_i^t := Θ_i^{t-1} + (Θ^{t-1} - Θ_i^{t-1}) ⊙ [1^{|Θ_i|-p}; W_i^p]
        for i in range(start_idx):
            self.weights[i] = torch.ones_like(params_l[i].data).to(self.device)

        # Create Temp Model for weight learning
        model_t = copy.deepcopy(self.task.model)
        model_t.to(self.device)
        
        # Freeze temp model parameters (we only train 'weights')
        for param in model_t.parameters():
            param.requires_grad = False

        # --- Get the correct DataLoader from task ---
        loader = self._get_data_loader()
        if loader is None:
            raise ValueError("ALA: No data loader found.")

        model_t.eval()
        losses = []
        cnt = 0
        
        # Weight learning loop
        while True:
            for batch in loader:
                # Move batch to device
                batch = batch.to(self.device)
                
                # 1. Initialize Temp Model: params_t = params_l + (params_g - params_l) * w
                params_tp = list(model_t.parameters())
                
                with torch.no_grad():
                    for idx, (param_t, param_l, param_g, weight) in enumerate(
                        zip(params_tp, params_l, params_g, self.weights)
                    ):
                        # Get data from parameters (handle both Parameter and Tensor)
                        param_g_data = param_g.data if hasattr(param_g, 'data') else param_g
                        param_t.data = param_l.data + (param_g_data - param_l.data) * weight

                # 2. Enable gradients for temp model
                for param in model_t.parameters():
                    param.requires_grad = True
                    if param.grad is not None:
                        param.grad.zero_()

                # 3. Forward pass
                output = model_t(batch)
                if isinstance(output, tuple):
                    embedding, logits = output
                else:
                    logits = output
                    embedding = None

                # 4. Compute Loss
                if hasattr(self.task, 'loss_fn') and embedding is not None:
                    loss = self.task.loss_fn(embedding, logits, batch.y, torch.ones_like(batch.y).bool())
                else:
                    loss = nn.functional.cross_entropy(logits, batch.y)

                # 5. Backward pass
                loss.backward()

                # 6. Update Weights (only for adaptive layers)
                with torch.no_grad():
                    for idx, (param_t, param_l, param_g, weight) in enumerate(
                        zip(params_tp, params_l, params_g, self.weights)
                    ):
                        # Skip lower layers (keep weight=1 for overwrite)
                        if idx < start_idx:
                            continue
                            
                        if param_t.grad is not None:
                            param_g_data = param_g.data if hasattr(param_g, 'data') else param_g
                            grad_weight = param_t.grad * (param_g_data - param_l.data)
                            weight.data = torch.clamp(weight.data - self.eta * grad_weight, 0, 1)

                    # Disable gradients for next iteration
                    for param in model_t.parameters():
                        param.requires_grad = False
                
                # 7. Update Temp Model with new weights
                with torch.no_grad():
                    for param_t, param_l, param_g, weight in zip(params_tp, params_l, params_g, self.weights):
                        param_g_data = param_g.data if hasattr(param_g, 'data') else param_g
                        param_t.data = param_l.data + (param_g_data - param_l.data) * weight

            # --- Loop Control ---
            losses.append(loss.item())
            cnt += 1

            # Exit conditions
            if not self.start_phase:
                # After initial convergence, only do 1 epoch per round
                break
            
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                print(f'Client {self.cid} ALA converged at epoch {cnt}')
                break
            
            if cnt > 50:
                print(f'Client {self.cid} ALA reached max epochs (50)')
                break

        self.start_phase = False

        # 8. Apply Final Learned Weights to REAL Local Model
        with torch.no_grad():
            for param_l, param_g, weight in zip(params_l, params_g, self.weights):
                param_g_data = param_g.data if hasattr(param_g, 'data') else param_g
                param_l.data = param_l.data + (param_g_data - param_l.data) * weight

    def _get_data_loader(self):
        """
        Get the appropriate data loader from the task object.
        
        Returns:
            DataLoader object or None if not found
        """
        # Try different possible locations for the data loader
        if hasattr(self.task, 'train_dataloader') and self.task.train_dataloader is not None:
            return self.task.train_dataloader
        elif hasattr(self.task, 'processed_data') and self.task.processed_data is not None:
            return self.task.processed_data.get('train_dataloader')
        elif hasattr(self.task, 'splitted_data') and self.task.splitted_data is not None:
            return self.task.splitted_data.get('train_dataloader')
        return None
