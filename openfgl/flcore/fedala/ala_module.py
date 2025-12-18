import torch
import torch.nn as nn
import copy
import numpy as np
from torch.utils.data import DataLoader

class ALA:
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
        Strictly aligned with official implementation (ALA.py) but patched for GCN/Subgraph.
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

    def adaptive_local_aggregation(self, global_model_params):
        if isinstance(global_model_params, dict):
            params_g = list(global_model_params.values())
        elif isinstance(global_model_params, list):
            params_g = global_model_params
        else:
            # Fallback if it's a model object
            params_g = list(global_model_params.parameters())
        params_l = list(self.task.model.parameters())

        # Initialize weights if first round
        if self.weights is None:
            self.weights = [torch.ones_like(p).to(self.device) for p in params_l]

        # Create Temp Model
        model_t = copy.deepcopy(self.task.model)
        
        # Freeze temp model parameters (we only train 'weights')
        for param in model_t.parameters():
            param.requires_grad = False

        # --- Data Loading (Handle Subgraph vs Graph) ---
        if hasattr(self.task, 'train_loader') and self.task.train_loader is not None:
            loader = self.task.train_loader # Graph-FL
        elif hasattr(self.task, 'data'):
             loader = [self.task.data]      # Subgraph-FL
        else:
             raise ValueError("ALA: No data found.")

        model_t.eval()
        losses = []
        cnt = 0
        
        while True:
            for batch in loader:
                batch = batch.to(self.device)
                
                # 1. Initialize Temp Model: params_t = params_l + (params_g - params_l) * w
                params_tp = list(model_t.parameters())
                params_p = params_l
                params_gp = params_g
                
                with torch.no_grad():
                    for param_t, param_l, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
                        param_t.data = param_l.data + (param_g.data - param_l.data) * weight

                # 2. Get Gradients (Manual Zero Grad)
                for param in model_t.parameters():
                    param.requires_grad = True
                    if param.grad is not None:
                        param.grad.zero_()

                # 3. Forward (Patched for GCN)
                if hasattr(batch, 'edge_index'):
                     output = model_t(batch.x, batch.edge_index)
                else:
                     output = model_t(batch)
                
                if isinstance(output, tuple): output = output[0]

                # 4. Loss (Patched for Subgraph Mask)
                if hasattr(batch, 'train_mask') and batch.train_mask is not None:
                    loss = nn.functional.cross_entropy(output[batch.train_mask], batch.y[batch.train_mask])
                elif hasattr(self.task, 'criterion'):
                    loss = self.task.criterion(output, batch.y)
                else:
                    if batch.y.is_floating_point():
                         loss = nn.functional.binary_cross_entropy_with_logits(output, batch.y)
                    else:
                         loss = nn.functional.cross_entropy(output, batch.y)

                # 5. Backward
                loss.backward()

                # 6. Update Weights (Strict Official Logic)
                # Formula: w_new = w - eta * grad_w
                # Chain rule: grad_w = grad_param_t * (param_g - param_l)
                with torch.no_grad():
                    for param_t, param_l, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
                        if param_t.grad is not None:
                            grad_weight = param_t.grad * (param_g.data - param_l.data)
                            weight.data = torch.clamp(weight.data - self.eta * grad_weight, 0, 1)

                    # Reset temp model grads
                    for param in model_t.parameters():
                        param.requires_grad = False
                
                # 7. Update Temp Model IMMEDIATELY (Official ALA.py logic)
                # The official code updates the temp model inside the batch loop
                with torch.no_grad():
                    for param_t, param_l, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
                         param_t.data = param_l.data + (param_g.data - param_l.data) * weight

            # --- Loop Control (Strict Official Logic) ---
            losses.append(loss.item())
            cnt += 1

            # "only train one epoch in the subsequent iterations" (from official ALA.py)
            if not self.start_phase:
                break
            
            # Start Phase Convergence
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                print(f'Client {self.cid} ALA converged at epoch {cnt}')
                break
            
            if cnt > 50: 
                break

        self.start_phase = False

        # 8. Apply Final Learned Weights to REAL Local Model
        with torch.no_grad():
             for param_l, param_g, weight in zip(params_l, params_g, self.weights):
                param_l.data = param_l.data + (param_g.data - param_l.data) * weight