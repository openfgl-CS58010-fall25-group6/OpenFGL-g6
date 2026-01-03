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
            params_g = list(global_model_params.parameters())
        
        params_l = list(self.task.model.parameters())

        # Initialize weights if first round
        if self.weights is None:
            self.weights = [torch.ones_like(p).to(self.device) for p in params_l]

        # Create Temp Model
        model_t = copy.deepcopy(self.task.model)
        model_t.to(self.device)
        
        # Freeze temp model parameters (we only train 'weights')
        for param in model_t.parameters():
            param.requires_grad = False

        # --- Get the correct DataLoader from task ---
        if hasattr(self.task, 'train_dataloader') and self.task.train_dataloader is not None:
            loader = self.task.train_dataloader
        elif hasattr(self.task, 'processed_data') and self.task.processed_data is not None:
            loader = self.task.processed_data.get('train_dataloader')
        elif hasattr(self.task, 'splitted_data') and self.task.splitted_data is not None:
            loader = self.task.splitted_data.get('train_dataloader')
        else:
            raise ValueError("ALA: No data loader found.")

        model_t.eval()
        losses = []
        cnt = 0
        
        while True:
            for batch in loader:
                # Batch from DataLoader is already a PyG Batch object
                batch = batch.to(self.device)
                
                # 1. Initialize Temp Model: params_t = params_l + (params_g - params_l) * w
                params_tp = list(model_t.parameters())
                
                with torch.no_grad():
                    for param_t, param_l, param_g, weight in zip(params_tp, params_l, params_g, self.weights):
                        param_t.data = param_l.data + (param_g.data - param_l.data) * weight

                # 2. Enable gradients for temp model
                for param in model_t.parameters():
                    param.requires_grad = True
                    if param.grad is not None:
                        param.grad.zero_()

                # 3. Forward
                output = model_t(batch)
                if isinstance(output, tuple):
                    embedding, logits = output
                else:
                    logits = output
                    embedding = None

                # 4. Loss
                if hasattr(self.task, 'loss_fn') and embedding is not None:
                    loss = self.task.loss_fn(embedding, logits, batch.y, torch.ones_like(batch.y).bool())
                else:
                    loss = nn.functional.cross_entropy(logits, batch.y)

                # 5. Backward
                loss.backward()

                # 6. Update Weights
                with torch.no_grad():
                    for param_t, param_l, param_g, weight in zip(params_tp, params_l, params_g, self.weights):
                        if param_t.grad is not None:
                            grad_weight = param_t.grad * (param_g.data - param_l.data)
                            weight.data = torch.clamp(weight.data - self.eta * grad_weight, 0, 1)

                    for param in model_t.parameters():
                        param.requires_grad = False
                
                # 7. Update Temp Model
                with torch.no_grad():
                    for param_t, param_l, param_g, weight in zip(params_tp, params_l, params_g, self.weights):
                        param_t.data = param_l.data + (param_g.data - param_l.data) * weight

            # --- Loop Control ---
            losses.append(loss.item())
            cnt += 1

            if not self.start_phase:
                break
            
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