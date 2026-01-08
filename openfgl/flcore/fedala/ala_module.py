import numpy as np
import torch
import torch.nn as nn
import copy

class ALA:
    def __init__(self,
                 client_id,
                 task,
                 device,
                 batch_size=32,
                 rand_percent=80,
                 layer_idx=1,
                 eta=1.0,
                 threshold=0.1,
                 num_pre_loss=10):
        """
        FedALA Module for OpenFGL.
        Aligned with official FedALA implementation (https://github.com/TsingZ0/FedALA).
        
        Args:
            client_id: Client ID.
            task: The OpenFGL task object containing model and data.
            device: Using cuda or cpu.
            batch_size: Weight learning batch size.
            rand_percent: The percent of the local training data to sample.
            layer_idx: Control the weight range. layer_idx=1 means adapt only last layer. 
                       layer_idx=0 means adapt all layers. Default: 1
            eta: Weight learning rate. Default: 1.0
            threshold: Train the weight until the std of recorded losses < threshold. Default: 0.1
            num_pre_loss: Number of recorded losses for std calculation. Default: 10
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

        self.weights = None  # Learnable local aggregation weights
        self.start_phase = True

    def adaptive_local_aggregation(self, global_model_params):
        """
        Adaptive local aggregation following official FedALA implementation.
        
        Args:
            global_model_params: The received global model parameters (list or dict).
        """
        # Obtain the references of the parameters
        if isinstance(global_model_params, dict):
            params_g = list(global_model_params.values())
        elif isinstance(global_model_params, list):
            params_g = global_model_params
        else:
            params_g = list(global_model_params.parameters())

        params = list(self.task.model.parameters())

        # Deactivate ALA at the 1st communication iteration (when global == local)
        if torch.sum(params_g[0].data - params[0].data) == 0:
            return

        # Get data loader
        if hasattr(self.task, 'train_dataloader') and self.task.train_dataloader is not None:
            rand_loader = self.task.train_dataloader
        elif hasattr(self.task, 'processed_data') and self.task.processed_data is not None:
            rand_loader = self.task.processed_data.get('train_dataloader')
        elif hasattr(self.task, 'splitted_data') and self.task.splitted_data is not None:
            rand_loader = self.task.splitted_data.get('train_dataloader')
        else:
            raise ValueError("ALA: No data loader found.")

        # Handle layer_idx logic (official implementation uses negative indexing)
        if self.layer_idx == 0:
            # Adapt all layers
            params_p = params
            params_gp = params_g
        else:
            # Preserve all the updates in the lower layers (copy global to local)
            for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
                param.data = param_g.data.clone()

            # Only consider higher layers for ALA
            params_p = params[-self.layer_idx:]
            params_gp = params_g[-self.layer_idx:]

        # Temp local model only for weight learning
        model_t = copy.deepcopy(self.task.model)
        model_t.to(self.device)
        params_t = list(model_t.parameters())

        # Get higher layer params from temp model
        if self.layer_idx == 0:
            params_tp = params_t
        else:
            params_tp = params_t[-self.layer_idx:]
            # Freeze the lower layers to reduce computational cost
            for param in params_t[:-self.layer_idx]:
                param.requires_grad = False

        # Used to obtain the gradient of higher layers
        # No need to use optimizer.step(), so lr=0
        optimizer = torch.optim.SGD(params_tp, lr=0)

        # Initialize the weight to all ones in the beginning
        if self.weights is None:
            self.weights = [torch.ones_like(param.data).to(self.device) for param in params_p]

        # Initialize the higher layers in the temp local model
        for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
            param_t.data = param.data + (param_g.data - param.data) * weight

        # Weight learning
        losses = []  # Record losses
        cnt = 0  # Weight training iteration counter

        while True:
            for batch in rand_loader:
                batch = batch.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass - OpenFGL models return (embedding, logits)
                output = model_t(batch)
                if isinstance(output, tuple):
                    embedding, logits = output
                else:
                    logits = output
                    embedding = None

                # Compute loss
                if hasattr(self.task, 'loss_fn') and embedding is not None:
                    loss_value = self.task.loss_fn(embedding, logits, batch.y, torch.ones_like(batch.y).bool())
                else:
                    loss_value = nn.functional.cross_entropy(logits, batch.y)

                loss_value.backward()

                # Update weight in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
                    if param_t.grad is not None:
                        weight.data = torch.clamp(
                            weight - self.eta * (param_t.grad * (param_g.data - param.data)), 0, 1)

                # Update temp local model in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
                    param_t.data = param.data + (param_g.data - param.data) * weight

            losses.append(loss_value.item())
            cnt += 1

            # Only train one epoch in the subsequent iterations
            if not self.start_phase:
                break

            # Train the weight until convergence
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                print(f'Client {self.cid}: Std={np.std(losses[-self.num_pre_loss:]):.4f}, ALA epochs={cnt}')
                break

            # Safety break
            if cnt > 50:
                break

        self.start_phase = False

        # Obtain initialized local model (apply learned weights)
        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()