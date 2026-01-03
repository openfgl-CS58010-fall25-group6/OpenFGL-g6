import torch
import copy
from openfgl.flcore.fedavg.server import FedAvgServer


class FedALARCServer(FedAvgServer):
    """
    FedALARC Server: FedALA with Adaptive Robust Clipping (ARC).
    
    Key features:
    1. Standard FedAvg aggregation (inherited)
    2. Optional ARC pre-aggregation clipping for Byzantine robustness
    
    ARC Algorithm:
    - Computes gradient norms from all clients
    - Clips the largest k gradients where k = floor(2*(f/n)*(n-f))
    - Clipping threshold C = norm of (k+1)-th largest gradient
    - Preserves theoretical robustness guarantees
    """
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedALARCServer, self).__init__(args, global_data, data_dir, message_pool, device)
        
        # ========================================
        # ARC Configuration
        # ========================================
        self._setup_arc(args)
        
    def _setup_arc(self, args):
        """
        Setup ARC parameters with robust parsing.
        
        Args:
            args: Command-line arguments
        """
        # Parse use_arc (handle both bool and string from YAML)
        use_arc = getattr(args, 'use_arc', False)
        if isinstance(use_arc, str):
            self.use_arc = use_arc.lower() in ('true', '1', 'yes')
        else:
            self.use_arc = bool(use_arc)
        
        # Parse max_byzantine (f parameter in ARC paper)
        self.max_byzantine = int(getattr(args, 'max_byzantine', 1))
        
        if self.use_arc:
            print(f"╔{'═'*58}╗")
            print(f"║ FedALARC Server: ARC ENABLED (f={self.max_byzantine:2d})                      ║")
            print(f"║ Tolerating up to {self.max_byzantine} Byzantine worker(s)                    ║")
            print(f"╚{'═'*58}╝")
        else:
            print(f"FedALARC Server: ARC DISABLED (operating as standard FedALA)")
    
    def arc_clip(self, client_weights, f, n):
        """
        Adaptive Robust Clipping (ARC).
        
        Algorithm from "Adaptive Gradient Clipping for Robust Federated Learning" (ICLR 2025):
        1. Compute gradient g_i = w_i - w_global for each client i
        2. Compute norms ||g_i|| for all clients
        3. Sort norms in descending order
        4. Compute k = floor(2 * (f/n) * (n-f))
        5. Set clipping threshold C = ||g_{(k+1)}|| (norm of (k+1)-th largest)
        6. For each gradient: if ||g_i|| > C, scale to C
        
        Args:
            client_weights: List of client model weights (list of parameter lists)
            f: Maximum number of Byzantine workers to tolerate
            n: Total number of clients in this round
            
        Returns:
            List of clipped client weights
        """
        # Get global model weights for gradient computation
        global_weights = list(self.task.model.parameters())
        
        # ========================================
        # Step 1-2: Compute gradients and norms
        # ========================================
        gradients = []
        norms = []
        
        for client_weight in client_weights:
            grad = []
            total_norm_sq = 0.0
            
            for local_param, global_param in zip(client_weight, global_weights):
                # Gradient = local - global
                g = local_param.data - global_param.data
                grad.append(g)
                
                # Accumulate squared norm
                total_norm_sq += g.norm(2).item() ** 2
            
            gradients.append(grad)
            norms.append(total_norm_sq ** 0.5)
        
        # ========================================
        # Step 3: Sort by norm (descending)
        # ========================================
        sorted_indices = sorted(range(len(norms)), key=lambda i: norms[i], reverse=True)
        
        # ========================================
        # Step 4-5: Compute k and clipping threshold
        # ========================================
        k = int(2 * (f / n) * (n - f))
        
        if k >= len(norms) - 1:
            # Not enough clients to perform clipping
            # This happens when n is too small or f is too large
            print(f"  ⚠ ARC: Not enough clients for clipping (k={k}, n={n})")
            return client_weights
        
        # Clipping threshold C = norm of (k+1)-th largest gradient
        C = norms[sorted_indices[k]]
        
        # ========================================
        # Step 6: Clip gradients and reconstruct weights
        # ========================================
        clipped_weights = []
        num_clipped = 0
        
        for i, (client_weight, grad, norm) in enumerate(zip(client_weights, gradients, norms)):
            if norm <= C:
                # No clipping needed - gradient is within threshold
                clipped_weights.append(client_weight)
            else:
                # Clip: scale gradient to threshold, then add back to global
                scale = C / norm
                clipped_grad = [g * scale for g in grad]
                clipped_weight = [
                    global_param.data + cg 
                    for global_param, cg in zip(global_weights, clipped_grad)
                ]
                
                # Convert to Parameter objects
                clipped_weight = [
                    w if isinstance(w, torch.nn.Parameter) else torch.nn.Parameter(w)
                    for w in clipped_weight
                ]
                clipped_weights.append(clipped_weight)
                num_clipped += 1
        
        # Log clipping statistics
        if num_clipped > 0:
            max_norm = max(norms)
            min_norm = min(norms)
            print(f"  🔒 ARC: Clipped {num_clipped}/{n} clients")
            print(f"      Threshold C = {C:.4f}")
            print(f"      Norm range: [{min_norm:.4f}, {max_norm:.4f}]")
        else:
            print(f"  ✓ ARC: No clipping needed (all norms ≤ {C:.4f})")
        
        return clipped_weights
    
    def execute(self):
        """
        Server aggregation with optional ARC.
        
        Steps:
        1. Collect client weights from message pool
        2. Apply ARC if enabled (clips Byzantine gradients)
        3. Perform FedAvg aggregation (inherited from FedAvgServer)
        """
        # Get sampled clients for this round
        sampled_clients = self.message_pool.get("sampled_clients", [])
        
        if len(sampled_clients) == 0:
            print("⚠ Warning: No clients sampled for this round")
            return
        
        # ========================================
        # Apply ARC if enabled
        # ========================================
        if self.use_arc and len(sampled_clients) > self.max_byzantine:
            # Collect client weights
            client_weights = [
                self.message_pool[f"client_{client_id}"]["weight"]
                for client_id in sampled_clients
            ]
            
            # Apply ARC clipping
            clipped_weights = self.arc_clip(
                client_weights, 
                f=self.max_byzantine, 
                n=len(sampled_clients)
            )
            
            # Update message pool with clipped weights
            for i, client_id in enumerate(sampled_clients):
                self.message_pool[f"client_{client_id}"]["weight"] = clipped_weights[i]
        
        elif self.use_arc:
            print(f"  ⚠ ARC: Skipping (need > {self.max_byzantine} clients, got {len(sampled_clients)})")
        
        # ========================================
        # Standard FedAvg aggregation
        # ========================================
        super(FedALARCServer, self).execute()