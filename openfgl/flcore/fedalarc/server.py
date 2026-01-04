"""
FedALARC Server - Corrected Version

Implements Adaptive Robust Clipping (ARC) from ICLR 2025 paper.
Key corrections:
1. Handle edge case when k=0 (no clipping needed)
2. Better validation of inputs
3. More robust gradient computation
"""

import torch
import copy
from openfgl.flcore.fedavg.server import FedAvgServer


class FedALARCServer(FedAvgServer):
    """
    FedALARC Server: FedALA with Adaptive Robust Clipping (ARC).
    
    Key features:
    1. Standard FedAvg aggregation (inherited)
    2. Optional ARC pre-aggregation clipping for Byzantine robustness
    
    ARC Algorithm (from Algorithm 2 in paper):
    - Input: f (max Byzantine workers) and x_1, ..., x_n (client gradients)
    - Find permutation π such that ||x_π(1)|| ≥ ||x_π(2)|| ≥ ... ≥ ||x_π(n)||
    - Set k = floor(2 * (f/n) * (n-f))
    - Set C = ||x_π(k+1)|| (clipping threshold)
    - Output: Clip_C(x_1, ..., x_n)
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
    
    def _compute_gradient_norm(self, client_weight, global_weights):
        """
        Compute the gradient (update) and its L2 norm for a client.
        
        Gradient is defined as: g = w_local - w_global
        
        Args:
            client_weight: List of client model parameters
            global_weights: List of global model parameters
            
        Returns:
            Tuple of (gradient list, L2 norm)
        """
        grad = []
        total_norm_sq = 0.0
        
        for local_param, global_param in zip(client_weight, global_weights):
            # Handle both Parameter and Tensor types
            local_data = local_param.data if hasattr(local_param, 'data') else local_param
            global_data = global_param.data if hasattr(global_param, 'data') else global_param
            
            # Gradient = local - global (the update direction)
            g = local_data - global_data
            grad.append(g)
            
            # Accumulate squared norm
            total_norm_sq += g.norm(2).item() ** 2
        
        return grad, total_norm_sq ** 0.5
    
    def arc_clip(self, client_weights, f, n):
        """
        Adaptive Robust Clipping (ARC).
        
        Algorithm from "Adaptive Gradient Clipping for Robust Federated Learning" (ICLR 2025):
        1. Compute gradient g_i = w_i - w_global for each client i
        2. Compute norms ||g_i|| for all clients
        3. Sort norms in descending order to get permutation π
        4. Compute k = floor(2 * (f/n) * (n-f))
        5. Set clipping threshold C = ||g_{π(k+1)}|| (norm of (k+1)-th largest)
        6. For each gradient: if ||g_i|| > C, scale g_i to have norm C
        
        Args:
            client_weights: List of client model weights (list of parameter lists)
            f: Maximum number of Byzantine workers to tolerate
            n: Total number of clients in this round
            
        Returns:
            List of clipped client weights
        """
        # Validate inputs
        if n <= 0:
            print(f"  ⚠ ARC: Invalid n={n}, skipping clipping")
            return client_weights
        
        if f < 0:
            print(f"  ⚠ ARC: Invalid f={f}, skipping clipping")
            return client_weights
        
        if f >= n / 2:
            print(f"  ⚠ ARC: f={f} violates Byzantine assumption (f < n/2 where n={n})")
            print(f"       ARC cannot guarantee robustness, proceeding anyway...")
        
        # Get global model weights for gradient computation
        global_weights = list(self.task.model.parameters())
        
        # ========================================
        # Step 1-2: Compute gradients and norms
        # ========================================
        gradients = []
        norms = []
        
        for client_weight in client_weights:
            grad, norm = self._compute_gradient_norm(client_weight, global_weights)
            gradients.append(grad)
            norms.append(norm)
        
        # ========================================
        # Step 3: Sort by norm (descending) to get permutation π
        # ========================================
        # sorted_indices[i] gives the index of the (i+1)-th largest gradient
        sorted_indices = sorted(range(len(norms)), key=lambda i: norms[i], reverse=True)
        
        # ========================================
        # Step 4: Compute k = floor(2 * (f/n) * (n-f))
        # ========================================
        k = int(2 * (f / n) * (n - f))  # int() truncates toward zero = floor for positive
        
        # ========================================
        # Edge case handling
        # ========================================
        
        # Edge case 1: k = 0 means no clipping needed
        if k == 0:
            print(f"  ✓ ARC: k=0, no clipping needed (f/n ratio too small)")
            return client_weights
        
        # Edge case 2: k >= n means we can't determine a valid threshold
        if k >= n:
            print(f"  ⚠ ARC: k={k} >= n={n}, cannot compute threshold")
            print(f"       This happens when f is too large relative to n")
            return client_weights
        
        # ========================================
        # Step 5: Set clipping threshold C = ||g_{π(k+1)}||
        # ========================================
        # In 0-based indexing: sorted_indices[k] is the (k+1)-th largest
        threshold_idx = sorted_indices[k]
        C = norms[threshold_idx]
        
        # Handle edge case where C = 0 (all remaining gradients are zero)
        if C == 0:
            print(f"  ⚠ ARC: Threshold C=0, all gradients at or below index {k} are zero")
            # In this case, clip all non-zero gradients to zero
            C = 1e-10  # Small epsilon to avoid division by zero
        
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
                # Clip: scale gradient to have norm C, then add back to global
                # clip_C(g) = g * min(1, C / ||g||) = g * (C / ||g||) when ||g|| > C
                scale = C / norm
                clipped_grad = [g * scale for g in grad]
                
                # Reconstruct weights: w_clipped = w_global + clipped_grad
                clipped_weight = []
                for global_param, cg in zip(global_weights, clipped_grad):
                    global_data = global_param.data if hasattr(global_param, 'data') else global_param
                    new_weight = global_data + cg
                    
                    # Ensure output is a Parameter
                    if not isinstance(new_weight, torch.nn.Parameter):
                        new_weight = torch.nn.Parameter(new_weight.clone())
                    clipped_weight.append(new_weight)
                
                clipped_weights.append(clipped_weight)
                num_clipped += 1
        
        # ========================================
        # Log clipping statistics
        # ========================================
        if num_clipped > 0:
            max_norm = max(norms)
            min_norm = min(norms)
            avg_norm = sum(norms) / len(norms)
            print(f"  🔒 ARC Clipping Statistics:")
            print(f"      - Clipped: {num_clipped}/{n} clients")
            print(f"      - Threshold C = {C:.6f}")
            print(f"      - k = {k}")
            print(f"      - Norm range: [{min_norm:.6f}, {max_norm:.6f}]")
            print(f"      - Avg norm: {avg_norm:.6f}")
        else:
            print(f"  ✓ ARC: No clipping needed (all {n} norms ≤ {C:.6f})")
        
        return clipped_weights
    
    def execute(self):
        """
        Server aggregation with optional ARC.
        
        Steps:
        1. Collect client weights from message pool
        2. Apply ARC if enabled (clips potentially Byzantine gradients)
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
        if self.use_arc:
            n = len(sampled_clients)
            f = self.max_byzantine
            
            # Check if we have enough clients for ARC to be meaningful
            if n <= f:
                print(f"  ⚠ ARC: Skipping (need > {f} clients, got {n})")
            elif n <= 2 * f:
                print(f"  ⚠ ARC: Warning - n={n} ≤ 2f={2*f}, Byzantine assumption violated")
                print(f"       Proceeding with clipping but robustness not guaranteed")
                self._apply_arc_clipping(sampled_clients)
            else:
                self._apply_arc_clipping(sampled_clients)
        
        # ========================================
        # Standard FedAvg aggregation
        # ========================================
        super(FedALARCServer, self).execute()
    
    def _apply_arc_clipping(self, sampled_clients):
        """
        Helper method to apply ARC clipping to client weights.
        
        Args:
            sampled_clients: List of client IDs participating in this round
        """
        # Collect client weights
        client_weights = []
        for client_id in sampled_clients:
            weight = self.message_pool[f"client_{client_id}"]["weight"]
            client_weights.append(weight)
        
        # Apply ARC clipping
        clipped_weights = self.arc_clip(
            client_weights, 
            f=self.max_byzantine, 
            n=len(sampled_clients)
        )
        
        # Update message pool with clipped weights
        for i, client_id in enumerate(sampled_clients):
            self.message_pool[f"client_{client_id}"]["weight"] = clipped_weights[i]
