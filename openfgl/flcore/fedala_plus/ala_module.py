import torch
import torch.nn as nn
import copy
import numpy as np
import time

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
                 num_pre_loss=10,
                 selection_frequency=1,  # NEW: How often to recompute disagreement
                 min_disagreement_samples=None):  # NEW: Minimum samples needed
        """
        FedALA Module for OpenFGL with enhanced disagreement sampling.
        
        Args:
            client_id: Client identifier
            task: Training task object
            device: Computation device
            batch_size: Batch size for training
            rand_percent: Percentage of samples to use (20-80)
            layer_idx: Which layers to apply ALA (0 = all)
            eta: Learning rate for weight updates
            threshold: Convergence threshold
            num_pre_loss: Number of losses to check for convergence
            selection_frequency: Recompute disagreement every N rounds (default: 1)
            min_disagreement_samples: Minimum samples needed, fallback to random if not met
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
        
        # NEW: Caching and fallback parameters
        self.selection_frequency = selection_frequency
        self.min_disagreement_samples = min_disagreement_samples or int(rand_percent * 0.5)  # Default: at least 50% of target
        
        self.weights = None 
        self.start_phase = True
        
        # NEW: Caching structures
        self.cached_selected_batches = None
        self.cached_disagreement_scores = None
        self.cache_round = -1  # Track which round the cache is from
        self.current_round = 0
        
        # NEW: Metrics tracking
        self.metrics = {
            'selection_time': [],
            'training_time': [],
            'convergence_epochs': [],
            'disagreement_stats': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'fallback_to_random': 0
        }

    def select_disagreement_samples(self, global_model_params, sample_percent=40):
        """
        Select samples where local and global models disagree.
        Implements caching and fallback mechanisms.
        """
        start_time = time.time()
        
        # **SOLUTION 1: Cache-based selection**
        if (self.cached_selected_batches is not None and 
            self.current_round - self.cache_round < self.selection_frequency):
            self.metrics['cache_hits'] += 1
            print(f"Client {self.cid}: Using cached disagreement samples from round {self.cache_round}")
            return self.cached_selected_batches, self.cached_disagreement_scores, True
        
        # Need to recompute
        self.metrics['cache_misses'] += 1
        print(f"Client {self.cid}: Computing disagreement samples for round {self.current_round}")
        
        # Get models
        model_l = self.task.model
        model_g = copy.deepcopy(model_l)
        
        # Load global parameters into model_g
        if isinstance(global_model_params, dict):
            params_g = list(global_model_params.values())
        elif isinstance(global_model_params, list):
            params_g = global_model_params
        else:
            params_g = list(global_model_params.parameters())
        
        # === DEBUG: Check parameter loading ===
        #print(f"Client {self.cid}: Loading {len(params_g)} parameter tensors into global model")
        
        with torch.no_grad():
            for idx, (param_g_model, param_g_data) in enumerate(zip(model_g.parameters(), params_g)):
                if param_g_model.shape != param_g_data.shape:
                    print(f"Client {self.cid}: WARNING - Shape mismatch at layer {idx}: "
                        f"{param_g_model.shape} vs {param_g_data.shape}")
                param_g_model.data.copy_(param_g_data.data)
        
        print(f"Client {self.cid}: Global model loaded successfully")
        
        model_l.eval()
        model_g.eval()
        
        # Get full training data
        if hasattr(self.task, 'train_dataloader') and self.task.train_dataloader is not None:
            full_loader = self.task.train_dataloader
        elif hasattr(self.task, 'processed_data') and self.task.processed_data is not None:
            full_loader = self.task.processed_data.get('train_dataloader')
        elif hasattr(self.task, 'splitted_data') and self.task.splitted_data is not None:
            full_loader = self.task.splitted_data.get('train_dataloader')
        else:
            raise ValueError("ALA: No data loader found.")
        
        # === DEBUG: Check data loader ===
        """try:
            total_batches = len(full_loader)
            print(f"Client {self.cid}: Data loader has {total_batches} batches")
        except:
            print(f"Client {self.cid}: Data loader length unknown (generator?)")"""
        
        disagreement_scores = []
        batch_indices = []
        batch_sizes = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(full_loader):
                batch = batch.to(self.device)
                
                # === DEBUG: First batch ===
                """if batch_idx == 0:
                    print(f"Client {self.cid}: First batch - {len(batch.y)} samples")"""
                
                # Get predictions from both models
                try:
                    output_l = model_l(batch)
                    output_g = model_g(batch)
                except Exception as e:
                    print(f"Client {self.cid}: ERROR in forward pass at batch {batch_idx}: {e}")
                    continue
                
                # Handle tuple outputs (embedding, logits)
                if isinstance(output_l, tuple):
                    _, logits_l = output_l
                else:
                    logits_l = output_l
                    
                if isinstance(output_g, tuple):
                    _, logits_g = output_g
                else:
                    logits_g = output_g
                
                # === DEBUG: Check logits ===
                """if batch_idx == 0:
                    print(f"Client {self.cid}: Logits shape - local: {logits_l.shape}, global: {logits_g.shape}")"""
                
                # Compute disagreement metrics
                pred_l = torch.argmax(logits_l, dim=1)
                pred_g = torch.argmax(logits_g, dim=1)
                label_disagreement = (pred_l != pred_g).float()
                
                # KL divergence
                prob_l = torch.softmax(logits_l, dim=1)
                prob_g = torch.softmax(logits_g, dim=1)
                kl_div = torch.sum(prob_l * torch.log((prob_l + 1e-10) / (prob_g + 1e-10)), dim=1)
                
                # Combined score
                disagreement = label_disagreement + 0.5 * kl_div
                
                # Average disagreement for this batch
                batch_disagreement = disagreement.mean().item()
                
                # === DEBUG: First batch disagreement ===
                """if batch_idx == 0:
                    print(f"Client {self.cid}: First batch disagreement score: {batch_disagreement:.4f}")
                    print(f"  Label disagreements: {label_disagreement.sum().item()}/{len(label_disagreement)}")
                    print(f"  Avg KL divergence: {kl_div.mean().item():.4f}")"""
                
                # Store batch-level information
                disagreement_scores.append(batch_disagreement)
                batch_indices.append(batch_idx)
                batch_sizes.append(len(batch.y))
        
        # === DEBUG: Summary statistics ===
        #print(f"Client {self.cid}: Processed {len(batch_indices)} batches total")
        
        if len(disagreement_scores) == 0:
            print(f"Client {self.cid}: ERROR - No disagreement scores computed!")
            return [], np.array([]), False
        
        # Convert to arrays
        disagreement_scores = np.array(disagreement_scores)
        batch_indices = np.array(batch_indices)
        batch_sizes = np.array(batch_sizes)
        
        """print(f"Client {self.cid}: Disagreement stats - "
            f"mean: {disagreement_scores.mean():.4f}, "
            f"std: {disagreement_scores.std():.4f}, "
            f"min: {disagreement_scores.min():.4f}, "
            f"max: {disagreement_scores.max():.4f}")"""
        
        # Calculate target number of samples
        total_samples = sum(batch_sizes)
        target_samples = int(total_samples * sample_percent / 100)
        
        #print(f"Client {self.cid}: Total samples: {total_samples}, Target: {target_samples} ({sample_percent}%)")
        
        # Sort batches by disagreement score (descending)
        sorted_indices = np.argsort(disagreement_scores)[::-1]
        
        #print(f"Client {self.cid}: Top 5 disagreement scores: {disagreement_scores[sorted_indices[:5]]}")
        #print(f"Client {self.cid}: Bottom 5 disagreement scores: {disagreement_scores[sorted_indices[-5:]]}")
        
        # Select batches until we reach target samples
        selected_batches = []
        selected_samples = 0
        
        for idx in sorted_indices:
            selected_batches.append(batch_indices[idx])
            selected_samples += batch_sizes[idx]
            if selected_samples >= target_samples:
                break
        
        # **SOLUTION 2: Fallback to random sampling**
        actual_percent = (selected_samples / total_samples) * 100
        min_percent_threshold = self.min_disagreement_samples
        
        print(f"Client {self.cid}: Selected {len(selected_batches)} batches "
            f"({selected_samples} samples = {actual_percent:.1f}%)")
        
        if actual_percent < min_percent_threshold:
            print(f"Client {self.cid}: Insufficient disagreement samples "
                f"({actual_percent:.1f}% < {min_percent_threshold}%). "
                f"Falling back to random sampling.")
            self.metrics['fallback_to_random'] += 1
            
            # Random sampling fallback
            all_batch_indices = list(range(len(batch_indices)))
            np.random.shuffle(all_batch_indices)
            
            selected_batches = []
            selected_samples = 0
            
            for idx in all_batch_indices:
                selected_batches.append(idx)
                selected_samples += batch_sizes[idx]
                if selected_samples >= target_samples:
                    break
            
            # Create fake disagreement scores for random selection
            disagreement_scores = np.random.rand(len(batch_indices))
            
            print(f"Client {self.cid}: Random fallback selected {len(selected_batches)} batches")
        
        # Update cache
        self.cached_selected_batches = selected_batches
        self.cached_disagreement_scores = disagreement_scores
        self.cache_round = self.current_round
        
        selection_time = time.time() - start_time
        self.metrics['selection_time'].append(selection_time)
        
        # Store disagreement statistics
        self.metrics['disagreement_stats'].append({
            'round': self.current_round,
            'mean_disagreement': np.mean(disagreement_scores),
            'std_disagreement': np.std(disagreement_scores),
            'max_disagreement': np.max(disagreement_scores),
            'selected_batches': len(selected_batches),
            'selected_samples': selected_samples,
            'selection_time': selection_time
        })
        
        print(f"Client {self.cid}: Disagreement selection completed in {selection_time:.2f}s")
        print(f"Client {self.cid}: Returning {len(selected_batches)} batches: {selected_batches[:10]}{'...' if len(selected_batches) > 10 else ''}")
        
        return selected_batches, disagreement_scores, False    

    """def select_disagreement_samples(self, global_model_params, sample_percent=40):
        """"""
        Select samples where local and global models disagree.
        Implements caching and fallback mechanisms.
        
        Args:
            global_model_params: Parameters from global model
            sample_percent: Percentage of samples to select
        
        Returns:
            selected_batches: List of batch indices to use
            disagreement_scores: Array of disagreement scores for all samples
            used_cache: Boolean indicating if cache was used
        """"""
        start_time = time.time()
        
        # **SOLUTION 1: Cache-based selection**
        # Check if we can use cached selection
        if (self.cached_selected_batches is not None and 
            self.current_round - self.cache_round < self.selection_frequency):
            self.metrics['cache_hits'] += 1
            print(f"Client {self.cid}: Using cached disagreement samples from round {self.cache_round}")
            return self.cached_selected_batches, self.cached_disagreement_scores, True
        
        # Need to recompute
        self.metrics['cache_misses'] += 1
        print(f"Client {self.cid}: Computing disagreement samples for round {self.current_round}")
        
        # Get models
        model_l = self.task.model
        model_g = copy.deepcopy(model_l)
        
        # Load global parameters into model_g
        if isinstance(global_model_params, dict):
            params_g = list(global_model_params.values())
        elif isinstance(global_model_params, list):
            params_g = global_model_params
        else:
            params_g = list(global_model_params.parameters())
        
        with torch.no_grad():
            for param_g_model, param_g_data in zip(model_g.parameters(), params_g):
                param_g_model.data.copy_(param_g_data.data)
        
        model_l.eval()
        model_g.eval()
        
        # Get full training data
        if hasattr(self.task, 'train_dataloader') and self.task.train_dataloader is not None:
            full_loader = self.task.train_dataloader
        elif hasattr(self.task, 'processed_data') and self.task.processed_data is not None:
            full_loader = self.task.processed_data.get('train_dataloader')
        elif hasattr(self.task, 'splitted_data') and self.task.splitted_data is not None:
            full_loader = self.task.splitted_data.get('train_dataloader')
        else:
            raise ValueError("ALA: No data loader found.")
        
        disagreement_scores = []
        batch_indices = []
        batch_sizes = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(full_loader):
                batch = batch.to(self.device)
                
                # Get predictions from both models
                output_l = model_l(batch)
                output_g = model_g(batch)
                
                # Handle tuple outputs (embedding, logits)
                if isinstance(output_l, tuple):
                    _, logits_l = output_l
                else:
                    logits_l = output_l
                    
                if isinstance(output_g, tuple):
                    _, logits_g = output_g
                else:
                    logits_g = output_g
                
                # Compute disagreement metrics
                # Option 1: Label disagreement (hard labels)
                pred_l = torch.argmax(logits_l, dim=1)
                pred_g = torch.argmax(logits_g, dim=1)
                label_disagreement = (pred_l != pred_g).float()
                
                # Option 2: Distribution disagreement (soft labels) - KL divergence
                prob_l = torch.softmax(logits_l, dim=1)
                prob_g = torch.softmax(logits_g, dim=1)
                kl_div = torch.sum(prob_l * torch.log((prob_l + 1e-10) / (prob_g + 1e-10)), dim=1)
                
                # Option 3: Combined score (you can tune the weighting)
                disagreement = label_disagreement + 0.5 * kl_div
                
                # Average disagreement for this batch
                batch_disagreement = disagreement.mean().item()
                
                # Store batch-level information
                disagreement_scores.append(batch_disagreement)
                batch_indices.append(batch_idx)
                batch_sizes.append(len(batch.y))
        
        # Convert to arrays
        disagreement_scores = np.array(disagreement_scores)
        batch_indices = np.array(batch_indices)
        batch_sizes = np.array(batch_sizes)
        
        # Calculate target number of samples
        total_samples = sum(batch_sizes)
        target_samples = int(total_samples * sample_percent / 100)
        
        # Sort batches by disagreement score (descending)
        sorted_indices = np.argsort(disagreement_scores)[::-1]
        
        # Select batches until we reach target samples
        selected_batches = []
        selected_samples = 0
        
        for idx in sorted_indices:
            selected_batches.append(batch_indices[idx])
            selected_samples += batch_sizes[idx]
            if selected_samples >= target_samples:
                break
        
        # **SOLUTION 2: Fallback to random sampling**
        # Check if we have enough disagreement samples
        actual_percent = (selected_samples / total_samples) * 100
        min_percent_threshold = self.min_disagreement_samples  # Treat as percentage
        
        if actual_percent < min_percent_threshold:
            print(f"Client {self.cid}: Insufficient disagreement samples "
                  f"({actual_percent:.1f}% < {min_percent_threshold}%). "
                  f"Falling back to random sampling.")
            self.metrics['fallback_to_random'] += 1
            
            # Random sampling fallback
            all_batch_indices = list(range(len(batch_indices)))
            np.random.shuffle(all_batch_indices)
            
            selected_batches = []
            selected_samples = 0
            
            for idx in all_batch_indices:
                selected_batches.append(idx)
                selected_samples += batch_sizes[idx]
                if selected_samples >= target_samples:
                    break
            
            # Create fake disagreement scores for random selection
            disagreement_scores = np.random.rand(len(batch_indices))
        
        # Update cache
        self.cached_selected_batches = selected_batches
        self.cached_disagreement_scores = disagreement_scores
        self.cache_round = self.current_round
        
        selection_time = time.time() - start_time
        self.metrics['selection_time'].append(selection_time)
        
        # Store disagreement statistics
        self.metrics['disagreement_stats'].append({
            'round': self.current_round,
            'mean_disagreement': np.mean(disagreement_scores),
            'std_disagreement': np.std(disagreement_scores),
            'max_disagreement': np.max(disagreement_scores),
            'selected_batches': len(selected_batches),
            'selected_samples': selected_samples,
            'selection_time': selection_time
        })
        
        print(f"Client {self.cid}: Selected {len(selected_batches)} batches "
              f"({selected_samples} samples, {actual_percent:.1f}%) in {selection_time:.2f}s")
        
        return selected_batches, disagreement_scores, False"""

    def adaptive_local_aggregation(self, global_model_params, use_disagreement=True):
        """
        Adaptive Local Aggregation with optional disagreement-based sampling.
        
        Args:
            global_model_params: Global model parameters
            use_disagreement: If True, use disagreement-based sampling
        """
        training_start_time = time.time()
        
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
        
        # Freeze temp model parameters
        for param in model_t.parameters():
            param.requires_grad = False

        # Select samples based on disagreement or random
        selected_batches = None
        used_cache = False
        
        if use_disagreement:
            selected_batches, disagreement_scores, used_cache = self.select_disagreement_samples(
                global_model_params, 
                sample_percent=self.rand_percent
            )
            
            # === CHECK: Handle empty selection ===
            if not selected_batches or len(selected_batches) == 0:
                print(f"Client {self.cid}: WARNING - No batches selected by disagreement! "
                      f"Falling back to random sampling.")
                selected_batches = None
                use_disagreement = False
            else:
                print(f"Client {self.cid}: Using {len(selected_batches)} batches selected by disagreement")
            
            # Store for external analysis
            self.disagreement_scores = disagreement_scores
        
        # Get data loader
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
            epoch_loss = 0.0
            batches_processed = 0
            
            for batch_idx, batch in enumerate(loader):
                # Filter batches based on selection method
                if use_disagreement and selected_batches is not None:
                    if batch_idx not in selected_batches:
                        continue
                elif not use_disagreement:
                    # Random sampling (original FedALA behavior)
                    if np.random.rand() > (self.rand_percent / 100):
                        continue
                
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

                # 3. Forward - OpenFGL models return (embedding, logits)
                output = model_t(batch)
                if isinstance(output, tuple):
                    embedding, logits = output
                else:
                    logits = output
                    embedding = None

                # 4. Loss - use task's loss function or default cross entropy
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
                
                # Accumulate loss
                epoch_loss += loss.item()
                batches_processed += 1

            # --- Loop Control ---
            if batches_processed == 0:
                print(f"Client {self.cid}: WARNING - No batches processed in epoch {cnt}! "
                      f"Selected batches: {len(selected_batches) if selected_batches else 'N/A'}")
                print(f"Client {self.cid}: Breaking ALA early due to no batches")
                break
            
            # Average loss for this epoch
            avg_epoch_loss = epoch_loss / batches_processed
            losses.append(avg_epoch_loss)
            cnt += 1

            if not self.start_phase:
                break
            
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                print(f'Client {self.cid} ALA converged at epoch {cnt}')
                self.metrics['convergence_epochs'].append(cnt)
                break
            
            if cnt > 50: 
                print(f'Client {self.cid} ALA reached max epochs (50)')
                self.metrics['convergence_epochs'].append(50)
                break

        self.start_phase = False

        # 8. Apply Final Learned Weights to REAL Local Model
        with torch.no_grad():
            for param_l, param_g, weight in zip(params_l, params_g, self.weights):
                param_l.data = param_l.data + (param_g.data - param_l.data) * weight
        
        training_time = time.time() - training_start_time
        self.metrics['training_time'].append(training_time)
        
        print(f"Client {self.cid}: ALA training completed in {training_time:.2f}s "
              f"(cache_used={used_cache}, batches/epoch={batches_processed})")
        
        # Increment round counter for caching logic
        self.current_round += 1
    
    def get_metrics_summary(self):
        """
        Get summary of metrics for analysis.
        
        Returns:
            Dictionary with metric summaries
        """
        summary = {
            'avg_selection_time': np.mean(self.metrics['selection_time']) if self.metrics['selection_time'] else 0,
            'avg_training_time': np.mean(self.metrics['training_time']) if self.metrics['training_time'] else 0,
            'avg_convergence_epochs': np.mean(self.metrics['convergence_epochs']) if self.metrics['convergence_epochs'] else 0,
            'cache_hits': self.metrics['cache_hits'],
            'cache_misses': self.metrics['cache_misses'],
            'fallback_to_random': self.metrics['fallback_to_random'],
            'cache_hit_rate': self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses']) if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0,
        }
        
        if self.metrics['disagreement_stats']:
            summary['avg_disagreement'] = np.mean([s['mean_disagreement'] for s in self.metrics['disagreement_stats']])
            summary['avg_selected_batches'] = np.mean([s['selected_batches'] for s in self.metrics['disagreement_stats']])
        
        return summary
    
    def reset_round_counter(self):
        """Reset the round counter (useful for experiments)"""
        self.current_round = 0
        self.cache_round = -1
        self.cached_selected_batches = None
        self.cached_disagreement_scores = None