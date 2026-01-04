"""
TRULY FIXED VERSION - Attack happens AFTER client.send_message()

The previous version had a timing bug:
1. Clients train → client.execute()
2. Clients upload weights → client.send_message()  
3. Attack patches server.execute() ← TOO LATE! Clients already uploaded clean weights
4. Server aggregates

This version:
1. Clients train → client.execute()
2. Clients upload weights → client.send_message()
3. ✅ ATTACK modifies uploaded weights in message_pool
4. Server aggregates (using attacked weights)
"""

import os
import sys
import copy
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random

# Global counter for debugging
_attack_call_count = 0
_attack_applied_count = 0

def apply_correct_attack_patch(trainer, args, attack_type, byzantine_ids, attack_strength=1.0):
    """
    Patches the TRAINING LOOP to inject attacks AFTER client.send_message().
    
    This is the correct timing: attack happens after clients upload their weights
    but before server aggregates them.
    """
    global _attack_call_count, _attack_applied_count
    _attack_call_count = 0
    _attack_applied_count = 0
    
    print(f"\n[ATTACK PATCH] Setting up attack:")
    print(f"  - attack_type: {attack_type}")
    print(f"  - byzantine_ids: {byzantine_ids}")
    print(f"  - attack_strength: {attack_strength}")
    
    if not byzantine_ids or attack_type == 'none':
        print(f"  - SKIPPING: No attack to apply")
        return

    # Store original train method
    original_train = trainer.train

    def attacked_train():
        """Modified training loop that injects attacks at the RIGHT time"""
        for round_id in range(trainer.args.num_rounds):
            sampled_clients = sorted(random.sample(
                list(range(trainer.args.num_clients)), 
                int(trainer.args.num_clients * trainer.args.client_frac)
            ))
            print(f"round # {round_id}\t\tsampled_clients: {sampled_clients}")
            trainer.message_pool["round"] = round_id
            trainer.message_pool["sampled_clients"] = sampled_clients
            
            # Step 1: Server sends global model to clients
            trainer.server.send_message()
            
            # Step 2 & 3: Clients train and upload their weights
            for client_id in sampled_clients:
                trainer.clients[client_id].execute()
                trainer.clients[client_id].send_message()
            
            # ✅ Step 3.5: ATTACK! Modify uploaded weights in message_pool
            global _attack_call_count, _attack_applied_count
            _attack_call_count += 1
            
            active_attackers = [cid for cid in byzantine_ids if cid in sampled_clients]
            
            if active_attackers:
                _attack_applied_count += 1
                
                # Print debug info for first 3 rounds
                if round_id < 3:
                    print(f"\n[ATTACK DEBUG Round {round_id + 1}]")
                    print(f"  sampled_clients: {sampled_clients}")
                    print(f"  byzantine_ids: {byzantine_ids}")
                    print(f"  active_attackers: {active_attackers}")
                
                for client_id in active_attackers:
                    client_msg = trainer.message_pool[f"client_{client_id}"]
                    weights = client_msg["weight"]
                    attacked_weights = []
                    
                    # Debug: show weight BEFORE attack
                    if round_id < 3 and len(weights) > 0:
                        w0 = weights[0].data if hasattr(weights[0], 'data') else weights[0]
                        print(f"  Client {client_id} weight[0] BEFORE: norm={w0.norm().item():.6f}, mean={w0.mean().item():.6f}")
                    
                    # Apply attack
                    if attack_type == 'sign_flip':
                        global_weights = list(trainer.server.task.model.parameters())
                        for w_local, w_global in zip(weights, global_weights):
                            data_local = w_local.data if hasattr(w_local, 'data') else w_local
                            data_global = w_global.data if hasattr(w_global, 'data') else w_global
                            w_attacked = 2 * data_global - data_local
                            
                            if isinstance(w_local, torch.nn.Parameter):
                                w_attacked = torch.nn.Parameter(w_attacked)
                            attacked_weights.append(w_attacked)
                    
                    elif attack_type == 'scaled_sign_flip':
                        global_weights = list(trainer.server.task.model.parameters())
                        for w_local, w_global in zip(weights, global_weights):
                            data_local = w_local.data if hasattr(w_local, 'data') else w_local
                            data_global = w_global.data if hasattr(w_global, 'data') else w_global
                            
                            gradient = data_local - data_global
                            w_attacked = data_global - attack_strength * gradient
                            
                            if isinstance(w_local, torch.nn.Parameter):
                                w_attacked = torch.nn.Parameter(w_attacked)
                            attacked_weights.append(w_attacked)
                    
                    elif attack_type == 'gaussian_noise':
                        for w_local in weights:
                            data_local = w_local.data if hasattr(w_local, 'data') else w_local
                            noise_scale = attack_strength * data_local.std().item() * 10
                            noise = torch.randn_like(data_local) * max(noise_scale, 0.1)
                            w_attacked = data_local + noise
                            
                            if isinstance(w_local, torch.nn.Parameter):
                                w_attacked = torch.nn.Parameter(w_attacked)
                            attacked_weights.append(w_attacked)
                    
                    elif attack_type == 'label_flip_proxy':
                        global_weights = list(trainer.server.task.model.parameters())
                        for w_local, w_global in zip(weights, global_weights):
                            data_local = w_local.data if hasattr(w_local, 'data') else w_local
                            data_global = w_global.data if hasattr(w_global, 'data') else w_global
                            
                            gradient = data_local - data_global
                            w_attacked = data_global - attack_strength * 5 * gradient
                            
                            if isinstance(w_local, torch.nn.Parameter):
                                w_attacked = torch.nn.Parameter(w_attacked)
                            attacked_weights.append(w_attacked)
                    
                    elif attack_type == 'random':
                        for w_local in weights:
                            data_local = w_local.data if hasattr(w_local, 'data') else w_local
                            w_attacked = torch.randn_like(data_local) * attack_strength
                            
                            if isinstance(w_local, torch.nn.Parameter):
                                w_attacked = torch.nn.Parameter(w_attacked)
                            attacked_weights.append(w_attacked)
                    
                    else:
                        attacked_weights = weights
                    
                    # Actually replace the weights in the pool
                    trainer.message_pool[f"client_{client_id}"]["weight"] = attacked_weights
                    
                    # Debug: verify attack worked
                    if round_id < 3 and len(attacked_weights) > 0:
                        w0_after = attacked_weights[0].data if hasattr(attacked_weights[0], 'data') else attacked_weights[0]
                        print(f"  Client {client_id} weight[0] AFTER:  norm={w0_after.norm().item():.6f}, mean={w0_after.mean().item():.6f}")
                        
                        # Verify it's actually in the pool
                        w0_verify = trainer.message_pool[f"client_{client_id}"]["weight"][0]
                        w0_verify_data = w0_verify.data if hasattr(w0_verify, 'data') else w0_verify
                        print(f"  Client {client_id} VERIFY in pool: norm={w0_verify_data.norm().item():.6f}")
            
            # Step 4: Server aggregates (using attacked weights from pool)
            trainer.server.execute()
            
            # Debug: show global model after aggregation
            if round_id < 3:
                global_w0 = list(trainer.server.task.model.parameters())[0]
                print(f"  Global weight[0] AFTER aggregation: norm={global_w0.data.norm().item():.6f}")
            
            # Step 5: Evaluate
            trainer.evaluate()
            print("-"*50)
        
        # ✅ SAVE LAST ROUND ACCURACY
        # Re-evaluate after training to get final accuracy
        trainer.evaluate()
        # Now the evaluation_result has been updated with the last round's "current" accuracy
        # We need to capture this BEFORE it gets lost
        metric_name = 'accuracy' if trainer.args.task in ['graph_cls', 'node_cls'] else 'mse'
        
        # Manually compute last round accuracy
        last_test_acc = 0
        tot_samples = 0
        for client_id in range(trainer.args.num_clients):
            result = trainer.clients[client_id].task.evaluate()
            num_samples = trainer.clients[client_id].task.num_samples
            last_test_acc += result[f'{metric_name}_test'] * num_samples
            tot_samples += num_samples
        
        last_test_acc = last_test_acc / tot_samples
        
        # Store in evaluation_result for later access
        trainer.evaluation_result[f'last_test_{metric_name}'] = last_test_acc
        
        trainer.logger.save()

    # Replace the train method
    trainer.train = attacked_train
    print(f"  - Attack patch installed on trainer.train()!")


def print_attack_summary():
    global _attack_call_count, _attack_applied_count
    print(f"\n[ATTACK SUMMARY] Rounds: {_attack_call_count}, Attacked: {_attack_applied_count}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_single_experiment(config, seed):
    """Run a single experiment and return LAST round accuracy"""
    import openfgl.config as openfgl_config
    from openfgl.flcore.trainer import FGLTrainer
    
    set_seed(seed)
    
    args = copy.deepcopy(openfgl_config.args)
    args.root = os.path.join(os.getcwd(), "data")
    args.seed = seed
    
    # Apply configuration
    dataset = config['dataset']
    args.dataset = [dataset] if isinstance(dataset, str) else dataset
    args.task = config['task']
    args.scenario = config['scenario']
    args.fl_algorithm = config['algorithm']
    model = config['model']
    args.model = [model] if isinstance(model, str) else model
    args.simulation_mode = config['simulation_mode']
    args.num_clients = config['num_clients']
    args.num_rounds = config['rounds']
    args.num_epochs = config.get('local_epochs', 3)
    args.batch_size = config['batch_size']
    args.lr = config['lr']
    args.weight_decay = config['weight_decay']
    args.dropout = config['dropout']
    args.optim = config.get('optimizer', 'adam')
    metrics = config.get('metrics', ['accuracy'])
    args.metrics = [metrics] if isinstance(metrics, str) else metrics
    args.evaluation_mode = config.get('evaluation_mode', 'local_model_on_local_data')
    
    args.dirichlet_alpha = config.get('dirichlet_alpha', 0.5)
    args.skew_alpha = config.get('skew_alpha', config.get('dirichlet_alpha', 0.5))
    args.louvain_resolution = config.get('louvain_resolution', 1.0)
    args.louvain_delta = config.get('louvain_delta', 20)
    args.metis_num_coms = config.get('metis_num_coms', 100)
    
    # FedALA parameters
    args.eta = config.get('eta', 1.0)
    args.layer_idx = config.get('layer_idx', 1)
    args.rand_percent = config.get('rand_percent', 80)
    args.threshold = config.get('threshold', 0.1)
    args.num_pre_loss = config.get('num_pre_loss', 10)
    
    # FedALA-ARC / Byzantine parameters
    args.use_arc = config.get('use_arc', False)
    args.max_byzantine = config.get('max_byzantine', 0)
    args.byzantine_ids = config.get('byzantine_ids', [])
    args.attack_type = config.get('attack_type', 'none')
    args.attack_params = json.dumps(config.get('attack_params', {}))
    
    # Logging
    args.debug = True
    args.log_root = config.get('log_root', './logs_fedalarc')
    args.log_name = f"{config['name']}_seed{seed}"
    
    print(f"\n{'#'*70}")
    print(f"# {config['name']} | Seed: {seed}")
    print(f"# Algorithm: {args.fl_algorithm} | Attack: {config.get('external_attack_type', 'none')}")
    print(f"{'#'*70}")
    
    start_time = time.time()
    trainer = FGLTrainer(args)
    
    # Apply attack with CORRECT timing
    real_attack_type = config.get('external_attack_type', 'none')
    byz_ids = config.get('byzantine_ids', [])
    attack_strength = config.get('attack_strength', 5.0)
    
    if real_attack_type != 'none':
        print(f"[ATTACK] Installing {real_attack_type} on clients {byz_ids} (strength={attack_strength})")
    else:
        print(f"[ATTACK] No attack configured")
    
    apply_correct_attack_patch(trainer, args, real_attack_type, byz_ids, attack_strength)

    # Patch torch.load
    original_torch_load = torch.load
    def patched_torch_load(*a, **kw):
        if "weights_only" not in kw:
            kw["weights_only"] = False
        return original_torch_load(*a, **kw)
    torch.load = patched_torch_load
    
    trainer.train()
    running_time = time.time() - start_time
    
    print_attack_summary()
    
    # Extract results
    results = trainer.evaluation_result
    metric_name = 'accuracy' if config['task'] in ['graph_cls', 'node_cls'] else 'mse'
    
    best_test_acc = results.get(f'best_test_{metric_name}', 0)
    last_test_acc = results.get(f'last_test_{metric_name}', best_test_acc)  # Use saved last round accuracy
    
    if metric_name == 'accuracy':
        if best_test_acc < 1:
            best_test_acc *= 100
        if last_test_acc < 1:
            last_test_acc *= 100
    
    print(f"\n[RESULT] Last Round: {last_test_acc:.2f}% | Best Round: {best_test_acc:.2f}%")
    
    return {
        'last_acc': last_test_acc,
        'best_acc': best_test_acc,
        'time': running_time,
        'seed': seed
    }


def run_experiment_with_seeds(config, seeds):
    """Run experiment with multiple seeds"""
    all_results = []
    
    for seed in seeds:
        try:
            result = run_single_experiment(config, seed)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR in seed {seed}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_results:
        return None
    
    # Aggregate results - focus on LAST round accuracy (where attack effects show)
    last_accs = [r['last_acc'] for r in all_results]
    best_accs = [r['best_acc'] for r in all_results]
    times = [r['time'] for r in all_results]
    
    return {
        'name': config['name'],
        'last_acc_mean': np.mean(last_accs),
        'last_acc_std': np.std(last_accs),
        'best_acc_mean': np.mean(best_accs),
        'best_acc_std': np.std(best_accs),
        'time_mean': np.mean(times),
        'seeds': seeds,
        'all_last_accs': last_accs,
        'all_best_accs': best_accs
    }


def main():
    """Main experiment runner - DEBUG VERSION (quick test)"""
    
    # Quick test with just 10 rounds, 2 experiments
    base_config = {
        'dataset': 'PROTEINS',
        'task': 'graph_cls',
        'scenario': 'graph_fl',
        'model': 'gin',
        'simulation_mode': 'graph_fl_label_skew',
        'num_clients': 10,
        'rounds': 10,  # Quick test
        'local_epochs': 3,
        'batch_size': 16,
        'lr': 0.01,
        'weight_decay': 5e-4,
        'dropout': 0.5,
        'metrics': 'accuracy',
        'evaluation_mode': 'local_model_on_local_data',
        'dirichlet_alpha': 0.5,
    }
    
    # Test scenarios
    experiments = [
        # Baseline (no attack)
        {
            **base_config,
            'name': 'PROTEINS_no_attack_FedAvg',
            'algorithm': 'fedavg',
            'external_attack_type': 'none',
            'byzantine_ids': [],
        },
        # FedAvg with f=2 attack
        {
            **base_config,
            'name': 'PROTEINS_f2_attack_FedAvg',
            'algorithm': 'fedavg',
            'external_attack_type': 'scaled_sign_flip',
            'byzantine_ids': [0, 1],
            'attack_strength': 5.0,
        },
    ]
    
    print("="*70)
    print("DEBUG: Verifying attack is working")
    print("="*70)
    
    seeds = [0]  # Single seed for debug
    results = []
    
    for config in experiments:
        result = run_experiment_with_seeds(config, seeds)
        if result:
            results.append(result)
    
    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON (LAST ROUND ACCURACY - where attack effects show)")
    print("="*70)
    for r in results:
        print(f"{r['name']}: {r['last_acc_mean']:.2f}%")
    print("="*70)


if __name__ == "__main__":
    main()
