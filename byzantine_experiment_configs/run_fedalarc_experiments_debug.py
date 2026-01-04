"""
DEBUG VERSION - Experiment Runner for FedALA-ARC Evaluation

This version includes extensive debugging to understand why attacks aren't working.
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

def apply_universal_attack_patch(trainer, args, attack_type, byzantine_ids):
    """
    Injects attacks directly into the message pool before server execution.
    Ensures identical attack logic for ALL algorithms (FedAvg, FedALA, FedALARC).
    """
    global _attack_call_count, _attack_applied_count
    _attack_call_count = 0
    _attack_applied_count = 0
    
    print(f"\n[DEBUG PATCH] Setting up attack patch:")
    print(f"  - attack_type: {attack_type}")
    print(f"  - byzantine_ids: {byzantine_ids}")
    print(f"  - algorithm: {args.fl_algorithm}")
    
    if not byzantine_ids or attack_type == 'none':
        print(f"  - SKIPPING: No attack to apply (byzantine_ids={byzantine_ids}, attack_type={attack_type})")
        return

    # Capture the original execution method
    original_server_execute = trainer.server.execute
    print(f"  - Original server.execute captured: {original_server_execute}")

    def attacked_execute():
        global _attack_call_count, _attack_applied_count
        _attack_call_count += 1
        
        # Access the message pool where clients deposited their weights
        pool = trainer.message_pool
        
        # DEBUG: Print pool keys
        if _attack_call_count <= 3:  # Only print first 3 rounds
            print(f"\n[DEBUG Round {_attack_call_count}] Pool keys: {list(pool.keys())}")
        
        # Identify which active clients are Byzantine
        sampled_clients = pool.get("sampled_clients", [])
        
        if _attack_call_count <= 3:
            print(f"  - sampled_clients: {sampled_clients}")
            print(f"  - byzantine_ids: {byzantine_ids}")
        
        active_attackers = [cid for cid in byzantine_ids if cid in sampled_clients]
        
        if _attack_call_count <= 3:
            print(f"  - active_attackers: {active_attackers}")

        if active_attackers:
            _attack_applied_count += 1
            
            if _attack_call_count <= 3:
                print(f"  - APPLYING ATTACK to {len(active_attackers)} clients!")
            
            for client_id in active_attackers:
                client_msg = pool[f"client_{client_id}"]
                weights = client_msg["weight"]
                
                if _attack_call_count <= 3:
                    print(f"    - Client {client_id}: {len(weights)} weight tensors")
                    if weights:
                        print(f"      - First weight shape: {weights[0].shape if hasattr(weights[0], 'shape') else 'N/A'}")
                        print(f"      - First weight norm BEFORE attack: {weights[0].data.norm().item() if hasattr(weights[0], 'data') else weights[0].norm().item():.4f}")
                
                attacked_weights = []
                
                if attack_type == 'sign_flip':
                    global_weights = list(trainer.server.task.model.parameters())
                    
                    for w_local, w_global in zip(weights, global_weights):
                        data_local = w_local.data if hasattr(w_local, 'data') else w_local
                        data_global = w_global.data if hasattr(w_global, 'data') else w_global
                        
                        # Apply Sign Flip: w_new = 2 * w_global - w_local
                        w_attacked = 2 * data_global - data_local
                        
                        if isinstance(w_local, torch.nn.Parameter):
                            w_attacked = torch.nn.Parameter(w_attacked)
                        attacked_weights.append(w_attacked)
                    
                    if _attack_call_count <= 3:
                        print(f"      - First weight norm AFTER attack: {attacked_weights[0].data.norm().item() if hasattr(attacked_weights[0], 'data') else attacked_weights[0].norm().item():.4f}")
                
                elif attack_type == 'gaussian_noise':
                    noise_scale = 10.0
                    for w_local in weights:
                        data_local = w_local.data if hasattr(w_local, 'data') else w_local
                        noise = torch.randn_like(data_local) * noise_scale
                        w_attacked = data_local + noise
                        if isinstance(w_local, torch.nn.Parameter):
                            w_attacked = torch.nn.Parameter(w_attacked)
                        attacked_weights.append(w_attacked)
                
                elif attack_type == 'zero':
                    for w_local in weights:
                        data_local = w_local.data if hasattr(w_local, 'data') else w_local
                        w_attacked = torch.zeros_like(data_local)
                        if isinstance(w_local, torch.nn.Parameter):
                            w_attacked = torch.nn.Parameter(w_attacked)
                        attacked_weights.append(w_attacked)
                
                elif attack_type == 'random':
                    for w_local in weights:
                        data_local = w_local.data if hasattr(w_local, 'data') else w_local
                        w_attacked = torch.randn_like(data_local)
                        if isinstance(w_local, torch.nn.Parameter):
                            w_attacked = torch.nn.Parameter(w_attacked)
                        attacked_weights.append(w_attacked)
                
                else:
                    attacked_weights = weights

                # Update the pool with the malicious weights
                pool[f"client_{client_id}"]["weight"] = attacked_weights
        else:
            if _attack_call_count <= 3:
                print(f"  - NO ATTACK this round (no overlap between sampled and byzantine)")

        # Run the actual server aggregation
        original_server_execute()

    # Apply the hook
    trainer.server.execute = attacked_execute
    print(f"  - Attack patch installed successfully!")


def print_attack_summary():
    """Print summary of attack application"""
    global _attack_call_count, _attack_applied_count
    print(f"\n[ATTACK SUMMARY]")
    print(f"  - Total server.execute calls: {_attack_call_count}")
    print(f"  - Rounds with attack applied: {_attack_applied_count}")
    print(f"  - Attack rate: {_attack_applied_count/_attack_call_count*100:.1f}%" if _attack_call_count > 0 else "  - No calls made")


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_single_experiment(config, seed):
    """
    Run a single experiment with given configuration and seed.
    """
    import openfgl.config as openfgl_config
    from openfgl.flcore.trainer import FGLTrainer
    
    set_seed(seed)
    
    # Create fresh args
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
    args.num_epochs = config.get('local_steps', 1)
    args.batch_size = config['batch_size']
    args.lr = config['lr']
    args.weight_decay = config['weight_decay']
    args.dropout = config['dropout']
    args.optim = config.get('optimizer', 'adam')
    metrics = config.get('metrics', ['accuracy'])
    args.metrics = [metrics] if isinstance(metrics, str) else metrics
    args.evaluation_mode = config.get('evaluation_mode', 'local_model_on_local_data')
    
    # Simulation-specific parameters
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
    
    print(f"\n{'='*70}")
    print(f"[DEBUG] Starting experiment: {config['name']}")
    print(f"  - Algorithm: {args.fl_algorithm}")
    print(f"  - External attack: {config.get('external_attack_type', 'none')}")
    print(f"  - Byzantine IDs: {config.get('byzantine_ids', [])}")
    print(f"  - use_arc: {args.use_arc}")
    print(f"  - max_byzantine: {args.max_byzantine}")
    print(f"{'='*70}")
    
    # Run training
    start_time = time.time()
    trainer = FGLTrainer(args)
    
    # --- external attack block ---
    real_attack_type = config.get('external_attack_type', 'none')
    byz_ids = config.get('byzantine_ids', [])
    
    apply_universal_attack_patch(trainer, args, real_attack_type, byz_ids)
    # ----------------------

    # Patch torch.load for PyTorch 2.6+ compatibility
    original_torch_load = torch.load
    def patched_torch_load(*a, **kw):
        if "weights_only" not in kw:
            kw["weights_only"] = False
        return original_torch_load(*a, **kw)
    torch.load = patched_torch_load
    
    trainer.train()
    running_time = time.time() - start_time
    
    # Print attack summary
    print_attack_summary()
    
    # Extract results
    results = trainer.evaluation_result
    metric_name = 'accuracy' if config['task'] in ['graph_cls', 'node_cls'] else 'mse'
    
    test_acc = results.get(f'best_test_{metric_name}', 0)
    val_acc = results.get(f'best_val_{metric_name}', 0)
    best_round = results.get('best_round', 0)
    
    if metric_name == 'accuracy':
        if test_acc < 1:
            test_acc *= 100
        if val_acc < 1:
            val_acc *= 100
    
    return {
        'test_acc': test_acc,
        'val_acc': val_acc,
        'best_round': best_round,
        'time': running_time,
        'seed': seed
    }


def run_experiment_with_seeds(config, seeds):
    """Run experiment with multiple seeds and aggregate results"""
    all_results = []
    
    for seed in seeds:
        print(f"\n{'#'*70}")
        print(f"# Running: {config['name']} | Seed: {seed}")
        print(f"{'#'*70}")
        
        result = run_single_experiment(config, seed)
        all_results.append(result)
        
        print(f"\n[RESULT] Test Acc: {result['test_acc']:.2f}%")
        print(f"[RESULT] Time: {result['time']:.2f}s")
    
    # Aggregate
    test_accs = [r['test_acc'] for r in all_results]
    
    return {
        'name': config['name'],
        'mean_acc': np.mean(test_accs),
        'std_acc': np.std(test_accs),
        'all_accs': test_accs,
        'config': config
    }


def get_byzantine_configs(dataset='PROTEINS'):
    """Generate configs for Byzantine robustness experiments - DEBUG VERSION with fewer experiments"""
    configs = []
    
    base = {
        'dataset': dataset,
        'scenario': 'graph_fl',
        'task': 'graph_cls',
        'model': 'gin',
        'simulation_mode': 'graph_fl_label_skew',
        'dirichlet_alpha': 0.5,
        'skew_alpha': 0.5,
        'num_clients': 10,
        'rounds': 10,  # REDUCED for debugging
        'local_steps': 1,
        'batch_size': 128,
        'lr': 0.001,
        'weight_decay': 0.0005,
        'dropout': 0.5,
        'optimizer': 'adam',
        'metrics': ['accuracy'],
        'eta': 1.0,
        'layer_idx': 1,
        'rand_percent': 80,
        'threshold': 0.1,
        'num_pre_loss': 10,
    }
    
    # DEBUG: Only test a few combinations
    attacks = [
        ('no_attack', 0, [], 'none'),
        ('f2_sign_flip', 2, [0, 1], 'sign_flip'),
    ]
    
    methods = [
        ('FedAvg', 'fedavg', False, 0),
        ('FedALA', 'fedala', False, 0),
        ('FedALARC-f2', 'fedalarc', True, 2),
    ]
    
    for attack_name, num_byz, byz_ids, attack_type in attacks:
        for method_name, alg, use_arc, max_byz in methods:
            config = {
                **base,
                'name': f"{dataset}_{attack_name}_{method_name}",
                'algorithm': alg,
                'use_arc': use_arc,
                'max_byzantine': max_byz,
                'byzantine_ids': byz_ids,
                'external_attack_type': attack_type, 
                'attack_type': 'none',
            }
            configs.append(config)
            
    return configs


def format_byzantine_results(results):
    """Format results as Byzantine robustness table"""
    df_data = []
    
    for result in results:
        cfg = result['config']
        
        real_attack = cfg.get('external_attack_type', 'none')
        num_attackers = len(cfg.get('byzantine_ids', []))
        
        if real_attack == 'none' or num_attackers == 0:
            attack_label = 'No Attack'
        else:
            attack_label = f"f={num_attackers} {real_attack}"
        
        method_name = cfg['name'].split('_')[-1]
        
        df_data.append({
            'Dataset': cfg['dataset'],
            'Attack': attack_label,
            'Method': method_name,
            'Accuracy': f"{result['mean_acc']:.2f}±{result['std_acc']:.2f}",
            'Mean': result['mean_acc'],
        })
    
    df = pd.DataFrame(df_data)
    
    pivot = df.pivot_table(
        index=['Dataset', 'Method'],
        columns='Attack',
        values='Accuracy',
        aggfunc='first'
    )
    
    cols = list(pivot.columns)
    if 'No Attack' in cols:
        cols.remove('No Attack')
        cols = ['No Attack'] + sorted(cols)
        pivot = pivot[cols]
    
    return pivot


def main():
    parser = argparse.ArgumentParser(description='FedALA-ARC Experiments - DEBUG VERSION')
    parser.add_argument('--dataset', type=str, default='PROTEINS')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])  # Single seed for debugging
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("DEBUG: BYZANTINE ROBUSTNESS EXPERIMENTS")
    print("="*70)
    
    configs = get_byzantine_configs(args.dataset)
    
    print(f"\nWill run {len(configs)} experiments:")
    for cfg in configs:
        ext_attack = cfg.get('external_attack_type', 'none')
        byz_ids = cfg.get('byzantine_ids', [])
        print(f"  - {cfg['name']} | attack={ext_attack} | byzantine_ids={byz_ids}")
    
    byzantine_results = []
    for config in configs:
        result = run_experiment_with_seeds(config, args.seeds)
        byzantine_results.append(result)
    
    table = format_byzantine_results(byzantine_results)
    print("\n" + "="*70)
    print("BYZANTINE ROBUSTNESS RESULTS")
    print("="*70)
    print(table.to_string())


if __name__ == "__main__":
    main()