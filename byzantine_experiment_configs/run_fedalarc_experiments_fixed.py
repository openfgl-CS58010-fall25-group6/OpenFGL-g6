"""
FIXED VERSION - Experiment Runner for FedALA-ARC Evaluation

Key fixes:
1. Stronger attacks (scaled sign-flip, Gaussian noise)
2. Use LAST round accuracy for fair comparison (not best round)
3. More local epochs to create larger gradient differences
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

def apply_universal_attack_patch(trainer, args, attack_type, byzantine_ids, attack_strength=1.0):
    """
    Injects attacks directly into the message pool before server execution.
    
    Args:
        trainer: The FGLTrainer object
        args: Arguments
        attack_type: Type of attack ('sign_flip', 'gaussian_noise', 'scaled_sign_flip', etc.)
        byzantine_ids: List of Byzantine client IDs
        attack_strength: Multiplier for attack intensity (default 1.0)
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

    original_server_execute = trainer.server.execute

    def attacked_execute():
        global _attack_call_count, _attack_applied_count
        _attack_call_count += 1
        
        pool = trainer.message_pool
        sampled_clients = pool.get("sampled_clients", [])
        active_attackers = [cid for cid in byzantine_ids if cid in sampled_clients]

        if active_attackers:
            _attack_applied_count += 1
            
            for client_id in active_attackers:
                client_msg = pool[f"client_{client_id}"]
                weights = client_msg["weight"]
                attacked_weights = []
                
                if attack_type == 'sign_flip':
                    # Standard sign flip: flip the gradient direction
                    global_weights = list(trainer.server.task.model.parameters())
                    for w_local, w_global in zip(weights, global_weights):
                        data_local = w_local.data if hasattr(w_local, 'data') else w_local
                        data_global = w_global.data if hasattr(w_global, 'data') else w_global
                        
                        # w_attacked = 2 * w_global - w_local
                        w_attacked = 2 * data_global - data_local
                        
                        if isinstance(w_local, torch.nn.Parameter):
                            w_attacked = torch.nn.Parameter(w_attacked)
                        attacked_weights.append(w_attacked)
                
                elif attack_type == 'scaled_sign_flip':
                    # STRONGER: Scale up the flipped gradient
                    global_weights = list(trainer.server.task.model.parameters())
                    for w_local, w_global in zip(weights, global_weights):
                        data_local = w_local.data if hasattr(w_local, 'data') else w_local
                        data_global = w_global.data if hasattr(w_global, 'data') else w_global
                        
                        # Compute gradient and flip with scaling
                        gradient = data_local - data_global
                        # w_attacked = w_global - scale * gradient (flip and amplify)
                        w_attacked = data_global - attack_strength * gradient
                        
                        if isinstance(w_local, torch.nn.Parameter):
                            w_attacked = torch.nn.Parameter(w_attacked)
                        attacked_weights.append(w_attacked)
                
                elif attack_type == 'gaussian_noise':
                    # Add Gaussian noise proportional to weight magnitude
                    for w_local in weights:
                        data_local = w_local.data if hasattr(w_local, 'data') else w_local
                        # Scale noise by weight std to be proportionally disruptive
                        noise_scale = attack_strength * data_local.std().item() * 10
                        noise = torch.randn_like(data_local) * max(noise_scale, 0.1)
                        w_attacked = data_local + noise
                        
                        if isinstance(w_local, torch.nn.Parameter):
                            w_attacked = torch.nn.Parameter(w_attacked)
                        attacked_weights.append(w_attacked)
                
                elif attack_type == 'label_flip_proxy':
                    # Simulate label flipping by sending very wrong gradients
                    # This sends gradients in opposite direction with large magnitude
                    global_weights = list(trainer.server.task.model.parameters())
                    for w_local, w_global in zip(weights, global_weights):
                        data_local = w_local.data if hasattr(w_local, 'data') else w_local
                        data_global = w_global.data if hasattr(w_global, 'data') else w_global
                        
                        gradient = data_local - data_global
                        # Flip and scale significantly
                        w_attacked = data_global - attack_strength * 5 * gradient
                        
                        if isinstance(w_local, torch.nn.Parameter):
                            w_attacked = torch.nn.Parameter(w_attacked)
                        attacked_weights.append(w_attacked)
                
                elif attack_type == 'random':
                    # Replace with random weights
                    for w_local in weights:
                        data_local = w_local.data if hasattr(w_local, 'data') else w_local
                        w_attacked = torch.randn_like(data_local) * attack_strength
                        
                        if isinstance(w_local, torch.nn.Parameter):
                            w_attacked = torch.nn.Parameter(w_attacked)
                        attacked_weights.append(w_attacked)
                
                else:
                    attacked_weights = weights

                pool[f"client_{client_id}"]["weight"] = attacked_weights

        original_server_execute()

    trainer.server.execute = attacked_execute
    print(f"  - Attack patch installed!")


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
    """Run a single experiment and return LAST round accuracy (not best)"""
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
    args.num_epochs = config.get('local_epochs', 3)  # MORE LOCAL EPOCHS for bigger gradients
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
    args.attack_type = config.get('attack_type', 'none')  # Internal attack disabled
    args.attack_params = json.dumps(config.get('attack_params', {}))
    
    # Logging
    args.debug = True
    args.log_root = config.get('log_root', './logs_fedalarc')
    args.log_name = f"{config['name']}_seed{seed}"
    
    print(f"\n{'='*60}")
    print(f"[EXP] {config['name']} | Seed: {seed}")
    print(f"  Algorithm: {args.fl_algorithm} | Attack: {config.get('external_attack_type', 'none')}")
    print(f"  Byzantine: {config.get('byzantine_ids', [])} | ARC: {args.use_arc}")
    print(f"{'='*60}")
    
    start_time = time.time()
    trainer = FGLTrainer(args)
    
    # Apply external attack
    real_attack_type = config.get('external_attack_type', 'none')
    byz_ids = config.get('byzantine_ids', [])
    attack_strength = config.get('attack_strength', 3.0)  # Default stronger attack
    
    apply_universal_attack_patch(trainer, args, real_attack_type, byz_ids, attack_strength)

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
    
    # Extract results - use LAST round, not best round for fair comparison
    results = trainer.evaluation_result
    metric_name = 'accuracy' if config['task'] in ['graph_cls', 'node_cls'] else 'mse'
    
    # Get last round accuracy (current) instead of best
    last_test_acc = results.get(f'curr_test_{metric_name}', results.get(f'best_test_{metric_name}', 0))
    best_test_acc = results.get(f'best_test_{metric_name}', 0)
    
    if metric_name == 'accuracy':
        if last_test_acc < 1:
            last_test_acc *= 100
        if best_test_acc < 1:
            best_test_acc *= 100
    
    print(f"\n[RESULT] Last Round Acc: {last_test_acc:.2f}% | Best Acc: {best_test_acc:.2f}%")
    
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
        result = run_single_experiment(config, seed)
        all_results.append(result)
    
    last_accs = [r['last_acc'] for r in all_results]
    best_accs = [r['best_acc'] for r in all_results]
    
    return {
        'name': config['name'],
        'mean_last_acc': np.mean(last_accs),
        'std_last_acc': np.std(last_accs),
        'mean_best_acc': np.mean(best_accs),
        'std_best_acc': np.std(best_accs),
        'config': config
    }


def get_byzantine_configs(dataset='PROTEINS'):
    """Generate configs for Byzantine robustness experiments"""
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
        'rounds': 30,  # Enough rounds to see degradation
        'local_epochs': 3,  # MORE LOCAL EPOCHS = bigger gradients = stronger attack effect
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
    
    # Attack scenarios - use SCALED sign flip for stronger effect
    attacks = [
        ('no_attack', [], 'none', 1.0),
        ('f2_attack', [0, 1], 'scaled_sign_flip', 3.0),  # Stronger attack
        ('f3_attack', [0, 1, 2], 'scaled_sign_flip', 3.0),
    ]
    
    # Methods
    methods = [
        ('FedAvg', 'fedavg', False, 0),
        ('FedALA', 'fedala', False, 0),
        ('FedALARC-f2', 'fedalarc', True, 2),
        ('FedALARC-f3', 'fedalarc', True, 3),
    ]
    
    for attack_name, byz_ids, attack_type, strength in attacks:
        for method_name, alg, use_arc, max_byz in methods:
            config = {
                **base,
                'name': f"{dataset}_{attack_name}_{method_name}",
                'algorithm': alg,
                'use_arc': use_arc,
                'max_byzantine': max_byz,
                'byzantine_ids': byz_ids,
                'external_attack_type': attack_type,
                'attack_strength': strength,
                'attack_type': 'none',  # Disable internal attack
            }
            configs.append(config)
            
    return configs


def format_byzantine_results(results):
    """Format results as table using LAST round accuracy"""
    df_data = []
    
    for result in results:
        cfg = result['config']
        
        real_attack = cfg.get('external_attack_type', 'none')
        num_attackers = len(cfg.get('byzantine_ids', []))
        
        if real_attack == 'none' or num_attackers == 0:
            attack_label = 'No Attack'
        else:
            attack_label = f"f={num_attackers}"
        
        method_name = cfg['name'].split('_')[-1]
        
        df_data.append({
            'Dataset': cfg['dataset'],
            'Attack': attack_label,
            'Method': method_name,
            'LastAcc': f"{result['mean_last_acc']:.2f}±{result['std_last_acc']:.2f}",
            'BestAcc': f"{result['mean_best_acc']:.2f}±{result['std_best_acc']:.2f}",
            'Mean': result['mean_last_acc'],
        })
    
    df = pd.DataFrame(df_data)
    
    # Pivot using Last Round Accuracy
    pivot = df.pivot_table(
        index=['Dataset', 'Method'],
        columns='Attack',
        values='LastAcc',
        aggfunc='first'
    )
    
    cols = list(pivot.columns)
    if 'No Attack' in cols:
        cols.remove('No Attack')
        cols = ['No Attack'] + sorted(cols)
        pivot = pivot[cols]
    
    return pivot


def main():
    parser = argparse.ArgumentParser(description='FedALA-ARC Experiments - FIXED VERSION')
    parser.add_argument('--dataset', type=str, default='PROTEINS')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--dry_run', action='store_true')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("BYZANTINE ROBUSTNESS EXPERIMENTS (FIXED)")
    print("  - Using LAST round accuracy (not best)")
    print("  - Using SCALED sign-flip attack (3x strength)")
    print("  - Using 3 local epochs (bigger gradients)")
    print("="*70)
    
    configs = get_byzantine_configs(args.dataset)
    
    if args.dry_run:
        print(f"\nWould run {len(configs)} experiments:")
        for cfg in configs:
            print(f"  - {cfg['name']}")
        return
    
    results = []
    for config in configs:
        result = run_experiment_with_seeds(config, args.seeds)
        results.append(result)
    
    table = format_byzantine_results(results)
    print("\n" + "="*70)
    print("RESULTS (Last Round Accuracy)")
    print("="*70)
    print(table.to_string())
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('results', exist_ok=True)
    with open(f'results/byzantine_fixed_{timestamp}.json', 'w') as f:
        json.dump([{
            'name': r['name'],
            'mean_last_acc': r['mean_last_acc'],
            'std_last_acc': r['std_last_acc'],
            'mean_best_acc': r['mean_best_acc'],
            'config': {k: v for k, v in r['config'].items() if k != 'byzantine_ids' or isinstance(v, list)}
        } for r in results], f, indent=2)


if __name__ == "__main__":
    main()
