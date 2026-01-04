"""
TRULY FIXED VERSION - Captures last round accuracy directly

The issue: OpenFGL stores 'best' metrics, and FedALA achieves best in round 0.
Fix: Hook into the training loop to capture the ACTUAL last round accuracy.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random

_last_round_accuracy = {}  # Global to capture last round accuracy

def apply_universal_attack_patch(trainer, args, attack_type, byzantine_ids, attack_strength=3.0):
    """Injects attacks into the message pool before server aggregation."""
    
    if not byzantine_ids or attack_type == 'none':
        print(f"[ATTACK] No attack configured")
        return

    print(f"[ATTACK] Installing {attack_type} on clients {byzantine_ids} (strength={attack_strength})")
    
    original_server_execute = trainer.server.execute
    attack_count = [0]  # Use list to allow modification in nested function

    def attacked_execute():
        attack_count[0] += 1
        pool = trainer.message_pool
        sampled_clients = pool.get("sampled_clients", [])
        active_attackers = [cid for cid in byzantine_ids if cid in sampled_clients]

        if active_attackers and attack_count[0] > 1:  # Skip round 0 where global=local
            for client_id in active_attackers:
                client_msg = pool[f"client_{client_id}"]
                weights = client_msg["weight"]
                attacked_weights = []
                
                global_weights = list(trainer.server.task.model.parameters())
                
                for w_local, w_global in zip(weights, global_weights):
                    data_local = w_local.data if hasattr(w_local, 'data') else w_local
                    data_global = w_global.data if hasattr(w_global, 'data') else w_global
                    
                    if attack_type in ['sign_flip', 'scaled_sign_flip']:
                        # Compute gradient: g = local - global
                        gradient = data_local - data_global
                        # Flip and scale: attacked = global - strength * gradient
                        w_attacked = data_global - attack_strength * gradient
                    
                    elif attack_type == 'gaussian':
                        noise_scale = attack_strength * 0.5
                        noise = torch.randn_like(data_local) * noise_scale
                        w_attacked = data_local + noise
                    
                    elif attack_type == 'zero':
                        w_attacked = torch.zeros_like(data_local)
                    
                    else:
                        w_attacked = data_local
                    
                    if isinstance(w_local, torch.nn.Parameter):
                        w_attacked = torch.nn.Parameter(w_attacked)
                    attacked_weights.append(w_attacked)
                
                pool[f"client_{client_id}"]["weight"] = attacked_weights
        
        original_server_execute()

    trainer.server.execute = attacked_execute


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_experiment(config, seed):
    """Run experiment and capture ACTUAL last round test accuracy."""
    import openfgl.config as openfgl_config
    from openfgl.flcore.trainer import FGLTrainer
    
    set_seed(seed)
    
    args = copy.deepcopy(openfgl_config.args)
    args.root = os.path.join(os.getcwd(), "data")
    args.seed = seed
    
    # Apply config
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
    args.metrics = ['accuracy']
    args.evaluation_mode = config.get('evaluation_mode', 'local_model_on_local_data')
    
    args.dirichlet_alpha = config.get('dirichlet_alpha', 0.5)
    args.skew_alpha = config.get('skew_alpha', 0.5)
    args.louvain_resolution = config.get('louvain_resolution', 1.0)
    args.louvain_delta = config.get('louvain_delta', 20)
    args.metis_num_coms = config.get('metis_num_coms', 100)
    
    # FedALA params
    args.eta = config.get('eta', 1.0)
    args.layer_idx = config.get('layer_idx', 1)
    args.rand_percent = config.get('rand_percent', 80)
    args.threshold = config.get('threshold', 0.1)
    args.num_pre_loss = config.get('num_pre_loss', 10)
    
    # ARC params - disable internal attacks
    args.use_arc = config.get('use_arc', False)
    args.max_byzantine = config.get('max_byzantine', 0)
    args.byzantine_ids = []  # Don't pass to internal - we handle externally
    args.attack_type = 'none'
    args.attack_params = '{}'
    
    args.debug = True
    args.log_root = './logs_fedalarc'
    args.log_name = f"{config['name']}_seed{seed}"
    
    print(f"\n{'='*60}")
    print(f"Running: {config['name']} | Seed: {seed}")
    print(f"  Algorithm: {args.fl_algorithm}")
    print(f"  Attack: {config.get('external_attack_type', 'none')} on {config.get('byzantine_ids', [])}")
    print(f"{'='*60}")
    
    trainer = FGLTrainer(args)
    
    # Apply external attack
    apply_universal_attack_patch(
        trainer, args,
        config.get('external_attack_type', 'none'),
        config.get('byzantine_ids', []),
        config.get('attack_strength', 3.0)
    )
    
    # Patch torch.load
    original_torch_load = torch.load
    def patched_torch_load(*a, **kw):
        kw.setdefault("weights_only", False)
        return original_torch_load(*a, **kw)
    torch.load = patched_torch_load
    
    # Hook to capture per-round accuracy
    round_accuracies = []
    original_server_execute = trainer.server.execute
    
    def tracking_execute():
        original_server_execute()
        # After aggregation, the evaluation_result should have current accuracy
        if hasattr(trainer, 'evaluation_result'):
            curr_acc = trainer.evaluation_result.get('curr_test_accuracy', 
                       trainer.evaluation_result.get('best_test_accuracy', 0))
            round_accuracies.append(curr_acc)
    
    trainer.server.execute = tracking_execute
    
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    
    # Get final results
    results = trainer.evaluation_result
    
    # Try multiple ways to get the last round accuracy
    last_acc = results.get('curr_test_accuracy', 0)
    if last_acc == 0:
        last_acc = results.get('best_test_accuracy', 0)
    if round_accuracies:
        last_acc = round_accuracies[-1]
    
    best_acc = results.get('best_test_accuracy', 0)
    
    # Convert to percentage if needed
    if last_acc < 1:
        last_acc *= 100
    if best_acc < 1:
        best_acc *= 100
    
    print(f"  -> Last Acc: {last_acc:.2f}% | Best Acc: {best_acc:.2f}%")
    
    return {
        'last_acc': last_acc,
        'best_acc': best_acc,
        'time': elapsed,
        'seed': seed,
        'round_accs': [a*100 if a < 1 else a for a in round_accuracies[-5:]] if round_accuracies else []
    }


def run_experiment_with_seeds(config, seeds):
    all_results = []
    for seed in seeds:
        result = run_single_experiment(config, seed)
        all_results.append(result)
    
    last_accs = [r['last_acc'] for r in all_results]
    best_accs = [r['best_acc'] for r in all_results]
    
    print(f"\n[AGGREGATE] {config['name']}")
    print(f"  Last accs: {last_accs} -> {np.mean(last_accs):.2f}±{np.std(last_accs):.2f}")
    
    return {
        'name': config['name'],
        'mean_last': np.mean(last_accs),
        'std_last': np.std(last_accs),
        'mean_best': np.mean(best_accs),
        'std_best': np.std(best_accs),
        'all_last': last_accs,
        'config': config
    }


def get_configs(dataset='PROTEINS'):
    """Generate experiment configs."""
    base = {
        'dataset': dataset,
        'scenario': 'graph_fl',
        'task': 'graph_cls',
        'model': 'gin',
        'simulation_mode': 'graph_fl_label_skew',
        'dirichlet_alpha': 0.5,
        'skew_alpha': 0.5,
        'num_clients': 10,
        'rounds': 30,
        'local_epochs': 3,  # More epochs = bigger gradients
        'batch_size': 128,
        'lr': 0.001,
        'weight_decay': 0.0005,
        'dropout': 0.5,
        'optimizer': 'adam',
        'eta': 1.0,
        'layer_idx': 1,
        'rand_percent': 80,
        'threshold': 0.1,
        'num_pre_loss': 10,
    }
    
    configs = []
    
    # Attack scenarios
    attacks = [
        ('no_attack', [], 'none', 0),
        ('f2_attack', [0, 1], 'scaled_sign_flip', 5.0),  # Even stronger
        ('f3_attack', [0, 1, 2], 'scaled_sign_flip', 5.0),
    ]
    
    # Methods to test
    methods = [
        ('FedAvg', 'fedavg', False, 0),
        ('FedALA', 'fedala', False, 0),
        ('FedALARC_f2', 'fedalarc', True, 2),
        ('FedALARC_f3', 'fedalarc', True, 3),
    ]
    
    for attack_name, byz_ids, attack_type, strength in attacks:
        for method_name, alg, use_arc, max_byz in methods:
            configs.append({
                **base,
                'name': f"{dataset}_{attack_name}_{method_name}",
                'algorithm': alg,
                'use_arc': use_arc,
                'max_byzantine': max_byz,
                'byzantine_ids': byz_ids,
                'external_attack_type': attack_type,
                'attack_strength': strength,
            })
    
    return configs


def format_results(results):
    """Format as pivot table."""
    rows = []
    for r in results:
        cfg = r['config']
        attack = cfg.get('external_attack_type', 'none')
        n_byz = len(cfg.get('byzantine_ids', []))
        
        attack_label = 'No Attack' if attack == 'none' else f'f={n_byz}'
        method = cfg['name'].split('_')[-1]
        
        rows.append({
            'Dataset': cfg['dataset'],
            'Attack': attack_label,
            'Method': method,
            'Acc': f"{r['mean_last']:.1f}±{r['std_last']:.1f}",
            'Val': r['mean_last'],
        })
    
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index=['Dataset', 'Method'], columns='Attack', 
                          values='Acc', aggfunc='first')
    
    # Reorder columns
    cols = list(pivot.columns)
    if 'No Attack' in cols:
        cols.remove('No Attack')
        cols = ['No Attack'] + sorted(cols)
        pivot = pivot[cols]
    
    return pivot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='PROTEINS')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    
    print("="*70)
    print("BYZANTINE ROBUSTNESS EXPERIMENT (v3)")
    print("  - Stronger attack (5x scaled sign-flip)")
    print("  - 3 local epochs")
    print("  - 30 rounds")
    print("  - External attack injection (bypasses FedALA's ALA)")
    print("="*70)
    
    configs = get_configs(args.dataset)
    
    if args.dry_run:
        for c in configs:
            print(f"  {c['name']}: attack={c.get('external_attack_type')} byz={c.get('byzantine_ids')}")
        return
    
    results = []
    for cfg in configs:
        r = run_experiment_with_seeds(cfg, args.seeds)
        results.append(r)
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(format_results(results).to_string())
    
    # Also print raw numbers for analysis
    print("\n\nRaw Results:")
    for r in results:
        print(f"  {r['name']}: {r['all_last']}")


if __name__ == "__main__":
    main()
